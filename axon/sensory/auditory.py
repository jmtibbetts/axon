"""
AXON — Auditory System
Microphone → VAD → Whisper STT → on_speech callback.

Key fixes:
  - Speaking lockout: mic ignores audio while Axon is talking (prevents self-hearing)
  - Whisper hallucination filter: rejects low-avg-logprob or known phantom phrases
  - Conservative VAD: -28 dBFS threshold, 1.5s silence window, 1.0s min speech
"""

import threading
import time
import queue
import re
import numpy as np
from typing import Callable


SAMPLE_RATE   = 16000
CHUNK         = 1024
SILENCE_DB    = -40        # lowered: catches normal speech at mic distance
MIN_DURATION  = 0.5        # min speech length before transcribing
MAX_DURATION  = 20.0
SILENCE_AFTER = 1.2        # silence gap to end utterance
LOCKOUT_PAD   = 0.6        # extra seconds to stay locked after voice stops

# Whisper sometimes hallucinates these on near-silence or room noise
HALLUCINATION_PATTERNS = re.compile(
    r'^\s*(thank\s+you\.?|thanks\.?|you\.?|okay\.?|ok\.?|'
    r'bye\.?|bye[-\s]bye\.?|'
    r'[\.\,\!\?\s\-]+|'          # only punctuation
    r'\[.*?\]|'                   # [BLANK_AUDIO] etc.
    r'\.{2,})\s*$',
    re.IGNORECASE
)

MIN_WORD_COUNT = 2    # must have at least 2 real words
MIN_AVG_LOGPROB = -1.2   # Whisper confidence gate (below = hallucination)


class AuditorySystem:
    def __init__(self, on_speech: Callable, on_volume: Callable,
                 device_index: int = None, on_audio_chunk: Callable = None):
        self.on_speech      = on_speech
        self.on_volume      = on_volume
        self.device_idx     = device_index
        self.on_audio_chunk = on_audio_chunk   # optional: receives float32 numpy chunk
        self.running        = False
        self._thread        = None
        self._whisper       = None
        self._audio_q       = queue.Queue()
        self.listening      = False
        self.volume_db      = -60.0
        self._device        = "cpu"

        # Speaking lockout — set True while Axon's voice is playing
        self._speaking_lockout = False
        self._lockout_until    = 0.0   # timestamp when lockout expires

    # ── Public: called by engine when voice starts/stops ──────────────────────

    def set_speaking(self, is_speaking: bool):
        """Call this when Axon starts/stops talking so the mic ignores its voice."""
        self._speaking_lockout = is_speaking
        if not is_speaking:
            # Keep lockout active for LOCKOUT_PAD seconds after voice ends
            self._lockout_until = time.time() + LOCKOUT_PAD

    @property
    def _is_locked_out(self) -> bool:
        if self._speaking_lockout:
            return True
        return time.time() < self._lockout_until

    # ── Device listing ─────────────────────────────────────────────────────────

    @staticmethod
    def list_devices() -> list:
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            results = []
            for i, d in enumerate(devices):
                if d["max_input_channels"] > 0:
                    results.append({
                        "index":    i,
                        "name":     d["name"],
                        "channels": d["max_input_channels"],
                        "label":    f"[{i}] {d['name']} ({d['max_input_channels']}ch)",
                    })
            return results
        except Exception as e:
            print(f"  [Auditory] list_devices error: {e}")
            return []

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_whisper(self):
        import whisper, torch
        # Respect installer's device choice (AXON_DEVICE env or gpu_config.json)
        import os, json
        from pathlib import Path
        _env_dev = os.environ.get("AXON_DEVICE", "").lower()
        if _env_dev in ("cuda", "mps", "cpu"):
            _auto = _env_dev
        else:
            _cfg = Path(__file__).parents[2] / "data" / "gpu_config.json"
            _auto = json.loads(_cfg.read_text()).get("gpu_type", "auto") if _cfg.exists() else "auto"

        if _auto == "cpu":
            self._device = "cpu"
        elif _auto == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            model_name = "medium.en"
            print(f"  [Auditory] CUDA ({gpu_name}) — loading Whisper {model_name} on GPU...")
        else:
            self._device = "cpu"
            model_name = "tiny.en"
            print(f"  [Auditory] CPU — loading Whisper {model_name}...")
        self._whisper = whisper.load_model(model_name, device=self._device)
        print(f"  [Auditory] Whisper {model_name} ready on {self._device.upper()}.")

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self):
        threading.Thread(target=self._load_whisper, daemon=True).start()
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    # ── Audio loop ─────────────────────────────────────────────────────────────

    def _rms_to_db(self, samples: np.ndarray) -> float:
        rms = np.sqrt(np.mean(samples.astype(np.float32)**2))
        if rms < 1e-10:
            return -100.0
        return 20 * np.log10(rms / 32768.0)

    def _loop(self):
        try:
            import sounddevice as sd
        except (ImportError, OSError):
            print("  [Auditory] sounddevice not installed — mic disabled.")
            return

        buffer    = []
        recording = False
        silence_t = 0.0
        rec_start = 0.0

        def callback(indata, frames, time_info, status):
            self._audio_q.put(indata.copy())
            # Share raw float32 samples with audio emotion detector (single stream)
            if self.on_audio_chunk is not None:
                try:
                    mono = indata[:, 0].astype(np.float32) / 32768.0
                    self.on_audio_chunk(mono)
                except Exception:
                    pass

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                dtype='int16', blocksize=CHUNK,
                                device=self.device_idx, callback=callback):
                while self.running:
                    try:
                        chunk = self._audio_q.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    samples = chunk[:, 0]
                    db      = self._rms_to_db(samples)
                    self.volume_db = db
                    self.on_volume(db)

                    # ── Speaking lockout: drop audio while Axon is talking ─────
                    if self._is_locked_out:
                        # Reset any partial recording so it doesn't get confused
                        if recording:
                            recording = False
                            buffer    = []
                            self.listening = False
                        continue

                    is_voice = db > SILENCE_DB
                    now      = time.time()

                    if is_voice and not recording:
                        recording = True
                        rec_start = now
                        buffer    = [samples]
                        silence_t = now
                        self.listening = True

                    elif is_voice and recording:
                        buffer.append(samples)
                        silence_t = now

                    elif not is_voice and recording:
                        buffer.append(samples)
                        silent_for = now - silence_t
                        rec_dur    = now - rec_start

                        if silent_for > SILENCE_AFTER or rec_dur > MAX_DURATION:
                            if rec_dur >= MIN_DURATION:
                                self._transcribe(buffer)
                            recording = False
                            buffer    = []
                            self.listening = False

        except Exception as e:
            print(f"  [Auditory] Error: {e}")

    # ── Transcription + hallucination filter ───────────────────────────────────

    def _transcribe(self, chunks: list):
        if self._whisper is None:
            return

        audio = np.concatenate(chunks).astype(np.float32) / 32768.0
        try:
            use_fp16 = (self._device == "cuda")
            result = self._whisper.transcribe(
                audio,
                language='en',
                fp16=use_fp16,
                condition_on_previous_text=False,   # prevents "thank you" looping
                no_speech_threshold=0.6,             # discard if probably silence
                logprob_threshold=-1.0,              # discard low-confidence segments
            )

            text = result.get('text', '').strip()
            if not text:
                return

            # ── Hallucination filter 1: known phantom phrases ─────────────────
            if HALLUCINATION_PATTERNS.match(text):
                print(f"  [Auditory] Filtered hallucination: {repr(text)}")
                return

            # ── Hallucination filter 2: word count ────────────────────────────
            words = [w for w in re.split(r'\s+', text) if re.search(r'[a-zA-Z]', w)]
            if len(words) < MIN_WORD_COUNT:
                print(f"  [Auditory] Too short ({len(words)} words): {repr(text)}")
                return

            # ── Hallucination filter 3: Whisper avg logprob ───────────────────
            segments = result.get('segments', [])
            if segments:
                avg_logprob = sum(s.get('avg_logprob', 0) for s in segments) / len(segments)
                if avg_logprob < MIN_AVG_LOGPROB:
                    print(f"  [Auditory] Low confidence (logprob={avg_logprob:.2f}): {repr(text)}")
                    return

            print(f"  [Auditory] Heard: {text}")
            self.on_speech(text, 0.9)

        except Exception as e:
            print(f"  [Auditory] Transcription error: {e}")

    # ── Status ─────────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "running":   self.running,
            "listening": self.listening,
            "volume_db": round(self.volume_db, 1),
            "whisper":   self._whisper is not None,
            "locked_out": self._is_locked_out,
        }
