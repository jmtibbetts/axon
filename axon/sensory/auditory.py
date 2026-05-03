"""
AXON — Auditory System
Microphone → VAD → Whisper STT → phoneme/word neurons.
Uses sounddevice for capture, openai-whisper for transcription.
"""

import threading
import time
import queue
import numpy as np
from typing import Callable


SAMPLE_RATE  = 16000
CHUNK        = 1024
SILENCE_DB   = -35       # dBFS threshold for voice activity
MIN_DURATION = 0.5       # seconds of speech before transcribing
MAX_DURATION = 15.0      # max recording length


class AuditorySystem:
    def __init__(self, on_speech: Callable, on_volume: Callable,
                 device_index: int = None):
        self.on_speech  = on_speech    # callback(text: str, confidence: float)
        self.on_volume  = on_volume    # callback(db: float)
        self.device_idx = device_index
        self.running    = False
        self._thread    = None
        self._whisper   = None
        self._audio_q   = queue.Queue()
        self.listening  = False
        self.volume_db  = -60.0

    @staticmethod
    def list_devices() -> list:
        """Return all input audio devices available on this system."""
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

    def _load_whisper(self):
        import whisper
        print("  [Auditory] Loading Whisper tiny.en model...")
        self._whisper = whisper.load_model("tiny.en")
        print("  [Auditory] Whisper ready.")

    def start(self):
        threading.Thread(target=self._load_whisper, daemon=True).start()
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _rms_to_db(self, samples: np.ndarray) -> float:
        rms = np.sqrt(np.mean(samples.astype(np.float32)**2))
        if rms < 1e-10:
            return -100.0
        return 20 * np.log10(rms / 32768.0)

    def _loop(self):
        try:
            import sounddevice as sd
        except ImportError:
            print("  [Auditory] sounddevice not installed — mic disabled.")
            return

        buffer    = []
        recording = False
        silence_t = 0.0
        rec_start = 0.0

        def callback(indata, frames, time_info, status):
            self._audio_q.put(indata.copy())

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
                        if silent_for > 0.8 or rec_dur > MAX_DURATION:
                            if rec_dur >= MIN_DURATION:
                                self._transcribe(buffer)
                            recording = False
                            buffer    = []
                            self.listening = False
        except Exception as e:
            print(f"  [Auditory] Error: {e}")

    def _transcribe(self, chunks: list):
        if self._whisper is None:
            return
        audio = np.concatenate(chunks).astype(np.float32) / 32768.0
        try:
            result = self._whisper.transcribe(audio, language='en', fp16=False)
            text   = result['text'].strip()
            if text and len(text) > 2:
                self.on_speech(text, 0.9)
        except Exception as e:
            print(f"  [Auditory] Transcription error: {e}")

    def get_status(self) -> dict:
        return {
            "running":   self.running,
            "listening": self.listening,
            "volume_db": round(self.volume_db, 1),
            "whisper":   self._whisper is not None,
        }
