"""
AXON — Audio Emotion Detector
Real-time voice prosody analysis: extracts emotional signals from mic audio
without sending any audio to an external service.

Features extracted per 1-second window:
  - Energy (RMS)           → arousal / excitement level
  - Zero-crossing rate     → consonant density / speech texture
  - Pitch (F0 via YIN)     → stress / happiness indicator
  - Pitch variance         → emotional expressiveness vs monotone
  - Speaking rate proxy    → energy event count
  - Spectral centroid       → brightness / tension

State outputs (emitted via callback every 1s):
  {
    "audio_emotion":    "calm" | "stressed" | "excited" | "sad" | "neutral",
    "arousal":          0.0–1.0,
    "valence":          -1.0–1.0,
    "pitch_hz":         float,
    "energy":           float,
    "speaking":         bool,
    "confidence":       float,
  }
"""

import threading
import time
import queue
import numpy as np
from typing import Callable, Optional

# ── Optional librosa import ───────────────────────────────────────────────────
_LIBROSA_OK = False
try:
    import librosa
    _LIBROSA_OK = True
    print("  [AudioEmo] librosa loaded — full prosody analysis")
except ImportError:
    print("  [AudioEmo] librosa not available — using basic energy/ZCR analysis")

try:
    import sounddevice as sd
    _SD_OK = True
except ImportError:
    _SD_OK = False
    print("  [AudioEmo] sounddevice not available")

SAMPLE_RATE    = 16000
WINDOW_SEC     = 1.0     # analysis window
HOP_SEC        = 0.5     # update every 0.5s (50% overlap)
SILENCE_THRESH = 0.008   # RMS below this = silence

# ── Feature → emotion mapping ─────────────────────────────────────────────────
# Simple rule-based classifier. Good enough for real-time feedback.
# Returns (emotion_label, arousal, valence, confidence)

def _classify(energy: float, zcr: float, pitch: float,
              pitch_var: float, centroid: float,
              is_speaking: bool) -> tuple:
    if not is_speaking:
        return ("neutral", 0.05, 0.0, 0.7)

    arousal = min(1.0, energy * 14.0 + zcr * 0.4 + (centroid / 4000.0) * 0.3)
    arousal = round(float(arousal), 3)

    # Pitch heuristics (F0 in Hz, typical speech range 85–255)
    # Higher pitch → more positive / excited
    # Very low, monotone → sad / tired
    pitch_score = 0.0
    if pitch > 0:
        # Normalise: 80–300 Hz range → -1 to 1
        pitch_score = (pitch - 180.0) / 120.0
        pitch_score = max(-1.0, min(1.0, pitch_score))

    valence = round(float(pitch_score * 0.6 + (arousal - 0.4) * 0.4), 3)
    valence = max(-1.0, min(1.0, valence))

    # Rule-based label
    if arousal > 0.65 and valence > 0.1:
        label = "excited"
        conf  = 0.70
    elif arousal > 0.65 and valence <= 0.0:
        label = "stressed"
        conf  = 0.68
    elif arousal < 0.25 and valence < -0.1:
        label = "sad"
        conf  = 0.60
    elif arousal < 0.30:
        label = "calm"
        conf  = 0.65
    elif pitch_var > 25 and arousal > 0.35:
        label = "excited"
        conf  = 0.60
    else:
        label = "neutral"
        conf  = 0.55

    return (label, arousal, valence, conf)


# ── Emoji map ─────────────────────────────────────────────────────────────────
AUDIO_EMO_EMOJI = {
    "excited":  "🤩",
    "stressed": "😤",
    "calm":     "😌",
    "sad":      "😢",
    "neutral":  "😐",
}


class AudioEmotionDetector:
    def __init__(self, on_emotion: Callable, device_index: int = None):
        """
        on_emotion — callback(dict) called every ~0.5s with audio emotion state
        """
        self.on_emotion   = on_emotion
        self.device_index = device_index
        self.running      = False
        self._thread      = None
        self._buf         = np.zeros(0, dtype=np.float32)
        self._win_samples = int(SAMPLE_RATE * WINDOW_SEC)
        self._hop_samples = int(SAMPLE_RATE * HOP_SEC)
        self._last_state  = {}
        self._smoothed_arousal = 0.0
        self._smoothed_valence = 0.0

    def start(self):
        if not _SD_OK:
            print("  [AudioEmo] Cannot start — sounddevice unavailable")
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("  [AudioEmo] Audio emotion detector started")

    def stop(self):
        self.running = False

    def _loop(self):
        """Continuously read mic via sounddevice InputStream and analyse."""
        chunk = int(SAMPLE_RATE * 0.1)  # 100ms chunks

        def _audio_cb(indata, frames, time_info, status):
            if not self.running:
                return
            mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            self._buf = np.concatenate([self._buf, mono])

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32',
                blocksize=chunk,
                device=self.device_index,
                callback=_audio_cb,
            ):
                t_last = time.time()
                while self.running:
                    time.sleep(0.05)
                    # Analyse every hop interval
                    if time.time() - t_last >= HOP_SEC and len(self._buf) >= self._win_samples:
                        window = self._buf[-self._win_samples:]
                        self._buf = self._buf[-self._win_samples:]  # keep last window
                        t_last = time.time()
                        self._analyse(window)
        except Exception as e:
            print(f"  [AudioEmo] Stream error: {e}")

    def _analyse(self, window: np.ndarray):
        try:
            energy = float(np.sqrt(np.mean(window ** 2)))
            is_speaking = energy > SILENCE_THRESH
            zcr = float(np.mean(np.abs(np.diff(np.sign(window)))) * 0.5)

            pitch    = 0.0
            pitch_var = 0.0
            centroid  = 1500.0  # default mid-range

            if _LIBROSA_OK and is_speaking:
                # Pitch via pyin (more robust than yin for speech)
                try:
                    f0, voiced, _ = librosa.pyin(
                        window,
                        fmin=librosa.note_to_hz('C2'),
                        fmax=librosa.note_to_hz('C7'),
                        sr=SAMPLE_RATE,
                        frame_length=512,
                    )
                    voiced_f0 = f0[voiced == True] if f0 is not None else np.array([])
                    if len(voiced_f0) > 0:
                        pitch     = float(np.median(voiced_f0))
                        pitch_var = float(np.std(voiced_f0))
                except Exception:
                    pass

                # Spectral centroid
                try:
                    spec  = np.abs(np.fft.rfft(window))
                    freqs = np.fft.rfftfreq(len(window), 1.0 / SAMPLE_RATE)
                    total = spec.sum()
                    if total > 0:
                        centroid = float(np.dot(freqs, spec) / total)
                except Exception:
                    pass
            elif not _LIBROSA_OK and is_speaking:
                # Naive pitch: zero-crossing based (rough but fast)
                zc_count = np.sum(np.abs(np.diff(np.sign(window))) > 0)
                pitch = float(zc_count / (2 * WINDOW_SEC))
                pitch_var = 0.0

            label, arousal, valence, conf = _classify(
                energy, zcr, pitch, pitch_var, centroid, is_speaking
            )

            # Smooth arousal/valence with exponential moving average
            α = 0.35
            self._smoothed_arousal = α * arousal + (1 - α) * self._smoothed_arousal
            self._smoothed_valence = α * valence + (1 - α) * self._smoothed_valence

            state = {
                "audio_emotion": label,
                "emoji":         AUDIO_EMO_EMOJI.get(label, "😐"),
                "arousal":       round(self._smoothed_arousal, 3),
                "valence":       round(self._smoothed_valence, 3),
                "pitch_hz":      round(pitch, 1),
                "pitch_variance":round(pitch_var, 1),
                "energy":        round(energy, 4),
                "zcr":           round(zcr, 4),
                "spectral_centroid": round(centroid, 1),
                "speaking":      is_speaking,
                "confidence":    round(conf, 2),
                "backend":       "librosa" if _LIBROSA_OK else "basic",
            }
            self._last_state = state
            self.on_emotion(state)

        except Exception as e:
            print(f"  [AudioEmo] Analysis error: {e}")

    def get_last_state(self) -> dict:
        return self._last_state
