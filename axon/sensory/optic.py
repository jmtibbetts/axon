"""
AXON — Optic System (GPU-accelerated)
Webcam → CUDA frame processing → YOLOv8-face + FER emotion detection → visual neurons.

GPU pipeline:
  - YOLOv8n-face  : real-time face detection on CUDA
  - FER           : pretrained FER2013 emotion classifier (7 emotions)
  - Frame tensors : brightness, motion, edge map — all torch ops
  - Pixel neurons : frame downsampled to 64x48, flattened to neuron activations
"""

import cv2
import threading
import time
import numpy as np
from typing import Callable, Optional

import torch
import torchvision.transforms as T

# ── Device ───────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Emotion labels (FER2013 standard order) ───────────────────────────────────
EMOTION_LABELS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Emoji mapping for UI display
EMOTION_EMOJI = {
    'angry':     '😠',
    'disgusted': '🤢',
    'fearful':   '😨',
    'happy':     '😊',
    'neutral':   '😐',
    'sad':       '😢',
    'surprised': '😲',
}

# How each detected emotion maps to neural stimulation
# (region/neuromod, amount)
EMOTION_NEURAL_MAP = {
    'happy': {
        'stimulate': [('amygdala_reward', 0.55), ('reward_anticipation', 0.45)],
        'reward': 0.15,
        'stress': 0.0,
    },
    'surprised': {
        'stimulate': [('attention_spotlight', 0.60), ('curiosity_drive', 0.50),
                      ('amygdala_fear', 0.20)],
        'reward': 0.05,
        'stress': 0.03,
    },
    'fearful': {
        'stimulate': [('amygdala_fear', 0.70), ('threat_detection', 0.65),
                      ('attention_spotlight', 0.50)],
        'reward': 0.0,
        'stress': 0.20,
    },
    'angry': {
        'stimulate': [('amygdala_fear', 0.50), ('threat_detection', 0.55),
                      ('social_pain', 0.40), ('inhibitory_control', 0.35)],
        'reward': 0.0,
        'stress': 0.15,
    },
    'sad': {
        'stimulate': [('social_pain', 0.55), ('self_referential', 0.45),
                      ('mind_wandering', 0.40)],
        'reward': 0.0,
        'stress': 0.10,
    },
    'disgusted': {
        'stimulate': [('amygdala_fear', 0.35), ('threat_detection', 0.40)],
        'reward': 0.0,
        'stress': 0.08,
    },
    'neutral': {
        'stimulate': [('consciousness_gate', 0.20)],
        'reward': 0.0,
        'stress': 0.0,
    },
}


# ── Frame tensor helpers ──────────────────────────────────────────────────────

_to_tensor = T.ToTensor()

def _frame_to_gpu(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return _to_tensor(rgb).to(DEVICE)

def _gpu_brightness(t: torch.Tensor) -> float:
    return float(t.mean().item())

def _gpu_motion(prev: Optional[torch.Tensor], curr: torch.Tensor) -> float:
    if prev is None:
        return 0.0
    return float((curr.mean(dim=0) - prev.mean(dim=0)).abs().mean().item())

def _gpu_pixel_neurons(t: torch.Tensor, w=64, h=48) -> list:
    small = torch.nn.functional.interpolate(
        t.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
    ).squeeze(0)
    gray = small.mean(dim=0)
    return gray.cpu().numpy().tolist()


# ── FER Emotion Detector ──────────────────────────────────────────────────────

class FERDetector:
    """
    Wraps the `fer` library (FER2013-trained model).
    Returns emotion label + per-class probabilities.
    Falls back to heuristic if fer is unavailable.
    """
    def __init__(self):
        self._fer = None
        self._ready = False

    def load(self):
        # Try fer library first, then deepface as fallback
        try:
            from fer import FER
            self._fer = FER(mtcnn=False)
            self._ready = True
            self._backend = "fer"
            print("  [Optic] FER emotion detector loaded (FER2013 pretrained)")
            return
        except Exception as e:
            print(f"  [Optic] fer import failed ({e}), trying deepface...")

        try:
            from deepface import DeepFace  # noqa: F401 — just verify import
            self._fer = None
            self._ready = True
            self._backend = "deepface"
            print("  [Optic] DeepFace emotion detector ready")
            return
        except Exception as e2:
            print(f"  [Optic] DeepFace unavailable ({e2}) — using heuristic fallback")

        self._ready = False
        self._backend = "heuristic"

    def predict(self, face_bgr: np.ndarray) -> tuple[str, dict]:
        """
        Returns (dominant_emotion, {emotion: probability, ...})
        face_bgr: cropped face region in BGR color
        """
        if not self._ready:
            return self._heuristic(face_bgr)

        backend = getattr(self, "_backend", "heuristic")

        # ── fer backend ───────────────────────────────────────────────────────
        if backend == "fer" and self._fer is not None:
            try:
                face = cv2.resize(face_bgr, (96, 96))
                result = self._fer.detect_emotions(face)
                if not result:
                    return "neutral", {e: 0.0 for e in EMOTION_LABELS}
                emotions = result[0]["emotions"]
                dominant = max(emotions, key=emotions.get)
                return dominant, emotions
            except Exception as e:
                print(f"  [Optic] FER predict error: {e}")
                return self._heuristic(face_bgr)

        # ── deepface backend ──────────────────────────────────────────────────
        if backend == "deepface":
            try:
                from deepface import DeepFace
                face = cv2.resize(face_bgr, (96, 96))
                result = DeepFace.analyze(face, actions=["emotion"],
                                          enforce_detection=False, silent=True)
                # result is a list; take first entry
                if isinstance(result, list):
                    result = result[0]
                raw_emo = result.get("emotion", {})
                # normalize to 0–1 range (deepface returns percentages)
                total = sum(raw_emo.values()) or 1.0
                emotions = {k: v / total for k, v in raw_emo.items()}
                dominant = result.get("dominant_emotion", max(emotions, key=emotions.get))
                return dominant, emotions
            except Exception as e:
                print(f"  [Optic] DeepFace predict error: {e}")
                return self._heuristic(face_bgr)

        return self._heuristic(face_bgr)

    def _heuristic(self, face_gray_or_bgr: np.ndarray) -> tuple[str, dict]:
        """Simple brightness/variance heuristic when FER unavailable."""
        if face_gray_or_bgr.ndim == 3:
            gray = cv2.cvtColor(face_gray_or_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_gray_or_bgr
        h = gray.shape[0]
        lower_var = float(gray[h//2:].std())
        upper_var = float(gray[:h//2].std())
        brightness = gray.mean() / 255.0
        if lower_var > upper_var * 1.4:
            emo = 'surprised'
        elif brightness < 0.28:
            emo = 'sad'
        else:
            emo = 'neutral'
        probs = {e: 0.05 for e in EMOTION_LABELS}
        probs[emo] = 0.70
        return emo, probs


# ── OpticSystem ───────────────────────────────────────────────────────────────

class OpticSystem:
    def __init__(self, on_frame: Callable, on_face: Callable,
                 camera_index: int = -1, fps: int = 12):
        self.on_frame   = on_frame
        self.on_face    = on_face
        self.camera_idx = camera_index
        self.fps        = fps
        self.running    = False
        self._thread    = None
        self._cap       = None

        self.last_emotion       = 'neutral'
        self.last_emotion_probs = {}
        self.face_present       = False
        self.motion_level       = 0.0
        self.frame_count        = 0
        self._prev_tensor: Optional[torch.Tensor] = None

        # Models (lazy-loaded in background thread)
        self._yolo        = None
        self._fer         = FERDetector()
        self._models_ready = False

        # Haar fallback
        self._haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # ── Camera discovery ─────────────────────────────────────────────────────

    def _find_webcam(self) -> int:
        print("  [Optic] Scanning cameras...")
        candidates = []
        for idx in range(10):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release(); continue
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release(); continue
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            print(f"  [Optic]   [{idx}] {w}x{h} OK")
            candidates.append((idx, w, h))
        if not candidates:
            return 0
        sane = [c for c in candidates if 320 <= c[1] <= 1920] or candidates
        chosen = sorted(sane, key=lambda c: c[0])[-1][0]
        print(f"  [Optic] Selected camera index: {chosen}")
        return chosen

    @staticmethod
    def list_cameras() -> list:
        results = []
        for idx in range(10):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release(); continue
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release(); continue
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            results.append({
                "index": idx, "width": w, "height": h,
                "label": f"Camera {idx} ({w}x{h})"
            })
        return results

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_models(self):
        try:
            from ultralytics import YOLO
            import urllib.request, os
            model_path = "yolov8n-face.pt"
            if not os.path.exists(model_path):
                # Candidate URLs — try in order; first success wins
                candidates = [
                    "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt",
                    "https://github.com/akanametov/yolo-face/releases/latest/download/yolov8n-face.pt",
                    "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt",
                ]
                downloaded = False
                for url in candidates:
                    try:
                        print(f"  [Optic] Downloading yolov8n-face.pt from {url} ...")
                        urllib.request.urlretrieve(url, model_path)
                        print(f"  [Optic] Download complete.")
                        downloaded = True
                        break
                    except Exception as dl_err:
                        print(f"  [Optic] URL failed ({dl_err}), trying next...")
                        if os.path.exists(model_path):
                            os.remove(model_path)
                if not downloaded:
                    raise RuntimeError("All yolov8n-face.pt download mirrors failed")
            self._yolo = YOLO(model_path)
            self._yolo.to(DEVICE)
            print(f"  [Optic] YOLOv8-face loaded on {DEVICE}")
        except Exception as e:
            print(f"  [Optic] YOLOv8 unavailable ({e}), using Haar fallback")
            self._yolo = None

        self._fer.load()
        self._models_ready = True
        print(f"  [Optic] Vision pipeline ready on {DEVICE}")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def start(self, camera_index: int = None):
        if camera_index is not None:
            self.camera_idx = camera_index
        if self.camera_idx < 0:
            self.camera_idx = self._find_webcam()

        self._cap = cv2.VideoCapture(self.camera_idx, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.camera_idx)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        print(f"  [Optic] Opened camera {self.camera_idx} at 640x480")

        threading.Thread(target=self._load_models, daemon=True).start()

        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._cap:
            self._cap.release()

    def _loop(self):
        interval = 1.0 / self.fps
        while self.running:
            t0 = time.time()
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            self.frame_count += 1
            frame_t      = _frame_to_gpu(frame)
            brightness   = _gpu_brightness(frame_t)
            motion       = _gpu_motion(self._prev_tensor, frame_t)
            self.motion_level   = motion
            self._prev_tensor   = frame_t.clone()
            pixel_neurons = _gpu_pixel_neurons(frame_t)

            # Encode frame as JPEG for the UI canvas
            _, buf = cv2.imencode('.jpg', cv2.resize(frame, (160, 120)),
                                  [cv2.IMWRITE_JPEG_QUALITY, 70])
            import base64
            frame_b64 = base64.b64encode(buf).decode()

            self.on_frame({
                "frame_b64":    frame_b64,
                "brightness":   round(brightness, 3),
                "motion":       round(motion, 3),
                "pixel_neurons": pixel_neurons,
            })

            # Face detection every 4th frame (3 FPS @ 12fps camera)
            if self.frame_count % 4 == 0 and self._models_ready:
                face_data = self._detect_face(frame)
                if face_data:
                    self.face_present    = True
                    self.last_emotion    = face_data['emotion']
                    self.last_emotion_probs = face_data.get('emotion_probs', {})
                    self.on_face(face_data)
                else:
                    self.face_present = False

            elapsed = time.time() - t0
            time.sleep(max(0.0, interval - elapsed))

    # ── Face + Emotion Detection ──────────────────────────────────────────────

    def _detect_face(self, frame_bgr: np.ndarray) -> Optional[dict]:
        if self._yolo is not None:
            return self._detect_yolo(frame_bgr)
        return self._detect_haar(frame_bgr)

    def _detect_yolo(self, frame_bgr: np.ndarray) -> Optional[dict]:
        try:
            H, W = frame_bgr.shape[:2]
            results = self._yolo(frame_bgr, verbose=False, device=DEVICE)
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return None
            confs = boxes.conf
            if len(confs) == 0:
                return None
            best  = int(confs.argmax())
            x1,y1,x2,y2 = boxes.xyxy[best].cpu().numpy().astype(int)
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(W,x2), min(H,y2)
            if x2 <= x1 or y2 <= y1:
                return None

            face_crop = frame_bgr[y1:y2, x1:x2]
            emotion, probs = self._fer.predict(face_crop)

            # Annotate frame for display (draw box + label)
            frame_annotated = frame_bgr.copy()
            color = self._emotion_color(emotion)
            cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{EMOTION_EMOJI.get(emotion, '')} {emotion}"
            cv2.putText(frame_annotated, label, (x1, max(y1-8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            return {
                "x": int(x1/W*100), "y": int(y1/H*100),
                "w": int((x2-x1)/W*100), "h": int((y2-y1)/H*100),
                "emotion":       emotion,
                "emotion_probs": probs,
                "emoji":         EMOTION_EMOJI.get(emotion, '😐'),
                "eyes_open":     True,
                "smiling":       emotion == 'happy',
                "face_brightness": round(float(face_crop.mean())/255.0, 3)
                                   if face_crop.size > 0 else 0.5,
                "confidence":    round(float(confs[best]), 3),
                "detector":      "yolo-gpu",
            }
        except Exception as e:
            print(f"  [Optic] YOLO error: {e}")
            return self._detect_haar(frame_bgr)

    def _detect_haar(self, frame_bgr: np.ndarray) -> Optional[dict]:
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        H, W  = gray.shape
        faces = self._haar.detectMultiScale(gray, 1.1, 5, minSize=(40,40))
        if len(faces) == 0:
            return None
        fx, fy, fw, fh = faces[0]
        face_crop = frame_bgr[fy:fy+fh, fx:fx+fw]
        emotion, probs = self._fer.predict(face_crop)
        return {
            "x": int(fx/W*100), "y": int(fy/H*100),
            "w": int(fw/W*100),  "h": int(fh/H*100),
            "emotion":       emotion,
            "emotion_probs": probs,
            "emoji":         EMOTION_EMOJI.get(emotion, '😐'),
            "eyes_open":     True,
            "smiling":       emotion == 'happy',
            "face_brightness": round(float(face_crop.mean())/255.0, 3),
            "confidence":    0.5,
            "detector":      "haar-cpu",
        }

    @staticmethod
    def _emotion_color(emotion: str) -> tuple:
        colors = {
            'happy':     (0, 220, 100),
            'surprised': (0, 200, 255),
            'fearful':   (120, 0, 255),
            'angry':     (0, 0, 255),
            'sad':       (180, 100, 50),
            'disgusted': (0, 140, 0),
            'neutral':   (160, 160, 160),
        }
        return colors.get(emotion, (160, 160, 160))

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "running":       self.running,
            "camera_index":  self.camera_idx,
            "face_present":  self.face_present,
            "emotion":       self.last_emotion,
            "emotion_probs": self.last_emotion_probs,
            "emoji":         EMOTION_EMOJI.get(self.last_emotion, '😐'),
            "motion":        round(self.motion_level, 3),
            "frames":        self.frame_count,
            "detector":      "yolo-gpu" if self._yolo else "haar-cpu",
            "gpu":           str(DEVICE),
            "models_ready":  self._models_ready,
        }
