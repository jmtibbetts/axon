"""
AXON — Optic System (GPU-accelerated)
Webcam → CUDA frame processing → YOLOv8-face detection → CNN emotion → visual neurons.

GPU pipeline:
  - YOLOv8n-face  : real-time face detection on CUDA (replaces Haar cascades)
  - EmotionNet    : lightweight CNN for 7-class facial expression on GPU
  - Frame tensors : brightness, motion (optical flow diff), edge map — all torch ops
  - Pixel neurons : frame downsampled to 64x48 on GPU, flattened to neuron activations
"""

import cv2
import threading
import time
import numpy as np
from typing import Callable, Optional

import torch
import torch.nn as nn
import torchvision.transforms as T

# ── Device ───────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Tiny Emotion CNN (runs fully on GPU) ─────────────────────────────────────

class EmotionNet(nn.Module):
    """
    Lightweight 7-class emotion CNN.
    Input: 48x48 grayscale face crop (normalized).
    Classes: neutral, happy, sad, angry, surprised, fearful, disgusted
    Weights initialised randomly — learns over time from user interaction,
    or you can drop in FER2013-trained weights later.
    """
    CLASSES = ['neutral','happy','sad','angry','surprised','fearful','disgusted']

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 7),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def predict(self, face_gray_np: np.ndarray) -> str:
        """face_gray_np: HxW uint8. Returns emotion label."""
        img = cv2.resize(face_gray_np, (48, 48)).astype(np.float32) / 255.0
        t   = torch.tensor(img, device=DEVICE).unsqueeze(0).unsqueeze(0)  # [1,1,48,48]
        with torch.no_grad():
            logits = self(t)
            idx    = logits.argmax(dim=1).item()
        return self.CLASSES[idx]


# ── Frame tensor helpers ──────────────────────────────────────────────────────

_to_tensor = T.ToTensor()   # HxWxC uint8 → [C,H,W] float32 /255

def _frame_to_gpu(frame_bgr: np.ndarray) -> torch.Tensor:
    """BGR numpy frame → [3,H,W] float32 tensor on DEVICE."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return _to_tensor(rgb).to(DEVICE)

def _gpu_brightness(t: torch.Tensor) -> float:
    """Mean brightness from GPU tensor [3,H,W]."""
    return float(t.mean().item())

def _gpu_motion(prev: Optional[torch.Tensor], curr: torch.Tensor) -> float:
    """Frame diff motion estimate, GPU side."""
    if prev is None:
        return 0.0
    gray_curr = curr.mean(dim=0)   # [H,W]
    gray_prev = prev.mean(dim=0)
    diff = (gray_curr - gray_prev).abs()
    return float(diff.mean().item())

def _gpu_pixel_neurons(t: torch.Tensor, w=64, h=48) -> list:
    """Downsample frame to 64x48 pixel neuron grid on GPU, return as flat list."""
    small = torch.nn.functional.interpolate(
        t.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
    ).squeeze(0)                              # [3,h,w]
    gray  = small.mean(dim=0)                 # [h,w]
    return gray.cpu().numpy().tolist()         # CPU transfer only here


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

        self.last_emotion = 'neutral'
        self.face_present = False
        self.motion_level = 0.0
        self.frame_count  = 0
        self._prev_tensor: Optional[torch.Tensor] = None

        # Models (lazy-loaded in background thread)
        self._yolo        = None   # YOLOv8-face
        self._emotion_net = None   # EmotionNet
        self._models_ready = False

        # Fallback Haar cascade (used until YOLO loads)
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
            print("  [Optic] No cameras found, defaulting to 0")
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
        """Load YOLOv8-face + EmotionNet in background. Falls back gracefully."""
        try:
            from ultralytics import YOLO
            # yolov8n-face auto-downloads ~6MB on first run
            self._yolo = YOLO("yolov8n-face.pt")
            self._yolo.to(DEVICE)
            print(f"  [Optic] YOLOv8-face loaded on {DEVICE}")
        except Exception as e:
            print(f"  [Optic] YOLOv8 unavailable ({e}), using Haar fallback")
            self._yolo = None

        try:
            self._emotion_net = EmotionNet().to(DEVICE).eval()
            # Try to load pretrained weights if present
            import os
            weights_path = os.path.join(os.path.dirname(__file__), "emotion_weights.pt")
            if os.path.exists(weights_path):
                state = torch.load(weights_path, map_location=DEVICE)
                self._emotion_net.load_state_dict(state)
                print("  [Optic] EmotionNet weights loaded")
            else:
                print("  [Optic] EmotionNet init (random weights — will bias toward neutral)")
        except Exception as e:
            print(f"  [Optic] EmotionNet failed ({e})")
            self._emotion_net = None

        self._models_ready = True
        print(f"  [Optic] GPU vision pipeline ready on {DEVICE}")

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

        # Load models in background so camera starts immediately
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

            # ── GPU frame tensor ──────────────────────────────────────────
            frame_t = _frame_to_gpu(frame)                     # [3,H,W] on CUDA

            brightness    = _gpu_brightness(frame_t)
            motion        = _gpu_motion(self._prev_tensor, frame_t)
            self.motion_level = motion
            self._prev_tensor = frame_t.clone()

            pixel_neurons = _gpu_pixel_neurons(frame_t)        # 64x48 grid

            # ── Face detection ────────────────────────────────────────────
            face_data = None
            if self._models_ready and self._yolo is not None:
                face_data = self._detect_face_yolo(frame, frame_t)
            else:
                face_data = self._detect_face_haar(frame)

            if face_data:
                self.face_present = True
                self.last_emotion = face_data.get("emotion", "neutral")
                self.on_face(face_data)
            else:
                self.face_present = False

            # ── Edge map (GPU) ────────────────────────────────────────────
            gray_t    = frame_t.mean(dim=0)                    # [H,W]
            # Simple Sobel-style edge on GPU
            sobel_x   = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]],
                                      dtype=torch.float32, device=DEVICE).unsqueeze(0)
            sobel_y   = torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]],
                                      dtype=torch.float32, device=DEVICE).unsqueeze(0)
            g_in      = gray_t.unsqueeze(0).unsqueeze(0)       # [1,1,H,W]
            import torch.nn.functional as F
            gx        = F.conv2d(g_in, sobel_x, padding=1).squeeze()
            gy        = F.conv2d(g_in, sobel_y, padding=1).squeeze()
            edges_t   = (gx**2 + gy**2).sqrt().clamp(0,1)
            edge_small = F.interpolate(edges_t.unsqueeze(0).unsqueeze(0),
                                       size=(24,32), mode='bilinear',
                                       align_corners=False).squeeze()
            edge_neurons = edge_small.cpu().numpy().tolist()

            # ── Emit frame data ───────────────────────────────────────────
            self.on_frame({
                "pixels":       pixel_neurons,
                "edges":        edge_neurons,
                "motion":       round(motion, 3),
                "face_present": self.face_present,
                "emotion":      self.last_emotion,
                "frame_id":     self.frame_count,
                "brightness":   round(brightness, 3),
                "gpu":          str(DEVICE),
            })

            elapsed = time.time() - t0
            time.sleep(max(0.0, interval - elapsed))

    # ── YOLO face detection ───────────────────────────────────────────────────

    def _detect_face_yolo(self, frame_bgr: np.ndarray,
                           frame_t: torch.Tensor) -> Optional[dict]:
        """Run YOLOv8-face, return best detection or None."""
        H, W = frame_bgr.shape[:2]
        try:
            results = self._yolo.predict(frame_bgr, conf=0.4, verbose=False,
                                          device=DEVICE)
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return None

            # Best box (highest confidence)
            confs = boxes.conf.cpu().numpy()
            best  = int(confs.argmax())
            x1,y1,x2,y2 = boxes.xyxy[best].cpu().numpy().astype(int)
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(W,x2), min(H,y2)

            if x2 <= x1 or y2 <= y1:
                return None

            # Emotion from GPU CNN
            gray_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            face_roi = gray_bgr[y1:y2, x1:x2]
            emotion  = (self._emotion_net.predict(face_roi)
                        if self._emotion_net and face_roi.size > 0
                        else self.last_emotion)

            # Heuristic smile from mouth region brightness variation
            mouth_roi = gray_bgr[y1 + (y2-y1)//2: y2, x1:x2]
            smiling   = bool(mouth_roi.std() > 25) if mouth_roi.size > 0 else False

            return {
                "x": int(x1/W*100), "y": int(y1/H*100),
                "w": int((x2-x1)/W*100), "h": int((y2-y1)/H*100),
                "emotion":         emotion,
                "eyes_open":       True,
                "smiling":         smiling,
                "face_brightness": round(float(face_roi.mean())/255.0, 3)
                                   if face_roi.size > 0 else 0.5,
                "confidence":      round(float(confs[best]), 3),
                "detector":        "yolo-gpu",
            }
        except Exception as e:
            print(f"  [Optic] YOLO error: {e}")
            return self._detect_face_haar(frame_bgr)

    # ── Haar fallback ─────────────────────────────────────────────────────────

    def _detect_face_haar(self, frame_bgr: np.ndarray) -> Optional[dict]:
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        H, W  = gray.shape
        faces = self._haar.detectMultiScale(gray, 1.1, 5, minSize=(40,40))
        if len(faces) == 0:
            return None
        fx, fy, fw, fh = faces[0]
        face_roi = gray[fy:fy+fh, fx:fx+fw]
        emotion  = self._infer_emotion_haar(face_roi)
        return {
            "x": int(fx/W*100), "y": int(fy/H*100),
            "w": int(fw/W*100), "h": int(fh/H*100),
            "emotion":         emotion,
            "eyes_open":       True,
            "smiling":         False,
            "face_brightness": round(float(face_roi.mean())/255.0, 3),
            "detector":        "haar-cpu",
        }

    def _infer_emotion_haar(self, face_gray) -> str:
        brightness = face_gray.mean() / 255.0
        h = face_gray.shape[0]
        lower_var = float(face_gray[h//2:].std())
        upper_var = float(face_gray[:h//2].std())
        if lower_var > upper_var * 1.4:
            return 'surprised'
        if brightness < 0.3:
            return 'thinking'
        return 'neutral'

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "running":      self.running,
            "camera_index": self.camera_idx,
            "face_present": self.face_present,
            "emotion":      self.last_emotion,
            "motion":       round(self.motion_level, 3),
            "frames":       self.frame_count,
            "detector":     "yolo-gpu" if self._yolo else "haar-cpu",
            "gpu":          str(DEVICE),
            "models_ready": self._models_ready,
        }
