"""
AXON -- Optic System
Webcam -> frame capture -> face detection -> expression -> visual neurons.
Runs in its own thread, emits events via callback.
Auto-detects physical webcam (skips virtual/phone cameras).
"""

import cv2
import threading
import time
import numpy as np
from typing import Callable, Optional


EMOTIONS = ['neutral','happy','sad','angry','surprised','fearful','disgusted','thinking']


class OpticSystem:
    def __init__(self, on_frame: Callable, on_face: Callable,
                 camera_index: int = -1, fps: int = 8):
        self.on_frame   = on_frame
        self.on_face    = on_face
        self.camera_idx = camera_index  # -1 = auto-detect
        self.fps        = fps
        self.running    = False
        self._thread    = None
        self._cap       = None

        self._face_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self._smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        self._eye_cascade   = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        self.last_emotion   = 'neutral'
        self.face_present   = False
        self.motion_level   = 0.0
        self._prev_gray     = None
        self.frame_count    = 0

    def _find_webcam(self) -> int:
        """
        Scan indices 0-9 via DirectShow.
        Returns the index of the best physical camera found.
        Strategy:
          - Must open AND return a valid frame
          - Prefer index with highest native resolution (phone cams
            often report very high or very low res)
          - Skip indices that only work via a virtual driver
        Falls back to 0 if nothing else found.
        """
        print("  [Optic] Scanning cameras...")
        candidates = []
        for idx in range(10):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                continue
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            print(f"  [Optic]   [{idx}] {w}x{h} - OK")
            candidates.append((idx, w, h))

        if not candidates:
            print("  [Optic] No cameras found, defaulting to 0")
            return 0

        # Pick the one closest to a standard webcam resolution.
        # Phone/virtual cams often show up at idx 0 on Windows when
        # using apps like iPhone Continuity Camera or EpocCam.
        # Prefer the LAST index with a sane resolution (640-1920w).
        sane = [c for c in candidates if 320 <= c[1] <= 1920]
        if not sane:
            sane = candidates

        # If idx 0 is the only option, use it.
        # Otherwise prefer higher indices (physical webcam usually > 0
        # when a phone cam is also connected).
        if len(sane) == 1:
            chosen = sane[0][0]
        else:
            # Sort by index descending -- take highest non-phone index
            chosen = sorted(sane, key=lambda c: c[0])[-1][0]

        print(f"  [Optic] Selected camera index: {chosen}")
        return chosen

    def start(self, camera_index: int = None):
        if camera_index is not None:
            self.camera_idx = camera_index

        if self.camera_idx < 0:
            self.camera_idx = self._find_webcam()

        self._cap = cv2.VideoCapture(self.camera_idx, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.camera_idx)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        print(f"  [Optic] Opened camera {self.camera_idx}")
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
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (64, 48))

            # Motion
            if self._prev_gray is not None:
                diff = cv2.absdiff(gray, self._prev_gray)
                self.motion_level = float(diff.mean()) / 255.0
            self._prev_gray = gray.copy()

            # Face detection
            faces = self._face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40,40))
            self.face_present = len(faces) > 0

            face_data = None
            if self.face_present:
                fx, fy, fw, fh = faces[0]
                face_roi = gray[fy:fy+fh, fx:fx+fw]
                smiles   = self._smile_cascade.detectMultiScale(face_roi, 1.7, 20)
                eyes     = self._eye_cascade.detectMultiScale(face_roi, 1.1, 3)
                emotion  = self._infer_emotion(face_roi, len(smiles), len(eyes))
                self.last_emotion = emotion
                face_data = {
                    "x": int(fx/frame.shape[1]*100),
                    "y": int(fy/frame.shape[0]*100),
                    "w": int(fw/frame.shape[1]*100),
                    "h": int(fh/frame.shape[0]*100),
                    "emotion":         emotion,
                    "eyes_open":       len(eyes) >= 2,
                    "smiling":         len(smiles) > 0,
                    "face_brightness": float(face_roi.mean()) / 255.0,
                }
                self.on_face(face_data)

            pixel_neurons = (small / 255.0).tolist()
            edges         = cv2.Canny(gray, 50, 150)
            edge_small    = cv2.resize(edges, (32, 24))
            edge_neurons  = (edge_small / 255.0).tolist()

            frame_data = {
                "pixels":       pixel_neurons,
                "edges":        edge_neurons,
                "motion":       round(self.motion_level, 3),
                "face_present": self.face_present,
                "emotion":      self.last_emotion,
                "frame_id":     self.frame_count,
            }
            self.on_frame(frame_data)

            elapsed = time.time() - t0
            time.sleep(max(0, interval - elapsed))

    def _infer_emotion(self, face_gray, smile_count, eye_count) -> str:
        brightness = face_gray.mean() / 255.0
        h = face_gray.shape[0]
        upper_var = float(face_gray[:h//2].std())
        lower_var = float(face_gray[h//2:].std())
        if smile_count > 0:
            return 'happy'
        if eye_count == 0 and h > 60:
            return 'neutral'
        if lower_var > upper_var * 1.4:
            return 'surprised'
        if brightness < 0.3:
            return 'thinking'
        return 'neutral'

    def get_status(self) -> dict:
        return {
            "running":      self.running,
            "camera_index": self.camera_idx,
            "face_present": self.face_present,
            "emotion":      self.last_emotion,
            "motion":       round(self.motion_level, 3),
            "frames":       self.frame_count,
        }
