"""
AXON — Optic System
Webcam → frame capture → face detection → expression → visual neurons.
Runs in its own thread, emits events via callback.
"""

import cv2
import threading
import time
import numpy as np
from typing import Callable, Optional


EMOTIONS = ['neutral','happy','sad','angry','surprised','fearful','disgusted','thinking']


class OpticSystem:
    def __init__(self, on_frame: Callable, on_face: Callable,
                 camera_index: int = 0, fps: int = 8):
        self.on_frame  = on_frame   # callback(frame_data: dict)
        self.on_face   = on_face    # callback(face_data: dict)
        self.camera_idx= camera_index
        self.fps       = fps
        self.running   = False
        self._thread   = None
        self._cap      = None

        # Load face cascade (no extra deps, pure OpenCV)
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self._smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
        self._eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

        self.last_emotion   = 'neutral'
        self.face_present   = False
        self.motion_level   = 0.0
        self._prev_gray     = None
        self.frame_count    = 0

    def start(self):
        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
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
            gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small  = cv2.resize(gray, (64, 48))  # pixel neuron grid

            # Motion detection
            if self._prev_gray is not None:
                diff = cv2.absdiff(gray, self._prev_gray)
                self.motion_level = float(diff.mean()) / 255.0
            self._prev_gray = gray.copy()

            # Face detection
            faces = self._face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40,40))
            self.face_present = len(faces) > 0

            face_data = None
            if self.face_present:
                fx,fy,fw,fh = faces[0]
                face_roi    = gray[fy:fy+fh, fx:fx+fw]

                # Smile detection → happiness signal
                smiles = self._smile_cascade.detectMultiScale(face_roi, 1.7, 20)
                # Eye detection → open/closed
                eyes   = self._eye_cascade.detectMultiScale(face_roi, 1.1, 3)

                # Derive simple emotion from geometry
                emotion = self._infer_emotion(face_roi, len(smiles), len(eyes))
                self.last_emotion = emotion

                face_data = {
                    "x": int(fx/frame.shape[1]*100),
                    "y": int(fy/frame.shape[0]*100),
                    "w": int(fw/frame.shape[1]*100),
                    "h": int(fh/frame.shape[0]*100),
                    "emotion": emotion,
                    "eyes_open": len(eyes) >= 2,
                    "smiling":   len(smiles) > 0,
                    "face_brightness": float(face_roi.mean()) / 255.0,
                }
                self.on_face(face_data)

            # Build pixel neuron grid (64x48 normalized)
            pixel_neurons = (small / 255.0).tolist()  # 48 rows of 64 values

            # Edge map (visual cortex — simple gradient)
            edges   = cv2.Canny(gray, 50, 150)
            edge_small = cv2.resize(edges, (32, 24))
            edge_neurons = (edge_small / 255.0).tolist()

            frame_data = {
                "pixels":       pixel_neurons,  # 48×64 grayscale
                "edges":        edge_neurons,    # 24×32 edge map
                "motion":       round(self.motion_level, 3),
                "face_present": self.face_present,
                "emotion":      self.last_emotion,
                "frame_id":     self.frame_count,
            }
            self.on_frame(frame_data)

            elapsed = time.time() - t0
            sleep   = max(0, interval - elapsed)
            time.sleep(sleep)

    def _infer_emotion(self, face_gray, smile_count, eye_count) -> str:
        """
        Heuristic emotion from cascade detections + brightness.
        In future: swap for a real expression classifier.
        """
        brightness = face_gray.mean() / 255.0
        # Variance in upper vs lower face
        h = face_gray.shape[0]
        upper_var = float(face_gray[:h//2].std())
        lower_var = float(face_gray[h//2:].std())

        if smile_count > 0:
            return 'happy'
        if eye_count == 0 and h > 60:
            return 'neutral'  # eyes not detected = looking away
        if lower_var > upper_var * 1.4:
            return 'surprised'
        if brightness < 0.3:
            return 'thinking'
        return 'neutral'

    @property
    def camera_index(self):
        return self.camera_idx

    def get_status(self) -> dict:
        return {
            "running":      self.running,
            "face_present": self.face_present,
            "emotion":      self.last_emotion,
            "motion":       round(self.motion_level, 3),
            "frames":       self.frame_count,
        }
