"""
AXON — Core Engine
Orchestrates all systems via an event bus.
Sensory → Cognition → Output, with Hebbian learning throughout.
"""

import threading
import time
import os
from pathlib import Path

from axon.cognition.memory   import MemorySystem
from axon.cognition.language import LanguageCore
from axon.sensory.optic      import OpticSystem
from axon.sensory.auditory   import AuditorySystem
from axon.output.speech      import SpeechSystem


class AxonEngine:
    def __init__(self, socketio=None, api_key: str = None, lm_studio_url: str = 'http://localhost:1234', lm_studio_model: str = None, prefer_local: bool = True):
        self.socketio = socketio
        self.running         = False
        self.lm_studio_url   = lm_studio_url
        self.lm_studio_model = lm_studio_model

        # Systems
        self.memory   = MemorySystem()
        self.language = LanguageCore(
            self.memory,
            api_key=api_key,
            lm_studio_url=lm_studio_url,
            lm_studio_model=lm_studio_model,
            prefer_local=prefer_local,
        )
        self.speech   = SpeechSystem(
            on_speaking = self._on_speaking,
            on_done     = self._on_speech_done,
        )
        self.optic    = OpticSystem(
            on_frame = self._on_frame,
            on_face  = self._on_face,
        )
        self.auditory = AuditorySystem(
            on_speech = self._on_speech_heard,
            on_volume = self._on_volume,
        )

        # State
        self.thinking       = False
        self.current_emotion= "neutral"
        self.face_present   = False
        self.volume_db      = -60.0
        self.motion_level   = 0.0
        self._last_frame    = None

        # Neuron activation state (for UI)
        self.neuron_state = {
            "optic":      0.0,
            "auditory":   0.0,
            "thalamus":   0.0,
            "columns":    [0.0]*6,
            "hippocampus":0.0,
            "prefrontal": 0.0,
            "dopamine":   0.0,
            "language":   0.0,
            "memory":     0.0,
            "speech":     0.0,
        }

        # Background emitter
        self._emit_thread = threading.Thread(target=self._emit_loop, daemon=True)

    def start(self, enable_camera: bool = True, enable_mic: bool = True):
        self.running = True
        self._emit_thread.start()

        if enable_camera:
            try:
                self.optic.start()
                self._emit("log", {"msg": "👁 Optic system online."})
            except Exception as e:
                self._emit("log", {"msg": f"⚠ Camera unavailable: {e}"})

        if enable_mic:
            try:
                self.auditory.start()
                self._emit("log", {"msg": "👂 Auditory system online. Listening..."})
            except Exception as e:
                self._emit("log", {"msg": f"⚠ Mic unavailable: {e}"})

        # Greet
        mem_count = self.memory.count_episodes()
        if mem_count > 0:
            name = self.memory.recall("user_name")
            greeting = f"Welcome back{', ' + name if name else ''}. My memory contains {mem_count} episodes."
        else:
            greeting = "Initializing. I am AXON. My neurons are forming."
        self.speech.say(greeting)
        self._emit("log", {"msg": f"🧠 AXON online. {greeting}"})

    def stop(self):
        self.running = False
        self.optic.stop()
        self.auditory.stop()
        self.memory.close()

    # ── Sensory callbacks ──────────────────────────────────────

    def _on_frame(self, frame_data: dict):
        self._last_frame = frame_data
        self.motion_level = frame_data["motion"]
        self.face_present = frame_data["face_present"]

        # Activate optic neurons proportional to motion + face
        act = min(1.0, frame_data["motion"] * 5 + (0.3 if frame_data["face_present"] else 0))
        self.neuron_state["optic"] = act
        self.neuron_state["thalamus"] = act * 0.8

        # Emit pixel data every 4th frame (throttle bandwidth)
        if frame_data["frame_id"] % 4 == 0:
            self._emit("optic_frame", {
                "pixels":   frame_data["pixels"],
                "edges":    frame_data["edges"],
                "motion":   frame_data["motion"],
                "emotion":  frame_data["emotion"],
                "face":     frame_data["face_present"],
            })

        # Hebbian: optic + thalamus co-fire
        if act > 0.2:
            self.memory.coactivate("optic_cortex", "thalamus")

    def _on_face(self, face_data: dict):
        self.current_emotion = face_data["emotion"]
        # Fire hippocampal memory when we see a face
        self.neuron_state["hippocampus"] = min(1.0, self.neuron_state["hippocampus"] + 0.4)
        self.memory.coactivate("optic_cortex", "hippocampus")
        self.memory.store_episode(
            "visual",
            {"description": f"face detected, emotion: {face_data['emotion']}", **face_data},
            emotion=face_data["emotion"],
            importance=0.6,
        )
        self._emit("face_detected", face_data)

    def _on_speech_heard(self, text: str, confidence: float):
        if self.thinking:
            return  # ignore while processing
        self._emit("speech_heard", {"text": text})
        self._emit("log", {"msg": f"👂 Heard: \"{text}\""})

        # Fire auditory neurons
        self.neuron_state["auditory"] = 1.0
        self.neuron_state["thalamus"] = 1.0

        # Think + respond in thread
        threading.Thread(target=self._process_input, args=(text,), daemon=True).start()

    def _on_volume(self, db: float):
        self.volume_db = db
        # Map dBFS to 0-1 activation
        norm = max(0.0, min(1.0, (db + 60) / 40))
        self.neuron_state["auditory"] = norm * 0.5  # background level

    # ── Language processing ────────────────────────────────────

    def _process_input(self, text: str):
        self.thinking = True
        self._emit("thinking_start", {"text": text})

        # Activate cognition neurons
        self.neuron_state["prefrontal"]  = 0.9
        self.neuron_state["hippocampus"] = 0.8
        self.neuron_state["language"]    = 1.0
        self.neuron_state["memory"]      = 0.7
        self.neuron_state["dopamine"]    = 0.5

        visual_ctx = {
            "face_present": self.face_present,
            "emotion":      self.current_emotion,
            "motion":       self.motion_level,
        } if self._last_frame else None

        response = self.language.think(text, visual_context=visual_ctx)

        # Fire output neurons
        self.neuron_state["speech"]   = 1.0
        self.neuron_state["dopamine"] = 0.8

        self._emit("axon_response", {"text": response})
        self._emit("log", {"msg": f"🧠 AXON: \"{response}\""})

        self.speech.say(response)
        self.thinking = False

    def process_text(self, text: str):
        """Called when user types in the UI."""
        threading.Thread(target=self._process_input, args=(text,), daemon=True).start()

    # ── Speech callbacks ───────────────────────────────────────

    def _on_speaking(self, text: str):
        self.neuron_state["speech"]   = 1.0
        self.neuron_state["language"] = 0.8
        self._emit("speaking_start", {"text": text})

    def _on_speech_done(self):
        self.neuron_state["speech"] = 0.0
        self._emit("speaking_done", {})

    # ── Emit loop ──────────────────────────────────────────────

    def _emit_loop(self):
        """Emit brain state at ~20fps."""
        while self.running:
            # Natural decay of neuron activations
            for k in self.neuron_state:
                if k == "columns":
                    self.neuron_state[k] = [max(0, v*0.88) for v in self.neuron_state[k]]
                else:
                    self.neuron_state[k] = max(0.0, self.neuron_state[k] * 0.90)

            # Memory neuron pulses on memory access
            mem_top = self.memory.top_connections(5)
            mem_act = min(1.0, sum(c["weight"] for c in mem_top) * 0.3)
            self.neuron_state["memory"] = max(self.neuron_state["memory"], mem_act * 0.3)

            self._emit("neuron_state", {
                "state":     self.neuron_state,
                "thinking":  self.thinking,
                "listening": self.auditory.listening,
                "speaking":  self.speech.speaking,
                "emotion":   self.current_emotion,
                "volume_db": round(self.volume_db, 1),
                "face":      self.face_present,
                "memory": {
                    "episodes":    self.memory.count_episodes(),
                    "connections": mem_top[:3],
                }
            })
            time.sleep(0.05)

    def _emit(self, event: str, data: dict):
        if self.socketio:
            try:
                self.socketio.emit(event, data)
            except:
                pass

    def get_status(self) -> dict:
        return {
            "running":   self.running,
            "thinking":  self.thinking,
            "optic":     self.optic.get_status(),
            "auditory":  self.auditory.get_status(),
            "speech":    self.speech.get_status(),
            "memory":    self.memory.memory_summary(),
            "language":  self.language.get_status(),
        }
