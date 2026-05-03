"""
AXON — Core Engine
Orchestrates all systems: optic, auditory, language, neural fabric, voice output.
"""

import threading
import time
import os
from typing import Optional

from pathlib import Path
from axon.sensory.optic    import OpticSystem
from axon.sensory.auditory import AuditorySystem
from axon.cognition.language    import LanguageCore
from axon.cognition.memory      import MemorySystem
from axon.cognition.neural_fabric import NeuralFabric
from axon.cognition.voice_output  import VoiceOutput


class AxonEngine:
    def __init__(self, socketio=None, api_key: str = None,
                 lm_studio_url: str = "http://localhost:1234",
                 lm_studio_model: str = None,
                 prefer_local: bool = True):
        self.socketio        = socketio
        self.running         = False
        self.lm_studio_url   = lm_studio_url
        self.lm_studio_model = lm_studio_model

        data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(data_dir, exist_ok=True)

        print("  [Engine] Initializing memory...")
        self.memory   = MemorySystem(db_path=Path(data_dir) / "memory" / "axon.db")

        print("  [Engine] Initializing neural fabric...")
        self.fabric   = NeuralFabric(data_dir=os.path.join(data_dir, "neural"))
        self.fabric.add_callback(self._on_fabric_state)

        print("  [Engine] Initializing language core...")
        self.language = LanguageCore(
            self.memory,
            api_key=api_key,
            lm_studio_url=lm_studio_url,
            lm_studio_model=lm_studio_model,
            prefer_local=prefer_local,
            neural_fabric=self.fabric,
        )

        print("  [Engine] Initializing voice output...")
        self.voice    = VoiceOutput()

        print("  [Engine] Initializing sensory systems...")
        self.optic    = OpticSystem(
            on_frame=self._on_frame,
            on_face=self._on_face,
        )
        self.auditory = AuditorySystem(on_speech=lambda text, conf: self._on_transcript(text), on_volume=lambda v: None)

        self._last_visual_ctx = {}

    # ── Start / Stop ──────────────────────────────────────────

    def start(self, enable_camera: bool = True, enable_mic: bool = True,
              camera_index: int = -1):
        self.running = True
        self.fabric.start()

        if enable_camera:
            try:
                self.optic.start(camera_index=camera_index)
                self._emit("log", {"msg": "👁 Optic system online"})
            except Exception as e:
                self._emit("log", {"msg": f"⚠ Camera failed: {e}"})

        if enable_mic:
            try:
                self.auditory.start()
                self._emit("log", {"msg": "🎤 Auditory system online"})
            except Exception as e:
                self._emit("log", {"msg": f"⚠ Mic failed: {e}"})

        status = self.language.get_status()
        self._emit("lm_status", status)
        self._emit("log", {"msg": f"🧠 Neural fabric online — {self.fabric.get_state_snapshot()['total_neurons']:,} virtual neurons"})
        self._emit("log", {"msg": f"🤖 LLM: {'LM Studio (' + status['lm_model'] + ')' if status['lm_studio'] else 'Claude'}"})

        # Wake thought
        threading.Timer(2.0, self._wake_thought).start()

    def _wake_thought(self):
        self.fabric.stimulate_for_input("reward", 0.3)
        self.fabric.stimulate_region("identity_core", 0.5)
        self.fabric.stimulate_region("consciousness_gate", 0.6)
        self.fabric.stimulate_region("self_referential", 0.4)

    def stop(self):
        self.running = False
        self.optic.stop()
        self.auditory.stop()
        self.fabric.stop()
        self.voice.stop()

    # ── Sensory callbacks ─────────────────────────────────────

    def _on_frame(self, frame_data: dict):
        self._last_visual_ctx = frame_data
        self._emit("frame", frame_data)
        if frame_data.get("motion", 0) > 0.05:
            self.fabric.stimulate_for_input("visual", frame_data["motion"] * 0.5)

    def _on_face(self, face_data: dict):
        self._emit("face", face_data)
        self.fabric.stimulate_for_input("face", 0.4)
        emotion = face_data.get("emotion", "neutral")
        if emotion in ("happy", "surprised"):
            self.fabric.neuromod.reward(0.1)
        elif emotion in ("angry", "fearful"):
            self.fabric.neuromod.stress(0.1)

    def _on_transcript(self, text: str):
        if not text.strip():
            return
        self._emit("transcript", {"text": text})
        self._emit("log",        {"msg": f"🎤 Heard: {text}"})
        self.fabric.stimulate_for_input("speech",   0.6)
        self.fabric.stimulate_for_input("question", 0.4)
        self._think(text)

    # ── Thinking ──────────────────────────────────────────────

    def _think(self, user_input: str):
        def _run():
            self._emit("thinking", {"state": True})
            visual_ctx = {
                "face_present": self._last_visual_ctx.get("face_present", False),
                "emotion":      self._last_visual_ctx.get("emotion", "neutral"),
            }
            try:
                response = self.language.think(user_input, visual_context=visual_ctx)
                self._emit("response",  {"text": response})
                self._emit("thinking",  {"state": False})
                # Voice output
                self.voice.speak(response)
                # Stimulate language output neurons
                self.fabric.stimulate_for_input("language_out", 0.5)
                self.fabric.neuromod.reward(0.15)
            except Exception as e:
                self._emit("log",      {"msg": f"⚠ Think error: {e}"})
                self._emit("thinking", {"state": False})

        threading.Thread(target=_run, daemon=True).start()

    def chat(self, user_input: str):
        """Called from UI text input."""
        self.fabric.stimulate_for_input("speech",   0.6)
        self.fabric.stimulate_for_input("question", 0.5)
        self._think(user_input)

    # ── Neural fabric state → UI ──────────────────────────────

    def _on_fabric_state(self, state: dict):
        self._emit("neural_state", state)
        # Bubble thoughts to UI
        thoughts = state.get("thoughts", [])
        if thoughts:
            self._emit("thought", {"text": thoughts[-1]})

    # ── Helpers ───────────────────────────────────────────────

    def _emit(self, event: str, data: dict):
        if self.socketio:
            try:
                self.socketio.emit(event, data)
            except:
                pass

    def get_status(self) -> dict:
        fabric_state = self.fabric.get_state_snapshot()
        return {
            "running":     self.running,
            "optic":       self.optic.get_status(),
            "auditory":    {"running": self.auditory.running},
            "language":    self.language.get_status(),
            "voice":       self.voice.get_status(),
            "emotion":     fabric_state["emotion"],
            "personality": fabric_state["personality"],
            "neuromod":    fabric_state["neuromod"],
            "neurons":     fabric_state["total_neurons"],
            "connections": fabric_state["total_connections"],
        }
