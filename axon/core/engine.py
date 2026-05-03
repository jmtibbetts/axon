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
        self.auditory = AuditorySystem(on_speech=lambda text, conf: self._on_transcript(text), on_volume=lambda v: None, device_index=None)

        self._last_visual_ctx = {}

    # ── Start / Stop ──────────────────────────────────────────

    def start(self, enable_camera: bool = True, enable_mic: bool = True,
              camera_index: int = -1, mic_index: int = None):
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
                if mic_index is not None:
                    self.auditory.device_idx = mic_index
                self.auditory.start()
                self._emit("log", {"msg": "🎤 Auditory system online"})
            except Exception as e:
                self._emit("log", {"msg": f"⚠ Mic failed: {e}"})

        status = self.language.get_status()
        self._emit("lm_status", status)
        self._emit("log", {"msg": f"🧠 Neural fabric online — {self.fabric.get_state_snapshot()['total_neurons']:,} virtual neurons"})
        self._emit("log", {"msg": f"🤖 LLM: {'LM Studio (' + status['lm_model'] + ')' if status['lm_studio'] else 'Claude'}"})
        vs = self.voice.get_status()
        self._emit("voice_speaking", {"speaking": False, "enabled": vs["enabled"], "playback": vs.get("playback","none")})
        if not vs["enabled"]:
            self._emit("log", {"msg": f"⚠ Voice output disabled — edge-tts or audio playback missing"})

        # Wake thought
        threading.Timer(2.0, self._wake_thought).start()

    def _wake_thought(self):
        self.fabric.stimulate_for_input("reward", 0.25)
        self.fabric.stimulate_region("identity_core", 0.35)
        self.fabric.stimulate_region("consciousness_gate", 0.40)
        self.fabric.stimulate_region("self_referential", 0.30)

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
        # Always stimulate visual cortex — camera is always active
        motion     = frame_data.get("motion", 0)
        brightness = frame_data.get("brightness", 0.5)
        # Gate: only stimulate every 4th frame to avoid per-frame flooding
        self._frame_tick = getattr(self, "_frame_tick", 0) + 1
        if self._frame_tick % 4 != 0:
            return
        # Reduced amounts — camera is always-on so values must stay low
        base_visual = 0.04 + motion * 0.12
        self.fabric.stimulate_for_input("visual", base_visual)
        if frame_data.get("face_present"):
            self.fabric.stimulate_for_input("face", 0.20)
        self.fabric.stimulate_region("primary_visual",   0.08 + brightness * 0.18)
        self.fabric.stimulate_region("motion_detection", 0.05 + motion   * 0.25)

    def _on_face(self, face_data: dict):
        self._emit("face", face_data)
        emotion = face_data.get("emotion", "neutral")
        probs   = face_data.get("emotion_probs", {})
        conf    = face_data.get("confidence", 0.5)

        # Always stimulate social/face recognition areas
        self.fabric.stimulate_for_input("face", 0.25)

        # Emotion → neural mapping (import inline to avoid circular)
        from axon.sensory.optic import EMOTION_NEURAL_MAP
        mapping = EMOTION_NEURAL_MAP.get(emotion, EMOTION_NEURAL_MAP["neutral"])

        # Scale stimulation by emotion probability (confidence in the emotion)
        emo_conf = probs.get(emotion, conf) if probs else conf
        scale    = max(0.3, min(1.0, emo_conf * 1.5))   # boost weak signals

        for cluster, amount in mapping.get("stimulate", []):
            self.fabric.stimulate_region(cluster, amount * scale)

        if mapping.get("reward", 0) > 0:
            self.fabric.neuromod.reward(mapping["reward"] * scale)
        if mapping.get("stress", 0) > 0:
            self.fabric.neuromod.stress(mapping["stress"] * scale)

        # Log significant emotion changes
        if emotion != "neutral" and emo_conf > 0.45:
            self._emit("log", {"msg": f"😶 Emotion: {face_data.get('emoji','')} {emotion} ({int(emo_conf*100)}%)"})

    def _on_transcript(self, text: str):
        if not text.strip():
            return
        self._emit("transcript", {"text": text})
        self._emit("log",        {"msg": f"🎤 Heard: {text}"})
        self.fabric.stimulate_for_input("speech",   0.75)
        self.fabric.stimulate_for_input("question", 0.60)
        self._think(text)

    # ── Thinking ──────────────────────────────────────────────

    def _think(self, user_input: str):
        def _run():
            self._emit("thinking", {"state": True})
            # Light up cognitive regions during LLM inference
            self.fabric.stimulate_for_input("thinking", 0.70)
            self.fabric.stimulate_for_input("memory",   0.55)
            visual_ctx = {
                "face_present": self._last_visual_ctx.get("face_present", False),
                "emotion":      self._last_visual_ctx.get("emotion", "neutral"),
            }
            try:
                response = self.language.think(user_input, visual_context=visual_ctx)
                self._emit("response",  {"text": response})
                self._emit("thinking",  {"state": False})
                # Voice output
                self._emit("voice_speaking", {
                    "speaking": True,
                    "enabled":  self.voice.enabled,
                    "playback": self.voice.get_status().get("playback","none")
                })
                self.voice.speak(response)
                self._emit("voice_speaking", {
                    "speaking": False,
                    "enabled":  self.voice.enabled,
                    "playback": self.voice.get_status().get("playback","none")
                })
                # Stimulate language output neurons
                self.fabric.stimulate_for_input("language_out", 0.78)
                self.fabric.stimulate_for_input("thinking",     0.55)
                self.fabric.neuromod.reward(0.08)
            except Exception as e:
                self._emit("log",      {"msg": f"⚠ Think error: {e}"})
                self._emit("thinking", {"state": False})

        if self.socketio:
            self.socketio.start_background_task(_run)
        else:
            threading.Thread(target=_run, daemon=True).start()

    def chat(self, user_input: str):
        """Called from UI text input."""
        self.fabric.stimulate_for_input("speech",   0.75)
        self.fabric.stimulate_for_input("question", 0.60)
        self._think(user_input)

    # ── Neural fabric state → UI ──────────────────────────────

    def _on_fabric_state(self, state: dict):
        self._emit("neural_state", state)
        # Push synapse count to header counter every tick
        self._emit("synapse_count", {
            "connections": state.get("total_connections", 0),
            "neurons":     state.get("total_neurons", 0),
        })
        # Bubble thoughts to UI
        thoughts = state.get("thoughts", [])
        if thoughts:
            self._emit("thought", {"text": thoughts[-1]})

    # ── Helpers ───────────────────────────────────────────────

    def _emit(self, event: str, data: dict):
        if self.socketio:
            try:
                self.socketio.emit(event, data, broadcast=True)
            except Exception as e:
                try:
                    # Fallback: push via server-side emit with namespace
                    self.socketio.emit(event, data, namespace="/")
                except Exception:
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
