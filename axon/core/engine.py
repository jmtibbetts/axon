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

        self._last_visual_ctx            = {}
        self._emotion_before_response    = "neutral"
        self._emotion_history: list      = []
        self._last_face_data: dict       = {}

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
        self._last_visual_ctx = {**frame_data, "camera_running": True}
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

    # Valence scores — positive emotions score high, negative score low
    _EMOTION_VALENCE = {
        "happy":     1.0, "surprised": 0.3, "neutral":   0.0,
        "fearful":  -0.4, "disgusted": -0.6, "sad":      -0.7, "angry": -0.9,
    }

    def _on_face(self, face_data: dict):
        self._emit("face", face_data)
        import time as _time
        emotion  = face_data.get("emotion", "neutral")
        probs    = face_data.get("emotion_probs", {})
        conf     = face_data.get("confidence", 0.5)
        emo_conf = probs.get(emotion, conf) if probs else conf

        # ── Store face data + rolling history ────────────────────────────
        self._last_face_data = face_data
        self._emotion_history.append((emotion, emo_conf, _time.time()))
        if len(self._emotion_history) > 10:
            self._emotion_history.pop(0)

        # Update visual context with emotion trend
        trend = self._emotion_trend()
        self._last_visual_ctx.update({
            "face_present":   True,
            "emotion":        emotion,
            "emotion_conf":   round(emo_conf, 3),
            "emotion_trend":  trend,
        })

        # Always stimulate social/face recognition areas
        self.fabric.stimulate_for_input("face", 0.25)

        # Emotion → neural mapping (import inline to avoid circular)
        from axon.sensory.optic import EMOTION_NEURAL_MAP
        mapping = EMOTION_NEURAL_MAP.get(emotion, EMOTION_NEURAL_MAP["neutral"])

        # Scale stimulation by emotion probability (confidence in the emotion)
        scale = max(0.3, min(1.0, emo_conf * 1.5))

        for cluster, amount in mapping.get("stimulate", []):
            self.fabric.stimulate_region(cluster, amount * scale)

        if mapping.get("reward", 0) > 0:
            self.fabric.neuromod.reward(mapping["reward"] * scale)
        if mapping.get("stress", 0) > 0:
            self.fabric.neuromod.stress(mapping["stress"] * scale)

        # Log significant emotion changes
        if emotion != "neutral" and emo_conf > 0.45:
            self._emit("log", {"msg": f"😶 Emotion: {face_data.get('emoji','')} {emotion} ({int(emo_conf*100)}%) trend:{trend}"})

    def _emotion_trend(self) -> str:
        """Compute trajectory of last N emotions: improving / declining / stable."""
        if len(self._emotion_history) < 3:
            return "stable"
        recent  = [self._EMOTION_VALENCE.get(e, 0) * c for e, c, _ in self._emotion_history[-5:]]
        delta   = recent[-1] - recent[0]
        if delta >  0.25: return "improving"
        if delta < -0.25: return "declining"
        return "stable"

    def _current_face_valence(self) -> float:
        """Return valence score of the current detected emotion."""
        if not self._emotion_history:
            return 0.0
        emotion, conf, _ = self._emotion_history[-1]
        return self._EMOTION_VALENCE.get(emotion, 0.0) * conf

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
            # Snapshot user emotion BEFORE we respond (baseline for feedback)
            valence_before = self._current_face_valence()
            emotion_before = (self._emotion_history[-1][0] if self._emotion_history else "neutral")

            # Light up cognitive regions during LLM inference
            self.fabric.stimulate_for_input("thinking", 0.70)
            self.fabric.stimulate_for_input("memory",   0.55)
            optic_status = self.optic.get_status()
            visual_ctx = {
                "camera_running": optic_status.get("running", False),
                "face_present":   self._last_visual_ctx.get("face_present", False),
                "emotion":        self._last_visual_ctx.get("emotion", "neutral"),
                "emotion_conf":   self._last_visual_ctx.get("emotion_conf", 0.5),
                "emotion_trend":  self._last_visual_ctx.get("emotion_trend", "stable"),
                "motion":         optic_status.get("motion", 0.0),
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
                # Lock mic while Axon speaks — prevents self-hearing
                self.auditory.set_speaking(True)
                self.voice.speak(response)
                self.auditory.set_speaking(False)
                self._emit("voice_speaking", {
                    "speaking": False,
                    "enabled":  self.voice.enabled,
                    "playback": self.voice.get_status().get("playback","none")
                })
                # Stimulate language output neurons
                self.fabric.stimulate_for_input("language_out", 0.78)
                self.fabric.stimulate_for_input("thinking",     0.55)
                self.fabric.neuromod.reward(0.08)

                # ── Emotional Reinforcement Loop ──────────────────────────
                # Give the face sensor ~2 seconds to react to our response
                import time as _t
                _t.sleep(2.0)
                valence_after  = self._current_face_valence()
                emotion_after  = (self._emotion_history[-1][0] if self._emotion_history else "neutral")
                delta_valence  = valence_after - valence_before
                face_present   = bool(self._emotion_history)

                if face_present:
                    if delta_valence >= 0.3:
                        # Positive reaction — reward the pathways that fired
                        reward_amt = min(0.25, delta_valence * 0.5)
                        self.fabric.neuromod.reward(reward_amt)
                        self.fabric.stimulate_for_input("reward_signal", 0.60)
                        self._emit("log", {"msg": f"😊 Emotional reward +{reward_amt:.2f} — user {emotion_before}→{emotion_after}"})
                        self._emit("memory_event", {
                            "type":   "fact",
                            "label":  f"Positive reaction: {emotion_before}→{emotion_after}",
                            "detail": f"Response produced +{delta_valence:.2f} emotional shift",
                        })
                        # Strengthen language↔social Hebbian link — talking this way worked
                        result = self.memory.coactivate("language", "social_empathy")
                        self._emit("hebbian_event", {**result, "type": "new" if result["is_new"] else "strengthen"})

                    elif delta_valence <= -0.3:
                        # Negative reaction — apply stress penalty
                        stress_amt = min(0.20, abs(delta_valence) * 0.4)
                        self.fabric.neuromod.stress(stress_amt)
                        self._emit("log", {"msg": f"😟 Emotional penalty -{stress_amt:.2f} — user {emotion_before}→{emotion_after}"})
                        self._emit("memory_event", {
                            "type":   "episode",
                            "label":  f"Negative reaction: {emotion_before}→{emotion_after}",
                            "detail": f"Response produced {delta_valence:.2f} emotional shift — will adjust",
                        })
                        # Store what caused a negative reaction so the LLM avoids it
                        self.memory.learn(
                            f"negative_reaction_{emotion_after}",
                            f"User responded negatively (became {emotion_after}) to a response on: {user_input[:80]}",
                            confidence=0.8, source="emotional_feedback"
                        )
                    else:
                        # Neutral / stable — small baseline reward for engagement
                        self.fabric.neuromod.reward(0.04)

                    # Store emotional context of this exchange in episodic memory
                    self.memory.store_episode(
                        modality="emotional_feedback",
                        content={
                            "before":   emotion_before,
                            "after":    emotion_after,
                            "delta":    round(delta_valence, 3),
                            "trend":    self._emotion_trend(),
                            "input":    user_input[:120],
                        },
                        topics=["emotion", "reinforcement", emotion_after],
                        importance=0.5 + abs(delta_valence) * 0.5,
                        emotion=emotion_after,
                    )

                # ── Emit memory/pathway events for activity feed ──────────
                ep_count = self.memory.count_episodes()
                self._emit("memory_event", {
                    "type":   "episode",
                    "label":  f"Episode #{ep_count} stored",
                    "detail": "User input encoded to episodic memory",
                })
                top = self.memory.top_connections(5)
                for conn in top[:3]:
                    self._emit("hebbian_event", {
                        "type":   "strengthen",
                        "a":      conn["a"],
                        "b":      conn["b"],
                        "weight": conn["weight"],
                        "fires":  conn["fires"],
                    })
                state_snap = self.fabric.get_state_snapshot()
                for region, act in (state_snap.get("regions") or {}).items():
                    if act > 0.55:
                        self._emit("region_spike", {
                            "region":     region,
                            "activation": round(act, 3),
                            "reason":     "High activation during response generation",
                        })
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

    def get_diagnostic(self) -> dict:
        """Full self-diagnostic — everything AXON knows about itself."""
        import torch, platform, datetime
        try:
            return self._get_diagnostic_impl()
        except Exception as e:
            import traceback
            return {"error": traceback.format_exc()}

    def _get_diagnostic_impl(self) -> dict:
        import torch, platform, datetime
        fabric_state  = self.fabric.get_state_snapshot()
        optic         = self.optic.get_status()
        lang_status   = self.language.get_status()
        voice_status  = self.voice.get_status()
        mem           = self.memory

        # Memory stats
        episodes      = mem.count_episodes()
        facts         = mem.all_facts() or {}
        connections   = [
            {"cluster_a": c.get("a","?"), "cluster_b": c.get("b","?"),
             "strength": c.get("weight", 0), "fires": c.get("fires", 0)}
            for c in (mem.top_connections(5) or [])
        ]

        # Neural fabric details
        clusters      = {name: c.size for name, c in self.fabric.clusters.items()}
        total_neurons = sum(clusters.values())
        total_conns   = fabric_state.get("total_connections", 0)
        try:
            with self.fabric._lock:
                active_conns = int(self.fabric.weight_mat.count_nonzero().item())
        except Exception:
            active_conns = total_conns

        # GPU info
        gpu_name  = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        gpu_mem_used = 0
        gpu_mem_total = 0
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_mem_total = round(props.total_memory / 1024**3, 1)
            gpu_mem_used  = round(torch.cuda.memory_allocated(0) / 1024**3, 2)

        # Capabilities list
        capabilities = []
        if optic.get("running"):
            capabilities.append(f"Vision ({optic.get('detector','unknown')} @ {optic.get('gpu','CPU')})")
        if self.auditory.running:
            capabilities.append("Auditory (Whisper speech-to-text)")
        if voice_status.get("engine"):
            capabilities.append(f"Voice synthesis ({voice_status.get('engine','unknown')})")
        capabilities.append("Episodic + semantic memory (SQLite)")
        capabilities.append("Hebbian learning (fire-together-wire-together)")
        capabilities.append("Ebbinghaus forgetting curve (3-day decay)")
        capabilities.append("Neuromodulator system (6 chemicals)")
        capabilities.append("Facial emotion recognition (FER/YOLOv8)")
        capabilities.append("Web search (DuckDuckGo + Wikipedia)")
        capabilities.append("User modeling (passive preference learning)")
        capabilities.append(f"LLM backend ({lang_status.get('model','unknown')})")

        # Personality snapshot
        pers = fabric_state.get("personality", {})
        neuro = fabric_state.get("neuromod", {})
        emo   = fabric_state.get("emotion", {})

        # Uptime (approximate from memory oldest episode)
        oldest = None
        try:
            row = mem.conn.execute(
                "SELECT MIN(timestamp) FROM episodic"
            ).fetchone()
            if row and row[0]:
                oldest = row[0]
        except Exception:
            pass

        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "neural": {
                "total_neurons":     total_neurons,
                "total_connections": active_conns,
                "num_clusters":      len(clusters),
                "clusters":          clusters,
                "gpu":               gpu_name,
                "gpu_mem_used_gb":   gpu_mem_used,
                "gpu_mem_total_gb":  gpu_mem_total,
            },
            "memory": {
                "episodic_count":   episodes,
                "semantic_facts":   len(facts),
                "hebbian_pathways": len(connections),
                "top_pathways":     connections,
                "oldest_memory":    oldest,
            },
            "senses": {
                "vision":   optic.get("running", False),
                "vision_detector": optic.get("detector","none"),
                "auditory": self.auditory.running,
                "voice_out": bool(voice_status.get("engine")),
            },
            "state": {
                "emotion":     emo,
                "personality": pers,
                "neuromod":    neuro,
            },
            "capabilities": capabilities,
            "platform": {
                "python":   platform.python_version(),
                "os":       platform.system() + " " + platform.release(),
                "torch":    torch.__version__,
            },
        }
