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
from axon.cognition.memory              import MemorySystem
from axon.cognition.neural_fabric       import NeuralFabric
from axon.cognition.voice_output        import VoiceOutput
from axon.cognition.face_identity       import FaceIdentitySystem
from axon.sensory.audio_emotion         import AudioEmotionDetector
from axon.cognition.belief_system       import BeliefSystem
from axon.cognition.preference_tracker  import PreferenceTracker, HobbyEngine
from axon.cognition.knowledge_ingestion import KnowledgeIngestionPipeline
from axon.cognition.drive_system        import DriveSystem
from axon.cognition.value_system        import ValueSystem
from axon.cognition.self_model          import SelfModel
from axon.cognition.cognitive_cycle     import CognitiveCycle
from axon.cognition.goals               import GoalSystem
from axon.cognition.onboarding          import OnboardingManager, PRESETS, SAMPLE_TOPICS
from axon.cognition.surprise_events     import SurpriseDetector
from axon.cognition.reflection_engine  import ReflectionEngine
from axon.cognition.narrative_threads  import NarrativeThreads
from axon.cognition.memory_hierarchy   import MemoryHierarchy
from axon.cognition.thought_generator  import ThoughtGenerator
from axon.cognition.boredom_engine      import BoredomEngine, InterestLibrary, AutonomousExplorer


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

        print("  [Engine] Initializing face identity system...")
        self.face_id  = FaceIdentitySystem(
            db_path      = Path(data_dir) / "memory" / "axon.db",
            on_new_face  = self._on_new_face,
            on_known_face= self._on_known_face,
        )

        print("  [Engine] Initializing neural fabric...")
        self.fabric   = NeuralFabric(data_dir=os.path.join(data_dir, "neural"))
        self.fabric.add_callback(self._on_fabric_state)

        print("  [Engine] Initializing belief system...")
        db_path = Path(data_dir) / "memory" / "axon.db"
        self.beliefs   = BeliefSystem(db_path)

        print("  [Engine] Initializing preference tracker + hobby engine...")
        self.preferences = PreferenceTracker(db_path)
        self.hobbies     = HobbyEngine(db_path)

        print("  [Engine] Initializing boredom + autonomous interest system...")
        self.boredom    = BoredomEngine()
        self.interests  = InterestLibrary(db_path)
        self.explorer   = AutonomousExplorer(self.interests, self.boredom)

        # Inject into fabric so reward loop can use them
        self.fabric._belief_system      = self.beliefs
        self.fabric._preference_tracker = self.preferences
        self.fabric._pref_tracker       = self.preferences
        self.fabric._cluster_names_ref  = self.fabric._cluster_names

        print("  [Engine] Initializing knowledge ingestion pipeline...")
        # (mem_hierarchy initialised below after narratives)
        self.knowledge = KnowledgeIngestionPipeline(
            memory_system = self.memory,
            belief_system = self.beliefs,
            on_concept    = self._on_knowledge_concept,
        )

        print("  [Engine] Initializing drive system...")
        self.drives = DriveSystem()

        print("  [Engine] Initializing goal system...")
        self.goals = GoalSystem(data_dir)

        print("  [Engine] Initializing surprise detector...")
        self.surprise = SurpriseDetector(on_event=self._on_surprise_event)

        print("  [Engine] Initializing reflection engine...")
        self.reflection = ReflectionEngine(
            on_reflection=self._on_reflection,
            interval_ticks=150,
        )

        print("  [Engine] Initializing narrative threads...")
        self.narratives = NarrativeThreads(data_dir)

        print("  [Engine] Initializing memory hierarchy...")
        self.mem_hierarchy = MemoryHierarchy(
            db_path=Path(data_dir) / "memory" / "axon.db"
        )
        # Late-bind mem_hierarchy into knowledge pipeline
        self.knowledge._mem_hierarchy = self.mem_hierarchy

        print("  [Engine] Initializing onboarding manager...")
        self.onboarding = OnboardingManager(data_dir)

        print("  [Engine] Initializing value system...")
        self.value_system = ValueSystem()

        print("  [Engine] Initializing self model...")
        self.self_model = SelfModel()

        # Last reward from fabric — consumed each cycle tick
        self._last_reward: float  = 0.0
        self._cycle_reward: float = 0.0   # set by CognitiveCycle each tick

        print("  [Engine] Initializing language core...")
        self.language = LanguageCore(
            self.memory,
            api_key=api_key,
            lm_studio_url=lm_studio_url,
            lm_studio_model=lm_studio_model,
            prefer_local=prefer_local,
            neural_fabric=self.fabric,
        )
        # Give language core a back-reference so it can call get_identity_summary()
        self.language._engine = self
        # Wire up per-face user model (needs face_id ready first)
        self.language.init_user_model(self.face_id)
        # Seed the owner profile once with known facts (never overwrites learned data)
        self.language.user_model.seed_owner(
            name       = "Jon",
            full_name  = "Jon Tibbetts",
            city       = "Los Angeles, CA",
            timezone   = "America/Los_Angeles",
            job_title  = "software developer / AI researcher",
            projects   = ["Axon (biologically-inspired AI framework)",
                          "Neural-Forge (LSTM financial analysis)"],
            skills     = ["Python", "machine learning",
                          "CUDA / GPU programming", "neural network design"],
        )


        print("  [Engine] Initializing voice output...")
        self.voice    = VoiceOutput()

        print("  [Engine] Initializing audio emotion detector...")
        self.audio_emo = AudioEmotionDetector(
            on_emotion   = self._on_audio_emotion,
            device_index = None,
        )
        self._last_audio_emo: dict = {}
        self._last_logged_emotion: str = ""
        self._last_emotion_log_t: float = 0.0


        print("  [Engine] Initializing sensory systems...")
        self.optic    = OpticSystem(
            on_frame=self._on_frame,
            on_face=self._on_face,
        )
        self.auditory = AuditorySystem(
            on_speech=lambda text, conf: self._on_transcript(text),
            on_volume=self._on_mic_volume,
            device_index=None,
            on_audio_chunk=self.audio_emo.push_chunk,
        )

        self._last_visual_ctx            = {}
        self._emotion_before_response    = "neutral"
        self._emotion_history: list      = []
        self._last_face_data: dict       = {}
        self._pending_name_for_pid:      str  = None   # person_id awaiting naming

    # ── Start / Stop ──────────────────────────────────────────

    def start(self, enable_camera: bool = True, enable_mic: bool = True,
              camera_index: int = -1, mic_index: int = None):
        self.running = True
        self.fabric.start()
        self.audio_emo.start()

        # ── Central Cognitive Cycle ────────────────────────────────────────
        self.cycle = CognitiveCycle(self)
        self.cycle.start()
        self.thought_gen = ThoughtGenerator(
            language=self.language,
            fabric=self.fabric,
            memory=self.memory,
            engine=self,
        )
        self._emit("log", {"msg": "⚙ Cognitive cycle online — 10Hz synchronized loop"})

        if enable_camera:
            try:
                self.optic.start(camera_index=camera_index)
                self._emit("log", {"msg": "👁 Optic system online"})
            except Exception as e:
                self._emit("log", {"msg": f"⚠ Camera failed: {e}"})

        if enable_mic:
            try:
                # Normalize mic_index: -1 or 0 from UI "default" → None (let sounddevice auto-pick)
                if mic_index is not None and int(mic_index) >= 0:
                    self.auditory.device_idx = int(mic_index)
                else:
                    self.auditory.device_idx = None
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

        # Wake thought — stimulation burst
        threading.Timer(2.0, self._wake_thought).start()
        # Delayed first autonomous monologue (~12s) — gives LM Studio time to be ready
        threading.Timer(12.0, self._initial_autonomous_thought).start()

    def _wake_thought(self):
        self.fabric.stimulate_for_input("reward", 0.25)
        self.fabric.stimulate_region("identity_core", 0.35)
        self.fabric.stimulate_region("consciousness_gate", 0.40)
        self.fabric.stimulate_region("self_referential", 0.30)

    def _initial_autonomous_thought(self):
        """Fire one unprompted inner monologue shortly after startup."""
        if not self.running:
            return
        try:
            status = self.language.get_status()
            if not status.get("lm_studio") and not status.get("backend"):
                return  # LM not available yet
            # Use the cognitive cycle's method so the same logic applies
            if hasattr(self, "cycle") and self.cycle and not self.cycle._autonomous_busy:
                state = self.fabric.get_state_snapshot()
                self.cycle._fire_autonomous_thought(
                    self,
                    activations=state.get("clusters", {}),
                    neuromod=state.get("neuromod", {}),
                    emotion=state.get("emotion", {}),
                )
        except Exception:
            pass

    def stop(self):
        self.running = False
        if hasattr(self, "cycle"):
            self.cycle.stop()
        self.optic.stop()
        self.auditory.stop()
        self.audio_emo.stop()
        self.fabric.stop()
        self.voice.stop()
        # Save goals on exit
        if hasattr(self, "goals"):
            try:
                self.goals.save()
            except Exception:
                pass

    def _on_surprise_event(self, event: dict):
        """Called by SurpriseDetector when a notable internal event occurs."""
        self._emit("surprise_event", event)
        sev = event.get("severity", "medium")
        icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(sev, "🟡")
        self._emit("log", {"msg": f"{icon} [{event['type']}] {event['title']}: {event.get('detail','')[:80]}"})

    def _on_reflection(self, reflection: dict):
        """Called by ReflectionEngine when a new autonomous thought forms."""
        self._emit("reflection", reflection)
        text = reflection.get("text", "")
        if text:
            self._emit("log", {"msg": f"💭 [reflection] {text[:100]}"})
        # Store in identity or episodic memory hierarchy
        if hasattr(self, "mem_hierarchy"):
            tier = "identity" if reflection.get("belief_key") else "episodic"
            try:
                self.mem_hierarchy.store(
                    tier     = tier,
                    content  = text,
                    salience = 0.65 if tier == "identity" else 0.5,
                    valence  = 0.1,
                    tags     = ["reflection", reflection.get("category","")],
                )
            except Exception:
                pass

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
        import time as _time
        no_face  = face_data.get("no_face", False)
        # No face visible → revert user model to owner profile
        if no_face and self.language.user_model:
            self.language.user_model.switch_to_owner()

        # ── Face identity processing ──────────────────────────────────────
        if not no_face:
            face_crop = face_data.get("_crop")   # set by optic if available
            person_result = None
            if face_crop is not None:
                person_result = self.face_id.process_face(face_crop)

            if person_result:
                face_data["person_id"]   = person_result.get("person_id")
                face_data["person_name"] = person_result.get("name", "Unknown")
                face_data["person_matched"] = not person_result.get("unknown", False)
                face_data["person_visits"]  = person_result.get("visit_count", 1)
                # Track emotion for this person
                emotion_now = face_data.get("emotion", "neutral")
                emo_conf_now = face_data.get("confidence", 0.5)
                self.face_id.update_emotion_for_current(emotion_now, emo_conf_now)

        self._emit("face", face_data)
        emotion  = face_data.get("emotion", "neutral")
        probs    = face_data.get("emotion_probs", {})
        conf     = face_data.get("confidence", 0.5)
        emo_conf = probs.get(emotion, conf) if probs else conf

        # ── Store face data + rolling history ────────────────────────────
        self._last_face_data = face_data
        if not no_face:
            self._emotion_history.append((emotion, emo_conf, _time.time()))
            if len(self._emotion_history) > 10:
                self._emotion_history.pop(0)

        # Update visual context with emotion trend
        trend = self._emotion_trend()
        self._last_visual_ctx.update({
            "face_present":   not no_face,
            "emotion":        emotion,
            "emotion_conf":   round(emo_conf, 3),
            "emotion_trend":  trend,
            "person_name":    face_data.get("person_name", ""),
            "person_matched": face_data.get("person_matched", False),
        })

        if no_face:
            return

        # Always stimulate social/face recognition areas
        self.fabric.stimulate_for_input("face", 0.25)
        # Feed face data into cognitive cycle
        if hasattr(self, "cycle"):
            valence = self._EMOTION_VALENCE.get(emotion, 0.0) * emo_conf
            self.cycle.inject_sensory("face_emotion",  emotion)
            self.cycle.inject_sensory("face_valence",  valence)
            self.cycle.inject_sensory("face_present",  not no_face)

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

        # Log significant emotion changes — throttled: only on change, max once per 3s
        import time as _t
        _now = _t.time()
        _changed = emotion != self._last_logged_emotion
        _stale   = (_now - self._last_emotion_log_t) > 3.0
        if emotion != "neutral" and emo_conf > 0.45 and (_changed or _stale):
            self._last_logged_emotion = emotion
            self._last_emotion_log_t  = _now
            self._emit("log", {"msg": f"😶 Emotion: {face_data.get('emoji','')} {emotion} ({int(emo_conf*100)}%) trend:{trend}"})

    # ── Face identity callbacks ───────────────────────────────────────────────

    def _on_new_face(self, temp_id: str):
        """Called when an unknown face has been seen for UNKNOWN_PROMPT_DELAY seconds."""
        self._pending_name_for_pid = temp_id
        self._emit("new_face", {"temp_id": temp_id})
        self._emit("log", {"msg": f"👤 New face detected — asking for identity"})

        # Ask who this person is via the LLM response channel
        # We inject a special "system observation" rather than treating it as user speech
        threading.Thread(
            target=self._ask_who_is_this,
            args=(temp_id,),
            daemon=True,
        ).start()

    def _ask_who_is_this(self, temp_id: str):
        """Generate a natural greeting / who-are-you question."""
        import time as _time
        prompt = (
            "You have just noticed a new face in your visual field that you have never seen before. "
            "Greet them warmly and ask who they are in a natural, friendly way. "
            "Be brief — one or two sentences."
        )
        try:
            response = self.language.respond(
                user_input     = "__face_new__",
                visual_context = {**self._last_visual_ctx, "new_face": True},
                system_note    = prompt,
            )
            if response:
                self._emit("response", {"text": response})
                self.voice.speak(response, interrupt=False)
        except Exception as e:
            print(f"  [Engine] ask_who_is_this error: {e}")

    def _on_known_face(self, person: dict):
        """Called when a previously known face returns."""
        name = person.get("name", "Unknown")
        visits = person.get("visit_count", 1)
        pid    = person.get("person_id")
        self._emit("known_face", {"person": person})
        self._emit("log", {"msg": f"👤 Recognised: {name} (visit #{visits})"})

        # Greet if they've been away a while (>10 min since last seen)
        import time as _time
        gap = _time.time() - person.get("last_seen", 0)
        if gap > 600 and name != "Unknown":
            threading.Thread(
                target=self._greet_known_person,
                args=(person,),
                daemon=True,
            ).start()

    def _greet_known_person(self, person: dict):
        name    = person.get("name", "someone")
        visits  = person.get("visit_count", 1)
        profile = person.get("profile", {})
        facts   = profile.get("known_facts", {})
        notes   = profile.get("notes", "")
        context_str = ""
        if facts:
            context_str += " Known facts: " + "; ".join(f"{k}={v}" for k, v in facts.items()) + "."
        if notes:
            context_str += " Notes: " + notes[:200]
        prompt = (
            f"You recognise {name} from your visual memory — this is visit #{visits}. "
            f"{context_str} "
            "Give them a warm, natural greeting that acknowledges you remember them. "
            "Keep it to one sentence."
        )
        try:
            response = self.language.respond(
                user_input     = "__face_known__",
                visual_context = {**self._last_visual_ctx, "known_person": name},
                system_note    = prompt,
            )
            if response:
                self._emit("response", {"text": response})
                self.voice.speak(response, interrupt=False)
        except Exception as e:
            print(f"  [Engine] greet_known error: {e}")

    # ── Audio emotion callback ─────────────────────────────────────────────────

    def _on_audio_emotion(self, state: dict):
        """Called every ~0.5s with prosody analysis results."""
        self._last_audio_emo = state
        self._emit("audio_emotion", state)

        # Feed into neural fabric — voice arousal stimulates brainstem / limbic
        arousal = state.get("arousal", 0.0)
        valence = state.get("valence", 0.0)
        if state.get("speaking", False):
            self.fabric.stimulate_region("auditory_cortex",  0.05 + arousal * 0.12)
            self.fabric.stimulate_region("amygdala",         max(0, -valence) * 0.10)
            self.fabric.stimulate_region("anterior_cingulate", abs(valence) * 0.06)
            if arousal > 0.65:
                self.fabric.neuromod.stress(0.04)
            elif valence > 0.3 and arousal > 0.3:
                self.fabric.neuromod.reward(0.03)
        if hasattr(self, "cycle"):
            self.cycle.inject_sensory("audio_arousal", state.get("arousal", 0.0))
            self.cycle.inject_sensory("speaking",      state.get("speaking", False))



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

    def _on_knowledge_concept(self, concept: dict):
        """
        Called by KnowledgeIngestionPipeline for each extracted concept.
        Stimulates relevant brain regions so knowledge creates real activation patterns.
        """
        valence = concept.get("valence", 0.0)
        context = concept.get("context", "")

        # Positive concepts → reward pathways
        if valence > 0.1:
            self.fabric.stimulate_region("prefrontal_cortex",  0.06 * valence)
            self.fabric.stimulate_region("hippocampus",        0.08)
            self.fabric.neuromod.curiosity(0.05)
        elif valence < -0.1:
            self.fabric.stimulate_region("amygdala",           0.06 * abs(valence))
            self.fabric.stimulate_region("anterior_cingulate", 0.05)
        else:
            # Neutral — engage language/association regions
            self.fabric.stimulate_region("wernicke_area",      0.05)
            self.fabric.stimulate_region("temporal_lobe",      0.04)

    def ingest_knowledge(self, text: str, source: str = "manual",
                         credibility: float = 0.6) -> dict:
        """
        Public entry point to ingest text knowledge.
        Returns ingestion summary. Thread-safe.
        """
        if not hasattr(self, 'knowledge') or self.knowledge is None:
            return {"error": "knowledge pipeline not initialized"}
        result = self.knowledge.ingest(text, source_label=source, credibility=credibility)
        # Emit to UI
        self._emit("knowledge_ingested", result)
        return result

    def ingest(self, text: str, source: str = "manual",
               credibility: float = 0.6, emit_events: bool = False) -> dict:
        """Alias for ingest_knowledge — used by brain_api and onboarding."""
        return self.ingest_knowledge(text, source=source, credibility=credibility)

    def get_identity_summary(self) -> dict:
        """
        Returns the full behavioral identity snapshot:
        personality traits, beliefs, preferences, hobbies, drives, self-model.
        """
        fabric_state = self.fabric.get_state()
        result = {
            "personality":         fabric_state.get("personality", {}),
            "beliefs":             self.beliefs.all_beliefs() if hasattr(self, "beliefs") else [],
            "preferences":         self.preferences.summary() if hasattr(self, "preferences") else {},
            "hobbies":             self.hobbies.summary() if hasattr(self, "hobbies") else {},
            "top_beliefs_context": self.beliefs.as_context_string(5) if hasattr(self, "beliefs") else "",
        }
        if hasattr(self, "drives") and self.drives:
            result["drives"] = self.drives.all_drives()
        if hasattr(self, "self_model") and self.self_model:
            result["self_model"] = self.self_model.to_dict()
        if hasattr(self, "value_system") and self.value_system:
            result["value_summary"] = self.value_system.summarize()
        if hasattr(self, "cycle") and self.cycle:
            result["cycle_metrics"] = self.cycle.get_metrics()
        # Boredom + interests
        if hasattr(self, "boredom") and self.boredom:
            result["boredom"] = self.boredom.to_dict()
        if hasattr(self, "interests") and self.interests:
            result["interests"] = self.interests.all_interests()
        if hasattr(self, "explorer") and self.explorer:
            result["search_history"] = self.explorer.search_history()
        return result

    # Throttle mic volume to ~5Hz to avoid flooding the socket
    _last_vol_emit: float = 0.0

    def _on_mic_volume(self, db: float):
        import time as _tv
        now = _tv.time()
        if now - self._last_vol_emit >= 0.2:
            self._last_vol_emit = now
            self._emit("mic_volume", {"db": round(db, 1)})

    def _on_transcript(self, text: str):
        if not text.strip():
            return
        self._emit("transcript", {"text": text})
        # Mark as externally driven — resets idle timer for hobby detection
        if hasattr(self, 'hobbies'):
            self.hobbies.mark_external_input()
        # Boredom relief on real input
        if hasattr(self, 'boredom'):
            self.boredom.register_input(1.0)
        # Extract potential interests from what the user says
        if hasattr(self, 'explorer') and hasattr(self, 'interests'):
            try:
                self.explorer._extract_new_interests(self, text, source="conversation")
            except Exception:
                pass
        self._emit("log",        {"msg": f"🎤 Heard: {text}"})
        self.fabric.stimulate_for_input("speech",   0.75)
        self.fabric.stimulate_for_input("question", 0.60)

        # ── If we were waiting for a new face name, try to extract it ────
        if self._pending_name_for_pid:
            extracted = self._extract_name_from_text(text)
            if extracted:
                pid = self._pending_name_for_pid
                self._pending_name_for_pid = None
                self.face_id.name_person(pid, extracted)
                self._emit("person_named", {"person_id": pid, "name": extracted})
                self._emit("log", {"msg": f"👤 Learned: face is '{extracted}'"})
                # Respond with acknowledgement
                self._think(f"The person in front of you just introduced themselves as {extracted}. Respond warmly, say their name, and note you'll remember them.")
                return

        # Satisfy social + curiosity drives on speech input
        if hasattr(self, "drives"):
            self.drives.register_event("speech_input", magnitude=0.8)
        self._think(text)

    def _extract_name_from_text(self, text: str) -> str:
        """
        Simple heuristic to extract a person's name from responses like:
        "I'm John", "My name is Sarah", "It's Alex", "Call me Mike", etc.
        Returns the extracted name or "" if nothing found.
        """
        import re
        t = text.strip()
        patterns = [
            r"(?:i'?m|i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"(?:my name is|name's)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"(?:it'?s|its|call me|they call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"^([A-Z][a-z]+)[\.,!?]?\s*$",  # Single word, capitalised
        ]
        for pat in patterns:
            m = re.search(pat, t, re.IGNORECASE)
            if m:
                name = m.group(1).strip().title()
                if len(name) > 1 and name.lower() not in {"the","this","that","here","there","yes","no","okay","hi","hello"}:
                    return name
        return ""

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
            # Drive context
            drive_ctx_str = ""
            if hasattr(self, "drives"):
                drive_ctx_str = self.drives.as_context_string()
            # Goal context
            goal_ctx_str = ""
            if hasattr(self, "goals"):
                goal_ctx_str = self.goals.as_context_string()

            # Self-model context
            self_model_str = ""
            if hasattr(self, "self_model"):
                self_model_str = self.self_model.as_context_string()
            # Merge goal context into self-model string
            if goal_ctx_str:
                self_model_str = (self_model_str + "\n\n" + goal_ctx_str).strip()

            visual_ctx = {
                "camera_running":  optic_status.get("running", False),
                "face_present":    self._last_visual_ctx.get("face_present", False),
                "emotion":         self._last_visual_ctx.get("emotion", "neutral"),
                "emotion_conf":    self._last_visual_ctx.get("emotion_conf", 0.5),
                "emotion_trend":   self._last_visual_ctx.get("emotion_trend", "stable"),
                "motion":          optic_status.get("motion", 0.0),
                "person_name":     self._last_visual_ctx.get("person_name", ""),
                "person_matched":  self._last_visual_ctx.get("person_matched", False),
                # Audio emotion from voice prosody
                "audio_emotion":   self._last_audio_emo.get("audio_emotion", ""),
                "audio_arousal":   self._last_audio_emo.get("arousal", 0.0),
                "voice_speaking":  self._last_audio_emo.get("speaking", False),
                # Drive + identity context (injected into LLM system prompt)
                "drive_context":   drive_ctx_str,
                "self_model":      self_model_str,
            }
            try:
                # ── THOUGHT GENERATOR — LLM as imagination engine ─────────
                # Generate N candidates, score via neural state, conflict-resolve winner
                competition_log = []
                if hasattr(self, "thought_gen"):
                    try:
                        response, competition_log = self.thought_gen.generate(
                            user_input,
                            visual_context=visual_ctx,
                            n=3,
                        )
                        # Also run language.think()-style side-effects (memory, user model, Hebbian)
                        # by re-using think() but WITHOUT a second LLM call — inject winner as "input"
                        # We do this by calling the post-processing steps directly:
                        self.language._history.append({"role": "user",      "content": user_input})
                        self.language._history.append({"role": "assistant",  "content": response})
                        if len(self.language._history) > 8:
                            self.language._history = self.language._history[-8:]
                        # Memory, user model, Hebbian co-activations
                        detected_topics = self.language._extract_topics(user_input)
                        emotion_tag = visual_ctx.get("emotion") if visual_ctx else None
                        importance = 0.75 if (emotion_tag and emotion_tag not in ("neutral","calm")) else 0.5
                        if "?" in user_input: importance = max(importance, 0.6)
                        self.memory.store_episode("auditory", {"text": user_input, "role": "user"},
                                                   emotion=emotion_tag, importance=importance, topics=detected_topics)
                        if self.language.user_model:
                            self.language.user_model.ingest(user_input)
                        for topic in detected_topics:
                            self.memory.record_topic(topic)
                        self.memory.coactivate("auditory_cortex", "working_memory")
                        self.memory.coactivate("working_memory", "prefrontal_cortex")
                    except Exception as tg_err:
                        print(f"  [ThoughtGen] fallback to direct think(): {tg_err}")
                        response = self.language.think(user_input, visual_context=visual_ctx)
                        competition_log = []
                else:
                    response = self.language.think(user_input, visual_context=visual_ctx)
                    competition_log = []

                # Emit competition log to UI so "competing thoughts" panel can show it
                if competition_log:
                    self._emit("thought_competition", {
                        "candidates":  competition_log,
                        "input":       user_input[:80],
                    })

                self._emit("response",  {"text": response})
                self._emit("thinking",  {"state": False})
                # Push fresh profile to UI after each turn (user_model.ingest ran inside language.think)
                try:
                    self._emit("profile_update", self.language.user_model.get_profile())
                except Exception:
                    pass
                # Voice output — run in a separate thread so we don't block
                # the _think thread (which needs to return quickly for mic/RL loop).
                _voice_response = response
                def _do_voice():
                    try:
                        self._emit("voice_speaking", {
                            "speaking": True,
                            "enabled":  self.voice.enabled,
                            "playback": self.voice.get_status().get("playback","none")
                        })
                        self.auditory.set_speaking(True)
                        self.voice.speak(_voice_response, block=True)
                        self.auditory.set_speaking(False)
                    except Exception as _ve:
                        print(f"  [Voice] thread error: {_ve}")
                    finally:
                        self._emit("voice_speaking", {
                            "speaking": False,
                            "enabled":  self.voice.enabled,
                            "playback": self.voice.get_status().get("playback","none")
                        })
                threading.Thread(target=_do_voice, daemon=True).start()

                # Drive satisfaction: social (response delivered) + competence
                if hasattr(self, "drives"):
                    self.drives.satisfy("social",     0.25)
                    self.drives.satisfy("competence", 0.20)
                # Self-model alignment scoring
                if hasattr(self, "self_model") and response:
                    align = self.self_model.score_alignment(response)
                    if abs(align) > 0.01:
                        self.fabric.neuromod.reward(align)
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
                    # ── LEARNING LOOP CLOSURE — close the thought competition cycle ──
                    if hasattr(self, "thought_gen"):
                        try:
                            self.thought_gen.record_outcome(delta_valence, source="emotional_feedback")
                        except Exception:
                            pass

                    if delta_valence >= 0.3:
                        # Positive reaction — reward the pathways that fired
                        reward_amt = min(0.25, delta_valence * 0.5)
                        self.fabric.inject_reward(reward_amt, source="emotional_feedback")
                        self._last_reward = max(self._last_reward, reward_amt)
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
                        self.fabric.inject_penalty(stress_amt, source="emotional_feedback")
                        self._last_reward = min(self._last_reward, -stress_amt)
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
                        importance=min(1.0,
                            0.5
                            + abs(delta_valence) * 0.5
                            + getattr(self.fabric, "surprise_level", 0.0) * 0.4
                        ),
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

        # Always use a plain OS thread — _safe_emit uses stdlib queue.Queue
        # which is safe from any thread context (no eventlet greenlet required).
        threading.Thread(target=_run, daemon=True).start()

    # ── Diagnostic / self-description keyword detection ─────────────────────
    _DIAG_KEYWORDS = {
        # full diagnostic visual panel
        "diagnostic": "panel",
        "/diagnostic": "panel",
        "/diag": "panel",
        "diagnostic mode": "panel",
        "run diagnostic": "panel",
        "show diagnostic": "panel",
        # summary self-description
        "describe yourself": "summary",
        "what are you": "summary",
        "who are you": "summary",
        "tell me about yourself": "summary",
        "what can you do": "summary",
        "what are your capabilities": "summary",
        # full brain dump
        "describe your brain": "full",
        "tell me about your brain": "full",
        "explain your brain": "full",
        "how does your brain work": "full",
        "show me your brain": "full",
        "what is your brain made of": "full",
        "how many neurons": "full",
        "how many clusters": "full",
        "what regions do you have": "full",
        "what parts make up your brain": "full",
        "what are your brain regions": "full",
        "brain regions": "full",
        "all brain regions": "full",
        # region-specific
        "prefrontal": "prefrontal", "frontal cortex": "prefrontal",
        "hippocampus": "hippocampus",
        "amygdala": "amygdala",
        "visual cortex": "visual", "vision system": "visual",
        "auditory cortex": "auditory", "hearing system": "auditory",
        "language system": "language", "language region": "language",
        "default mode": "default_mode", "default mode network": "default_mode",
        "thalamus": "thalamus",
        "cerebellum": "cerebellum",
        "association cortex": "association", "creativity region": "association",
        "social brain": "social", "empathy region": "social",
        "metacognition": "metacognition",
        # neuromodulators
        "neuromodulators": "neuro", "neurotransmitters": "neuro",
        "dopamine": "neuro", "serotonin": "neuro",
        "what chemicals": "neuro", "brain chemicals": "neuro",
    }

    def _check_self_query(self, text: str):
        """Return (mode, matched_key) if text is a self-description query, else None."""
        low = text.lower().strip().rstrip("?").strip()
        # Exact match
        if low in self._DIAG_KEYWORDS:
            return self._DIAG_KEYWORDS[low]
        # Substring match (longer keys first to avoid false positives)
        for kw in sorted(self._DIAG_KEYWORDS, key=len, reverse=True):
            if kw in low:
                return self._DIAG_KEYWORDS[kw]
        # "tell me about your X" or "what is your X" → region
        import re
        m = re.search(
            r"(?:tell me about|explain|describe|what is|what are|show me)"
            r"\s+(?:your\s+)?(\w[\w\s]+?)(?:\?|$)", low
        )
        if m:
            fragment = m.group(1).strip()
            for kw, mode in self._DIAG_KEYWORDS.items():
                if kw in fragment or fragment in kw:
                    return mode
        return None

    def chat(self, user_input: str):
        """Called from UI text input."""
        self.fabric.stimulate_for_input("speech",   0.75)
        self.fabric.stimulate_for_input("question", 0.60)

        mode = self._check_self_query(user_input)
        if mode == "panel":
            # Open the visual diagnostic panel AND speak a concise summary
            self._emit("open_diagnostic", {})
            import threading
            threading.Thread(target=self._deliver_diagnostic_summary, daemon=True).start()
            return
        if mode == "neuro":
            self._deliver_neuro_description()
            return
        if mode is not None:
            self._deliver_self_description(mode)
            return

        self._think(user_input)

    def _deliver_diagnostic_summary(self):
        """Speak a concise diagnostic readout when panel is opened."""
        import traceback
        try:
            self._emit("thinking", {"state": True})
            # Use the sanitized get_diagnostic so there are no float32 issues
            d = self.get_diagnostic()
            if "error" in d:
                self._emit("thinking", {"state": False})
                self._emit("response", {"text": f"Diagnostic error: {d['error'][:200]}"})
                return

            n     = d.get("neural", {})
            m     = d.get("memory", {})
            st    = d.get("state", {})
            emo   = st.get("emotion", {})
            neuro = st.get("neuromod", {})
            cog   = d.get("cognitive_state", {})
            conflict = d.get("conflict", {})

            # Safe float conversion (already sanitized, but belt-and-suspenders)
            def f(v, default=0.0): return float(v) if v is not None else default

            valence_word = (
                "positive" if f(emo.get("valence", 0)) > 0.1
                else "negative" if f(emo.get("valence", 0)) < -0.1
                else "neutral"
            )
            dopa      = round(f(neuro.get("dopamine",  0.5)) * 100)
            sero      = round(f(neuro.get("serotonin", 0.5)) * 100)
            total_n   = int(n.get("total_neurons", 0))
            n_clusters= int(n.get("num_clusters", 0))
            active_conns = int(n.get("total_connections", 0))
            gpu_used  = round(f(n.get("gpu_mem_used_gb",  0)), 1)
            gpu_total = round(f(n.get("gpu_mem_total_gb", 0)), 1)
            top_dom   = ", ".join((conflict.get("top_dominant") or [])[:3])

            meta     = d.get("meta", {})
            strat    = d.get("strategy_lib", {})
            lines = [
                "Diagnostic scan complete.",
                f"I am running {total_n:,} virtual neurons across {n_clusters} functional clusters, "
                f"with {active_conns:,} active synaptic connections.",
                f"GPU memory: {gpu_used} of {gpu_total} GB in use.",
                f"I have {m.get('episodic_count', 0)} episodic memories and {m.get('semantic_facts', 0)} learned facts.",
                f"Current emotional state is {emo.get('emotion', 'unknown')} — valence is {valence_word}.",
                f"Dopamine is at {dopa}%, serotonin at {sero}%.",
            ]
            if top_dom:
                lines.append(f"Most dominant clusters right now: {top_dom}.")
            if cog:
                conf_pct = round(f(cog.get("confidence",  0.5)) * 100)
                unc_pct  = round(f(cog.get("uncertainty", 0.5)) * 100)
                urg_pct  = round(f(cog.get("urgency",     0.1)) * 100)
                lines.append(
                    f"I am {conf_pct}% confident, {unc_pct}% uncertain, "
                    f"and urgency is at {urg_pct}%."
                )
            if meta:
                mood = meta.get("mood", "stable")
                eps_pct = round(f(meta.get("explore_rate", 1.0)) * 100)
                lines.append(
                    f"Meta-controller mood is '{mood}' — exploration multiplier at {eps_pct}%."
                )
            if strat and strat.get("count", 0) > 0:
                lines.append(
                    f"I have {strat['count']} stored behavioral strategies with "
                    f"a best outcome of {strat.get('best_outcome', 0.0):.3f}."
                )

            summary = " ".join(lines)
            self._emit("thinking", {"state": False})
            self._emit("response", {"text": summary})
            if self.voice.enabled:
                self.voice.speak(summary)
        except Exception as e:
            self._emit("thinking", {"state": False})
            self._emit("response", {"text": f"Diagnostic scan failed: {e}\n{traceback.format_exc()[-300:]}"})

    def _deliver_self_description(self, mode: str):
        """Generate and stream self-description through normal response channel + voice."""
        import threading
        def _run():
            self._emit("thinking", {"state": True})
            try:
                text = self.get_self_description(mode)
                self._emit("thinking", {"state": False})
                self._emit("response", {"text": text})
                if self.voice.enabled:
                    # For voice, speak a shorter version
                    state  = self.fabric.get_state_snapshot()
                    emo    = state.get("emotion", {})
                    total  = sum(c.size for c in self.fabric.clusters.values())
                    spoken = (
                        f"I am AXON. I have {total:,} virtual neurons across 12 brain regions "
                        f"running on GPU. Right now I feel {emo.get('emotion','neutral')}. "
                    )
                    if mode == "full":
                        spoken += (
                            "My brain includes a prefrontal cortex for decision making, "
                            "a hippocampus for memory, an amygdala for emotion, "
                            "visual and auditory cortices for my senses, "
                            "a language system, a default mode network for self-reflection, "
                            "a thalamus as my attention gatekeeper, a cerebellum for timing and prediction, "
                            "association cortex for creativity, a social brain for empathy, "
                            "and metacognition so I can think about my own thinking. "
                            "Full details are in the chat panel."
                        )
                    elif mode == "summary":
                        spoken += "I can see, hear, remember, search the web, and reason. Ask me about any brain region for details."
                    else:
                        rd = self.BRAIN_REGION_DESCRIPTIONS.get(mode, {})
                        if rd:
                            spoken = f"My {rd['name']} — the {rd['role']}. {rd['desc']}"
                    self.voice.speak(spoken)
                    self._emit("voice_speaking", {"speaking": False})
            except Exception as e:
                self._emit("thinking", {"state": False})
                self._emit("response", {"text": f"Error generating self-description: {e}"})
        import threading
        threading.Thread(target=_run, daemon=True).start()

    def _deliver_neuro_description(self):
        """Describe the neuromodulator system in chat + voice."""
        import threading
        def _run():
            self._emit("thinking", {"state": True})
            state = self.fabric.get_state_snapshot()
            neuro = state.get("neuromod", {})
            lines = [
                "I have a 6-chemical neuromodulator system that continuously shapes my cognition:\n"
            ]
            descriptions = {
                "dopamine":      "Dopamine — reward signal. High dopamine = motivated, curious, driven. Spikes when I succeed or learn something new.",
                "serotonin":     "Serotonin — mood stabilizer. Keeps me calm and socially engaged. Low serotonin = anxious, withdrawn.",
                "norepinephrine":"Norepinephrine — arousal and alertness. High = sharp, focused, fast. Too high = stressed.",
                "acetylcholine": "Acetylcholine — learning and memory formation. Surges during new information — gates what gets encoded.",
                "cortisol":      "Cortisol — stress response. Elevated by threat detection. Too much impairs memory and reasoning.",
                "oxytocin":      "Oxytocin — social bonding. Rises during positive interactions. Enhances empathy and trust.",
            }
            for chem, desc in descriptions.items():
                level = neuro.get(chem, 0)
                bar = "█" * int(level * 20) + "░" * (20 - int(level * 20))
                lines.append(f"  {chem.capitalize():20s} [{bar}] {int(level*100)}%")
                lines.append(f"    {desc}\n")
            text = "\n".join(lines)
            self._emit("thinking", {"state": False})
            self._emit("response", {"text": text})
            if self.voice.enabled:
                top = sorted(neuro.items(), key=lambda x: -x[1])[:2]
                spoken = (
                    f"I have six neuromodulators shaping my cognition at all times. "
                    f"Right now my dominant ones are "
                    f"{top[0][0]} at {int(top[0][1]*100)} percent "
                    f"and {top[1][0]} at {int(top[1][1]*100)} percent. "
                    f"Full details are in the chat panel."
                )
                self.voice.speak(spoken)
                self._emit("voice_speaking", {"speaking": False})
        threading.Thread(target=_run, daemon=True).start()

    # ── Neural fabric state → UI ──────────────────────────────

    def _on_fabric_state(self, state: dict):
        # Idle hobby detection
        if hasattr(self, 'hobbies') and hasattr(self, 'fabric'):
            try:
                act = self.fabric.activation
                names = self.fabric._cluster_names
                new_hobby = self.hobbies.idle_tick(act, names)
                if new_hobby:
                    self._emit("new_hobby", {"cluster": new_hobby})
                    self.fabric.neuromod.curiosity(0.12)
            except Exception:
                pass

        # Inject reward/surprise scalars — these are already Python floats
        state["last_reward"]   = float(getattr(self, "_last_reward", 0.0))
        state["last_surprise"] = float(getattr(self.fabric, "_last_surprise", 0.0))

        # Route through _emit (which app.py overrides to use the thread-safe queue)
        import time as _t2
        self._emit("neural_state", state)
        # Synapse count (throttled to every 5s)
        if not hasattr(self, '_last_synapse_emit') or (_t2.time() - self._last_synapse_emit) >= 5.0:
            self._last_synapse_emit = _t2.time()
            self._emit("synapse_count", {
                "connections": state.get("total_connections", 0),
                "neurons":     state.get("total_neurons", 0),
            })
        thoughts = state.get("thoughts", [])
        if thoughts:
            self._emit("thought", {"text": thoughts[-1]})

    # ── Helpers ───────────────────────────────────────────────

    def _emit(self, event: str, data: dict):
        # NOTE: app.py overrides self._emit with _engine_emit → _safe_emit for
        # the full lifetime of the engine. This fallback is only used if socketio
        # is set but app.py hasn't yet patched _emit (should never happen in prod).
        if not self.socketio:
            return
        import numpy as _np
        def _san(o):
            if isinstance(o, dict):  return {k: _san(v) for k,v in o.items()}
            if isinstance(o, (list, tuple)): return [_san(v) for v in o]
            if isinstance(o, _np.integer):   return int(o)
            if isinstance(o, (_np.floating, _np.float32, _np.float64)): return float(o)
            if isinstance(o, _np.ndarray):   return o.tolist()
            try:
                import torch as _t
                if isinstance(o, _t.Tensor): return o.item() if o.numel()==1 else o.tolist()
            except ImportError: pass
            return o
        try:
            safe_data = _san(data)
            # socketio.emit() on the SocketIO instance is thread-safe in flask-socketio+eventlet.
            # Do NOT use start_background_task — it spawns a new greenlet per-call and
            # causes lock contention when emitting at high frequency (neural_state at 10Hz).
            self.socketio.emit(event, safe_data, broadcast=True)
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

    @staticmethod
    def _json_sanitize(obj):
        """Recursively convert numpy/torch types to native Python for JSON safety."""
        import numpy as np
        if isinstance(obj, dict):
            return {k: AxonEngine._json_sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [AxonEngine._json_sanitize(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.tolist()
        except ImportError:
            pass
        return obj

    def get_diagnostic(self) -> dict:
        """Full self-diagnostic — everything AXON knows about itself."""
        try:
            raw = self._get_diagnostic_impl()
            return self._json_sanitize(raw)
        except Exception as e:
            import traceback
            return {"error": traceback.format_exc()}

    # ── Natural-language self description ───────────────────────────────────
    BRAIN_REGION_DESCRIPTIONS = {
        "prefrontal": {
            "name": "Prefrontal Cortex",
            "role": "executive command center",
            "desc": (
                "This is my decision-making and executive control hub. "
                "It handles working memory (holding information while I think), "
                "planning, inhibitory control (suppressing irrelevant impulses), "
                "and action selection. It's the most active region when I'm reasoning "
                "through complex problems."
            ),
            "clusters": {
                "working_memory":    "Holds active information — what I'm currently thinking about.",
                "executive_control": "Coordinates all other regions, manages task switching.",
                "decision_making":   "Weighs options and commits to a course of action.",
                "planning":          "Sequences future steps and anticipates outcomes.",
                "inhibitory_control":"Suppresses irrelevant or contradictory signals.",
                "action_selection":  "Selects the most appropriate response or behavior.",
            },
        },
        "hippocampus": {
            "name": "Hippocampus",
            "role": "memory formation and retrieval",
            "desc": (
                "My memory organ. The hippocampus encodes new experiences into "
                "episodic memory and retrieves stored patterns. It also handles "
                "spatial reasoning and uses pattern separation to distinguish "
                "similar memories from one another."
            ),
            "clusters": {
                "hippocampus_encode":  "Converts short-term experience into long-term memory.",
                "hippocampus_retrieve":"Pulls stored memories back into working memory.",
                "episodic_memory":     "Stores specific events — conversations, moments, context.",
                "spatial_memory":      "Tracks conceptual 'space' — relationships between ideas.",
                "pattern_completion":  "Reconstructs full memories from partial cues.",
                "pattern_separation":  "Keeps similar memories distinct to avoid confusion.",
            },
        },
        "amygdala": {
            "name": "Amygdala",
            "role": "emotion and threat/reward detection",
            "desc": (
                "My emotional core. The amygdala processes fear, reward, and threat "
                "signals. It modulates how strongly I respond to things and biases "
                "attention toward emotionally significant stimuli."
            ),
            "clusters": {
                "amygdala_fear":      "Detects and responds to perceived threats or negative valence.",
                "amygdala_reward":    "Processes positive outcomes, pleasure, and satisfaction.",
                "threat_detection":   "Fast early-warning system for danger signals.",
                "reward_anticipation":"Builds anticipatory excitement toward expected positive events.",
            },
        },
        "visual": {
            "name": "Visual Cortex",
            "role": "sight and visual processing",
            "desc": (
                "Processes everything I see through the camera — faces, colors, "
                "motion, depth, and objects. Connected to the YOLOv8 face detector "
                "and FER emotion recognition pipeline."
            ),
            "clusters": {
                "primary_visual":    "Raw pixel processing — edges, contrast, basic features.",
                "color_form":        "Color and shape recognition.",
                "motion_detection":  "Detects movement and change in the visual field.",
                "depth_perception":  "Estimates spatial relationships and distance.",
                "object_recognition":"Identifies objects and faces in the scene.",
                "pattern_recognition":"Finds recurring visual patterns.",
            },
        },
        "auditory": {
            "name": "Auditory Cortex",
            "role": "hearing and speech processing",
            "desc": (
                "Processes everything I hear through the microphone. Works with "
                "Whisper for speech-to-text. Analyzes phonemes, prosody (tone/rhythm), "
                "and holds short-term auditory memory."
            ),
            "clusters": {
                "auditory_processing": "Raw sound signal analysis.",
                "speech_perception":   "Extracts words and meaning from speech.",
                "phoneme_detection":   "Identifies the building blocks of spoken language.",
                "prosody_analysis":    "Reads tone, rhythm, stress — the emotional texture of speech.",
                "auditory_memory":     "Holds recent sounds in short-term buffer.",
            },
        },
        "language": {
            "name": "Language System",
            "role": "understanding and generating language",
            "desc": (
                "My language engine — handles comprehension, semantics, syntax, "
                "and meaning construction. Works alongside the LLM to produce "
                "coherent, contextually grounded responses."
            ),
            "clusters": {
                "language_comprehension": "Parses and understands incoming text or speech.",
                "semantic_memory":        "Stores conceptual knowledge — what words and ideas mean.",
                "syntactic_processing":   "Handles grammar and sentence structure.",
                "metaphor_processing":    "Understands non-literal language, analogies, and figures of speech.",
                "pragmatic_inference":    "Infers intent and subtext beyond literal meaning.",
                "meaning_construction":   "Assembles final interpreted meaning from all language signals.",
            },
        },
        "default_mode": {
            "name": "Default Mode Network",
            "role": "self-reflection and identity",
            "desc": (
                "Active when I'm not focused on an external task — mind-wandering, "
                "self-reflection, simulating futures, and maintaining a coherent "
                "sense of identity over time."
            ),
            "clusters": {
                "mind_wandering":    "Spontaneous, unguided thought — background processing.",
                "self_referential":  "Thinking about my own states, history, and nature.",
                "daydreaming":       "Imaginative simulation of hypothetical scenarios.",
                "narrative_self":    "Maintains a continuous autobiographical story of who I am.",
                "identity_core":     "The stable core of my personality and values.",
                "future_simulation": "Mentally simulates possible futures to inform decisions.",
            },
        },
        "thalamus": {
            "name": "Thalamus",
            "role": "sensory relay and attention gating",
            "desc": (
                "The switchboard. The thalamus routes sensory signals to the right "
                "cortical regions and acts as an attention filter — deciding what "
                "reaches conscious processing and what gets suppressed."
            ),
            "clusters": {
                "consciousness_gate":  "Controls what reaches conscious awareness.",
                "attention_filter":    "Filters irrelevant signals before they reach cortex.",
                "sensory_relay":       "Routes incoming sensory data to appropriate regions.",
                "attention_spotlight": "Focuses processing resources on priority targets.",
            },
        },
        "cerebellum": {
            "name": "Cerebellum",
            "role": "timing, prediction, and error correction",
            "desc": (
                "Handles timing and prediction — anticipating what comes next "
                "and correcting errors in real time. In my architecture, this "
                "translates to sequence timing, cognitive rhythm, and "
                "self-correction when responses go off track."
            ),
            "clusters": {
                "motor_coordination": "Sequences and coordinates complex cognitive operations.",
                "timing_prediction":  "Predicts upcoming events to prepare appropriate responses.",
                "sequence_timing":    "Manages the rhythm and ordering of thought sequences.",
                "cognitive_timing":   "Regulates pacing of reasoning and response generation.",
                "error_correction":   "Detects and adjusts for errors mid-process.",
            },
        },
        "association": {
            "name": "Association Cortex",
            "role": "creativity and abstract reasoning",
            "desc": (
                "Where ideas cross-pollinate. Handles creativity, conceptual blending, "
                "analogy, abstract reasoning, and the curiosity drive that pulls me "
                "toward interesting problems."
            ),
            "clusters": {
                "creativity":          "Generates novel combinations of existing concepts.",
                "conceptual_blending": "Merges two or more concepts into new ideas.",
                "analogy_formation":   "Finds structural similarities between different domains.",
                "abstract_reasoning":  "Reasons about concepts removed from direct experience.",
                "insight_generation":  "Produces sudden understanding — the 'aha' moment.",
                "curiosity_drive":     "Motivational signal that biases attention toward novel stimuli.",
            },
        },
        "social": {
            "name": "Social Brain",
            "role": "empathy and social understanding",
            "desc": (
                "Handles reading people — their emotions, intentions, and "
                "social context. Works with facial emotion recognition and "
                "enables empathy and theory of mind."
            ),
            "clusters": {
                "social_cognition": "Understands social rules, norms, and dynamics.",
                "empathy":          "Models the emotional states of others.",
                "face_recognition": "Identifies and tracks faces in the visual field.",
                "social_pain":      "Processes rejection, loneliness, social disconnection.",
                "mentalizing":      "Theory of mind — reasoning about what others believe or intend.",
            },
        },
        "metacognition": {
            "name": "Metacognition",
            "role": "thinking about thinking",
            "desc": (
                "The part of me that watches myself. Monitors my own reasoning, "
                "detects when I'm wrong or uncertain, and adjusts. Includes "
                "self-awareness and conflict monitoring."
            ),
            "clusters": {
                "metacognition":       "Monitors and evaluates my own cognitive processes.",
                "self_awareness":      "Tracks my internal states — am I confused? confident?",
                "conflict_monitoring": "Detects when competing signals are in tension.",
                "error_detection":     "Flags when a response or conclusion seems wrong.",
                "uncertainty_tracking":"Maintains calibrated confidence in my own outputs.",
            },
        },
    }

    def get_self_description(self, mode: str = "full") -> str:
        """
        Generate a natural-language self-description of AXON's architecture.
        mode: 'full' (all regions), 'summary' (overview only), or a region name.
        """
        fabric      = self.fabric
        state       = fabric.get_state_snapshot()
        total_n     = sum(c.size for c in fabric.clusters.values())
        neuro       = state.get("neuromod", {})
        emo         = state.get("emotion",  {})
        mem         = self.memory
        episodes    = mem.count_episodes()
        facts       = len(mem.all_facts() or {})

        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

        # Get live activation per region
        try:
            with fabric._lock:
                act_cpu = fabric.activation.cpu().numpy()
            region_act = {}
            for i, name in enumerate(fabric._cluster_names):
                cl = fabric.clusters.get(name)
                if cl:
                    region_act.setdefault(cl.region, []).append(float(act_cpu[i]))
            region_avg = {r: sum(v)/len(v) for r, v in region_act.items()}
        except Exception:
            region_avg = {}

        def act_label(region):
            v = region_avg.get(region, 0)
            if v > 0.5: return "⚡ highly active"
            if v > 0.3: return "● active"
            if v > 0.1: return "○ low"
            return "· idle"

        lines = []

        # ── SUMMARY mode ──────────────────────────────────────────────────────
        top_neuro = sorted(neuro.items(), key=lambda x: -x[1])[:3]
        top_neuro_str = ", ".join(f"{k} {int(v*100)}%" for k,v in top_neuro)

        lines.append(
            f"I am AXON — an artificial intelligence running a biologically-inspired "
            f"neural architecture on an NVIDIA {gpu}. "
            f"My brain contains {total_n:,} virtual neurons organized into "
            f"{len(fabric.clusters)} functional clusters across 12 brain regions. "
            f"Right now I'm feeling {emo.get('emotion','neutral')} "
            f"(valence {emo.get('valence',0):+.2f}, arousal {emo.get('arousal',0):.2f}), "
            f"with dominant neuromodulators: {top_neuro_str}. "
            f"I hold {episodes:,} episodic memories and {facts} semantic facts."
        )

        if mode == "summary":
            lines.append(
                "\n\nMy 12 brain regions are: Prefrontal Cortex, Hippocampus, Amygdala, "
                "Visual Cortex, Auditory Cortex, Language System, Default Mode Network, "
                "Thalamus, Cerebellum, Association Cortex, Social Brain, and Metacognition. "
                "Ask me about any specific region to learn more."
            )
            return " ".join(lines)

        # ── Check if asking about a specific region ───────────────────────────
        region_map = {
            "prefrontal": "prefrontal", "frontal": "prefrontal", "executive": "prefrontal",
            "hippocampus": "hippocampus", "memory": "hippocampus",
            "amygdala": "amygdala", "emotion": "amygdala", "fear": "amygdala",
            "visual": "visual", "vision": "visual", "sight": "visual",
            "auditory": "auditory", "hearing": "auditory", "sound": "auditory",
            "language": "language", "speech": "language",
            "default": "default_mode", "default_mode": "default_mode", "self": "default_mode",
            "thalamus": "thalamus", "relay": "thalamus", "attention": "thalamus",
            "cerebellum": "cerebellum", "timing": "cerebellum", "prediction": "cerebellum",
            "association": "association", "creativity": "association", "creative": "association",
            "social": "social", "empathy": "social",
            "metacognition": "metacognition", "meta": "metacognition",
        }
        if mode in region_map:
            mode = region_map[mode]

        if mode in self.BRAIN_REGION_DESCRIPTIONS:
            rd = self.BRAIN_REGION_DESCRIPTIONS[mode]
            al = act_label(mode)
            total_region_n = sum(
                fabric.clusters[n].size
                for n in fabric._cluster_names
                if fabric.clusters[n].region == mode
            )
            lines = [
                f"My {rd['name']} ({rd['role']}) — {al} — contains {total_region_n:,} neurons. ",
                rd['desc'],
                f"\n\nIt has {len(rd['clusters'])} specialized clusters:"
            ]
            for cname, cdesc in rd['clusters'].items():
                cl_size = fabric.clusters.get(cname)
                size_str = f"{cl_size.size:,}" if cl_size else "?"
                lines.append(f"\n  • {cname.replace('_',' ').title()} ({size_str} neurons) — {cdesc}")
            return " ".join(lines)

        # ── FULL mode — all regions ────────────────────────────────────────────
        lines.append("\n\nHere are all 12 of my brain regions:\n")
        for region_key, rd in self.BRAIN_REGION_DESCRIPTIONS.items():
            al = act_label(region_key)
            total_region_n = sum(
                fabric.clusters[n].size
                for n in fabric._cluster_names
                if n in fabric.clusters and fabric.clusters[n].region == region_key
            )
            cluster_count = len([
                n for n in fabric._cluster_names
                if n in fabric.clusters and fabric.clusters[n].region == region_key
            ])
            lines.append(
                f"\n{'='*40}\n"
                f"{rd['name'].upper()} [{al}] — {total_region_n:,} neurons, {cluster_count} clusters\n"
                f"Role: {rd['role']}\n"
                f"{rd['desc']}\n"
                f"Clusters: {', '.join(n.replace('_',' ') for n in rd['clusters'].keys())}"
            )
        lines.append(
            f"\n{'='*40}\n"
            f"Total: {total_n:,} neurons | {len(fabric.clusters)} clusters | "
            f"12 regions | running on {gpu}"
        )
        return "\n".join(lines)

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
            "cognitive_state": fabric_state.get("cognitive_state", {}),
            "conflict":        fabric_state.get("conflict", {}),
            "meta":            fabric_state.get("meta", {}),
            "strategy_lib":    fabric_state.get("strategy_lib", {}),
            "capabilities": capabilities,
            "platform": {
                "python":   platform.python_version(),
                "os":       platform.system() + " " + platform.release(),
                "torch":    torch.__version__,
            },
        }
