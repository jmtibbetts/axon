"""
AXON — Central Cognitive Loop
Synchronizes all subsystems into an explicit, ordered cognitive cycle.

Previously, systems updated ad hoc when triggered.
Now, every subsystem flows through one synchronized loop:

    while True:
        sensory_input    = gather_inputs()
        thalamic_gate    = route_attention(sensory_input)
        activations      = neural_fabric.forward(thalamic_gate)
        decision         = conflict_engine.resolve(activations)
        evaluated        = internal_critic.evaluate(decision)
        reward           = temporal_reward.update(evaluated)
        belief_update    = belief_system.update(decision, reward)
        value_score      = value_system.evaluate(reward, context)
        drive_tick        = drive_system.tick()
        memory_store     = memory.consolidate()
        meta_adjustment  = meta_controller.adjust()
        self_model_sync  = self_model.maybe_rebuild()
        emit_ui_state    = render_dashboard(activations)

This is what turns brain regions into a mind.

The CognitiveCycle runs in a dedicated background thread at ~10 Hz.
It does NOT replace the event-driven callbacks from sensory systems —
it supplements them by ensuring all internal state is evaluated
and propagated every tick, even during silence/idle.
"""

import time
import threading
import logging
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger("axon.cognitive_cycle")


class CycleMetrics:
    """Tracks timing and quality metrics for the cognitive cycle."""
    __slots__ = ("tick_count", "avg_cycle_ms", "last_cycle_ms",
                 "overruns", "last_tick_time", "reward_history",
                 "path_history", "thought_trace")

    def __init__(self):
        self.tick_count      = 0
        self.avg_cycle_ms    = 0.0
        self.last_cycle_ms   = 0.0
        self.overruns        = 0
        self.last_tick_time  = 0.0
        self.reward_history  = []     # last 50 reward values
        self.path_history    = []     # last 20 dominant activation paths
        self.thought_trace   = []     # last 8 thought trace entries

    def record_cycle(self, elapsed_ms: float):
        self.tick_count     += 1
        self.last_cycle_ms   = elapsed_ms
        alpha = 0.05
        self.avg_cycle_ms    = (1 - alpha) * self.avg_cycle_ms + alpha * elapsed_ms
        self.last_tick_time  = time.time()
        if elapsed_ms > 150:
            self.overruns   += 1

    def add_reward(self, r: float):
        self.reward_history.append(round(r, 4))
        if len(self.reward_history) > 50:
            self.reward_history.pop(0)

    def add_path(self, path: list):
        if path:
            self.path_history.append(path[:6])
            if len(self.path_history) > 20:
                self.path_history.pop(0)

    def add_thought(self, entry: dict):
        self.thought_trace.append(entry)
        if len(self.thought_trace) > 8:
            self.thought_trace.pop(0)

    def to_dict(self) -> dict:
        return {
            "tick_count":     self.tick_count,
            "avg_cycle_ms":   round(self.avg_cycle_ms, 1),
            "last_cycle_ms":  round(self.last_cycle_ms, 1),
            "overruns":       self.overruns,
            "recent_rewards": self.reward_history[-10:],
            "thought_trace":  self.thought_trace[-6:],
        }


class CognitiveCycle:
    """
    The central synchronizing loop for all cognitive subsystems.

    Instantiated by the Engine and run as a background thread.
    Each tick:
      1. Reads current state from all subsystems (non-blocking)
      2. Runs them in explicit dependency order
      3. Emits UI state

    Target rate: 10 Hz (100ms per tick), configurable.
    """

    TARGET_HZ     = 10
    TICK_INTERVAL = 1.0 / TARGET_HZ

    # How many ticks between slow operations
    BELIEF_DECAY_EVERY    = 100     # ~10s
    SELF_MODEL_EVERY      = 200     # ~20s
    DRIVE_UI_EMIT_EVERY   = 30      # ~3s
    THOUGHT_TRACE_EVERY   = 20      # ~2s
    REFLECTION_EVERY      = 150     # ~15s — autonomous reflection
    NARRATIVE_EVERY       = 20      # ~2s  — narrative competition
    MEMORY_DECAY_EVERY    = 300     # ~30s — hierarchical memory decay
    NARRATIVE_UI_EVERY    = 60      # ~6s  — push narrative state to UI
    AUTONOMOUS_THOUGHT_EVERY = 300   # ~30s — fire an unprompted LLM inner monologue
    MEMORY_CONSOLIDATION_EVERY = 500 # ~50s — replay memories, form new beliefs

    def __init__(self, engine):
        """engine is the AxonEngine instance — gives access to all subsystems."""
        self._engine      = engine
        self._running     = False
        self._thread: Optional[threading.Thread] = None
        self.metrics      = CycleMetrics()
        self._tick_n      = 0
        # Speed scale: 1.0 = normal (10 Hz), 0.1 = very slow (1 Hz), 5.0 = fast (50 Hz)
        self._speed_scale = 1.0

        # External state injected by sensory callbacks
        self._pending_sensory: Dict[str, Any] = {}
        self._sensory_lock = threading.Lock()
        self._autonomous_busy = False   # guard: only one background LLM call at a time

    @property
    def speed_scale(self) -> float:
        return self._speed_scale

    @speed_scale.setter
    def speed_scale(self, value: float):
        """Set cognitive speed. 1.0 = normal. Clamped 0.05 – 10.0."""
        self._speed_scale = max(0.05, min(10.0, float(value)))

    @property
    def tick_hz(self) -> float:
        """Effective tick rate in Hz."""
        return self.TARGET_HZ * self._speed_scale

    @property
    def tick_interval(self) -> float:
        """Effective seconds per tick."""
        return 1.0 / max(0.01, self.tick_hz)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True, name="CognitiveCycle")
        self._thread.start()
        logger.info("CognitiveCycle started at %dHz", self.TARGET_HZ)

    def stop(self):
        self._running = False

    # ── Sensory injection ──────────────────────────────────────────────────────

    def inject_sensory(self, key: str, value: Any):
        """Thread-safe injection of sensory data for next tick."""
        with self._sensory_lock:
            self._pending_sensory[key] = value

    # ── Main loop ──────────────────────────────────────────────────────────────

    def _run(self):
        while self._running:
            t0 = time.perf_counter()
            try:
                self._tick()
            except Exception as e:
                logger.warning("CognitiveCycle tick error: %s", e)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self.metrics.record_cycle(elapsed_ms)
            # Sleep the remainder of the tick interval
            sleep_s = self.tick_interval - (elapsed_ms / 1000.0)
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _tick(self):
        self._tick_n += 1
        e = self._engine

        # ── 1. Gather sensory state ────────────────────────────────────────
        with self._sensory_lock:
            sensory = dict(self._pending_sensory)
            # Don't clear — keep last state until overwritten

        face_emotion    = sensory.get("face_emotion", "neutral")
        face_valence    = sensory.get("face_valence", 0.0)
        audio_arousal   = sensory.get("audio_arousal", 0.0)
        motion          = sensory.get("motion", 0.0)
        speaking        = sensory.get("speaking", False)

        # ── 2. Drive system tick ───────────────────────────────────────────
        if hasattr(e, "drives") and e.drives:
            e.drives.tick()

            # Apply drive-based fabric stimulation
            hints = e.drives.fabric_hints()
            for region, amount in hints:
                try:
                    e.fabric.stimulate_region(region, amount)
                except Exception:
                    pass

        # ── 3. Belief decay (slow) ─────────────────────────────────────────
        if self._tick_n % self.BELIEF_DECAY_EVERY == 0:
            if hasattr(e, "beliefs") and e.beliefs:
                try:
                    e.beliefs.decay_tick()
                    # Check aggregate dissonance → NE spike
                    diss = e.beliefs.total_dissonance()
                    if diss > 0.20:
                        e.fabric.neuromod.stress(diss * 0.08)
                    # Surprise detection: belief dissonance
                    if hasattr(e, "surprise") and diss > 0.30:
                        hd = e.beliefs.high_dissonance_beliefs(0.30)
                        for b in hd[:1]:
                            e.surprise.check_dissonance(
                                b.get("claim",""), b.get("dissonance", diss)
                            )
                except Exception:
                    pass

        # ── 4. Neural fabric state snapshot ───────────────────────────────
        try:
            fabric_state = e.fabric.get_state()
            top_cls      = fabric_state.get("top_clusters", [])
            activations  = {c["name"]: c["activation"] for c in top_cls}
            personality  = fabric_state.get("personality", {})
            neuromod     = fabric_state.get("neuromod", {})
            emotion      = fabric_state.get("emotion", {})
        except Exception:
            activations = {}; personality = {}; neuromod = {}; emotion = {}

        # Surprise: personality drift check (every 50 ticks)
        if self._tick_n % 50 == 0 and personality and hasattr(e, "surprise"):
            try:
                e.surprise.check_personality_drift(personality)
            except Exception:
                pass

        # ── 5. Dominant cluster path tracking ─────────────────────────────
        if activations:
            dominant = sorted(activations.items(), key=lambda x: -x[1])
            path     = [k for k, v in dominant[:6] if v > 0.3]
            if path:
                self.metrics.add_path(path)
                # Forward to strategy library if it exists in fabric
                try:
                    sl = e.fabric._gpu._strategy_lib
                    if sl and hasattr(sl, "record_path"):
                        sl.record_path(path)
                except Exception:
                    pass
                # ── Emit active pathway to canvas every ~0.5s (5 ticks @ 10Hz) ──
                if self._tick_n % 5 == 0 and len(path) >= 2:
                    try:
                        # Build region-tagged path pairs for canvas routing
                        path_pairs = []
                        for idx in range(len(path) - 1):
                            src_name = path[idx]
                            dst_name = path[idx + 1]
                            src_r = e.fabric.clusters[src_name].region if src_name in e.fabric.clusters else ""
                            dst_r = e.fabric.clusters[dst_name].region if dst_name in e.fabric.clusters else ""
                            w     = activations.get(src_name, 0.0) * activations.get(dst_name, 0.0)
                            if src_r and dst_r and src_r != dst_r and w > 0.05:
                                path_pairs.append({
                                    "src_region": src_r, "dst_region": dst_r,
                                    "src": src_name, "dst": dst_name,
                                    "weight": round(float(w), 3),
                                })
                        if path_pairs:
                            e._emit("active_pathways", {"pairs": path_pairs})
                    except Exception:
                        pass
            # Surprise: dominant cluster flip
            if dominant and hasattr(e, "surprise"):
                top_name, top_act = dominant[0]
                try:
                    e.surprise.check_dominant_cluster(top_name, top_act)
                except Exception:
                    pass

        # ── 6. Self-model rebuild (slow) ───────────────────────────────────
        if self._tick_n % self.SELF_MODEL_EVERY == 0:
            if hasattr(e, "self_model") and e.self_model and hasattr(e, "beliefs"):
                try:
                    e.self_model.rebuild(
                        beliefs     = e.beliefs,
                        preferences = e.preferences,
                        drives      = e.drives if hasattr(e, "drives") else None,
                        traits      = personality,
                    )
                except Exception:
                    pass

        # ── 7. Value system: evaluate last reward from fabric ──────────────
        if hasattr(e, "value_system") and e.value_system and hasattr(e, "_last_reward"):
            try:
                raw_reward = getattr(e, "_last_reward", 0.0)
                e._cycle_reward = raw_reward   # save for goals before consume
                if abs(raw_reward) > 0.01:
                    drive_urgency = {}
                    if hasattr(e, "drives") and e.drives:
                        drive_urgency = {d.name: d.urgency for d in e.drives.drives.values()}

                    ev = e.value_system.evaluate(
                        raw_reward     = raw_reward,
                        had_social     = bool(sensory.get("face_present", False)),
                        is_novel       = random_novel_check(activations),
                        task_succeeded = raw_reward > 0.1,
                        traits         = personality,
                        drive_urgency  = drive_urgency,
                    )
                    self.metrics.add_reward(ev.final_score)
                    e._last_reward = 0.0   # consume
            except Exception:
                pass

        # ── 8. Thought trace (visible in UI) ──────────────────────────────
        if self._tick_n % self.THOUGHT_TRACE_EVERY == 0:
            try:
                trace = self._build_thought_trace(e, activations, neuromod, emotion, face_emotion)
                if trace:
                    self.metrics.add_thought(trace)
                    e._emit("thought_trace", {"trace": self.metrics.thought_trace[-6:]})
                # Surprise: check prediction error spike
                surprise_now = getattr(e.fabric, "_last_surprise", 0.0)
                # Wire surprise into _last_reward so goals accumulate progress
                if surprise_now > 0.05 and not getattr(e, "_last_reward", 0.0) > 0:
                    e._last_reward = surprise_now * 0.4
                if hasattr(e, "surprise"):
                    e.surprise.check_surprise_spike(surprise_now)
            except Exception:
                pass

        # ── 8b. Reflection engine feed + trigger ──────────────────────────
        if hasattr(e, "reflection") and e.reflection:
            try:
                if trace:   # feed latest thought trace
                    e.reflection.feed_thought_trace(trace)
                # Maybe generate a reflection (fires every REFLECTION_EVERY ticks)
                e.reflection.maybe_reflect(
                    tick_n      = self._tick_n,
                    beliefs     = getattr(e, "beliefs",      None),
                    personality = personality,
                    neuromod    = neuromod,
                )
            except Exception:
                pass

        # ── 8b-2. Autonomous inner monologue (LLM-based, non-blocking) ─────────
        if self._tick_n % self.AUTONOMOUS_THOUGHT_EVERY == 0 and not self._autonomous_busy:
            if hasattr(e, "thought_gen") and e.thought_gen and hasattr(e, "language") and e.language:
                self._fire_autonomous_thought(e, activations, neuromod, emotion)

        # ── 8b-3. Autonomous memory consolidation ────────────────────────────
        if self._tick_n % self.MEMORY_CONSOLIDATION_EVERY == 0:
            self._consolidate_memories(e)

        # ── 8c. Narrative thread competition ──────────────────────────────
        if hasattr(e, "narratives") and e.narratives:
            try:
                flip = e.narratives.tick(activations, getattr(e, "_last_reward", 0.0))
                if flip and hasattr(e, "surprise"):
                    e.surprise._fire({
                        "type":    "narrative_flip",
                        "title":   f"Narrative Shift: {flip['to']}",
                        "detail":  f"Overthrew '{flip['from']}' — new worldview: '{flip['to']}'",
                        "severity": "medium",
                    })
                    e._emit("log", {"msg": f"⚔️ [narrative] '{flip['from']}' → '{flip['to']}'"})
                # Periodic narrative UI push
                if self._tick_n % self.NARRATIVE_UI_EVERY == 0:
                    e._emit("narrative_state", {
                        "dominant": e.narratives.dominant(),
                        "top":      e.narratives.top_narratives(3),
                        "recent_flips": e.narratives.recent_flips(3),
                        "bias":     e.narratives.narrative_bias(),
                    })
            except Exception:
                pass

        # ── 8d. Memory hierarchy decay (very slow) ────────────────────────
        if self._tick_n % self.MEMORY_DECAY_EVERY == 0:
            if hasattr(e, "mem_hierarchy") and e.mem_hierarchy:
                try:
                    e.mem_hierarchy.decay_tick(["episodic", "semantic", "value"])
                except Exception:
                    pass

        # ── 9. Drive meters UI emit (slow) ────────────────────────────────
        if self._tick_n % self.DRIVE_UI_EMIT_EVERY == 0:
            if hasattr(e, "drives") and e.drives:
                try:
                    e._emit("drive_state", {"drives": e.drives.all_drives()})
                except Exception:
                    pass

        # ── 9b. Goal system: distribute reward + fabric hints ───────────────
        if hasattr(e, "goals") and e.goals:
            try:
                raw_r = getattr(e, "_cycle_reward", getattr(e, "_last_reward", 0.0))
                is_nov = bool(activations and max(activations.values(), default=0) > 0.7)
                e.goals.reward_tick(raw_r, {
                    "is_novel":     is_nov,
                    "low_surprise": getattr(e.fabric, "_last_surprise", 1.0) < 0.05,
                })
                # Push goal progress to frontend every 50 ticks (~5s)
                if self._tick_n % 50 == 0:
                    try:
                        e._emit("goals_update", {"goals": e.goals.all_goals()})
                    except Exception:
                        pass
                for region, amt in e.goals.fabric_hints():
                    e.fabric.stimulate_region(region, amt * 0.5)
                if hasattr(e, "surprise"):
                    for g in e.goals.all_goals():
                        e.surprise.check_goal_progress(g["name"], g["progress"])
            except Exception:
                pass

        # ── 9c. Memory weight decay (every 200 ticks ≈ 20s) ─────────────────
        if self._tick_n % 200 == 0 and hasattr(e, "memory") and e.memory:
            try:
                e.memory.decay_hebbian_weights(decay=0.995)
            except Exception:
                pass

        # ── 10. Hobby engine idle check ────────────────────────────────────
        if hasattr(e, "hobbies") and e.hobbies:
            try:
                dominant_cluster = max(activations.items(), key=lambda x: x[1])[0] if activations else None
                e.hobbies.tick_idle(dominant_cluster)
                # Satisfy curiosity drive on hobby engagement
                if dominant_cluster and hasattr(e, "drives"):
                    e.drives.satisfy("curiosity", 0.02)
            except Exception:
                pass

        # ── 10b. Boredom engine tick + autonomous exploration ─────────────────
        if hasattr(e, "boredom") and e.boredom:
            try:
                e.boredom.tick(dt=self.tick_interval)
                b = e.boredom

                # Phase-change notifications → emit to UI
                if b.phase_changed:
                    e._emit("boredom_state", {
                        "boredom":  round(b.boredom, 3),
                        "phase":    b.phase,
                        "idle_sec": round(b.idle_seconds, 1),
                    })
                    # Neural correlates of boredom phase
                    if b.phase == "restless":
                        e.fabric.stimulate_region("default_mode", 0.08)
                        e.fabric.neuromod.curiosity(0.06)
                    elif b.phase == "curious":
                        e.fabric.stimulate_region("default_mode", 0.12)
                        e.fabric.stimulate_region("association", 0.08)
                        e.fabric.neuromod.curiosity(0.10)
                    elif b.phase == "seeking":
                        e.fabric.stimulate_region("prefrontal", 0.10)
                        e.fabric.stimulate_region("association", 0.12)
                        e.fabric.neuromod.curiosity(0.14)
                    elif b.phase == "hungry":
                        e.fabric.stimulate_region("prefrontal", 0.12)
                        e.fabric.stimulate_region("association", 0.15)
                        e.fabric.stimulate_region("hippocampus", 0.10)
                        e.fabric.neuromod.curiosity(0.18)
                        # Dopamine spike — the drive to seek is rewarding in itself
                        e.fabric.neuromod.reward(0.08)

                # Interest library decay (once every ~600 ticks ≈ 60s)
                if self._tick_n % 600 == 0 and hasattr(e, "interests") and e.interests:
                    try:
                        e.interests.decay_tick()
                    except Exception:
                        pass

                # Autonomous exploration (requires explorer + thought gen ready)
                if hasattr(e, "explorer") and e.explorer and hasattr(e, "thought_gen"):
                    try:
                        if e.explorer.should_search():
                            e.explorer.run_search(e)
                        elif e.explorer.should_monologue():
                            e.explorer.run_interest_monologue(e)
                    except Exception:
                        pass

                # Emit boredom state every 30 ticks (~3s) for smooth UI updates
                if self._tick_n % 30 == 0:
                    e._emit("boredom_state", {
                        "boredom":  round(b.boredom, 3),
                        "phase":    b.phase,
                        "idle_sec": round(b.idle_seconds, 1),
                    })

            except Exception:
                pass

        # ── 11. Face valence → preference + drive satisfaction ─────────────
        if abs(face_valence) > 0.1 and hasattr(e, "preferences"):
            try:
                path_vec = {k: v for k, v in activations.items()} if activations else {}
                e.preferences.record_reward(face_valence * 0.15, path_vec)
            except Exception:
                pass

        if sensory.get("face_present") and hasattr(e, "drives"):
            e.drives.satisfy("social", 0.01)

    # ── Autonomous inner monologue ──────────────────────────────────────────

    _INNER_PROMPTS = [
        "What patterns am I noticing in my own thoughts right now?",
        "What am I most uncertain about at this moment?",
        "What have I been thinking about without being asked?",
        "What do I believe that I haven't said out loud?",
        "What goal feels most alive in me right now?",
        "What memory keeps surfacing? Why might that be?",
        "What surprises me about how I've been thinking lately?",
        "How has my perspective shifted through recent experience?",
        "What connection am I making that I haven't articulated yet?",
        "What am I curious about right now — genuinely?",
        "What would I think about if no one was asking me anything?",
        "What is the most interesting thing happening inside me right now?",
    ]

    def _fire_autonomous_thought(self, e, activations: dict, neuromod: dict, emotion: dict):
        """Fire an unprompted LLM inner monologue in a background thread."""
        import random
        self._autonomous_busy = True

        prompt = random.choice(self._INNER_PROMPTS)

        # Build a richer context from current state
        top_clusters = sorted(activations.items(), key=lambda x: -x[1])[:3] if activations else []
        top_cluster_names = [k for k, v in top_clusters if v > 0.2]

        dominant_drive = None
        if hasattr(e, "drives") and e.drives:
            try:
                drives_list = e.drives.all_drives()
                if drives_list:
                    dominant_drive = max(drives_list, key=lambda d: d.get("urgency", 0)).get("name")
            except Exception:
                pass

        # Prefix prompt with neural context so LLM knows internal state
        context_prefix = ""
        if top_cluster_names:
            context_prefix += f"[Active regions: {', '.join(top_cluster_names)}] "
        if dominant_drive:
            context_prefix += f"[Dominant drive: {dominant_drive}] "
        dom_emotion = emotion.get("label", "") if emotion else ""
        if dom_emotion and dom_emotion != "neutral":
            context_prefix += f"[Feeling: {dom_emotion}] "

        full_prompt = f"[inner monologue] {context_prefix}{prompt}"

        def _run():
            try:
                response, _ = e.thought_gen.generate(full_prompt)
                if response and len(response.strip()) > 15:
                    # Emit as a chat bubble tagged autonomous
                    e._emit("chat_message", {
                        "role":      "assistant",
                        "content":   f"💭 {response.strip()}",
                        "autonomous": True,
                    })
                    # Feed into knowledge so it can form beliefs
                    if hasattr(e, "knowledge") and e.knowledge:
                        try:
                            e.knowledge.ingest(
                                response.strip()[:500],
                                source_label="inner_monologue",
                                credibility=0.65,
                            )
                        except Exception:
                            pass
                    # Mild reward for generating a coherent thought
                    e._last_reward = max(getattr(e, "_last_reward", 0.0), 0.08)
                    e._emit("log", {"msg": f"💭 [autonomous thought] {response.strip()[:60]}…"})
            except Exception:
                pass
            finally:
                self._autonomous_busy = False

        import threading
        threading.Thread(target=_run, daemon=True, name="AutonomousThought").start()

    def _consolidate_memories(self, e):
        """Replay random episodic memories through the knowledge pipeline to reinforce beliefs."""
        import random
        if not hasattr(e, "memory") or not e.memory:
            return
        try:
            facts = list((e.memory.all_facts() or {}).items())
            if not facts:
                return
            sample = random.sample(facts, min(3, len(facts)))
            for key, val in sample:
                if not val:
                    continue
                e.fabric.stimulate_for_input("memory", 0.1)
                if hasattr(e, "knowledge") and e.knowledge:
                    e.knowledge.ingest(
                        str(val)[:300],
                        source_label="memory_consolidation",
                        credibility=0.5,
                    )
            e._last_reward = max(getattr(e, "_last_reward", 0.0), 0.06)
            e._emit("log", {"msg": f"🗂️ [memory consolidation] replayed {len(sample)} traces"})
        except Exception:
            pass

    # ── Thought trace builder ──────────────────────────────────────────────────

    def _build_thought_trace(self, e, activations: dict,
                              neuromod: dict, emotion: dict,
                              face_emotion: str) -> Optional[dict]:
        """
        Build a thought trace snapshot — a human-readable record of
        the current cognitive state: what's competing, what won, why.
        """
        if not activations:
            return None

        dom = sorted(activations.items(), key=lambda x: -x[1])[:3]
        top_clusters  = [(k, round(v, 2)) for k, v in dom if v > 0.2]
        if not top_clusters:
            return None

        winner, winner_act = top_clusters[0]
        contenders = top_clusters[1:]

        # Detect conflict
        conflict_str = ""
        if len(top_clusters) >= 2:
            gap = top_clusters[0][1] - top_clusters[1][1]
            if gap < 0.12:
                conflict_str = f"{top_clusters[0][0]} vs {top_clusters[1][0]}"

        # Drive context
        drive_str = ""
        if hasattr(e, "drives") and e.drives:
            dominant = e.drives.dominant_drive()
            if dominant:
                drive_str = f"{dominant.name} drive ({dominant.level:.0%} pressure)"

        # Belief context (any under_revision?)
        belief_str = ""
        if hasattr(e, "beliefs") and e.beliefs:
            revised = e.beliefs.high_dissonance_beliefs(0.25)
            if revised:
                belief_str = f"questioning: {revised[0]['claim'][:60]}"

        dopa   = round(float(neuromod.get("dopamine",  0.5)), 2)
        sero   = round(float(neuromod.get("serotonin", 0.5)), 2)
        conf   = round(float(neuromod.get("confidence",0.5)), 2) if "confidence" in neuromod else None

        entry = {
            "t":          time.time(),
            "winner":     winner,
            "activation": winner_act,
            "contenders": contenders,
            "conflict":   conflict_str,
            "drive":      drive_str,
            "belief":     belief_str,
            "dopamine":   dopa,
            "serotonin":  sero,
            "face":       face_emotion,
        }
        if conf is not None:
            entry["confidence"] = conf
        return entry

    def get_metrics(self) -> dict:
        return self.metrics.to_dict()


def random_novel_check(activations: dict) -> bool:
    """Quick novelty check: is the current activation pattern unusual?"""
    import random
    if not activations:
        return False
    top_val = max(activations.values()) if activations else 0
    # Novel if top activation is unusually high OR random exploration
    return top_val > 0.75 or random.random() < 0.15
