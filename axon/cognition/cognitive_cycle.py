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
            activations  = fabric_state.get("clusters", {})
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
                if hasattr(e, "surprise"):
                    e.surprise.check_surprise_spike(surprise_now)
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
        if hasattr(e, "goals") and e.goals and hasattr(e, "_last_reward"):
            try:
                raw_r = getattr(e, "_last_reward", 0.0)
                is_nov = bool(activations and max(activations.values(), default=0) > 0.7)
                e.goals.reward_tick(raw_r, {
                    "is_novel":     is_nov,
                    "low_surprise": getattr(e.fabric, "_last_surprise", 1.0) < 0.05,
                })
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

        # ── 11. Face valence → preference + drive satisfaction ─────────────
        if abs(face_valence) > 0.1 and hasattr(e, "preferences"):
            try:
                path_vec = {k: v for k, v in activations.items()} if activations else {}
                e.preferences.record_reward(face_valence * 0.15, path_vec)
            except Exception:
                pass

        if sensory.get("face_present") and hasattr(e, "drives"):
            e.drives.satisfy("social", 0.01)

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
