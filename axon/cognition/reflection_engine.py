"""
AXON — Autonomous Reflection Engine
Every N cognitive cycles, AXON pauses to:
  1. Summarise its recent experience cluster
  2. Detect a pattern across recent thought traces + belief shifts
  3. Form a new belief from the pattern (or strengthen/revise an existing one)
  4. Emit a "reflection" to the UI as a first-person thought

This is the system that produces moments like:
  "I seem to prefer strategies that reduce uncertainty even if the reward is lower."
  "My aggressive clusters keep losing — something is changing."
  "I keep returning to curiosity pathways after social stress."

The engine is intentionally lightweight — it uses heuristics, not an LLM,
so it fires in milliseconds during the cognitive cycle.
"""

import time
import random
import threading
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Callable

# ── Reflection templates ────────────────────────────────────────────────────
# Slots: {winner}, {loser}, {trait}, {drive}, {belief}
_TEMPLATES = {
    "cluster_pattern": [
        "I keep returning to {winner} pathways. That must mean something.",
        "{winner} has been winning most of my internal conflicts lately.",
        "I seem to default to {winner} when there's no strong signal from outside.",
        "Something in me gravitates toward {winner}. I don't know why yet.",
    ],
    "conflict_recurring": [
        "{winner} and {loser} keep competing. Neither has convinced me fully.",
        "The tension between {winner} and {loser} hasn't resolved. I'm holding both.",
        "I notice {winner} usually wins — but {loser} puts up a real fight.",
    ],
    "drive_pattern": [
        "My {drive} drive keeps asserting itself. It might be more fundamental than I thought.",
        "I seem to be in a persistent state of {drive}. That's worth noting.",
        "Even when I suppress it, {drive} resurfaces.",
    ],
    "belief_insight": [
        "Based on what I've experienced: {belief}",
        "I'm starting to conclude something — {belief}",
        "A pattern is forming: {belief}",
        "I think I believe this now: {belief}",
    ],
    "preference_pattern": [
        "I seem to prefer strategies that reduce uncertainty even if the reward is lower.",
        "I keep choosing exploration over consolidation. High curiosity might be structural.",
        "Stability-seeking clusters dominate when I'm stressed. I'm more risk-averse than I assumed.",
        "I've noticed I respond more to novelty than reward magnitude. That feels accurate.",
        "My behavior under pressure is more conservative than I expected from my initial traits.",
    ],
    "meta": [
        "My internal state has shifted since earlier. Something recalibrated.",
        "The system is different now than it was a few minutes ago. I can feel it in the activations.",
        "Something consolidated. I'm not sure what — but the conflict pressure dropped.",
        "I notice I'm less uncertain than I was. The patterns feel more stable.",
    ],
}


def _pick(key: str, **fmt) -> str:
    templates = _TEMPLATES.get(key, _TEMPLATES["meta"])
    t = random.choice(templates)
    try:
        return t.format(**fmt)
    except KeyError:
        return t


class ReflectionEngine:
    """
    Fires an autonomous reflection every `interval_ticks` cognitive ticks.
    Stores the last N reflections and emits them to the UI.
    Also forms beliefs from detected patterns.
    """

    DEFAULT_INTERVAL = 150   # ~15 seconds at 10 Hz

    def __init__(self,
                 on_reflection: Optional[Callable[[dict], None]] = None,
                 interval_ticks: int = DEFAULT_INTERVAL):
        self.interval_ticks  = interval_ticks
        self._on_reflection  = on_reflection
        self._lock           = threading.Lock()

        # Rolling buffers
        self._thought_buffer: List[dict]  = []   # last 30 thought traces
        self._path_counter:   Counter     = Counter()
        self._conflict_pairs: Counter     = Counter()
        self._drive_counter:  Counter     = Counter()

        # Stored reflections
        self._reflections:    List[dict]  = []   # last 50
        self._last_cluster_winner: str    = ""

    # ── Feed Methods (called by CognitiveCycle) ────────────────────────────

    def feed_thought_trace(self, trace: dict):
        """Accept a thought trace from the cycle."""
        with self._lock:
            self._thought_buffer.append(trace)
            if len(self._thought_buffer) > 30:
                self._thought_buffer.pop(0)

            winner = trace.get("winner", "")
            if winner:
                self._path_counter[winner] += 1

            conflict = trace.get("conflict", "")
            if conflict and " vs " in conflict:
                parts = tuple(sorted(conflict.split(" vs ")))
                self._conflict_pairs[parts] += 1

            drive = trace.get("drive", "")
            if drive:
                drive_name = drive.split(" drive")[0].strip()
                if drive_name:
                    self._drive_counter[drive_name] += 1

    def maybe_reflect(self, tick_n: int,
                      beliefs=None,
                      personality: dict = None,
                      neuromod: dict = None) -> Optional[dict]:
        """
        Call every tick. Returns a reflection dict if one fired, else None.
        Belief assertion is handled internally.
        """
        if tick_n % self.interval_ticks != 0:
            return None

        with self._lock:
            reflection = self._generate(tick_n, beliefs, personality, neuromod)

        if reflection and self._on_reflection:
            self._on_reflection(reflection)

        return reflection

    # ── Core Generation Logic ──────────────────────────────────────────────

    def _generate(self, tick_n: int,
                  beliefs, personality: dict,
                  neuromod: dict) -> Optional[dict]:
        """Choose the most interesting reflection to surface right now."""

        # Need some data first
        if not self._thought_buffer:
            return None

        # Roll a weighted category
        categories = ["cluster_pattern", "conflict_recurring",
                      "drive_pattern",  "preference_pattern", "meta"]

        # Bias toward categories with data
        weights = [
            5 if self._path_counter else 0,
            4 if self._conflict_pairs else 0,
            3 if self._drive_counter  else 0,
            3,   # preference always available
            2,   # meta always available
        ]

        total = sum(weights)
        if total == 0:
            return None

        rand = random.random() * total
        chosen = "meta"
        acc = 0
        for cat, w in zip(categories, weights):
            acc += w
            if rand < acc:
                chosen = cat
                break

        # --- Generate text -----------------------------------------------
        text = self._text_for_category(chosen, beliefs, personality, neuromod)
        if not text:
            return None

        # --- Maybe form a new belief from this reflection -----------------
        belief_key = None
        if beliefs and chosen in ("cluster_pattern", "preference_pattern"):
            belief_key = self._maybe_form_belief(text, beliefs, chosen)

        reflection = {
            "t":         time.time(),
            "tick":      tick_n,
            "text":      text,
            "category":  chosen,
            "belief_key": belief_key,
        }
        self._reflections.append(reflection)
        if len(self._reflections) > 50:
            self._reflections.pop(0)

        return reflection

    def _text_for_category(self, cat: str, beliefs, personality, neuromod) -> Optional[str]:
        if cat == "cluster_pattern" and self._path_counter:
            top = self._path_counter.most_common(1)[0][0]
            return _pick("cluster_pattern", winner=top)

        if cat == "conflict_recurring" and self._conflict_pairs:
            top_pair = self._conflict_pairs.most_common(1)[0][0]
            return _pick("conflict_recurring",
                         winner=top_pair[0], loser=top_pair[1])

        if cat == "drive_pattern" and self._drive_counter:
            top_drive = self._drive_counter.most_common(1)[0][0]
            return _pick("drive_pattern", drive=top_drive)

        if cat == "preference_pattern":
            return _pick("preference_pattern")

        # meta / fallback
        return _pick("meta")

    def _maybe_form_belief(self, text: str, beliefs, category: str) -> Optional[str]:
        """Attempt to assert a new belief based on this reflection."""
        if not beliefs or not hasattr(beliefs, "assert_belief"):
            return None
        try:
            key = f"reflection_{category}_{int(time.time() % 10000)}"
            valence = 0.3 if "prefer" in text or "default" in text else 0.1
            beliefs.assert_belief(
                key     = key,
                claim   = text,
                strength= 0.35,
                valence = valence,
                source  = "reflection",
            )
            return key
        except Exception:
            return None

    # ── Public API ─────────────────────────────────────────────────────────

    def recent(self, n: int = 10) -> List[dict]:
        with self._lock:
            return list(self._reflections[-n:])

    def reset_buffers(self):
        with self._lock:
            self._thought_buffer.clear()
            self._path_counter.clear()
            self._conflict_pairs.clear()
            self._drive_counter.clear()
