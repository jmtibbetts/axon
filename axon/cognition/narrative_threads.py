"""
AXON — Narrative Threads (Competing Internal Worldviews)
Each cluster group can represent a "worldview" that competes across experiences.
Worldviews are persistent belief constellations that fight for dominance.

Example worldviews:
  "Efficiency first"     → prune exploration, minimize variance
  "Explore at all costs" → exploration ε floor, tolerate low reward
  "Safety above all"     → reduce stress, avoid high-arousal states
  "Dominance-seeking"    → maximize reward, accept conflict
  "Social harmony"       → weight external social signals heavily

Over many cycles:
  - The narrative with the most confirmed predictions grows stronger
  - When a narrative loses enough predictions, it can be overturned
  - Dominant narrative biases ingestion credibility and belief formation
"""

import time
import json
import threading
import random
from pathlib import Path
from typing import Dict, List, Optional


class Narrative:
    NAMES = [
        "Efficiency First",
        "Explore at All Costs",
        "Safety Above All",
        "Dominance-Seeking",
        "Social Harmony",
        "Uncertainty is Fuel",
        "Consistency is Strength",
    ]

    CLUSTER_AFFINITIES: Dict[str, List[str]] = {
        "Efficiency First":        ["prefrontal_cortex", "cerebellum", "metacognition"],
        "Explore at All Costs":    ["association_cortex", "hippocampus", "default_mode_network"],
        "Safety Above All":        ["amygdala", "thalamus", "metacognition"],
        "Dominance-Seeking":       ["prefrontal_cortex", "amygdala", "social_brain"],
        "Social Harmony":          ["social_brain", "default_mode_network", "language_system"],
        "Uncertainty is Fuel":     ["association_cortex", "hippocampus", "visual_cortex"],
        "Consistency is Strength": ["thalamus", "cerebellum", "prefrontal_cortex"],
    }

    __slots__ = ("name", "strength", "valence", "wins", "losses",
                 "last_active", "cluster_affinities", "dominant_ticks")

    def __init__(self, name: str, strength: float = 0.3):
        self.name               = name
        self.strength           = float(strength)
        self.valence            = 0.0
        self.wins               = 0
        self.losses             = 0
        self.last_active        = time.time()
        self.cluster_affinities = self.CLUSTER_AFFINITIES.get(name, [])
        self.dominant_ticks     = 0

    def to_dict(self) -> dict:
        return {
            "name":        self.name,
            "strength":    round(self.strength, 3),
            "valence":     round(self.valence,  3),
            "wins":        self.wins,
            "losses":      self.losses,
            "dominant_ticks": self.dominant_ticks,
            "affinities":  self.cluster_affinities,
        }


class NarrativeThreads:
    """
    Manages a set of competing narrative worldviews.
    Each tick receives the cluster activation map and:
      - rewards the narrative whose affiliated clusters are most active
      - penalizes narratives whose clusters are suppressed
      - tracks dominance flips over time
    """

    COMPETITION_INTERVAL = 20   # ticks between narrative competition rounds
    DECAY_RATE           = 0.0008

    def __init__(self, data_dir: str):
        self._path   = Path(data_dir) / "narratives.json"
        self._lock   = threading.Lock()
        self._narratives: Dict[str, Narrative] = {}
        self._dominant: str = ""
        self._flip_history: List[dict] = []
        self._tick_n = 0
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────

    def _load(self):
        defaults = {n: Narrative(n, strength=0.25 + random.random() * 0.2)
                    for n in Narrative.NAMES}
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                for name, d in data.items():
                    n = Narrative(name, d.get("strength", 0.3))
                    n.valence          = d.get("valence", 0.0)
                    n.wins             = d.get("wins", 0)
                    n.losses           = d.get("losses", 0)
                    n.dominant_ticks   = d.get("dominant_ticks", 0)
                    defaults[name]     = n
            except Exception:
                pass
        self._narratives = defaults
        # Set initial dominant
        self._dominant = max(self._narratives, key=lambda k: self._narratives[k].strength)

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps({k: v.to_dict() for k, v in self._narratives.items()}, indent=2)
            )
        except Exception:
            pass

    # ── Main Tick ──────────────────────────────────────────────────────────

    def tick(self, activations: dict, reward: float = 0.0) -> Optional[str]:
        """
        Run one competition round. Returns the name of the dominant narrative,
        or None if no flip occurred.
        """
        self._tick_n += 1
        if self._tick_n % self.COMPETITION_INTERVAL != 0:
            return None

        with self._lock:
            return self._compete(activations, reward)

    def _compete(self, activations: dict, reward: float) -> Optional[str]:
        # Score each narrative by how much its affiliated clusters are active
        scores: Dict[str, float] = {}
        for name, narr in self._narratives.items():
            aff_sum = sum(activations.get(c, 0.0) for c in narr.cluster_affinities)
            count   = max(1, len(narr.cluster_affinities))
            scores[name] = aff_sum / count

        # Decay all narratives slightly
        for narr in self._narratives.values():
            narr.strength = max(0.02, narr.strength - self.DECAY_RATE)

        # Reward winner, penalize losers
        if scores:
            best_name  = max(scores, key=scores.get)
            best_score = scores[best_name]
            for name, score in scores.items():
                narr = self._narratives[name]
                if name == best_name:
                    gain = 0.01 + score * 0.03 + max(0, reward) * 0.02
                    narr.strength = min(0.95, narr.strength + gain)
                    narr.wins    += 1
                    narr.dominant_ticks += 1
                else:
                    loss = 0.005 * (best_score - score)
                    narr.strength = max(0.02, narr.strength - loss)
                    narr.losses  += 1

        # Detect dominance flip
        new_dominant = max(self._narratives, key=lambda k: self._narratives[k].strength)
        flip = None
        if new_dominant != self._dominant:
            flip = {
                "t":       time.time(),
                "from":    self._dominant,
                "to":      new_dominant,
                "old_str": round(self._narratives[self._dominant].strength, 3),
                "new_str": round(self._narratives[new_dominant].strength,  3),
            }
            self._flip_history.append(flip)
            if len(self._flip_history) > 20:
                self._flip_history.pop(0)
            self._dominant = new_dominant

        # Persist every 10 competition rounds
        if self._tick_n % (self.COMPETITION_INTERVAL * 10) == 0:
            self._save()

        return flip  # None if no flip

    # ── Public API ─────────────────────────────────────────────────────────

    def dominant(self) -> str:
        return self._dominant

    def top_narratives(self, n: int = 3) -> List[dict]:
        with self._lock:
            ranked = sorted(self._narratives.values(),
                            key=lambda x: -x.strength)
            return [r.to_dict() for r in ranked[:n]]

    def all_narratives(self) -> List[dict]:
        with self._lock:
            return [n.to_dict() for n in
                    sorted(self._narratives.values(), key=lambda x: -x.strength)]

    def recent_flips(self, n: int = 5) -> List[dict]:
        with self._lock:
            return list(self._flip_history[-n:])

    def narrative_bias(self) -> dict:
        """
        Returns behavioral hints based on the current dominant narrative.
        Used by the cognitive cycle to bias exploration + reward.
        """
        dom = self._narratives.get(self._dominant)
        if not dom:
            return {}
        name = dom.name
        return {
            "exploration_bias": {
                "Explore at All Costs":    +0.15,
                "Uncertainty is Fuel":     +0.10,
                "Efficiency First":        -0.10,
                "Safety Above All":        -0.15,
                "Dominance-Seeking":       -0.05,
                "Social Harmony":          +0.02,
                "Consistency is Strength": -0.08,
            }.get(name, 0.0),
            "social_weight": {
                "Social Harmony":       +0.20,
                "Dominance-Seeking":    +0.10,
                "Efficiency First":     -0.10,
            }.get(name, 0.0),
            "stress_sensitivity": {
                "Safety Above All":         +0.15,
                "Dominance-Seeking":        -0.10,
                "Uncertainty is Fuel":      -0.05,
            }.get(name, 0.0),
        }
