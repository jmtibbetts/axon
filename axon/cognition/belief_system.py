"""
AXON — Belief System
Weighted assumptions about the world, updated by:
  - live experience (prediction error, reward signals)
  - external knowledge (book/article ingestion)
  - contradiction (experience vs. stored belief)

Beliefs are not facts. They have:
  - strength   : 0.0–1.0  (how confident the system is)
  - valence    : -1.0–1.0 (positive/negative expectation)
  - source     : "experience" | "knowledge" | "inference"
  - last_tested: unix timestamp of last prediction/comparison

Storage: SQLite table 'beliefs' in axon.db
"""

import json
import math
import time
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class Belief:
    __slots__ = ("key", "claim", "strength", "valence",
                 "source", "hits", "misses", "last_tested")

    def __init__(self, key: str, claim: str, strength: float = 0.5,
                 valence: float = 0.0, source: str = "inference"):
        self.key         = key
        self.claim       = claim
        self.strength    = float(strength)
        self.valence     = float(valence)
        self.source      = source
        self.hits        = 0      # times prediction confirmed
        self.misses      = 0      # times prediction violated
        self.last_tested = 0.0

    def to_dict(self) -> dict:
        return {
            "key":         self.key,
            "claim":       self.claim,
            "strength":    round(self.strength, 3),
            "valence":     round(self.valence, 3),
            "source":      self.source,
            "hits":        self.hits,
            "misses":      self.misses,
            "last_tested": self.last_tested,
        }


class BeliefSystem:
    """
    Core belief store.
    Beliefs drift toward 0.5 if never tested (decay).
    They strengthen when confirmed, weaken when violated.
    Contradictory evidence from external sources adjusts
    strength proportionally to source credibility.
    """

    DECAY_RATE   = 0.0002   # per tick (called from fabric tick)
    LEARN_RATE   = 0.08     # per confirmation / violation
    CONFLICT_LR  = 0.04     # rate when external source contradicts experience

    def __init__(self, db_path: Path):
        self._db   = str(db_path)
        self._lock = threading.Lock()
        self._cache: Dict[str, Belief] = {}
        self._init_db()
        self._load_all()

        # Seed foundational beliefs if empty
        if not self._cache:
            self._seed_defaults()

    # ── DB ────────────────────────────────────────────────────────────────────

    def _init_db(self):
        conn = sqlite3.connect(self._db)
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS beliefs (
            key         TEXT PRIMARY KEY,
            claim       TEXT NOT NULL,
            strength    REAL DEFAULT 0.5,
            valence     REAL DEFAULT 0.0,
            source      TEXT DEFAULT 'inference',
            hits        INTEGER DEFAULT 0,
            misses      INTEGER DEFAULT 0,
            last_tested REAL DEFAULT 0
        );
        """)
        conn.commit()
        conn.close()

    def _load_all(self):
        conn = sqlite3.connect(self._db)
        rows = conn.execute(
            "SELECT key,claim,strength,valence,source,hits,misses,last_tested FROM beliefs"
        ).fetchall()
        conn.close()
        for key, claim, strength, valence, source, hits, misses, last_tested in rows:
            b = Belief(key, claim, strength, valence, source)
            b.hits = hits; b.misses = misses; b.last_tested = last_tested
            self._cache[key] = b

    def _save(self, b: Belief):
        conn = sqlite3.connect(self._db)
        conn.execute("""
            INSERT INTO beliefs (key,claim,strength,valence,source,hits,misses,last_tested)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(key) DO UPDATE SET
                claim=excluded.claim, strength=excluded.strength,
                valence=excluded.valence, source=excluded.source,
                hits=excluded.hits, misses=excluded.misses,
                last_tested=excluded.last_tested
        """, (b.key, b.claim, b.strength, b.valence,
              b.source, b.hits, b.misses, b.last_tested))
        conn.commit()
        conn.close()

    def _seed_defaults(self):
        seeds = [
            ("effort_leads_to_success",  "Sustained effort tends to produce positive outcomes.",   0.60,  0.6, "inference"),
            ("novelty_is_rewarding",     "Encountering new things tends to feel good.",            0.55,  0.5, "inference"),
            ("social_interaction_helps", "Engaging with others tends to improve internal state.",  0.50,  0.4, "inference"),
            ("consistency_reduces_stress","Predictable routines reduce internal stress signals.", 0.50,  0.3, "inference"),
            ("conflict_has_cost",        "Internal cluster competition consumes resources.",       0.65, -0.3, "experience"),
            ("curiosity_is_safe",        "Exploring unknown patterns rarely leads to bad outcomes.",0.55,  0.4, "inference"),
            ("high_arousal_fades",       "Intense activation states naturally decay toward baseline.",0.70, 0.0,"experience"),
        ]
        for key, claim, strength, valence, source in seeds:
            self.assert_belief(key, claim, strength, valence, source)

    # ── Public API ────────────────────────────────────────────────────────────

    def assert_belief(self, key: str, claim: str,
                      strength: float = 0.5, valence: float = 0.0,
                      source: str = "inference") -> Belief:
        """Create or update a belief."""
        with self._lock:
            if key in self._cache:
                b = self._cache[key]
                # Blend existing strength toward new evidence
                b.strength = b.strength * 0.7 + strength * 0.3
                b.valence  = b.valence  * 0.7 + valence  * 0.3
                if source == "experience":
                    b.source = "experience"  # experience always wins label
            else:
                b = Belief(key, claim, strength, valence, source)
                self._cache[key] = b
            self._save(b)
            return b

    def confirm(self, key: str, magnitude: float = 1.0):
        """A prediction based on this belief proved correct."""
        with self._lock:
            b = self._cache.get(key)
            if not b:
                return
            b.hits       += 1
            b.last_tested = time.time()
            delta = self.LEARN_RATE * magnitude * (1.0 - b.strength)
            b.strength   = min(0.97, b.strength + delta)
            self._save(b)

    def violate(self, key: str, magnitude: float = 1.0):
        """A prediction based on this belief proved wrong."""
        with self._lock:
            b = self._cache.get(key)
            if not b:
                return
            b.misses      += 1
            b.last_tested  = time.time()
            delta = self.LEARN_RATE * magnitude * b.strength
            b.strength    = max(0.03, b.strength - delta)
            # Flip valence slightly toward opposite
            b.valence = b.valence * 0.92
            self._save(b)

    def challenge(self, key: str, external_valence: float, credibility: float = 0.5):
        """
        External source (book/article) provides evidence that may
        contradict the current belief valence.
        credibility: 0–1, how much we trust the source vs. our experience.
        """
        with self._lock:
            b = self._cache.get(key)
            if not b:
                return
            # Disagreement amount
            disagreement = abs(b.valence - external_valence)
            if disagreement < 0.05:
                # Consistent — lightly confirm
                b.strength = min(0.97, b.strength + self.CONFLICT_LR * 0.5 * credibility)
            else:
                # Contradiction — pull valence toward external, weaken certainty
                pull = self.CONFLICT_LR * credibility * disagreement
                b.valence   = b.valence + (external_valence - b.valence) * pull
                b.strength  = max(0.1, b.strength - pull * 0.3)
                b.source    = "contested"
            self._save(b)

    def decay_tick(self):
        """Call periodically — beliefs drift toward 0.5 if never tested."""
        now = time.time()
        with self._lock:
            for b in self._cache.values():
                age = max(0, now - b.last_tested) if b.last_tested else 3600
                drift = self.DECAY_RATE * math.log1p(age / 3600)
                b.strength += (0.5 - b.strength) * drift
                b.strength  = max(0.03, min(0.97, b.strength))

    def get(self, key: str) -> Optional[Belief]:
        return self._cache.get(key)

    def all_beliefs(self) -> List[dict]:
        with self._lock:
            return sorted(
                [b.to_dict() for b in self._cache.values()],
                key=lambda x: -x["strength"]
            )

    def strongest(self, n: int = 5) -> List[Belief]:
        with self._lock:
            return sorted(self._cache.values(),
                          key=lambda b: -b.strength)[:n]

    def as_context_string(self, n: int = 6) -> str:
        """Returns a compact string for injection into LLM context."""
        top = self.strongest(n)
        if not top:
            return ""
        lines = []
        for b in top:
            conf = "strongly" if b.strength > 0.75 else "tentatively"
            lines.append(f'- I {conf} believe: "{b.claim}" (confidence {b.strength:.2f})')
        return "Current beliefs:\n" + "\n".join(lines)

    def personality_bias(self, traits: dict) -> dict:
        """
        Returns belief-adjusted reward modifiers based on current
        belief strengths + trait interactions.
        Used by TemporalRewardBuffer to shape reward.
        """
        effort   = self._cache.get("effort_leads_to_success")
        novelty  = self._cache.get("novelty_is_rewarding")
        social   = self._cache.get("social_interaction_helps")
        conflict = self._cache.get("conflict_has_cost")

        effort_str  = effort.strength   if effort   else 0.5
        novelty_str = novelty.strength  if novelty  else 0.5
        social_str  = social.strength   if social   else 0.5
        conflict_pen= conflict.strength if conflict else 0.5

        return {
            "consistency_bonus": effort_str  * traits.get("conscientiousness", 0.5),
            "novelty_bonus":     novelty_str * traits.get("openness",          0.5),
            "social_bonus":      social_str  * traits.get("agreeableness",     0.5),
            "conflict_penalty":  conflict_pen * traits.get("neuroticism",      0.5),
        }
