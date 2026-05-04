"""
AXON — Value System
Replaces shallow "I liked this outcome" preference scoring with a
multi-dimensional value evaluation that depends on personality.

For any outcome, compute:
    value = {
        "short_term_reward": x,   # immediate reinforcement signal
        "long_term_reward":  y,   # estimated future benefit
        "social_impact":     z,   # did it involve / benefit the user?
        "novelty":           n,   # was the path new?
        "competence":        c,   # did it demonstrate or build skill?
    }

Final score:
    score = w1*short + w2*long + w3*social + w4*novelty + w5*competence

Weights come from personality traits:
    extraversion     → amplifies short_term + social
    openness         → amplifies novelty
    conscientiousness→ amplifies long_term + competence
    agreeableness    → amplifies social
    neuroticism      → penalizes short_term variance

This means:
    - Two identical outcomes CAN score differently depending on state
    - Personality shifts gradually change what "feels good"
    - The system can learn to prefer certain value dimensions
"""

import math
import threading
import time
from typing import Dict, Optional


class ValueEvaluation:
    """Result of a single value assessment."""
    __slots__ = ("short_term", "long_term", "social_impact",
                 "novelty", "competence", "final_score",
                 "weights_used", "timestamp")

    def __init__(self, short_term: float, long_term: float, social_impact: float,
                 novelty: float, competence: float, weights: dict):
        self.short_term   = float(short_term)
        self.long_term    = float(long_term)
        self.social_impact= float(social_impact)
        self.novelty      = float(novelty)
        self.competence   = float(competence)
        self.weights_used = weights
        self.timestamp    = time.time()
        self.final_score  = self._compute(weights)

    def _compute(self, w: dict) -> float:
        return (
            w["short_term"]    * self.short_term   +
            w["long_term"]     * self.long_term     +
            w["social_impact"] * self.social_impact +
            w["novelty"]       * self.novelty       +
            w["competence"]    * self.competence
        )

    def to_dict(self) -> dict:
        return {
            "short_term":    round(self.short_term,    3),
            "long_term":     round(self.long_term,     3),
            "social_impact": round(self.social_impact, 3),
            "novelty":       round(self.novelty,       3),
            "competence":    round(self.competence,    3),
            "final_score":   round(self.final_score,   3),
            "weights":       {k: round(v, 3) for k, v in self.weights_used.items()},
        }


class ValueSystem:
    """
    Evaluates outcomes along multiple dimensions and weights them
    by personality to produce a final value score.

    Usage:
        evaluation = value_system.evaluate(
            raw_reward=0.3,
            had_social=True,
            is_novel=True,
            task_succeeded=True,
            traits=personality_traits
        )
        # evaluation.final_score is the adjusted reward
    """

    # Default weights (before personality adjustment)
    BASE_WEIGHTS = {
        "short_term":    0.25,
        "long_term":     0.25,
        "social_impact": 0.20,
        "novelty":       0.15,
        "competence":    0.15,
    }

    def __init__(self):
        self._lock = threading.Lock()
        self._history: list = []       # last 50 evaluations
        self._max_history = 50

        # Running dimension averages (for self-reporting)
        self._dim_sums   = dict.fromkeys(self.BASE_WEIGHTS, 0.0)
        self._dim_counts = 0

    def personality_weights(self, traits: dict) -> dict:
        """
        Compute personality-adjusted weight vector.
        Each trait shifts the importance of value dimensions.
        """
        w = dict(self.BASE_WEIGHTS)   # copy

        ext  = float(traits.get("extraversion",      0.5))
        opn  = float(traits.get("openness",          0.5))
        con  = float(traits.get("conscientiousness", 0.5))
        agr  = float(traits.get("agreeableness",     0.5))
        neu  = float(traits.get("neuroticism",       0.5))

        # Extraversion: live in the moment + care about social
        w["short_term"]    += 0.08 * (ext - 0.5)
        w["social_impact"] += 0.12 * (ext - 0.5)

        # Openness: drawn to novelty
        w["novelty"]       += 0.15 * (opn - 0.5)
        w["long_term"]     += 0.05 * (opn - 0.5)   # open minds invest in future

        # Conscientiousness: long-term + competence over quick rewards
        w["long_term"]     += 0.12 * (con - 0.5)
        w["competence"]    += 0.10 * (con - 0.5)
        w["short_term"]    -= 0.06 * (con - 0.5)   # less impulsive

        # Agreeableness: social impact matters most
        w["social_impact"] += 0.12 * (agr - 0.5)

        # Neuroticism: penalises short-term variance (uncertainty = bad)
        w["short_term"]    -= 0.06 * (neu - 0.5)
        w["competence"]    += 0.04 * (neu - 0.5)   # seeking mastery to reduce anxiety

        # Normalize so weights sum to 1.0
        total = sum(w.values())
        if total > 0:
            w = {k: max(0.05, v / total) for k, v in w.items()}
            # Re-normalize after flooring
            total2 = sum(w.values())
            w = {k: v / total2 for k, v in w.items()}

        return w

    def evaluate(self, raw_reward: float, had_social: bool = False,
                 is_novel: bool = False, task_succeeded: bool = False,
                 traits: Optional[dict] = None,
                 drive_urgency: Optional[dict] = None) -> ValueEvaluation:
        """
        Evaluate an outcome and return a ValueEvaluation.

        raw_reward     : base reinforcement signal (-1 to 1)
        had_social     : was a person involved / responding positively?
        is_novel       : was this an activation path not seen before?
        task_succeeded : did the system produce a successful output?
        traits         : personality trait dict
        drive_urgency  : dict of drive urgencies (amplifies relevant dimensions)
        """
        traits = traits or {}
        weights = self.personality_weights(traits)

        # Build component scores
        short_term    = float(raw_reward)
        long_term     = float(raw_reward) * 0.7 + (0.15 if task_succeeded else -0.05)
        social_impact = (0.6 if had_social else -0.05) * (1.0 + abs(raw_reward) * 0.2)
        novelty_score = 0.5 if is_novel else 0.1
        competence_sc = 0.5 if task_succeeded else (0.0 if raw_reward >= 0 else -0.2)

        # Drive amplification — unmet drive amplifies the corresponding dimension
        if drive_urgency:
            if drive_urgency.get("curiosity", 0) > 0.3:
                novelty_score  *= 1.0 + drive_urgency["curiosity"] * 0.4
            if drive_urgency.get("social", 0) > 0.3 and had_social:
                social_impact  *= 1.0 + drive_urgency["social"] * 0.5
            if drive_urgency.get("competence", 0) > 0.3 and task_succeeded:
                competence_sc  *= 1.0 + drive_urgency["competence"] * 0.4
            if drive_urgency.get("stability", 0) > 0.3:
                long_term      *= 1.0 + drive_urgency["stability"] * 0.2

        # Clip all to [-1, 1]
        def _clip(v): return max(-1.0, min(1.0, v))

        ev = ValueEvaluation(
            short_term    = _clip(short_term),
            long_term     = _clip(long_term),
            social_impact = _clip(social_impact),
            novelty       = _clip(novelty_score),
            competence    = _clip(competence_sc),
            weights       = weights,
        )

        with self._lock:
            self._history.append(ev)
            if len(self._history) > self._max_history:
                self._history.pop(0)
            for k in self._dim_sums:
                self._dim_sums[k] += getattr(ev, k.replace("_impact", "_impact").replace("short_term","short_term"), 0)
            self._dim_counts += 1

        return ev

    def recent_dimension_averages(self) -> dict:
        """Average value dimensions over recent evaluations."""
        if not self._dim_counts:
            return {}
        return {k: round(v / self._dim_counts, 3) for k, v in self._dim_sums.items()}

    def summarize(self, n: int = 10) -> dict:
        with self._lock:
            recent = self._history[-n:]
        if not recent:
            return {}
        avg_final = sum(e.final_score for e in recent) / len(recent)
        avg_social = sum(e.social_impact for e in recent) / len(recent)
        avg_novelty = sum(e.novelty for e in recent) / len(recent)
        return {
            "avg_final_score": round(avg_final, 3),
            "avg_social":      round(avg_social, 3),
            "avg_novelty":     round(avg_novelty, 3),
            "evaluations":     len(recent),
        }
