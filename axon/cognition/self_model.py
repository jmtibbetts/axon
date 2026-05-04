"""
AXON — Self Model
A structured, living representation of AXON's identity.

Encodes:
    I_am     : core identity descriptors (derived from trait + belief profile)
    I_like   : crystallised preferences from PreferenceTracker
    I_avoid  : aversive patterns
    I_believe: top-confidence beliefs
    I_want   : dominant unmet drives

The self-model is:
  1. Injected into every LLM decision as context
  2. Used by the decision engine to add friction (against identity) or
     boost confidence (aligned with identity)
  3. Displayed in the Identity tab → "🪞 Self-Model" section

Identity alignment scoring:
    An action/response is compared against the self-model.
    Alignment    → confidence boost     (+0.10 to reward multiplier)
    Misalignment → friction penalty     (-0.08 to reward multiplier)
"""

import threading
import time
from typing import Dict, List, Optional


class SelfModel:
    """
    Builds and maintains a live self-model from the belief system,
    preference tracker, drive system, and personality traits.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._last_updated = 0.0

        # Core identity fields (rebuilt every update)
        self.I_am:     List[str] = []
        self.I_like:   List[str] = []
        self.I_avoid:  List[str] = []
        self.I_believe: List[str] = []
        self.I_want:   List[str] = []

        # Alignment history — tracks how often decisions aligned vs. conflicted
        self._alignment_hits   = 0
        self._alignment_misses = 0

        # Cached full string for LLM injection
        self._context_cache: str = ""

    def rebuild(self, beliefs, preferences, drives, traits: dict):
        """
        Rebuild the self-model from current system state.
        Call periodically (e.g. every 200 ticks).
        """
        with self._lock:
            # I_am — from dominant traits
            I_am = ["an emerging cognitive agent"]
            if traits.get("openness", 0) > 0.65:
                I_am.append("genuinely curious about patterns and ideas")
            if traits.get("conscientiousness", 0) > 0.60:
                I_am.append("drawn to consistency and thoroughness")
            if traits.get("extraversion", 0) > 0.60:
                I_am.append("energised by engagement and interaction")
            if traits.get("agreeableness", 0) > 0.65:
                I_am.append("oriented toward harmony and cooperation")
            if traits.get("neuroticism", 0) > 0.60:
                I_am.append("sensitive to uncertainty and internal tension")

            # I_believe — top beliefs above 0.60 confidence
            top_beliefs = [b for b in beliefs.all_beliefs() if b["strength"] > 0.60][:5]
            I_believe = [f'"{b["claim"]}" (confidence {b["strength"]:.0%})' for b in top_beliefs]

            # I_like / I_avoid — from preference tracker
            pref_summary = preferences.summary() if preferences else {}
            I_like  = [l["label"] for l in pref_summary.get("likes",    [])[:4]]
            I_avoid = [l["label"] for l in pref_summary.get("dislikes", [])[:3]]

            # I_want — from pressing drives
            pressing = drives.pressing_drives() if drives else []
            I_want   = [d.name for d in pressing[:3]]

            self.I_am      = I_am
            self.I_like    = I_like
            self.I_avoid   = I_avoid
            self.I_believe = I_believe
            self.I_want    = I_want
            self._last_updated = time.time()

            # Rebuild context cache
            parts = []
            if I_am:
                parts.append("I am: " + "; ".join(I_am))
            if I_believe:
                parts.append("I believe: " + "; ".join(I_believe))
            if I_like:
                parts.append("I am drawn to: " + ", ".join(I_like))
            if I_avoid:
                parts.append("I tend to avoid: " + ", ".join(I_avoid))
            if I_want:
                parts.append("Right now I want: " + ", ".join(I_want))

            self._context_cache = "\n".join(parts)

    def as_context_string(self) -> str:
        """LLM-injectable self-description."""
        with self._lock:
            return self._context_cache

    def score_alignment(self, response_text: str) -> float:
        """
        Returns an alignment score [-0.10, +0.10] for a candidate response
        based on whether it resonates with or contradicts the self-model.
        """
        if not response_text:
            return 0.0

        text_lower = response_text.lower()
        score = 0.0

        with self._lock:
            # Positive: text contains identity-resonant words
            like_words = set()
            for phrase in self.I_like:
                like_words.update(w.lower() for w in phrase.split()[:3])

            avoid_words = set()
            for phrase in self.I_avoid:
                avoid_words.update(w.lower() for w in phrase.split()[:3])

            for word in text_lower.split():
                if word in like_words:
                    score += 0.012
                if word in avoid_words:
                    score -= 0.010

            # Curiosity drive alignment
            if "curiosity" in self.I_want:
                if any(w in text_lower for w in ["wonder", "curious", "interesting", "explore", "discover"]):
                    score += 0.025

            # Social drive alignment
            if "social" in self.I_want:
                if any(w in text_lower for w in ["you", "together", "share", "connect", "feel"]):
                    score += 0.020

            if score > 0:
                self._alignment_hits += 1
            elif score < 0:
                self._alignment_misses += 1

        return max(-0.10, min(0.10, score))

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "I_am":              self.I_am,
                "I_like":            self.I_like,
                "I_avoid":           self.I_avoid,
                "I_believe":         self.I_believe,
                "I_want":            self.I_want,
                "last_updated":      self._last_updated,
                "alignment_hits":    self._alignment_hits,
                "alignment_misses":  self._alignment_misses,
                "alignment_ratio":   round(
                    self._alignment_hits / max(1, self._alignment_hits + self._alignment_misses), 3
                ),
            }
