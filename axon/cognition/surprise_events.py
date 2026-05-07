"""
AXON — Surprise Event System
==============================
Detects and surfaces "moments of surprise" — internal state changes
significant enough to be shown as live notifications.

Events:
  - belief_shift          : a belief's strength changed by > threshold
  - cognitive_dissonance  : new info strongly conflicts with existing belief
  - contradiction_resolved: a dissonant belief was reconciled
  - unexpected_conclusion : prediction error spike while ingesting knowledge
  - personality_drift     : a personality trait shifted noticeably
  - goal_progress         : a goal crossed a milestone (25%, 50%, 75%, 100%)
  - cluster_dominance_flip: the dominant brain region changed unexpectedly

Each event has:
  type    : string identifier
  title   : short human-readable label
  detail  : 1-sentence explanation
  severity: "low" | "medium" | "high"
  ts      : unix timestamp
"""

import time
import threading
from collections import deque
from typing import Callable, Optional


class SurpriseEvent:
    __slots__ = ("type", "title", "detail", "severity", "ts", "data")

    def __init__(self, type_: str, title: str, detail: str,
                 severity: str = "medium", data: dict = None):
        self.type     = type_
        self.title    = title
        self.detail   = detail
        self.severity = severity
        self.ts       = time.time()
        self.data     = data or {}

    def to_dict(self) -> dict:
        return {
            "type":     self.type,
            "title":    self.title,
            "detail":   self.detail,
            "severity": self.severity,
            "ts":       self.ts,
            "data":     self.data,
        }


class SurpriseDetector:
    """
    Watches for notable internal events and fires callbacks.
    Throttled per-type to avoid flooding.
    """

    COOLDOWNS = {
        "belief_shift":           30.0,
        "cognitive_dissonance":   15.0,
        "contradiction_resolved": 20.0,
        "unexpected_conclusion":  10.0,
        "personality_drift":      60.0,
        "goal_progress":          20.0,
        "cluster_dominance_flip": 120.0,
    }

    def __init__(self, on_event: Optional[Callable] = None):
        self._on_event   = on_event
        self._lock       = threading.Lock()
        self._history    = deque(maxlen=100)
        self._last_fired = {}    # type → last ts

        # State tracking for delta detection
        self._last_dominant_cluster: str = ""
        self._last_personality: dict     = {}
        self._last_belief_strengths: dict = {}
        self._last_goal_milestones: dict  = {}

    def _should_fire(self, event_type: str) -> bool:
        cooldown = self.COOLDOWNS.get(event_type, 15.0)
        last     = self._last_fired.get(event_type, 0.0)
        return (time.time() - last) >= cooldown

    def _fire(self, event: SurpriseEvent):
        with self._lock:
            self._history.append(event)
            self._last_fired[event.type] = time.time()
        if self._on_event:
            try:
                self._on_event(event.to_dict())
            except Exception:
                pass

    # ── Check methods called by cognitive cycle ───────────────────────────

    def check_belief_shift(self, belief_key: str, claim: str,
                           old_strength: float, new_strength: float):
        delta = new_strength - old_strength
        if abs(delta) < 0.12:
            return
        if not self._should_fire("belief_shift"):
            return
        direction = "strengthened" if delta > 0 else "weakened"
        self._fire(SurpriseEvent(
            type_    = "belief_shift",
            title    = f"Belief {direction}",
            detail   = f'"{claim[:60]}" shifted {delta:+.2f} → {new_strength:.2f}',
            severity = "medium" if abs(delta) < 0.25 else "high",
            data     = {"key": belief_key, "delta": round(delta, 3)},
        ))

    def check_dissonance(self, claim: str, dissonance: float):
        if dissonance < 0.30:
            return
        if not self._should_fire("cognitive_dissonance"):
            return
        severity = "high" if dissonance > 0.55 else "medium"
        self._fire(SurpriseEvent(
            type_    = "cognitive_dissonance",
            title    = "⚡ Cognitive Dissonance Spike",
            detail   = f'New belief conflicts with existing model: "{claim[:60]}"',
            severity = severity,
            data     = {"dissonance": round(dissonance, 3)},
        ))

    def check_contradiction_resolved(self, claim: str, old_val: float, new_val: float):
        if abs(old_val - new_val) < 0.20:
            return
        if not self._should_fire("contradiction_resolved"):
            return
        self._fire(SurpriseEvent(
            type_    = "contradiction_resolved",
            title    = "🔗 Contradiction Resolved",
            detail   = f'Conflicting belief updated: "{claim[:60]}"',
            severity = "medium",
            data     = {"old": round(old_val,3), "new": round(new_val,3)},
        ))

    def check_surprise_spike(self, surprise: float, context: str = ""):
        if surprise < 0.45:
            return
        if not self._should_fire("unexpected_conclusion"):
            return
        self._fire(SurpriseEvent(
            type_    = "unexpected_conclusion",
            title    = "🌀 Unexpected Conclusion Detected",
            detail   = f"Prediction error spike: {surprise:.2f}" + (f" — {context[:60]}" if context else ""),
            severity = "high" if surprise > 0.65 else "medium",
            data     = {"surprise": round(surprise, 3)},
        ))

    def check_personality_drift(self, new_traits: dict):
        if not self._last_personality:
            self._last_personality = dict(new_traits)
            return
        changed = {}
        for k, v in new_traits.items():
            old = self._last_personality.get(k, v)
            if abs(v - old) >= 0.025:
                changed[k] = round(v - old, 3)
        if not changed or not self._should_fire("personality_drift"):
            return
        self._last_personality = dict(new_traits)
        desc_parts = [f"{k} {'↑' if d>0 else '↓'} {abs(d):.2f}" for k, d in list(changed.items())[:3]]
        self._fire(SurpriseEvent(
            type_    = "personality_drift",
            title    = "🎭 Personality Drift Detected",
            detail   = "Your AI is shifting: " + ", ".join(desc_parts),
            severity = "high" if max(abs(v) for v in changed.values()) > 0.05 else "medium",
            data     = {"changes": changed},
        ))

    def check_dominant_cluster(self, dominant: str, activation: float):
        if dominant == self._last_dominant_cluster:
            return
        if not self._should_fire("cluster_dominance_flip"):
            return
        old = self._last_dominant_cluster
        self._last_dominant_cluster = dominant
        if old:   # Only fire after first observation
            self._fire(SurpriseEvent(
                type_    = "cluster_dominance_flip",
                title    = "🧠 Dominance Shift",
                detail   = f"{old} → {dominant} ({activation:.0%} activation)",
                severity = "low",
                data     = {"from": old, "to": dominant},
            ))

    def check_goal_progress(self, goal_name: str, progress: float):
        milestones = [0.25, 0.50, 0.75, 1.0]
        last = self._last_goal_milestones.get(goal_name, 0.0)
        for m in milestones:
            if last < m <= progress:
                self._last_goal_milestones[goal_name] = progress
                if not self._should_fire("goal_progress"):
                    return
                label = "✅ Goal complete!" if m == 1.0 else f"Goal {int(m*100)}% reached"
                self._fire(SurpriseEvent(
                    type_    = "goal_progress",
                    title    = label,
                    detail   = f'"{goal_name}" reached {int(m*100)}% progress',
                    severity = "high" if m == 1.0 else "medium",
                    data     = {"goal": goal_name, "milestone": m},
                ))
                break

    # ── History ─────────────────────────────────────────────────────────────

    def recent_events(self, n: int = 10) -> list:
        with self._lock:
            return [e.to_dict() for e in list(self._history)[-n:]]
