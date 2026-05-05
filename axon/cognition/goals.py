"""
AXON — Goal System
==================
Goals give AXON direction and persistence.

Each goal has:
  name        : short identifier
  description : what AXON is trying to do
  priority    : 0.0–1.0 (higher = more urgent)
  progress    : 0.0–1.0 (updated by reward signals)
  satisfied   : bool

Built-in default goals that can be tuned or replaced:
  - explore_uncertainty   : seek novel states
  - reduce_error          : minimize prediction error
  - maintain_coherence    : keep belief system consistent
  - deepen_understanding  : accumulate high-confidence knowledge

Goals inject context into the LLM system prompt and bias neural stimulation.
Persisted to data/goals.json.
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional


class Goal:
    __slots__ = ("name", "description", "priority", "progress",
                 "satisfied", "created_at", "last_updated",
                 "total_reward", "steps_taken")

    def __init__(self, name: str, description: str,
                 priority: float = 0.5, progress: float = 0.0):
        self.name         = name
        self.description  = description
        self.priority     = max(0.0, min(1.0, priority))
        self.progress     = max(0.0, min(1.0, progress))
        self.satisfied    = False
        self.created_at   = time.time()
        self.last_updated = time.time()
        self.total_reward = 0.0
        self.steps_taken  = 0

    def update(self, reward: float, step_size: float = 0.02):
        """Nudge progress based on reward signal."""
        if self.satisfied:
            return
        if reward > 0:
            self.progress = min(1.0, self.progress + step_size * reward)
        else:
            # Small decay on penalty — goals don't collapse easily
            self.progress = max(0.0, self.progress + step_size * reward * 0.3)
        self.total_reward += reward
        self.steps_taken  += 1
        self.last_updated  = time.time()
        if self.progress >= 0.95:
            self.satisfied = True

    def reset(self):
        self.progress  = 0.0
        self.satisfied = False

    def to_dict(self) -> dict:
        return {
            "name":         self.name,
            "description":  self.description,
            "priority":     round(self.priority, 3),
            "progress":     round(self.progress, 3),
            "satisfied":    self.satisfied,
            "total_reward": round(self.total_reward, 3),
            "steps_taken":  self.steps_taken,
        }


DEFAULT_GOALS = [
    Goal("explore_uncertainty",
         "Seek novel patterns and explore areas of high uncertainty",
         priority=0.75),
    Goal("reduce_error",
         "Minimise prediction errors — understand what I don't know",
         priority=0.80),
    Goal("maintain_coherence",
         "Keep my belief system consistent — resolve contradictions",
         priority=0.70),
    Goal("deepen_understanding",
         "Accumulate high-confidence knowledge on topics I care about",
         priority=0.60),
]


class GoalSystem:
    """
    Manages AXON's active goals.
    Goals bias fabric stimulation and are injected into the LLM context.
    """

    def __init__(self, data_dir: str):
        self._path  = Path(data_dir) / "goals.json"
        self._lock  = threading.Lock()
        self._goals: Dict[str, Goal] = {}
        self._load()

    def _load(self):
        """Load persisted goals, fall back to defaults."""
        defaults = {g.name: g for g in DEFAULT_GOALS}
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                for item in data:
                    name = item["name"]
                    g    = Goal(name, item.get("description",""),
                                item.get("priority", 0.5),
                                item.get("progress", 0.0))
                    g.total_reward = item.get("total_reward", 0.0)
                    g.steps_taken  = item.get("steps_taken", 0)
                    g.satisfied    = item.get("satisfied", False)
                    defaults[name] = g   # override default with saved
            except Exception:
                pass
        self._goals = defaults

    def save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._path.write_text(
                json.dumps([g.to_dict() for g in self._goals.values()], indent=2)
            )

    # ── Query ──────────────────────────────────────────────────────────────

    def active_goals(self) -> List[Goal]:
        with self._lock:
            return sorted(
                [g for g in self._goals.values() if not g.satisfied],
                key=lambda g: -g.priority
            )

    def all_goals(self) -> List[dict]:
        with self._lock:
            return [g.to_dict() for g in
                    sorted(self._goals.values(), key=lambda g: -g.priority)]

    def top_goal(self) -> Optional[Goal]:
        active = self.active_goals()
        return active[0] if active else None

    # ── Update ─────────────────────────────────────────────────────────────

    def reward_tick(self, reward: float, context: dict = None):
        """
        Distribute reward to active goals based on relevance.
        context keys that matter: is_novel, resolved_conflict, learned_fact
        """
        with self._lock:
            for g in self._goals.values():
                if g.satisfied:
                    continue
                relevance = self._relevance(g, reward, context or {})
                if abs(relevance) > 0.001:
                    g.update(relevance)

    def _relevance(self, goal: Goal, reward: float, ctx: dict) -> float:
        name = goal.name
        if name == "explore_uncertainty":
            return reward * (1.5 if ctx.get("is_novel") else 0.3)
        if name == "reduce_error":
            return reward * (1.3 if ctx.get("low_surprise") else 0.5)
        if name == "maintain_coherence":
            return reward * (1.4 if ctx.get("resolved_conflict") else 0.2)
        if name == "deepen_understanding":
            return reward * (1.2 if ctx.get("learned_fact") else 0.3)
        return reward * 0.3

    # ── Mutation ───────────────────────────────────────────────────────────

    def add_goal(self, name: str, description: str, priority: float = 0.5) -> Goal:
        with self._lock:
            g = Goal(name, description, priority)
            self._goals[name] = g
        self.save()
        return g

    def remove_goal(self, name: str) -> bool:
        with self._lock:
            if name in self._goals:
                del self._goals[name]
                self.save()
                return True
        return False

    def reset_satisfied(self):
        """Reset all satisfied goals back to active."""
        with self._lock:
            for g in self._goals.values():
                if g.satisfied:
                    g.reset()
        self.save()

    # ── Context injection ──────────────────────────────────────────────────

    def as_context_string(self) -> str:
        active = self.active_goals()[:3]
        if not active:
            return ""
        lines = [f"- {g.name}: {g.description} (priority {g.priority:.0%}, progress {g.progress:.0%})"
                 for g in active]
        return "Active goals:\n" + "\n".join(lines)

    # ── Fabric hints ───────────────────────────────────────────────────────

    def fabric_hints(self) -> List[tuple]:
        """Return (region, amount) pairs to stimulate based on top goal."""
        top = self.top_goal()
        if not top:
            return []
        mapping = {
            "explore_uncertainty":  [("association_cortex", 0.04), ("prefrontal_cortex", 0.03)],
            "reduce_error":         [("prefrontal_cortex",  0.05), ("anterior_cingulate", 0.03)],
            "maintain_coherence":   [("prefrontal_cortex",  0.04), ("hippocampus", 0.03)],
            "deepen_understanding": [("hippocampus",        0.05), ("temporal_lobe", 0.03)],
        }
        return mapping.get(top.name, [])
