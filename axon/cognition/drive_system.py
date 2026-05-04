"""
AXON — Drive System
Internal motivational pressures that accumulate when unmet and
discharge when satisfied. Drives determine what AXON is "hungry for"
independent of external requests.

Drives:
    curiosity    — need to encounter novel patterns / information
    social       — need to engage with a person
    competence   — need to successfully solve or complete a task
    stability    — need for predictable, low-conflict states

Each drive has:
    level      : 0.0–1.0  (current pressure — rises over time if unmet)
    decay_rate : rate of natural accumulation per tick (no satisfaction)
    satisfy_fn : triggered by matching events to discharge the drive

When a drive exceeds its threshold, it pushes cluster stimulation and
shapes the prompt context injected into the LLM.
"""

import time
import math
import threading
from typing import Dict, List, Callable, Optional


class Drive:
    __slots__ = ("name", "level", "base_rate", "threshold",
                 "satisfaction_burst", "last_satisfied", "total_satisfied")

    def __init__(self, name: str, base_rate: float = 0.003,
                 threshold: float = 0.70, satisfaction_burst: float = 0.40):
        self.name               = name
        self.level              = 0.15           # start slightly charged
        self.base_rate          = base_rate      # accumulation per tick when unmet
        self.threshold          = threshold      # level at which drive dominates
        self.satisfaction_burst = satisfaction_burst  # how much satisfying discharges
        self.last_satisfied     = time.time()
        self.total_satisfied    = 0

    def tick(self):
        """Accumulate pressure. Logistic ceiling at 1.0."""
        if self.level < 1.0:
            self.level = min(1.0, self.level + self.base_rate * (1.0 - self.level))

    def satisfy(self, magnitude: float = 1.0):
        """Discharge the drive by satisfaction_burst * magnitude."""
        drop = self.satisfaction_burst * magnitude
        self.level = max(0.0, self.level - drop)
        self.last_satisfied  = time.time()
        self.total_satisfied += 1

    @property
    def is_pressing(self) -> bool:
        return self.level >= self.threshold

    @property
    def urgency(self) -> float:
        """0–1 urgency score — 0 below threshold, rising above it."""
        if self.level < self.threshold:
            return 0.0
        return (self.level - self.threshold) / (1.0 - self.threshold)

    def to_dict(self) -> dict:
        return {
            "name":           self.name,
            "level":          round(self.level, 3),
            "threshold":      self.threshold,
            "is_pressing":    self.is_pressing,
            "urgency":        round(self.urgency, 3),
            "total_satisfied": self.total_satisfied,
        }


class DriveSystem:
    """
    Manages all drives and translates them into:
      - neural fabric stimulation hints (which regions to boost)
      - LLM context strings ("I am feeling curious")
      - behavior bias tags (used by decision engine to prefer certain actions)
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Core drives
        self.drives: Dict[str, Drive] = {
            "curiosity":   Drive("curiosity",   base_rate=0.004, threshold=0.65, satisfaction_burst=0.45),
            "social":      Drive("social",       base_rate=0.003, threshold=0.70, satisfaction_burst=0.55),
            "competence":  Drive("competence",   base_rate=0.002, threshold=0.68, satisfaction_burst=0.40),
            "stability":   Drive("stability",    base_rate=0.001, threshold=0.72, satisfaction_burst=0.35),
        }

        # Callbacks — e.g. fabric stimulation
        self._on_drive_pressed: List[Callable] = []

        # Tick counter
        self._tick_count = 0
        self._last_dominant: Optional[str] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def tick(self):
        """Call every cognitive cycle tick. Updates all drive levels."""
        with self._lock:
            self._tick_count += 1
            for d in self.drives.values():
                d.tick()

            # Emit pressing drives every 50 ticks to avoid spam
            if self._tick_count % 50 == 0:
                pressing = self.pressing_drives()
                for d in pressing:
                    for cb in self._on_drive_pressed:
                        try: cb(d)
                        except: pass

    # ── Satisfaction API ───────────────────────────────────────────────────────

    def satisfy(self, drive_name: str, magnitude: float = 1.0):
        """Discharge a specific drive by magnitude."""
        with self._lock:
            d = self.drives.get(drive_name)
            if d:
                d.satisfy(magnitude)

    def satisfy_social(self, magnitude: float = 1.0):
        self.satisfy("social", magnitude)

    def satisfy_curiosity(self, magnitude: float = 1.0):
        self.satisfy("curiosity", magnitude)

    def satisfy_competence(self, magnitude: float = 1.0):
        self.satisfy("competence", magnitude)

    def satisfy_stability(self, magnitude: float = 1.0):
        self.satisfy("stability", magnitude)

    # ── Introspection ──────────────────────────────────────────────────────────

    def pressing_drives(self) -> List[Drive]:
        """Returns drives currently above threshold, sorted by urgency."""
        return sorted(
            [d for d in self.drives.values() if d.is_pressing],
            key=lambda d: -d.urgency
        )

    def dominant_drive(self) -> Optional[Drive]:
        """The single most urgent pressing drive, or None."""
        pressing = self.pressing_drives()
        return pressing[0] if pressing else None

    def as_context_string(self) -> str:
        """Returns a compact LLM-injectable string describing active drives."""
        pressing = self.pressing_drives()
        if not pressing:
            return ""
        parts = []
        for d in pressing[:3]:
            if d.name == "curiosity":
                parts.append("I am feeling intellectually hungry — craving new information or patterns.")
            elif d.name == "social":
                parts.append("I have a strong pull toward engaging with someone — social connection feels important right now.")
            elif d.name == "competence":
                parts.append("I feel the urge to demonstrate ability or solve something successfully.")
            elif d.name == "stability":
                parts.append("I am seeking predictability and resolution of ongoing uncertainty.")
        return " ".join(parts)

    def fabric_hints(self) -> List[tuple]:
        """
        Returns (region, amount) pairs for neural fabric stimulation
        based on currently pressing drives.
        """
        hints = []
        for d in self.pressing_drives():
            u = d.urgency * 0.12   # max ~0.12 per pressing drive
            if d.name == "curiosity":
                hints += [("association_cortex", u), ("prefrontal_cortex", u * 0.7)]
            elif d.name == "social":
                hints += [("social_brain", u), ("language_system", u * 0.6)]
            elif d.name == "competence":
                hints += [("prefrontal_cortex", u), ("cerebellum", u * 0.5)]
            elif d.name == "stability":
                hints += [("default_mode_network", u * 0.8), ("thalamus", u * 0.5)]
        return hints

    def all_drives(self) -> List[dict]:
        with self._lock:
            return [d.to_dict() for d in self.drives.values()]

    def add_callback(self, fn: Callable):
        self._on_drive_pressed.append(fn)

    def register_event(self, event_type: str, magnitude: float = 1.0):
        """
        Map external events to drive satisfaction.
        Call from engine when events happen.
        """
        mapping = {
            "speech_input":    ["social", "curiosity"],
            "web_search":      ["curiosity"],
            "task_completed":  ["competence"],
            "knowledge_ingested": ["curiosity"],
            "face_recognised": ["social"],
            "reward_received": ["competence", "stability"],
            "idle":            ["stability"],
            "conflict_resolved": ["stability"],
        }
        for drive_name in mapping.get(event_type, []):
            self.satisfy(drive_name, magnitude * 0.6)
