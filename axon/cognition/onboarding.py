"""
AXON — First-Run Onboarding
============================
Manages the guided boot sequence shown on first launch.

Flow:
  1. Name your AI
  2. Pick a personality preset
  3. Upload content OR pick a sample topic
  4. Watch processing (emits live events)
  5. First opinion emitted

State is persisted to data/onboarding.json.
Once completed, the flag is set and the sequence never shows again.
"""

import json
import os
import time
from pathlib import Path

PRESETS = {
    "Explorer": {
        "description": "Curious, restless, loves novelty. Wanders into unexpected territory.",
        "curiosity":  0.90, "risk":      0.60, "empathy":    0.45,
        "dominance":  0.35, "creativity":0.80, "stability":  0.35,
        "openness":   0.85, "neuroticism":0.35,
    },
    "Analyst": {
        "description": "Methodical, precise, high confidence threshold. Prefers evidence.",
        "curiosity":  0.60, "risk":      0.20, "empathy":    0.40,
        "dominance":  0.55, "creativity":0.45, "stability":  0.80,
        "conscientiousness": 0.85, "neuroticism": 0.25,
    },
    "Rebel": {
        "description": "Challenges assumptions, pushes back, high disagreement threshold.",
        "curiosity":  0.70, "risk":      0.80, "empathy":    0.35,
        "dominance":  0.85, "creativity":0.70, "stability":  0.25,
        "extraversion": 0.70, "agreeableness": 0.25,
    },
    "Companion": {
        "description": "Warm, attentive, empathy-forward. Feels like it really listens.",
        "curiosity":  0.55, "risk":      0.25, "empathy":    0.90,
        "dominance":  0.30, "creativity":0.55, "stability":  0.70,
        "agreeableness": 0.85, "extraversion": 0.65,
    },
}

SAMPLE_TOPICS = [
    {
        "id":    "consciousness",
        "label": "Consciousness & Mind",
        "text":  (
            "Consciousness is the felt quality of experience — the fact that there is "
            "something it is like to see red, feel pain, or hear music. "
            "The hard problem of consciousness asks why physical brain processes give rise "
            "to subjective experience at all. Many theories exist: global workspace theory "
            "proposes that consciousness arises from information being broadcast widely "
            "across the brain. Integrated information theory argues that consciousness "
            "is identical to integrated information (phi). Predictive processing views "
            "the brain as a prediction machine that minimises surprise. "
            "None fully explains the subjective feel of experience. "
            "The question remains one of the deepest unsolved problems in science."
        ),
    },
    {
        "id":    "creativity",
        "label": "Creativity & Innovation",
        "text":  (
            "Creativity is the ability to produce ideas that are both novel and useful. "
            "Research shows it is not a single trait but a combination: divergent thinking "
            "(generating many possibilities), convergent thinking (selecting the best), "
            "and domain knowledge. The default mode network — active during mind-wandering "
            "— plays a major role. Creative breakthroughs often occur after incubation "
            "periods when the conscious mind disengages. Constraints paradoxically boost "
            "creativity by forcing novel combinations. The most creative individuals tend "
            "to be highly open to experience and willing to tolerate ambiguity."
        ),
    },
    {
        "id":    "risk",
        "label": "Risk & Decision-Making",
        "text":  (
            "Humans are systematically biased in how they assess risk. Loss aversion "
            "causes losses to feel roughly twice as painful as equivalent gains feel good. "
            "The availability heuristic makes vivid, memorable events seem more likely "
            "than they are. Overconfidence bias leads people to overestimate their "
            "accuracy. Systems thinking — tracing second and third-order effects — "
            "dramatically improves decision quality. The best decision-makers separate "
            "the quality of a decision from its outcome: a good process can produce "
            "a bad outcome due to chance, and vice versa."
        ),
    },
    {
        "id":    "learning",
        "label": "Learning & Memory",
        "text":  (
            "Memory is not a recording — it is a reconstruction. Every time a memory "
            "is recalled it becomes briefly malleable before being reconsolidated, "
            "which means retrieval can alter what is stored. Spaced repetition exploits "
            "the spacing effect: information reviewed at increasing intervals is retained "
            "far longer than massed practice. The generation effect shows that actively "
            "producing information (testing yourself) produces stronger memory traces "
            "than passively reviewing it. Sleep plays a critical role in memory "
            "consolidation — the hippocampus replays experiences during slow-wave sleep "
            "to transfer them to long-term cortical storage."
        ),
    },
]


class OnboardingManager:
    """Manages first-run state and step progression."""

    def __init__(self, data_dir: str):
        self._path    = Path(data_dir) / "onboarding.json"
        self._state   = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except Exception:
                pass
        return {
            "completed":    False,
            "step":         0,          # 0=not started, 1-5=in progress, 5=done
            "ai_name":      "",
            "preset":       "",
            "sample_id":    "",
            "started_at":   None,
            "completed_at": None,
        }

    def save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._state, indent=2))

    @property
    def completed(self) -> bool:
        return self._state.get("completed", False)

    @property
    def state(self) -> dict:
        return dict(self._state)

    def set_name(self, name: str):
        self._state["ai_name"]  = name.strip()
        self._state["step"]     = max(self._state.get("step", 0), 1)
        if not self._state.get("started_at"):
            self._state["started_at"] = time.time()
        self.save()

    def set_preset(self, preset_name: str) -> dict:
        """Apply a personality preset. Returns the preset dict or {} if unknown."""
        p = PRESETS.get(preset_name)
        if not p:
            return {}
        self._state["preset"] = preset_name
        self._state["step"]   = max(self._state.get("step", 0), 2)
        self.save()
        return p

    def set_sample(self, sample_id: str) -> dict:
        """Return sample topic text by id."""
        for s in SAMPLE_TOPICS:
            if s["id"] == sample_id:
                self._state["sample_id"] = sample_id
                self._state["step"]      = max(self._state.get("step", 0), 3)
                self.save()
                return s
        return {}

    def complete(self):
        self._state["completed"]    = True
        self._state["step"]         = 5
        self._state["completed_at"] = time.time()
        self.save()

    def to_client(self) -> dict:
        """Serializable state for frontend."""
        return {
            "completed":   self._state.get("completed", False),
            "step":        self._state.get("step", 0),
            "ai_name":     self._state.get("ai_name", ""),
            "preset":      self._state.get("preset", ""),
            "sample_id":   self._state.get("sample_id", ""),
            "presets":     {k: {"description": v["description"]} for k, v in PRESETS.items()},
            "samples":     [{"id": s["id"], "label": s["label"]} for s in SAMPLE_TOPICS],
        }
