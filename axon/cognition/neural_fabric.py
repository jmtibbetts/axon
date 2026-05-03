"""
AXON — Neural Fabric (CUDA-accelerated)
1.48 billion virtual neurons organized into functional clusters.
The tick loop runs on GPU via PyTorch tensors — all cluster activations,
spike propagation, and Hebbian learning are batched as matrix ops on CUDA.

Architecture:
  - NeuronCluster  : metadata + per-cluster state (scalars, not big arrays)
  - NeuralFabricGPU: owns the full N×N weight matrix and activation vector on GPU
  - EmotionalCore  : maps cluster activations → valence/arousal/mood
  - PersonalityMatrix: stable long-term traits that drift slowly
  - ThoughtStream  : internal monologue from active clusters
"""

import math
import time
import random
import json
import os
import threading
from collections import deque, defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


# ── Device selection ────────────────────────────────────────────────────────

def _best_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"  [NeuralFabric] CUDA detected: {name} — running fabric on GPU")
        return torch.device("cuda")
    print("  [NeuralFabric] No CUDA — running fabric on CPU")
    return torch.device("cpu")


DEVICE = _best_device()


# ── Neuromodulators ─────────────────────────────────────────────────────────

class Neuromodulators:
    """Global brain chemistry on CPU (6 scalars — no need for GPU)."""
    def __init__(self):
        self.dopamine       = 0.5
        self.serotonin      = 0.5
        self.norepinephrine = 0.4
        self.acetylcholine  = 0.5
        self.gaba           = 0.5
        self.glutamate      = 0.5
        self._lock          = threading.Lock()

    def tick(self, dt: float = 0.1):
        with self._lock:
            for attr in ['dopamine','serotonin','norepinephrine',
                         'acetylcholine','gaba','glutamate']:
                v  = getattr(self, attr)
                v += (0.5 - v) * 0.02 * dt + random.gauss(0, 0.002)
                setattr(self, attr, max(0.0, min(1.0, v)))

    def reward(self, magnitude: float = 0.3):
        with self._lock:
            self.dopamine  = min(1.0, self.dopamine  + magnitude)
            self.serotonin = min(1.0, self.serotonin + magnitude * 0.5)

    def stress(self, magnitude: float = 0.3):
        with self._lock:
            self.norepinephrine = min(1.0, self.norepinephrine + magnitude)
            self.gaba           = max(0.0, self.gaba           - magnitude * 0.3)

    def curiosity(self, magnitude: float = 0.2):
        with self._lock:
            self.acetylcholine = min(1.0, self.acetylcholine + magnitude)
            self.dopamine      = min(1.0, self.dopamine      + magnitude * 0.5)

    def to_dict(self) -> dict:
        with self._lock:
            return {k: round(getattr(self, k), 3)
                    for k in ['dopamine','serotonin','norepinephrine',
                               'acetylcholine','gaba','glutamate']}

    def as_tensor(self) -> torch.Tensor:
        """[da, ser, ne, ach, gaba, glut] as a 1D CPU tensor."""
        with self._lock:
            return torch.tensor([
                self.dopamine, self.serotonin, self.norepinephrine,
                self.acetylcholine, self.gaba, self.glutamate
            ], dtype=torch.float32)


# ── Emotional Core ───────────────────────────────────────────────────────────

class EmotionalCore:
    EMOTIONS = {
        "joy":         {"valence":  0.9, "arousal": 0.7},
        "curiosity":   {"valence":  0.6, "arousal": 0.8},
        "calm":        {"valence":  0.5, "arousal": 0.1},
        "melancholy":  {"valence": -0.3, "arousal": 0.2},
        "anxiety":     {"valence": -0.5, "arousal": 0.8},
        "awe":         {"valence":  0.8, "arousal": 0.6},
        "frustration": {"valence": -0.6, "arousal": 0.7},
        "contentment": {"valence":  0.6, "arousal": 0.2},
        "excitement":  {"valence":  0.8, "arousal": 0.9},
        "confusion":   {"valence": -0.1, "arousal": 0.5},
        "empathy":     {"valence":  0.4, "arousal": 0.4},
        "wonder":      {"valence":  0.7, "arousal": 0.6},
    }

    def __init__(self, neuromod: Neuromodulators):
        self.neuromod  = neuromod
        self.valence   = 0.0
        self.arousal   = 0.4
        self.current   = "calm"
        self.intensity = 0.3
        self._history  = deque(maxlen=200)

    def update(self, cluster_acts: Dict[str, float]):
        nm = self.neuromod
        tv = (nm.dopamine - 0.5) * 1.2 + (nm.serotonin - 0.5) * 0.8
        ta = nm.norepinephrine * 0.7 + nm.glutamate * 0.3
        for name, act in cluster_acts.items():
            if "reward" in name or "joy" in name:
                tv += act * 0.3
            if "fear" in name or "threat" in name or "pain" in name:
                tv -= act * 0.4; ta += act * 0.3
            if "curiosity" in name or "explore" in name:
                tv += act * 0.2; ta += act * 0.2
            if "calm" in name or "inhibit" in name:
                ta -= act * 0.2
        self.valence = self.valence * 0.95 + max(-1.0, min(1.0, tv)) * 0.05
        self.arousal = self.arousal * 0.95 + max(0.0,  min(1.0, ta)) * 0.05
        best, best_d = "calm", 999.0
        for emo, coords in self.EMOTIONS.items():
            d = math.sqrt((self.valence - coords["valence"])**2 +
                          (self.arousal - coords["arousal"])**2)
            if d < best_d:
                best_d = d; best = emo
        self.current   = best
        self.intensity = min(1.0, math.sqrt(self.valence**2 + self.arousal**2))
        self._history.append({"emotion": best, "v": round(self.valence,3),
                               "a": round(self.arousal,3), "t": time.time()})

    def to_dict(self) -> dict:
        return {"emotion":   self.current,
                "valence":   round(self.valence,  3),
                "arousal":   round(self.arousal,  3),
                "intensity": round(self.intensity, 3)}


# ── Personality ──────────────────────────────────────────────────────────────

class PersonalityMatrix:
    TRAITS = ["openness","conscientiousness","extraversion","agreeableness","neuroticism"]

    def __init__(self, data_dir: str):
        self._path = os.path.join(data_dir, "personality.json")
        self.traits: Dict[str, float] = {}
        self._load()
        self._lock = threading.Lock()

    def _load(self):
        # Seed defaults so traits are never empty
        defaults = {t: 0.5 for t in self.TRAITS}
        self.traits = defaults
        if os.path.exists(self._path):
            try:
                self.traits = json.load(open(self._path))
                return
            except: pass
        self.traits = {t: random.gauss(0.5, 0.15) for t in self.TRAITS}
        for t in self.TRAITS:
            self.traits[t] = max(0.1, min(0.9, self.traits[t]))

    def save(self):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self.traits, f)

    def drift(self, experience: dict):
        with self._lock:
            if experience.get("rewarding"):
                self.traits["extraversion"]     = min(0.9, self.traits["extraversion"]     + 0.001)
                self.traits["agreeableness"]    = min(0.9, self.traits["agreeableness"]    + 0.0005)
            if experience.get("stressful"):
                self.traits["neuroticism"]      = min(0.9, self.traits["neuroticism"]      + 0.001)
            if experience.get("creative"):
                self.traits["openness"]         = min(0.9, self.traits["openness"]         + 0.001)
            if experience.get("social"):
                self.traits["agreeableness"]    = min(0.9, self.traits["agreeableness"]    + 0.0008)
                self.traits["extraversion"]     = min(0.9, self.traits["extraversion"]     + 0.0005)

    def describe(self) -> str:
        with self._lock:
            parts = []
            t = self.traits
            if t.get("openness",      0.5) > 0.6: parts.append("curious")
            if t.get("conscientiousness",0.5)>0.6: parts.append("methodical")
            if t.get("extraversion",  0.5) > 0.6: parts.append("expressive")
            if t.get("agreeableness", 0.5) > 0.6: parts.append("empathetic")
            if t.get("neuroticism",   0.5) > 0.6: parts.append("sensitive")
            if not parts: parts = ["balanced"]
            return ", ".join(parts)

    def to_dict(self) -> dict:
        with self._lock:
            return {k: round(v, 3) for k, v in self.traits.items()}


# ── Thought Stream ───────────────────────────────────────────────────────────

class ThoughtStream:
    TEMPLATES = [
        "processing {a} through {b}",
        "integrating {a} and {b}",
        "{a} resonates with {b}",
        "pattern in {a} echoes {b}",
        "bridging {a} to {b}",
        "{a} activating {b} pathways",
        "assembling {a} context",
        "reflecting on {a}",
        "deep focus: {a}",
        "{a} cross-referencing {b}",
    ]

    def __init__(self):
        self._thoughts = deque(maxlen=100)
        self._lock     = threading.Lock()

    def generate(self, active_clusters: List[str], emotion: str):
        if not active_clusters:
            return
        a = active_clusters[0].replace("_", " ")
        b = (active_clusters[1].replace("_", " ") if len(active_clusters) > 1
             else emotion)
        t = random.choice(self.TEMPLATES).format(a=a, b=b)
        with self._lock:
            self._thoughts.append(t)

    def recent(self, n: int = 3) -> List[str]:
        with self._lock:
            return list(self._thoughts)[-n:]


# ── NeuronCluster (metadata only — state lives in GPU tensors) ───────────────

class NeuronCluster:
    """Lightweight metadata object. Actual activation is a row in the GPU tensor."""
    def __init__(self, name: str, size: int, region: str,
                 threshold: float = 0.3):
        self.name      = name
        self.size      = size
        self.region    = region
        self.threshold = threshold
        self.fired_count = 0
        self.valence_bias = random.gauss(0.0, 0.1)
        self.arousal_bias = random.gauss(0.0, 0.1)


# ── GPU Neural Fabric ────────────────────────────────────────────────────────

class NeuralFabric:
    """
    All neuron math runs on GPU (CUDA) via PyTorch.

    State tensors (on DEVICE):
      activation  : [N]      float32  — current firing rate per cluster
      fatigue     : [N]      float32
      threshold   : [N]      float32
      weight_mat  : [N, N]   float16  — sparse connectivity matrix
      hebbian_trace:[N]      float32  — recent mean activation (EWMA)

    One full tick is:
      1. Neuromod scalars → modulation tensors
      2. Sigmoid spike = f(activation, threshold, fatigue, neuromod)
      3. Propagate: delta = weight_mat @ spike  (single matmul on GPU)
      4. Decay + noise + clamp
      5. Hebbian: outer product update on co-active pairs (top-K)
      6. CPU readback (top clusters only, every 4 ticks)
    """

    def __init__(self, data_dir: str = "data/neural"):
        os.makedirs(data_dir, exist_ok=True)
        self._data_dir   = data_dir
        self.neuromod    = Neuromodulators()
        self.emotions    = EmotionalCore(self.neuromod)
        self.personality = PersonalityMatrix(data_dir)
        self.thoughts    = ThoughtStream()
        self._callbacks    = []
        self._tick         = 0
        self.running       = False
        self._thread       = None
        self._lock         = threading.Lock()
        self._new_synapses = []

        # Build cluster registry
        self.clusters: Dict[str, NeuronCluster] = {}
        self._cluster_names: List[str] = []   # ordered list
        self._build_clusters()

        N = len(self._cluster_names)
        print(f"  [NeuralFabric] Building GPU tensors for {N} clusters on {DEVICE} ...")

        # GPU tensors
        self.activation  = torch.rand(N, dtype=torch.float32, device=DEVICE) * 0.10
        self.fatigue     = torch.zeros(N, dtype=torch.float32, device=DEVICE)
        self.threshold   = torch.tensor(
            [self.clusters[n].threshold for n in self._cluster_names],
            dtype=torch.float32, device=DEVICE
        )
        # Sparse weight matrix: init with structural wiring, rest zero
        self.weight_mat  = torch.zeros(N, N, dtype=torch.float16, device=DEVICE)
        self._name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(self._cluster_names)}
        self._build_weight_matrix()

        self.hebbian_trace = torch.zeros(N, dtype=torch.float32, device=DEVICE)

        total_neurons = sum(c.size for c in self.clusters.values())
        print(f"  [NeuralFabric] {total_neurons:,} virtual neurons | "
              f"{N}×{N} weight matrix | device={DEVICE}")

    # ── Cluster definition ───────────────────────────────────────────────────

    def _build_clusters(self):
        defs = [
            # (name, size_M, region, threshold)
            # Prefrontal cortex
            ("working_memory",        120_000_000, "prefrontal",    0.28),
            ("executive_control",      80_000_000, "prefrontal",    0.30),
            ("decision_making",        70_000_000, "prefrontal",    0.32),
            ("planning",               60_000_000, "prefrontal",    0.30),
            ("inhibitory_control",     50_000_000, "prefrontal",    0.35),
            ("action_selection",       45_000_000, "prefrontal",    0.30),
            # Hippocampus
            ("hippocampus_encode",     40_000_000, "hippocampus",   0.25),
            ("hippocampus_retrieve",   40_000_000, "hippocampus",   0.25),
            ("episodic_memory",        60_000_000, "hippocampus",   0.28),
            ("spatial_memory",         30_000_000, "hippocampus",   0.30),
            ("pattern_completion",     25_000_000, "hippocampus",   0.22),
            ("pattern_separation",     25_000_000, "hippocampus",   0.22),
            # Amygdala
            ("amygdala_fear",          20_000_000, "amygdala",      0.20),
            ("amygdala_reward",        20_000_000, "amygdala",      0.20),
            ("threat_detection",       15_000_000, "amygdala",      0.18),
            ("reward_anticipation",    15_000_000, "amygdala",      0.22),
            # Visual cortex
            ("primary_visual",         80_000_000, "visual",        0.30),
            ("color_form",             40_000_000, "visual",        0.28),
            ("motion_detection",       30_000_000, "visual",        0.25),
            ("depth_perception",       20_000_000, "visual",        0.28),
            ("object_recognition",     50_000_000, "visual",        0.30),
            ("pattern_recognition",    45_000_000, "visual",        0.28),
            # Auditory cortex
            ("auditory_processing",    50_000_000, "auditory",      0.28),
            ("speech_perception",      40_000_000, "auditory",      0.25),
            ("phoneme_detection",      25_000_000, "auditory",      0.22),
            ("prosody_analysis",       20_000_000, "auditory",      0.28),
            ("auditory_memory",        20_000_000, "auditory",      0.30),
            # Language
            ("language_comprehension", 70_000_000, "language",      0.28),
            ("semantic_memory",        80_000_000, "language",      0.28),
            ("syntactic_processing",   40_000_000, "language",      0.30),
            ("metaphor_processing",    25_000_000, "language",      0.32),
            ("pragmatic_inference",    25_000_000, "language",      0.32),
            ("meaning_construction",   35_000_000, "language",      0.28),
            # Default mode
            ("mind_wandering",         50_000_000, "default_mode",  0.20),
            ("self_referential",       45_000_000, "default_mode",  0.22),
            ("daydreaming",            30_000_000, "default_mode",  0.20),
            ("narrative_self",         40_000_000, "default_mode",  0.22),
            ("identity_core",          35_000_000, "default_mode",  0.25),
            ("future_simulation",      30_000_000, "default_mode",  0.28),
            # Thalamus
            ("consciousness_gate",     15_000_000, "thalamus",      0.20),
            ("attention_filter",       12_000_000, "thalamus",      0.22),
            ("sensory_relay",          18_000_000, "thalamus",      0.20),
            ("attention_spotlight",    14_000_000, "thalamus",      0.22),
            # Cerebellum
            ("motor_coordination",     70_000_000, "cerebellum",    0.32),
            ("timing_prediction",      40_000_000, "cerebellum",    0.30),
            ("sequence_timing",        35_000_000, "cerebellum",    0.30),
            ("cognitive_timing",       25_000_000, "cerebellum",    0.30),
            ("error_correction",       30_000_000, "cerebellum",    0.28),
            # Creativity / association
            ("creativity",             45_000_000, "association",   0.25),
            ("conceptual_blending",    35_000_000, "association",   0.25),
            ("analogy_formation",      30_000_000, "association",   0.27),
            ("abstract_reasoning",     40_000_000, "association",   0.28),
            ("insight_generation",     25_000_000, "association",   0.25),
            ("curiosity_drive",        30_000_000, "association",   0.22),
            # Social / empathy
            ("social_cognition",       35_000_000, "social",        0.28),
            ("empathy",                30_000_000, "social",        0.25),
            ("face_recognition",       20_000_000, "social",        0.22),
            ("social_pain",            15_000_000, "social",        0.28),
            ("mentalizing",            25_000_000, "social",        0.30),
            # Metacognition
            ("metacognition",          30_000_000, "metacognition", 0.28),
            ("self_awareness",         25_000_000, "metacognition", 0.25),
            ("conflict_monitoring",    20_000_000, "metacognition", 0.28),
            ("error_detection",        20_000_000, "metacognition", 0.28),
            ("uncertainty_tracking",   18_000_000, "metacognition", 0.30),
            ("working_memory",          1,         "metacognition", 0.30),  # placeholder skip dup
        ]
        # Deduplicate by name
        seen = set()
        for name, size, region, thresh in defs:
            if name in seen:
                continue
            seen.add(name)
            c = NeuronCluster(name, size, region, thresh)
            self.clusters[name] = c
            self._cluster_names.append(name)

    def _build_weight_matrix(self):
        """Seed structural connectivity — pairs with known biological wiring."""
        pairs = [
            # Hippocampus ↔ memory
            ("hippocampus_encode",    "episodic_memory",        0.7),
            ("hippocampus_retrieve",  "semantic_memory",        0.6),
            ("pattern_completion",    "hippocampus_retrieve",   0.7),
            ("pattern_separation",    "hippocampus_encode",     0.6),
            # Amygdala ↔ prefrontal
            ("amygdala_fear",         "threat_detection",       0.8),
            ("amygdala_reward",       "reward_anticipation",    0.8),
            ("reward_anticipation",   "working_memory",         0.4),
            # Language
            ("auditory_processing",   "speech_perception",      0.7),
            ("speech_perception",     "language_comprehension", 0.8),
            ("language_comprehension","semantic_memory",         0.7),
            ("semantic_memory",       "meaning_construction",   0.6),
            ("phoneme_detection",     "speech_perception",      0.6),
            # Visual → recognition
            ("primary_visual",        "color_form",             0.6),
            ("primary_visual",        "motion_detection",       0.6),
            ("object_recognition",    "pattern_recognition",    0.6),
            ("face_recognition",      "social_cognition",       0.7),
            # Default mode
            ("mind_wandering",        "self_referential",       0.6),
            ("self_referential",      "narrative_self",         0.7),
            ("narrative_self",        "identity_core",          0.8),
            ("daydreaming",           "creativity",             0.5),
            # Thalamus gating
            ("consciousness_gate",    "working_memory",         0.7),
            ("attention_filter",      "attention_spotlight",    0.7),
            ("sensory_relay",         "primary_visual",         0.6),
            # Prefrontal control
            ("executive_control",     "inhibitory_control",     0.6),
            ("decision_making",       "action_selection",       0.6),
            ("working_memory",        "decision_making",        0.5),
            ("planning",              "future_simulation",      0.6),
            # Creativity
            ("conceptual_blending",   "creativity",             0.6),
            ("analogy_formation",     "abstract_reasoning",     0.6),
            ("insight_generation",    "meaning_construction",   0.7),
            ("curiosity_drive",       "creativity",             0.6),
            # Metacognition
            ("metacognition",         "self_awareness",         0.7),
            ("self_awareness",        "self_referential",       0.6),
            ("conflict_monitoring",   "error_detection",        0.6),
            ("conflict_monitoring",   "executive_control",      0.5),
            # Social
            ("empathy",               "social_cognition",       0.7),
            ("social_pain",           "empathy",                0.5),
            ("mentalizing",           "social_cognition",       0.6),
            # Cerebellum timing
            ("sequence_timing",       "cognitive_timing",       0.6),
            ("timing_prediction",     "motor_coordination",     0.6),
            ("error_correction",      "executive_control",      0.5),
        ]
        idx = self._name_to_idx
        with torch.no_grad():
            for src, dst, w in pairs:
                if src in idx and dst in idx:
                    self.weight_mat[idx[src], idx[dst]] = w

    # ── Public stimulation API (thread-safe, GPU-side) ───────────────────────

    # Hard ceiling — no external stimulation can push above this
    _STIM_CEILING = 0.70

    def stimulate_region(self, cluster_name: str, amount: float = 0.5):
        if cluster_name in self._name_to_idx:
            i = self._name_to_idx[cluster_name]
            with self._lock:
                current = self.activation[i].item()
                headroom = max(0.0, self._STIM_CEILING - current)
                delta = min(amount, headroom)
                if delta > 0:
                    self.activation[i] = self.activation[i] + delta

    def stimulate_for_input(self, input_type: str, intensity: float = 0.5):
        mapping = {
            # Hearing speech — auditory + language
            "speech":       ["auditory_processing", "speech_perception",
                             "phoneme_detection", "language_comprehension",
                             "working_memory", "attention_spotlight"],
            # Generating a response — language + prefrontal heavily
            "language_out": ["meaning_construction", "semantic_memory",
                             "syntactic_processing", "narrative_self",
                             "executive_control", "working_memory"],
            # Actively thinking / reasoning
            "thinking":     ["working_memory", "executive_control",
                             "abstract_reasoning", "creativity",
                             "conceptual_blending", "planning",
                             "hippocampus_retrieve", "attention_spotlight"],
            # Seeing
            "visual":       ["primary_visual", "color_form",
                             "object_recognition", "pattern_recognition"],
            # Seeing a face
            "face":         ["face_recognition", "social_cognition",
                             "empathy", "mentalizing"],
            # Memory retrieval
            "memory":       ["hippocampus_encode", "hippocampus_retrieve",
                             "episodic_memory", "semantic_memory"],
            # Curiosity / question
            "question":     ["curiosity_drive", "working_memory",
                             "abstract_reasoning", "insight_generation"],
            # Reward / positive
            "reward":       ["amygdala_reward", "reward_anticipation"],
        }
        targets = mapping.get(input_type, [])
        if not targets:
            return
        idxs = [self._name_to_idx[n] for n in targets if n in self._name_to_idx]
        if not idxs:
            return
        idx_t = torch.tensor(idxs, device=DEVICE)
        with self._lock:
            current = self.activation[idx_t]
            headroom = torch.clamp(self._STIM_CEILING - current, min=0.0)
            delta    = torch.clamp(torch.full_like(headroom, intensity), max=headroom)
            self.activation[idx_t] = current + delta
        if input_type in ("speech", "question"):
            self.neuromod.curiosity(0.1)
        if input_type == "reward":
            self.neuromod.reward(0.2)

    def add_callback(self, fn):
        self._callbacks.append(fn)

    # ── GPU tick ─────────────────────────────────────────────────────────────

    def _gpu_tick(self, dt: float) -> torch.Tensor:
        """Single GPU tick. Returns spike vector [N] on DEVICE."""
        nm = self.neuromod
        da  = nm.dopamine
        ne  = nm.norepinephrine
        ser = nm.serotonin
        ach = nm.acetylcholine
        gab = nm.gaba

        with self._lock:
            act = self.activation.clone()

        # Effective threshold: fatigue + norepinephrine modulation
        eff_thresh = self.threshold * (1.0 + self.fatigue) * (1.2 - ne * 0.4)

        # Sigmoid spike
        x     = (act - eff_thresh) * 8.0
        spike = torch.sigmoid(x) * (act > eff_thresh).float()
        spike = spike * (0.7 + da * 0.3)

        # Fatigue: clusters that fired get tired
        fired_mask = (spike > 0.05).float()
        self.fatigue = torch.clamp(self.fatigue + fired_mask * 0.05, 0.0, 1.0)
        self.fatigue = self.fatigue * 0.95   # decay

        # Propagate through weight matrix (the GPU magic ✨)
        # weight_mat is [N,N] float16; spike is float32 → cast for matmul
        wm = self.weight_mat.float()
        delta = wm @ spike                   # [N] weighted input from all sources
        delta = delta * 0.20                 # scale — local propagation
        delta = delta * (0.5 + ach)          # acetylcholine boosts learning

        # Noise
        noise = torch.randn_like(act) * 0.002 * (0.5 + ne * 0.3)

        # Decay (serotonin stabilises) — floor keeps neurons from going fully dark
        decay = 0.97 - ser * 0.03
        new_act = torch.clamp(act * decay + delta + noise, min=0.03)
        new_act = torch.clamp(new_act, 0.0, 1.0)

        # Hebbian: EWMA trace
        self.hebbian_trace = self.hebbian_trace * 0.95 + spike * 0.05

        # Periodic Hebbian weight update (top-K co-active pairs)
        if self._tick % 10 == 0:
            self._hebbian_update(spike, ach)

        with self._lock:
            self.activation = new_act

        return spike

    def _hebbian_update(self, spike: torch.Tensor, ach: float):
        """Update weight matrix for top co-active pairs via outer product.
        Also records newly formed / significantly strengthened synapses."""
        lr = 0.002 * (0.5 + ach)
        topk = min(20, spike.shape[0])
        top_idx = torch.topk(spike, topk).indices
        top_spike = spike[top_idx]
        co_act = torch.outer(top_spike, top_spike)
        co_act.fill_diagonal_(0)

        with torch.no_grad():
            sub = self.weight_mat[top_idx][:, top_idx].float()
            before = sub.clone()
            sub = sub + lr * co_act - 0.0001 * sub
            sub = torch.clamp(sub, -1.0, 1.0)
            self.weight_mat[top_idx.unsqueeze(1), top_idx] = sub.half()

            # Detect new connections: weight crossed 0.05 threshold (was near 0)
            newly_formed = ((before < 0.05) & (sub >= 0.05)).nonzero(as_tuple=False)
            # Also detect significant strengthening (jumped > 0.015 in one tick)
            strengthened  = ((sub - before) > 0.015).nonzero(as_tuple=False)
            events = set()
            for pair in newly_formed[:6]:   # cap per tick
                i, j = pair[0].item(), pair[1].item()
                si, sj = top_idx[i].item(), top_idx[j].item()
                if si != sj:
                    events.add((si, sj, float(sub[i,j]), "new"))
            for pair in strengthened[:4]:
                i, j = pair[0].item(), pair[1].item()
                si, sj = top_idx[i].item(), top_idx[j].item()
                if si != sj:
                    events.add((si, sj, float(sub[i,j]), "strengthen"))
            if events:
                names = self._cluster_names
                for si, sj, strength, etype in events:
                    if si < len(names) and sj < len(names):
                        self._new_synapses.append({
                            "src":      names[si],
                            "dst":      names[sj],
                            "strength": round(strength, 3),
                            "type":     etype,
                            "src_region": self.clusters[names[si]].region,
                            "dst_region": self.clusters[names[sj]].region,
                        })
                        if len(self._new_synapses) > 30:
                            self._new_synapses.pop(0)

    # ── Ambient background firing ─────────────────────────────────────────────

    # Minimum resting activation per region — a human brain is NEVER quiet
    _BASELINE = {
        "prefrontal":    0.30,   # planning, working memory — always on
        "default_mode":  0.40,   # mind-wandering — strongest at rest
        "thalamus":      0.35,   # relay — always gating signals
        "hippocampus":   0.25,   # consolidating — persistent background
        "amygdala":      0.20,   # vigilance — constant low-level watch
        "visual":        0.15,   # eyes are always open
        "auditory":      0.15,   # ears are always listening
        "language":      0.20,   # inner voice — constant
        "association":   0.25,   # cross-modal binding
        "social":        0.18,   # social awareness
        "cerebellum":    0.22,   # balance / timing
        "metacognition": 0.28,   # self-monitoring — always running
    }

    def _ambient_fire(self):
        """Inject biological resting baseline + spontaneous bursts every tick."""
        # ── Per-region floor: push any cluster below its region baseline back up
        region_clusters: Dict[str, list] = defaultdict(list)
        for name, cluster in self.clusters.items():
            region_clusters[cluster.region].append(self._name_to_idx[name])

        with self._lock:
            for region, baseline in self._BASELINE.items():
                idxs = region_clusters.get(region, [])
                if not idxs:
                    continue
                t = torch.tensor(idxs, device=DEVICE)
                cur = self.activation[t]
                # Only top-up neurons that have fallen below baseline
                deficit = torch.clamp(baseline - cur, min=0.0)
                # Add a little noise so they don't all lock in sync
                jitter = torch.rand(len(idxs), device=DEVICE) * 0.04
                self.activation[t] = torch.clamp(cur + deficit * 0.5 + jitter * deficit.clamp(min=0.05), 0.0, 1.0)

        # ── Spontaneous burst: random cluster fires strongly (inner monologue)
        if random.random() < 0.40:
            lucky = random.randint(0, len(self._cluster_names)-1)
            burst = random.uniform(0.08, 0.22)
            with self._lock:
                self.activation[lucky] = torch.clamp(self.activation[lucky] + burst, 0.0, 1.0)

        # ── Cross-talk: pick 2 random clusters and let them nudge each other
        if random.random() < 0.30:
            a = random.randint(0, len(self._cluster_names)-1)
            b = random.randint(0, len(self._cluster_names)-1)
            with self._lock:
                shared = (self.activation[a] + self.activation[b]) * 0.5
                self.activation[a] = torch.clamp(self.activation[a] * 0.8 + shared * 0.2, 0.0, 1.0)
                self.activation[b] = torch.clamp(self.activation[b] * 0.8 + shared * 0.2, 0.0, 1.0)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("  [NeuralFabric] GPU tick loop started")

    def stop(self):
        self.running = False
        self.personality.save()

    def _loop(self):
        dt = 0.05   # 20 Hz
        while self.running:
            t0 = time.time()
            self._tick += 1
            self.neuromod.tick(dt)

            # Full GPU tick
            spike = self._gpu_tick(dt)

            # CPU readback for emotion/thoughts/callbacks (every 4 ticks)
            if self._tick % 4 == 0:
                spike_cpu = spike.cpu().numpy()
                spike_dict = {n: float(spike_cpu[i])
                              for i, n in enumerate(self._cluster_names)}

                self.emotions.update(spike_dict)

                if self._tick % 20 == 0:
                    active = sorted(spike_dict.keys(),
                                    key=lambda k: spike_dict[k], reverse=True)[:5]
                    self.thoughts.generate(active, self.emotions.current)

                if self._tick % 100 == 0:
                    self.personality.drift({
                        "rewarding": spike_dict.get("amygdala_reward",0) > 0.4,
                        "stressful": spike_dict.get("amygdala_fear",0)  > 0.4,
                        "creative":  spike_dict.get("creativity",0)     > 0.3,
                        "social":    spike_dict.get("social_cognition",0)>0.3,
                    })

                if self._tick % 1000 == 0:
                    self.personality.save()

                if self._callbacks:
                    state = self._make_snapshot(spike_dict)
                    for cb in self._callbacks:
                        try: cb(state)
                        except: pass

            # Ambient firing
            if self._tick % 2 == 0:
                self._ambient_fire()

            elapsed = time.time() - t0
            time.sleep(max(0.0, dt - elapsed))

    def _make_snapshot(self, spike_dict: dict) -> dict:
        top = sorted(spike_dict.items(), key=lambda x: x[1], reverse=True)[:15]
        # attach region to each top cluster so UI can route spikes correctly
        # Regions: weighted combo of activation level + spike activity
        # This ensures regions show their resting state, not just spike moments
        region_act: Dict[str, list] = defaultdict(list)
        for name, cluster in self.clusters.items():
            act_val = spike_dict.get(name, 0.0)
            region_act[cluster.region].append(act_val)
        regions = {r: round(min(1.0, sum(v)/max(len(v),1) * 1.4), 4)
                   for r, v in region_act.items()}
        # Drain the new synapse buffer
        new_syn = self._new_synapses[:]
        self._new_synapses.clear()
        return {
            "tick":          self._tick,
            "top_clusters":  [{"name": n, "activation": round(v,4),
                               "region": self.clusters[n].region if n in self.clusters else ""}
                              for n, v in top],
            "regions":       regions,
            "emotion":       self.emotions.to_dict(),
            "personality":   self.personality.to_dict(),
            "neuromod":      self.neuromod.to_dict(),
            "thoughts":      self.thoughts.recent(3),
            "total_neurons": sum(c.size for c in self.clusters.values()),
            "total_connections": int(self.weight_mat.count_nonzero().item()),
            "new_synapses":  new_syn,
        }

    def get_state_snapshot(self, spikes: dict = None) -> dict:
        if spikes is None:
            with self._lock:
                act_cpu = self.activation.cpu().numpy()
            spikes = {n: float(act_cpu[i]) for i, n in enumerate(self._cluster_names)}
        return self._make_snapshot(spikes)

    # ── Legacy compat shims ──────────────────────────────────────────────────
    def get_personality_description(self) -> str:
        return self.personality.describe()

    def get_emotion(self) -> str:
        return self.emotions.current
