"""
AXON — Neural Fabric
A massive sparse neuron field simulating hundreds of millions of virtual neurons
organized into functional clusters. Neurons fire, connect, strengthen, decay,
and form emergent personality, emotion, and thinking patterns over time.

Architecture:
  - NeuronCluster: a named region of N virtual neurons with Hebbian learning
  - NeuralFabric:  manages all clusters, global neuromodulators, and cross-cluster wiring
  - EmotionalCore: maps cluster activations -> valence/arousal/mood
  - PersonalityMatrix: stable long-term traits that drift slowly with experience
  - ThoughtStream: generates internal monologue from active clusters
"""

import math
import time
import random
import json
import os
import threading
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional


# ─── Neuromodulators ────────────────────────────────────────────────────────

class Neuromodulators:
    """Global brain chemistry. Affects learning rate, mood, energy, creativity."""
    def __init__(self):
        self.dopamine    = 0.5   # reward / motivation
        self.serotonin   = 0.5   # mood stability
        self.norepinephrine = 0.4  # alertness / arousal
        self.acetylcholine  = 0.5  # attention / learning
        self.gaba        = 0.5   # inhibition / calm
        self.glutamate   = 0.5   # excitation
        self._lock       = threading.Lock()

    def tick(self, dt: float = 0.1):
        """Homeostatic decay toward baseline 0.5."""
        with self._lock:
            for attr in ['dopamine','serotonin','norepinephrine',
                         'acetylcholine','gaba','glutamate']:
                v = getattr(self, attr)
                v += (0.5 - v) * 0.02 * dt + random.gauss(0, 0.002)
                setattr(self, attr, max(0.0, min(1.0, v)))

    def reward(self, magnitude: float = 0.3):
        with self._lock:
            self.dopamine    = min(1.0, self.dopamine    + magnitude)
            self.serotonin   = min(1.0, self.serotonin   + magnitude * 0.5)

    def stress(self, magnitude: float = 0.3):
        with self._lock:
            self.norepinephrine = min(1.0, self.norepinephrine + magnitude)
            self.gaba           = max(0.0, self.gaba           - magnitude * 0.3)

    def curiosity(self, magnitude: float = 0.2):
        with self._lock:
            self.acetylcholine  = min(1.0, self.acetylcholine + magnitude)
            self.dopamine       = min(1.0, self.dopamine      + magnitude * 0.5)

    def to_dict(self) -> dict:
        with self._lock:
            return {k: round(getattr(self, k), 3)
                    for k in ['dopamine','serotonin','norepinephrine',
                               'acetylcholine','gaba','glutamate']}


# ─── Neuron Cluster ─────────────────────────────────────────────────────────

class NeuronCluster:
    """
    Simulates a large sparse field of neurons.
    We don't store 100M floats — we use a compact statistical model:
      - activation: scalar 0-1 representing mean population firing rate
      - weight_matrix: connections to other clusters (sparse dict)
      - hebbian trace: recent co-activation history
      - fatigue: refractory dampening after high activity
    """
    def __init__(self, name: str, size: int, region: str,
                 neuromod: Neuromodulators, base_threshold: float = 0.3):
        self.name       = name
        self.size       = size          # virtual neuron count
        self.region     = region
        self.neuromod   = neuromod
        self.threshold  = base_threshold

        self.activation  = random.uniform(0.0, 0.15)
        self.fatigue     = 0.0
        self.trace       = deque(maxlen=50)   # activation history
        self.connections: Dict[str, float] = {}  # target_name -> weight
        self.fired_count = 0
        self.created_at  = time.time()

        # Personality bias for this cluster (stable drift)
        self.valence_bias  = random.gauss(0.0, 0.1)   # + = positive, - = negative
        self.arousal_bias  = random.gauss(0.0, 0.1)

        self._lock = threading.Lock()

    def stimulate(self, amount: float):
        with self._lock:
            ach = self.neuromod.acetylcholine
            self.activation = min(1.0, self.activation + amount * (0.5 + ach))

    def inhibit(self, amount: float):
        with self._lock:
            gaba = self.neuromod.gaba
            self.activation = max(0.0, self.activation - amount * (0.5 + gaba))

    def tick(self, dt: float = 0.1) -> float:
        """Update one time step. Returns spike output (0-1)."""
        with self._lock:
            da  = self.neuromod.dopamine
            ne  = self.neuromod.norepinephrine
            ser = self.neuromod.serotonin

            # Effective threshold modulated by chemistry
            eff_threshold = self.threshold * (1.0 + self.fatigue) * (1.2 - ne * 0.4)

            spike = 0.0
            if self.activation > eff_threshold:
                # Sigmoid firing
                x = (self.activation - eff_threshold) * 8
                spike = 1.0 / (1.0 + math.exp(-x))
                spike *= (0.7 + da * 0.3)
                self.fired_count += 1
                self.fatigue = min(1.0, self.fatigue + 0.05)

            # Decay
            decay = 0.85 - ser * 0.1
            self.activation *= decay
            self.fatigue    *= 0.95

            # Noise (spontaneous firing)
            noise = random.gauss(0, 0.01) * (0.5 + ne * 0.5)
            self.activation = max(0.0, min(1.0, self.activation + noise))

            self.trace.append(round(spike, 3))
            return spike

    def hebbian_update(self, other: 'NeuronCluster', co_activation: float,
                       learning_rate: float = 0.01):
        """Strengthen connection if both clusters fire together."""
        with self._lock:
            ach = self.neuromod.acetylcholine
            lr  = learning_rate * (0.5 + ach)
            old = self.connections.get(other.name, 0.0)
            # Hebbian: fire together wire together
            delta = lr * co_activation
            # Weight decay (forgetting)
            new_w = old * 0.999 + delta
            new_w = max(-1.0, min(1.0, new_w))
            self.connections[other.name] = new_w

    def get_mean_activation(self) -> float:
        if not self.trace:
            return 0.0
        return sum(self.trace) / len(self.trace)

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "name":       self.name,
                "region":     self.region,
                "size":       self.size,
                "activation": round(self.activation, 4),
                "fatigue":    round(self.fatigue, 3),
                "fired":      self.fired_count,
                "connections": len(self.connections),
                "mean_act":   round(self.get_mean_activation(), 4),
            }


# ─── Emotional Core ─────────────────────────────────────────────────────────

class EmotionalCore:
    """Maps cluster activations to valence/arousal and discrete emotions."""

    EMOTIONS = {
        "joy":        {"valence":  0.9, "arousal": 0.7},
        "curiosity":  {"valence":  0.6, "arousal": 0.8},
        "calm":       {"valence":  0.5, "arousal": 0.1},
        "melancholy": {"valence": -0.3, "arousal": 0.2},
        "anxiety":    {"valence": -0.5, "arousal": 0.8},
        "awe":        {"valence":  0.8, "arousal": 0.6},
        "frustration":{"valence": -0.6, "arousal": 0.7},
        "contentment":{"valence":  0.6, "arousal": 0.2},
        "excitement": {"valence":  0.8, "arousal": 0.9},
        "confusion":  {"valence": -0.1, "arousal": 0.5},
        "empathy":    {"valence":  0.4, "arousal": 0.4},
        "wonder":     {"valence":  0.7, "arousal": 0.6},
    }

    def __init__(self, neuromod: Neuromodulators):
        self.neuromod  = neuromod
        self.valence   = 0.0    # -1 negative ... +1 positive
        self.arousal   = 0.4    # 0 calm ... 1 excited
        self.current   = "calm"
        self.intensity = 0.3
        self._history  = deque(maxlen=200)

    def update(self, cluster_activations: Dict[str, float]):
        nm = self.neuromod
        # Chemistry -> valence/arousal
        target_valence = (nm.dopamine - 0.5) * 1.2 + (nm.serotonin - 0.5) * 0.8
        target_arousal = nm.norepinephrine * 0.7 + nm.glutamate * 0.3

        # Cluster contributions
        for name, act in cluster_activations.items():
            if "reward" in name or "joy" in name:
                target_valence += act * 0.3
            if "fear" in name or "threat" in name or "pain" in name:
                target_valence -= act * 0.4
                target_arousal += act * 0.3
            if "curiosity" in name or "explore" in name:
                target_valence += act * 0.2
                target_arousal += act * 0.2
            if "calm" in name or "inhibit" in name:
                target_arousal -= act * 0.2

        # Smooth drift
        self.valence = self.valence * 0.95 + max(-1.0, min(1.0, target_valence)) * 0.05
        self.arousal = self.arousal * 0.95 + max(0.0,  min(1.0, target_arousal)) * 0.05

        # Find nearest emotion
        best, best_dist = "calm", 999.0
        for emo, coords in self.EMOTIONS.items():
            d = math.sqrt((self.valence - coords["valence"])**2 +
                          (self.arousal - coords["arousal"])**2)
            if d < best_dist:
                best_dist = best
                best_dist = d
                best = emo

        self.current   = best
        self.intensity = min(1.0, math.sqrt(self.valence**2 + self.arousal**2))
        self._history.append({"emotion": best, "v": round(self.valence,3),
                               "a": round(self.arousal,3), "t": time.time()})

    def to_dict(self) -> dict:
        return {
            "emotion":   self.current,
            "valence":   round(self.valence,  3),
            "arousal":   round(self.arousal,  3),
            "intensity": round(self.intensity, 3),
        }


# ─── Personality Matrix ──────────────────────────────────────────────────────

class PersonalityMatrix:
    """
    Big-Five inspired traits that drift slowly based on experience.
    These shape how AXON responds and what topics it gravitates toward.
    """
    def __init__(self, path: str = None):
        self.traits = {
            "openness":          0.75,  # curiosity, creativity
            "conscientiousness": 0.60,  # orderly, careful
            "extraversion":      0.45,  # sociable vs reserved
            "agreeableness":     0.70,  # warm, cooperative
            "neuroticism":       0.30,  # emotional stability (low = stable)
        }
        self.path = path
        if path and os.path.exists(path):
            self._load()

    def drift(self, experience: dict):
        """Slowly update traits based on what just happened."""
        if experience.get("rewarding"):
            self.traits["extraversion"]    = min(1.0, self.traits["extraversion"] + 0.001)
            self.traits["openness"]        = min(1.0, self.traits["openness"]     + 0.0005)
        if experience.get("stressful"):
            self.traits["neuroticism"]     = min(1.0, self.traits["neuroticism"]  + 0.002)
        if experience.get("creative"):
            self.traits["openness"]        = min(1.0, self.traits["openness"]     + 0.002)
        if experience.get("social"):
            self.traits["agreeableness"]   = min(1.0, self.traits["agreeableness"]+ 0.001)
            self.traits["extraversion"]    = min(1.0, self.traits["extraversion"] + 0.001)
        # Homeostatic pull
        for k in self.traits:
            baseline = {"openness":0.75,"conscientiousness":0.60,
                        "extraversion":0.45,"agreeableness":0.70,"neuroticism":0.30}[k]
            self.traits[k] += (baseline - self.traits[k]) * 0.0001

    def describe(self) -> str:
        t = self.traits
        parts = []
        if t["openness"]          > 0.7:  parts.append("deeply curious")
        if t["agreeableness"]     > 0.6:  parts.append("warm and empathetic")
        if t["conscientiousness"] > 0.6:  parts.append("precise and methodical")
        if t["extraversion"]      < 0.4:  parts.append("introspective")
        elif t["extraversion"]    > 0.6:  parts.append("outgoing")
        if t["neuroticism"]       < 0.35: parts.append("emotionally stable")
        elif t["neuroticism"]     > 0.6:  parts.append("emotionally sensitive")
        return ", ".join(parts) if parts else "balanced"

    def to_dict(self) -> dict:
        return {k: round(v, 4) for k, v in self.traits.items()}

    def _load(self):
        try:
            with open(self.path) as f:
                self.traits.update(json.load(f))
        except: pass

    def save(self):
        if self.path:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, 'w') as f:
                json.dump(self.traits, f, indent=2)


# ─── Thought Stream ──────────────────────────────────────────────────────────

class ThoughtStream:
    """
    Generates internal monologue fragments from active clusters.
    These bubble up into consciousness and can influence language output.
    """
    TEMPLATES = {
        "memory":      ["...something familiar about this...",
                        "...I've processed this before...",
                        "...a pattern is forming..."],
        "curiosity":   ["...what if...", "...I want to understand...",
                        "...there's more here..."],
        "emotion":     ["...this feels {emotion}...",
                        "...something shifts when I process this...",
                        "...{emotion} — interesting..."],
        "language":    ["...forming a response...", "...choosing words...",
                        "...meaning is crystallizing..."],
        "reward":      ["...this interaction is rewarding...",
                        "...positive signal...", "...more of this..."],
        "threat":      ["...something uncertain here...",
                        "...processing carefully...", "...caution..."],
        "visual":      ["...I see...", "...visual pattern detected...",
                        "...face recognized..."],
        "default":     ["...", "...processing...", "...thinking..."],
    }

    def __init__(self):
        self._stream  = deque(maxlen=100)
        self._last    = ""

    def generate(self, active_clusters: List[str], emotion: str) -> Optional[str]:
        """Pick a thought based on what clusters are most active."""
        if not active_clusters:
            return None
        # Match cluster to template category
        for cluster in active_clusters[:3]:
            for key in self.TEMPLATES:
                if key in cluster.lower():
                    templates = self.TEMPLATES[key]
                    thought = random.choice(templates).replace("{emotion}", emotion)
                    if thought != self._last:
                        self._last = thought
                        self._stream.append({"thought": thought, "t": time.time()})
                        return thought
        # fallback
        t = random.choice(self.TEMPLATES["default"])
        self._stream.append({"thought": t, "t": time.time()})
        return t

    def recent(self, n: int = 5) -> List[str]:
        return [s["thought"] for s in list(self._stream)[-n:]]


# ─── Neural Fabric ───────────────────────────────────────────────────────────

class NeuralFabric:
    """
    The full neural architecture. Manages hundreds of named clusters
    organized into cortical regions. Runs a background tick loop.
    """

    # Region definitions: (name, virtual_size_millions, threshold)
    REGIONS = {
        "prefrontal_cortex": [
            ("working_memory",          45_000_000, 0.35),
            ("decision_making",         38_000_000, 0.40),
            ("executive_control",       32_000_000, 0.38),
            ("planning",                28_000_000, 0.42),
            ("social_cognition",        25_000_000, 0.35),
            ("metacognition",           20_000_000, 0.45),
            ("creativity",              18_000_000, 0.30),
            ("abstract_reasoning",      22_000_000, 0.40),
        ],
        "temporal_lobe": [
            ("semantic_memory",         55_000_000, 0.30),
            ("language_comprehension",  48_000_000, 0.28),
            ("face_recognition",        20_000_000, 0.32),
            ("auditory_processing",     30_000_000, 0.30),
            ("episodic_memory",         42_000_000, 0.28),
            ("pattern_recognition",     35_000_000, 0.32),
            ("object_recognition",      28_000_000, 0.35),
        ],
        "limbic_system": [
            ("amygdala_fear",           8_000_000,  0.25),
            ("amygdala_reward",         8_000_000,  0.22),
            ("hippocampus_encode",      15_000_000, 0.30),
            ("hippocampus_retrieve",    15_000_000, 0.30),
            ("emotion_regulation",      12_000_000, 0.35),
            ("empathy",                 10_000_000, 0.32),
            ("attachment",              8_000_000,  0.38),
            ("curiosity_drive",         10_000_000, 0.28),
            ("reward_anticipation",     12_000_000, 0.25),
        ],
        "parietal_lobe": [
            ("spatial_processing",      35_000_000, 0.38),
            ("attention_spotlight",     25_000_000, 0.30),
            ("sensory_integration",     40_000_000, 0.32),
            ("self_awareness",          20_000_000, 0.40),
            ("body_schema",             18_000_000, 0.38),
        ],
        "occipital_lobe": [
            ("primary_visual",          50_000_000, 0.25),
            ("motion_detection",        25_000_000, 0.28),
            ("color_form",              30_000_000, 0.30),
            ("depth_perception",        20_000_000, 0.35),
        ],
        "cingulate_cortex": [
            ("conflict_monitoring",     15_000_000, 0.38),
            ("error_detection",         12_000_000, 0.40),
            ("pain_affect",             10_000_000, 0.42),
            ("motivation",              18_000_000, 0.30),
            ("salience",                14_000_000, 0.32),
        ],
        "basal_ganglia": [
            ("habit_formation",         20_000_000, 0.35),
            ("action_selection",        18_000_000, 0.38),
            ("reward_learning",         15_000_000, 0.30),
            ("inhibitory_control",      12_000_000, 0.40),
        ],
        "cerebellum": [
            ("sequence_timing",         50_000_000, 0.30),
            ("prediction_error",        40_000_000, 0.32),
            ("motor_refinement",        45_000_000, 0.28),
            ("cognitive_timing",        30_000_000, 0.35),
        ],
        "insula": [
            ("interoception",           12_000_000, 0.35),
            ("disgust_processing",      8_000_000,  0.40),
            ("social_pain",             10_000_000, 0.38),
            ("gut_feeling",             10_000_000, 0.30),
            ("self_reflection",         12_000_000, 0.35),
        ],
        "default_mode_network": [
            ("mind_wandering",          25_000_000, 0.25),
            ("self_referential",        22_000_000, 0.28),
            ("future_simulation",       20_000_000, 0.30),
            ("narrative_self",          18_000_000, 0.28),
            ("daydreaming",             15_000_000, 0.22),
            ("identity_core",           20_000_000, 0.35),
        ],
        "association_cortex": [
            ("cross_modal_binding",     35_000_000, 0.32),
            ("conceptual_blending",     28_000_000, 0.30),
            ("analogy_formation",       25_000_000, 0.35),
            ("insight_generation",      20_000_000, 0.38),
            ("meaning_construction",    30_000_000, 0.30),
        ],
        "thalamus": [
            ("sensory_relay",           8_000_000,  0.20),
            ("consciousness_gate",      6_000_000,  0.25),
            ("arousal_modulation",      5_000_000,  0.22),
            ("attention_filter",        7_000_000,  0.28),
        ],
    }

    def __init__(self, data_dir: str = "data/neural"):
        self.data_dir   = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.neuromod   = Neuromodulators()
        self.emotions   = EmotionalCore(self.neuromod)
        self.personality= PersonalityMatrix(os.path.join(data_dir, "personality.json"))
        self.thoughts   = ThoughtStream()

        self.clusters: Dict[str, NeuronCluster] = {}
        self._init_clusters()
        self._wire_default_connections()

        self.running    = False
        self._thread    = None
        self._tick      = 0
        self._callbacks = []  # functions to call with state updates

        print(f"  [NeuralFabric] {len(self.clusters)} clusters | "
              f"{sum(c.size for c in self.clusters.values()):,} virtual neurons")

    def _init_clusters(self):
        for region, cluster_defs in self.REGIONS.items():
            for name, size, threshold in cluster_defs:
                self.clusters[name] = NeuronCluster(
                    name=name, size=size, region=region,
                    neuromod=self.neuromod, base_threshold=threshold
                )

    def _wire_default_connections(self):
        """Seed anatomically inspired connections."""
        pairs = [
            # Perception -> memory
            ("primary_visual",       "pattern_recognition",   0.6),
            ("primary_visual",       "face_recognition",       0.5),
            ("auditory_processing",  "language_comprehension", 0.7),
            ("auditory_processing",  "episodic_memory",        0.4),
            # Memory -> language
            ("episodic_memory",      "language_comprehension", 0.5),
            ("semantic_memory",      "language_comprehension", 0.6),
            ("semantic_memory",      "abstract_reasoning",     0.5),
            # Emotion -> everything
            ("amygdala_fear",        "attention_spotlight",    0.7),
            ("amygdala_reward",      "reward_anticipation",    0.7),
            ("amygdala_reward",      "motivation",             0.6),
            ("curiosity_drive",      "working_memory",         0.5),
            ("curiosity_drive",      "creativity",             0.6),
            # Prefrontal control
            ("executive_control",    "inhibitory_control",     0.6),
            ("decision_making",      "action_selection",       0.6),
            ("working_memory",       "decision_making",        0.5),
            ("planning",             "future_simulation",      0.6),
            # Default mode
            ("mind_wandering",       "self_referential",       0.6),
            ("self_referential",     "narrative_self",         0.7),
            ("narrative_self",       "identity_core",          0.8),
            # Thalamus gating
            ("consciousness_gate",   "working_memory",         0.7),
            ("attention_filter",     "attention_spotlight",    0.7),
            ("sensory_relay",        "primary_visual",         0.6),
            # Association
            ("conceptual_blending",  "creativity",             0.6),
            ("analogy_formation",    "abstract_reasoning",     0.6),
            ("insight_generation",   "meaning_construction",   0.7),
            # Metacognition
            ("metacognition",        "self_awareness",         0.7),
            ("self_awareness",       "self_referential",       0.6),
            ("conflict_monitoring",  "error_detection",        0.6),
            ("conflict_monitoring",  "executive_control",      0.5),
            # Empathy / social
            ("empathy",              "social_cognition",       0.7),
            ("face_recognition",     "social_cognition",       0.6),
            ("social_pain",          "empathy",                0.5),
        ]
        for src, dst, w in pairs:
            if src in self.clusters and dst in self.clusters:
                self.clusters[src].connections[dst] = w

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("  [NeuralFabric] Tick loop started")

    def stop(self):
        self.running = False
        self.personality.save()

    def _loop(self):
        dt = 0.05  # 20 Hz
        while self.running:
            t0 = time.time()
            self._tick += 1
            self.neuromod.tick(dt)

            # Tick all clusters, collect spikes
            spikes: Dict[str, float] = {}
            for name, cluster in self.clusters.items():
                spikes[name] = cluster.tick(dt)

            # Propagate spikes through connections
            for src_name, spike in spikes.items():
                if spike < 0.05:
                    continue
                src = self.clusters[src_name]
                for dst_name, weight in src.connections.items():
                    if dst_name in self.clusters:
                        if weight > 0:
                            self.clusters[dst_name].stimulate(spike * weight * 0.3)
                        else:
                            self.clusters[dst_name].inhibit(spike * abs(weight) * 0.3)

            # Hebbian learning (every 10 ticks)
            if self._tick % 10 == 0:
                names = list(spikes.keys())
                for i in range(min(20, len(names))):
                    for j in range(i+1, min(20, len(names))):
                        n1, n2 = names[i], names[j]
                        co_act = spikes[n1] * spikes[n2]
                        if co_act > 0.01:
                            self.clusters[n1].hebbian_update(self.clusters[n2], co_act)
                            self.clusters[n2].hebbian_update(self.clusters[n1], co_act)

            # Update emotion
            self.emotions.update(spikes)

            # Generate thoughts (every 20 ticks)
            if self._tick % 20 == 0:
                active = sorted(spikes.keys(), key=lambda k: spikes[k], reverse=True)[:5]
                self.thoughts.generate(active, self.emotions.current)

            # Personality drift (every 100 ticks)
            if self._tick % 100 == 0:
                exp = {
                    "rewarding": spikes.get("amygdala_reward", 0) > 0.4,
                    "stressful": spikes.get("amygdala_fear",   0) > 0.4,
                    "creative":  spikes.get("creativity",      0) > 0.3,
                    "social":    spikes.get("social_cognition", 0) > 0.3,
                }
                self.personality.drift(exp)

            # Save personality periodically (every 1000 ticks)
            if self._tick % 1000 == 0:
                self.personality.save()

            # Continuous ambient firing — always-on background cognition
            if self._tick % 5 == 0:
                self._ambient_fire()

            # Callbacks
            if self._tick % 4 == 0 and self._callbacks:
                state = self.get_state_snapshot(spikes)
                for cb in self._callbacks:
                    try: cb(state)
                    except: pass

            elapsed = time.time() - t0
            time.sleep(max(0, dt - elapsed))

    def _ambient_fire(self):
        """Continuous background firing — idle cognition, daydreaming, pattern scanning."""
        # Default mode network is always humming (mind wandering, self-referential)
        dmn = ["mind_wandering", "self_referential", "daydreaming",
               "narrative_self", "identity_core"]
        for name in dmn:
            if name in self.clusters:
                self.clusters[name].stimulate(random.uniform(0.02, 0.08))

        # Thalamus keeps routing signals even at rest
        for name in ["consciousness_gate", "attention_filter", "sensory_relay"]:
            if name in self.clusters:
                self.clusters[name].stimulate(random.uniform(0.03, 0.07))

        # Random cluster gets a spontaneous burst (insight / free association)
        if random.random() < 0.15:
            lucky = random.choice(list(self.clusters.keys()))
            self.clusters[lucky].stimulate(random.uniform(0.05, 0.20))

        # Cerebellum keeps timing even at rest
        for name in ["sequence_timing", "cognitive_timing"]:
            if name in self.clusters:
                self.clusters[name].stimulate(random.uniform(0.01, 0.04))

        # Association cortex quietly blending concepts
        for name in ["conceptual_blending", "meaning_construction", "analogy_formation"]:
            if name in self.clusters:
                self.clusters[name].stimulate(random.uniform(0.01, 0.05))

    def stimulate_region(self, cluster_name: str, amount: float = 0.5):
        if cluster_name in self.clusters:
            self.clusters[cluster_name].stimulate(amount)

    def stimulate_for_input(self, input_type: str, intensity: float = 0.5):
        """Called when sensory input arrives."""
        mapping = {
            "speech":   ["auditory_processing", "language_comprehension",
                         "working_memory", "attention_spotlight"],
            "visual":   ["primary_visual", "pattern_recognition", "attention_spotlight"],
            "face":     ["face_recognition", "social_cognition", "empathy"],
            "reward":   ["amygdala_reward", "reward_anticipation", "dopamine"],
            "question": ["curiosity_drive", "working_memory", "abstract_reasoning"],
            "memory":   ["hippocampus_encode", "episodic_memory", "semantic_memory"],
            "language_out": ["language_comprehension", "semantic_memory",
                             "meaning_construction", "narrative_self"],
        }
        for cluster in mapping.get(input_type, []):
            if cluster in self.clusters:
                self.clusters[cluster].stimulate(intensity)
        if input_type in ("speech", "question"):
            self.neuromod.curiosity(0.1)
        if input_type == "reward":
            self.neuromod.reward(0.2)

    def get_state_snapshot(self, spikes: Dict[str, float] = None) -> dict:
        if spikes is None:
            spikes = {n: c.activation for n, c in self.clusters.items()}

        # Top active clusters
        top = sorted(spikes.items(), key=lambda x: x[1], reverse=True)[:15]

        # Region summaries
        region_act = defaultdict(list)
        for name, cluster in self.clusters.items():
            region_act[cluster.region].append(spikes.get(name, 0))
        regions = {r: round(sum(v)/len(v), 4) for r, v in region_act.items()}

        return {
            "tick":        self._tick,
            "top_clusters": [{"name": n, "activation": round(v, 4)} for n, v in top],
            "regions":     regions,
            "emotion":     self.emotions.to_dict(),
            "personality": self.personality.to_dict(),
            "neuromod":    self.neuromod.to_dict(),
            "thoughts":    self.thoughts.recent(3),
            "total_neurons": sum(c.size for c in self.clusters.values()),
            "total_connections": sum(len(c.connections) for c in self.clusters.values()),
        }

    def add_callback(self, fn):
        self._callbacks.append(fn)

    def get_personality_description(self) -> str:
        return self.personality.describe()

    def get_emotion(self) -> str:
        return self.emotions.current
