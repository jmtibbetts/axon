"""
AXON — Neural Fabric (CUDA-accelerated)
2.342 billion virtual neurons organized into 64 functional clusters.
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




# ── Conflict Engine ───────────────────────────────────────────────────────────
# Clusters compete via lateral inhibition + softmax gating.
# Winners suppress losers. Dominance history biases future competition.

class ConflictEngine:
    """
    Implements cluster competition:
      - Lateral inhibition: high-activation clusters suppress neighbours
      - Softmax gating with context bias: only the confident survive
      - Dominance memory: past winners are harder to beat (but can tire)
      - Confidence signal: per-cluster track record of "being right"
    """
    def __init__(self, n_clusters: int, cluster_names: list):
        self.n  = n_clusters
        self.names = cluster_names
        # How much each cluster has "won" historically (0-1)
        self.dominance = torch.ones(n_clusters, dtype=torch.float32) * 0.5
        # Per-cluster confidence track record
        self.confidence = torch.ones(n_clusters, dtype=torch.float32) * 0.5
        # Recent activation history for inconsistency penalty
        self._history = torch.zeros((16, n_clusters), dtype=torch.float32)
        self._hist_ptr = 0

    # ── "Use it or lose it" ticker ──────────────────────────────────────────
    _dominance_decay_rate = 0.0003   # per compete() call (20 Hz → ~17% per 10 min)
    _stagnation_counter   = 0        # ticks since last winner set changed
    _last_winner_set: set = set()

    def compete(self, activation: torch.Tensor,
                context_bias: torch.Tensor,
                temperature: float = 1.2) -> torch.Tensor:
        """
        Run one competition step.
        Returns gated activation where winners are amplified, losers suppressed.

        New:
          - Continuous dominance decay ("use it or lose it")
          - Stagnation detector → underdog rescue boost
          - Dominance soft-cap: >0.82 triggers extra decay to prevent calcification
        """
        dev = activation.device
        dom = self.dominance.to(dev)
        conf = self.confidence.to(dev)

        # ── Dominance decay: every cluster bleeds dominance unless it fires ──
        decay = self._dominance_decay_rate
        # Extra bleed for calcified clusters (dominance > 0.82)
        calcified = (self.dominance > 0.82).float()
        self.dominance = torch.clamp(
            self.dominance - decay - calcified * decay * 2.0,
            0.10, 0.90
        )
        dom = self.dominance.to(dev)

        # Weighted score: raw activation * dominance * confidence * context
        score = activation * (0.5 + dom * 0.5) * (0.5 + conf * 0.5) * (0.8 + context_bias * 0.4)

        # Softmax competition
        weights = F.softmax(score / temperature, dim=0)

        # Lateral inhibition: top 20% suppress rest
        top_thresh = torch.quantile(activation, 0.80)
        winners  = (activation >= top_thresh).float()
        losers   = 1.0 - winners

        # ── Stagnation detector: if same clusters keep winning, boost underdogs
        winner_set = set(torch.where(winners > 0)[0].tolist())
        if winner_set == self._last_winner_set:
            self._stagnation_counter += 1
        else:
            self._stagnation_counter = 0
            self._last_winner_set = winner_set

        # After 60 ticks of same winners (~3 sec) — inject underdog pressure
        if self._stagnation_counter > 60:
            underdog_boost = min(0.12, self._stagnation_counter * 0.0008)
            loser_mask = losers
            with torch.no_grad():
                self.dominance = torch.clamp(
                    self.dominance + loser_mask.cpu() * underdog_boost,
                    0.10, 0.90
                )
            # Also randomly spike a non-winner to create genuine competition
            loser_indices = torch.where(loser_mask > 0)[0]
            if len(loser_indices) > 0:
                lucky = loser_indices[torch.randint(len(loser_indices), (1,)).item()]
                activation = activation.clone()
                activation[lucky] = torch.clamp(activation[lucky] + 0.08, 0.0, 1.0)

        gated = activation * (1.0 + winners * 0.15) * (1.0 - losers * 0.08)
        gated = torch.clamp(gated, 0.0, 1.0)

        # Update dominance: winners grow proportional to their activation
        lr_dom = 0.005
        self.dominance = torch.clamp(
            self.dominance + lr_dom * winners.cpu() * activation.detach().cpu()
                           - lr_dom * 0.2 * losers.cpu(),
            0.10, 0.90
        )

        # Record history
        self._history[self._hist_ptr] = activation.detach().cpu()
        self._hist_ptr = (self._hist_ptr + 1) % 16

        return gated

    def penalise_inconsistency(self, activation: torch.Tensor, penalty: float = 0.01):
        """
        Reduce dominance of clusters that flip-flop (high variance in history).
        Stable, consistent clusters are rewarded.
        """
        dev = activation.device
        var = self._history.var(dim=0)   # [N] — how much each cluster bounced
        # High variance → dominance penalty
        self.dominance = torch.clamp(
            self.dominance - penalty * var,
            0.1, 0.9
        )

    def reward_cluster(self, idx: int, magnitude: float = 0.05):
        self.dominance[idx]  = min(0.9, self.dominance[idx]  + magnitude)
        self.confidence[idx] = min(0.9, self.confidence[idx] + magnitude * 0.8)

    def punish_cluster(self, idx: int, magnitude: float = 0.05):
        self.dominance[idx]  = max(0.1, self.dominance[idx]  - magnitude)
        self.confidence[idx] = max(0.1, self.confidence[idx] - magnitude * 0.5)

    def to_dict(self) -> dict:
        top_idx = torch.topk(self.dominance, 5).indices.tolist()
        return {
            "top_dominant": [self.names[i] for i in top_idx],
            "dominance_mean": round(float(self.dominance.mean()), 3),
            "confidence_mean": round(float(self.confidence.mean()), 3),
        }


# ── Prediction Engine (continuous learning loop) ──────────────────────────────
# Tracks expected vs actual cluster activity and fires error signals.

class PredictionEngine:
    """
    Predictive coding loop with structural path tracking:
      - Node-level: prediction error per cluster (as before)
      - Route-level: tracks co-activation pairs and their success history
        so the system can reinforce or weaken entire pathways, not just nodes
    """
    def __init__(self, n_clusters: int):
        self.n = n_clusters
        self.prediction = torch.zeros(n_clusters, dtype=torch.float32)
        self.error      = torch.zeros(n_clusters, dtype=torch.float32)
        self._alpha     = 0.12

        # Route-level tracking: success score per (src, dst) cluster pair
        # Stored as a dense [N, N] float16 matrix on CPU (written to GPU only on update)
        self.route_success = torch.zeros(n_clusters, n_clusters, dtype=torch.float16)
        # Last seen top-K active pairs (for regret / path replay)
        self._last_active_pairs: list = []   # list of (i, j) tuples

    def step(self, actual: torch.Tensor) -> torch.Tensor:
        """Returns error signal [N] and updates internal prediction."""
        dev = actual.device
        pred = self.prediction.to(dev)
        error = actual.detach() - pred              # positive = more than expected
        # Update running prediction
        self.prediction = (pred + self._alpha * error).cpu()
        self.error = error.cpu()
        return error   # on same device as `actual`

    def surprise_score(self) -> float:
        """Scalar: how surprised the system is right now (mean abs error)."""
        return float(self.error.abs().mean())

    def adjust_weights(self, weight_mat: torch.Tensor,
                       error: torch.Tensor, spike: torch.Tensor,
                       lr: float = 0.0003):
        """
        Two-level weight update:
          Node-level:  Δw_ij ∝ spike_i * error_j   (as before)
          Route-level: track (src→dst) co-activation success; boost good routes
        """
        dev = weight_mat.device
        err = error.to(dev)
        sp  = spike.to(dev)
        topk = min(16, sp.shape[0])
        top_idx = torch.topk(sp.abs(), topk).indices
        sp_top  = sp[top_idx]
        err_top = err[top_idx]

        # ── Node-level error update ────────────────────────────────────────────
        delta = torch.outer(sp_top, err_top) * lr
        with torch.no_grad():
            sub = weight_mat[top_idx][:, top_idx].float()
            sub = torch.clamp(sub + delta, -1.0, 1.0)
            weight_mat[top_idx.unsqueeze(1), top_idx] = sub.half()

        # ── Route-level: record active pairs ──────────────────────────────────
        # Track which (src, dst) pairs are co-firing — this IS the path
        pairs = []
        ti = top_idx.tolist()
        for ii, si in enumerate(ti):
            for jj, sj in enumerate(ti):
                if ii != jj and sp_top[ii] > 0.1 and sp_top[jj] > 0.1:
                    pairs.append((si, sj))
        self._last_active_pairs = pairs[:20]

    def reinforce_routes(self, weight_mat: torch.Tensor,
                         pairs: list, signal: float):
        """
        Structural reinforcement: adjust whole route weights by signal.
        signal > 0 → strengthen, signal < 0 → weaken.
        """
        if not pairs or signal == 0:
            return
        dev = weight_mat.device
        lr  = abs(signal) * 0.001
        direction = 1.0 if signal > 0 else -1.0
        src_idx = torch.tensor([p[0] for p in pairs[:12]], dtype=torch.long, device=dev)
        dst_idx = torch.tensor([p[1] for p in pairs[:12]], dtype=torch.long, device=dev)
        with torch.no_grad():
            current = weight_mat[src_idx, dst_idx].float()
            updated = torch.clamp(current + direction * lr, -1.0, 1.0)
            weight_mat[src_idx, dst_idx] = updated.half()
            # Update route success matrix
            rs = self.route_success[src_idx.cpu(), dst_idx.cpu()].float()
            self.route_success[src_idx.cpu(), dst_idx.cpu()] = torch.clamp(
                rs + direction * 0.01, -1.0, 1.0
            ).half()

    def top_routes(self, names: list, n: int = 5) -> list:
        """Return top-N strongest positive routes by route_success score."""
        flat = self.route_success.float().view(-1)
        topk = torch.topk(flat, min(n, flat.shape[0])).indices.tolist()
        N = self.n
        return [(names[i // N], names[i % N], round(float(flat[i]), 3))
                for i in topk if i // N != i % N and flat[i] > 0.0]


# ── Temporal Reward Buffer ─────────────────────────────────────────────────────
# Multi-step reward: accumulates actions, fires reward ONLY after N steps.

class TemporalRewardBuffer:
    """
    Delayed reward system:
      - Accumulates (cluster_activations, context) tuples
      - After `horizon` steps, evaluates the sequence and assigns credit
      - Successful paths reinforce dominance; failed paths are penalised
      - Penalises inconsistency across steps (flip-flopping = bad)
    """
    def __init__(self, horizon: int = 8):
        self.horizon = horizon
        self._buffer: list = []       # list of (activation_snapshot, emotion)
        self._total_reward = 0.0
        self._total_penalty = 0.0

    def push(self, activation: torch.Tensor, emotion: str, valence: float):
        snap = activation.detach().cpu().clone()
        self._buffer.append((snap, emotion, valence))
        if len(self._buffer) > self.horizon * 2:
            self._buffer.pop(0)

    # Ring buffer of past path fingerprints for novelty scoring
    _path_history: list = []
    _NOVELTY_ALPHA = 0.15     # weight of novelty in reward
    _REPETITION_PENALTY = 0.04

    def _path_fingerprint(self, acts: torch.Tensor) -> torch.Tensor:
        """Mean activation vector across the window — compact path signature."""
        return F.normalize(acts.mean(dim=0).unsqueeze(0), dim=1).squeeze(0)

    def _novelty_score(self, fingerprint: torch.Tensor) -> float:
        """
        1.0 = completely novel path never seen before
        0.0 = identical to a recent path
        Uses cosine similarity against last 20 path fingerprints.
        """
        if not self._path_history:
            return 1.0
        sims = [float(torch.dot(fingerprint, p).clamp(-1, 1))
                for p in self._path_history[-20:]]
        max_sim = max(sims)
        return 1.0 - max_sim   # high sim = low novelty

    def evaluate(self, conflict: "ConflictEngine") -> tuple:
        """
        Called every `horizon` ticks.
        Returns (reward, penalty, dominant_clusters).

        New:
          - Novelty bonus: novel activation paths get extra reward
          - Anti-repetition: highly repeated paths get penalised
          - Regret signal: compare chosen path vs best-seen (hindsight)
        """
        if len(self._buffer) < self.horizon:
            return 0.0, 0.0, []

        window = self._buffer[-self.horizon:]
        acts   = torch.stack([w[0] for w in window])    # [H, N]
        vals   = [w[2] for w in window]

        # ── Base reward from valence trajectory ───────────────────────────────
        valence_change = vals[-1] - vals[0]
        valence_mean   = sum(vals) / len(vals)
        reward  = max(0.0, valence_change * 0.5 + max(0.0, valence_mean) * 0.3)
        penalty = max(0.0, -valence_change * 0.3 + max(0.0, -valence_mean) * 0.2)

        # Consistency (moderate range is ideal — not too rigid, not chaotic)
        variance = acts.var(dim=0).mean().item()
        if variance < 0.005:
            reward  += 0.02
        elif variance < 0.02:
            reward  += 0.04   # sweet spot
        elif variance > 0.10:
            penalty += 0.03

        # ── Novelty bonus ─────────────────────────────────────────────────────
        fp = self._path_fingerprint(acts)
        novelty = self._novelty_score(fp)
        reward += self._NOVELTY_ALPHA * novelty

        # Anti-repetition penalty: if novelty is very low, this path is worn out
        if novelty < 0.10:
            penalty += self._REPETITION_PENALTY

        # Store fingerprint
        self._path_history.append(fp.detach().cpu())
        if len(self._path_history) > 50:
            self._path_history.pop(0)

        # ── Regret signal (hindsight) ─────────────────────────────────────────
        # Best possible outcome = max valence seen across buffer
        best_possible_val = max(vals)
        actual_final_val  = vals[-1]
        regret = max(0.0, best_possible_val - actual_final_val)
        # Regret creates extra penalty for leaving reward on the table
        if regret > 0.2:
            penalty += regret * 0.25
            self._last_regret = round(regret, 3)
        else:
            self._last_regret = 0.0

        # ── Credit assignment ─────────────────────────────────────────────────
        mean_act = acts.mean(dim=0)
        topk_idx = torch.topk(mean_act, min(5, len(mean_act))).indices.tolist()
        dominant = [conflict.names[i] for i in topk_idx]

        for i in topk_idx:
            if reward > 0:
                conflict.reward_cluster(i, reward * 0.3)
            if penalty > 0:
                conflict.punish_cluster(i, penalty * 0.3)

        self._total_reward  += reward
        self._total_penalty += penalty
        return reward, penalty, dominant

    def stats(self) -> dict:
        return {
            "total_reward":   round(self._total_reward,  3),
            "total_penalty":  round(self._total_penalty, 3),
            "buffer_len":     len(self._buffer),
            "last_regret":    getattr(self, "_last_regret", 0.0),
            "path_diversity": len(self._path_history),
        }



# ── Cognitive State ──────────────────────────────────────────────────────────
# Global internal variables that persist and shape ALL downstream behavior.
# Unlike emotions (fast, reactive), cognitive state is slow-moving and structural.

class CognitiveState:
    """
    Three persistent state variables:
      confidence  — how sure the system is about its current direction (0-1)
      uncertainty — epistemic: how much the system doesn't know (0-1)
      urgency     — temporal pressure, builds up when goals are unmet (0-1)

    These influence:
      exploration rate (uncertainty ↑ → more exploration)
      competition sharpness (confidence ↑ → sharper softmax)
      reward sensitivity (urgency ↑ → bigger swings on reward/penalty)
    """
    def __init__(self):
        self.confidence  = 0.5
        self.uncertainty = 0.5
        self.urgency     = 0.1
        self._history    = deque(maxlen=100)

    def update(self, surprise: float, reward: float, penalty: float,
               valence: float, novelty_rate: float):
        """
        Update all three state variables from recent signals.
        Called every evaluation cycle.
        """
        # Confidence: rises with reward and consistent predictions,
        #             falls with surprise and penalty
        target_conf = 0.5 + (reward - penalty) * 0.3 - surprise * 0.4
        self.confidence += (max(0.05, min(0.95, target_conf)) - self.confidence) * 0.08

        # Uncertainty: driven by prediction surprise and novelty
        target_unc = surprise * 0.6 + novelty_rate * 0.4
        self.uncertainty += (max(0.05, min(0.95, target_unc)) - self.uncertainty) * 0.06

        # Urgency: builds when valence is negative and no progress is made
        if valence < -0.1:
            self.urgency = min(0.95, self.urgency + 0.02)
        elif valence > 0.2:
            self.urgency = max(0.05, self.urgency - 0.03)
        else:
            self.urgency = max(0.05, self.urgency - 0.005)  # slow bleed

        self._history.append({
            "confidence": round(self.confidence, 3),
            "uncertainty": round(self.uncertainty, 3),
            "urgency": round(self.urgency, 3),
        })

    def explore_boost(self) -> float:
        """How much extra exploration the system wants right now."""
        # High uncertainty or high urgency → more willingness to try new paths
        return self.uncertainty * 0.6 + self.urgency * 0.4

    def competition_temperature(self, base: float = 1.2) -> float:
        """
        High confidence → sharper competition (lower temperature → winner takes more).
        High uncertainty → softer competition (higher temperature → spread bets).
        """
        return base * (1.0 + self.uncertainty * 0.5 - self.confidence * 0.3)

    def reward_sensitivity(self) -> float:
        """
        Urgency amplifies the impact of rewards and penalties.
        A desperate system cares more about every signal.
        """
        return 1.0 + self.urgency * 0.8

    def to_dict(self) -> dict:
        return {
            "confidence":  round(self.confidence,  3),
            "uncertainty": round(self.uncertainty, 3),
            "urgency":     round(self.urgency,     3),
        }

    def describe(self) -> str:
        parts = []
        if self.confidence > 0.7:
            parts.append("confident")
        elif self.confidence < 0.35:
            parts.append("uncertain about direction")
        if self.uncertainty > 0.65:
            parts.append("exploring actively")
        if self.urgency > 0.6:
            parts.append("under pressure")
        elif self.urgency < 0.2:
            parts.append("relaxed")
        return ", ".join(parts) if parts else "stable"



# ── Internal Critic ──────────────────────────────────────────────────────────
# The system evaluates its own recent output BEFORE committing.
# Two subsystems (fast/slow) can disagree — if they do, the slow one wins.

class InternalCritic:
    """
    Self-evaluation loop:
      fast_eval  — quick heuristic based on recent activation pattern
      slow_eval  — deeper check using path history + route success
      disagreement → hesitation (exploration boost, no commitment)
      regret_log  — what paths were abandoned and why (for transparency)

    Called after every `horizon`-step sequence.
    """
    def __init__(self, n_clusters: int, cluster_names: list):
        self.n     = n_clusters
        self.names = cluster_names
        # Simple baseline: expected value of each cluster (EWMA)
        self.expected = torch.zeros(n_clusters, dtype=torch.float32) + 0.1
        self._regret_log: list = []     # recent regrets for UI
        self._hesitation_count = 0

    def fast_eval(self, activation: torch.Tensor) -> float:
        """
        Heuristic score: are active clusters the ones we expect?
        Returns [-1, 1] — positive = on track, negative = unexpected.
        """
        dev = activation.device
        exp = self.expected.to(dev)
        alignment = float(torch.cosine_similarity(
            activation.unsqueeze(0), exp.unsqueeze(0)
        ).clamp(-1, 1))
        return alignment

    def slow_eval(self, acts: torch.Tensor,
                  route_success: torch.Tensor) -> float:
        """
        Structural evaluation: how successful were the routes that fired?
        Returns mean route success score for active pairs.
        """
        dev = acts.device
        topk = min(8, acts.shape[0])
        top_idx = torch.topk(acts, topk).indices
        rs = route_success.float().to(dev)
        scores = []
        ti = top_idx.tolist()
        for i in ti:
            for j in ti:
                if i != j:
                    scores.append(float(rs[i, j]))
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def evaluate(self, activation: torch.Tensor,
                 route_success: torch.Tensor,
                 cog_state: "CognitiveState",
                 valence: float) -> dict:
        """
        Full evaluation cycle.
        Returns dict with: score, hesitate, regret, message.
        """
        fast  = self.fast_eval(activation)
        slow  = self.slow_eval(activation, route_success)

        # Disagreement: fast says good, slow says bad (or vice versa)
        disagreement = abs(fast - slow) > 0.35
        if disagreement:
            self._hesitation_count += 1

        # Combined score
        score = fast * 0.4 + slow * 0.6 + valence * 0.2

        # Regret: if score is low but confidence was high, that's regret
        regret = max(0.0, cog_state.confidence - 0.6) * max(0.0, -score)
        if regret > 0.1:
            active_names = [self.names[i]
                            for i in torch.topk(activation, min(3, self.n)).indices.tolist()]
            entry = {
                "regret":    round(regret, 3),
                "clusters":  active_names,
                "fast_eval": round(fast, 3),
                "slow_eval": round(slow, 3),
            }
            self._regret_log.append(entry)
            if len(self._regret_log) > 20:
                self._regret_log.pop(0)

        # Update expected baseline toward current (slow drift)
        self.expected = self.expected * 0.98 + activation.detach().cpu() * 0.02

        return {
            "score":        round(score, 3),
            "hesitate":     disagreement,
            "regret":       round(regret, 3),
            "fast_eval":    round(fast, 3),
            "slow_eval":    round(slow, 3),
            "hesitations":  self._hesitation_count,
        }

    def recent_regrets(self, n: int = 3) -> list:
        return self._regret_log[-n:]

    def to_dict(self) -> dict:
        last = self._regret_log[-1] if self._regret_log else {}
        return {
            "last_regret":    last.get("regret", 0.0),
            "last_clusters":  last.get("clusters", []),
            "hesitations":    self._hesitation_count,
            "fast_eval":      last.get("fast_eval", 0.0),
            "slow_eval":      last.get("slow_eval", 0.0),
        }

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

        # ── New subsystems ────────────────────────────────────────────────────
        self.conflict     = ConflictEngine(N, self._cluster_names)
        self.predictor    = PredictionEngine(N)
        self.temp_reward  = TemporalRewardBuffer(horizon=10)
        self.cog_state    = CognitiveState()
        self.critic       = InternalCritic(N, self._cluster_names)
        # Context bias vector: memory + personality shape cluster gating
        self.context_bias = torch.zeros(N, dtype=torch.float32)
        # Exploration noise: epsilon now driven by CognitiveState
        self._explore_eps  = 0.18
        self._explore_step = 0

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
    _STIM_CEILING = 0.90  # let stimulation push regions bright

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
        """
        Single GPU tick — now with:
          1. Conflict gating (lateral inhibition + softmax competition)
          2. Prediction error → continuous weight updates
          3. Exploration noise (epsilon-greedy bursts)
          4. Memory context bias shapes cluster competition
        Returns spike vector [N] on DEVICE.
        """
        nm = self.neuromod
        da  = nm.dopamine
        ne  = nm.norepinephrine
        ser = nm.serotonin
        ach = nm.acetylcholine
        gab = nm.gaba

        with self._lock:
            act = self.activation.clone()

        # ── 1. Exploration noise (driven by CognitiveState) ─────────────────
        self._explore_step += 1
        # Base epsilon anneals, but CognitiveState can re-open it
        base_eps = max(0.04, 0.18 * math.exp(-self._explore_step / 10000))
        cog_boost = self.cog_state.explore_boost()
        self._explore_eps = min(0.35, base_eps + cog_boost * 0.15)
        if random.random() < self._explore_eps:
            burst_idx = random.randint(0, act.shape[0] - 1)
            burst_mag = random.gauss(0, 0.08) * (1.0 + self.cog_state.urgency * 0.5)
            act[burst_idx] = torch.clamp(act[burst_idx] + burst_mag, 0.0, 1.0)

        # ── 2. Conflict engine: lateral inhibition + softmax gating ─────────
        ctx_bias = self.context_bias.to(DEVICE)
        # Temperature: NE-driven AND CognitiveState-driven (uncertainty → softer)
        temp = self.cog_state.competition_temperature(base=1.4 - ne * 0.6)
        act = self.conflict.compete(act, ctx_bias, temperature=temp)

        # GABA-mediated global inhibition: when GABA is high, suppress weak clusters
        if gab > 0.55:
            low_mask = (act < 0.15).float()
            act = act * (1.0 - low_mask * (gab - 0.55) * 0.4)

        # ── 3. Standard spike computation ────────────────────────────────────
        eff_thresh = self.threshold * (1.0 + self.fatigue) * (1.2 - ne * 0.4)
        x     = (act - eff_thresh) * 8.0
        spike = torch.sigmoid(x) * (act > eff_thresh).float()
        spike = spike * (0.8 + da * 0.2)

        # Fatigue
        fired_mask = (spike > 0.05).float()
        self.fatigue = torch.clamp(self.fatigue + fired_mask * 0.05, 0.0, 1.0)
        self.fatigue = self.fatigue * 0.95

        # ── 4. Propagate through weight matrix ───────────────────────────────
        wm    = self.weight_mat.float()
        delta = wm @ spike
        delta = delta * 0.12 * (0.4 + ach * 0.6)

        # ── 5. Noise ─────────────────────────────────────────────────────────
        noise = torch.randn_like(act) * 0.002 * (0.5 + ne * 0.3)

        # ── 6. Decay ─────────────────────────────────────────────────────────
        decay   = 0.95 - ser * 0.03
        new_act = torch.clamp(act * decay + delta + noise, min=0.01, max=1.0)

        # ── 7. Prediction error → continuous micro-adaptation ────────────────
        error = self.predictor.step(new_act)
        if self._tick % 5 == 0:
            self.predictor.adjust_weights(self.weight_mat, error, spike, lr=0.0003)

        # ── 8. Hebbian update ─────────────────────────────────────────────────
        self.hebbian_trace = self.hebbian_trace * 0.95 + spike * 0.05
        if self._tick % 10 == 0:
            self._hebbian_update(spike, ach)

        # ── 9. Inconsistency penalty every 32 ticks ──────────────────────────
        if self._tick % 32 == 0:
            self.conflict.penalise_inconsistency(new_act)

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
        "prefrontal":    0.12,   # planning, working memory
        "default_mode":  0.18,   # mind-wandering — strongest at rest
        "thalamus":      0.14,   # relay
        "hippocampus":   0.10,   # memory consolidation
        "amygdala":      0.08,   # vigilance
        "visual":        0.06,   # eyes open
        "auditory":      0.06,   # listening
        "language":      0.09,   # inner voice
        "association":   0.10,   # cross-modal binding
        "social":        0.07,   # social awareness
        "cerebellum":    0.09,   # timing
        "metacognition": 0.12,   # self-monitoring
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
                self.activation[t] = torch.clamp(cur + deficit * 0.25 + jitter * 0.02, 0.0, 1.0)

        # ── Spontaneous burst: random cluster fires strongly (inner monologue)
        if random.random() < 0.15:
            lucky = random.randint(0, len(self._cluster_names)-1)
            burst = random.uniform(0.04, 0.12)
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


    def update_context_from_memory(self, memory_success: dict):
        """
        Feed memory success/failure data into context_bias vector.
        memory_success: {cluster_name: success_rate_0_to_1}
        This biases future softmax competition — clusters that historically
        led to good outcomes get a higher context bias.
        """
        N = len(self._cluster_names)
        new_bias = torch.zeros(N, dtype=torch.float32)
        for name, rate in memory_success.items():
            idx = self._name_to_idx.get(name)
            if idx is not None:
                # Sigmoid-centered: 0.5 success = neutral bias
                new_bias[idx] = (rate - 0.5) * 2.0   # range [-1, 1]
        # Smooth update — don't jump sharply
        self.context_bias = self.context_bias * 0.9 + new_bias * 0.1

    def inject_reward(self, magnitude: float = 0.3, source: str = "external"):
        """Reward signal: boost neuromod dopamine + credit active clusters."""
        self.neuromod.reward(magnitude)
        # Credit the currently most-active clusters
        with self._lock:
            act = self.activation.clone()
        topk = torch.topk(act, min(5, act.shape[0])).indices.tolist()
        for i in topk:
            self.conflict.reward_cluster(i, magnitude * 0.4)

    def inject_penalty(self, magnitude: float = 0.2, source: str = "external"):
        """Penalty signal: stress neuromod + punish active clusters."""
        self.neuromod.stress(magnitude)
        with self._lock:
            act = self.activation.clone()
        topk = torch.topk(act, min(5, act.shape[0])).indices.tolist()
        for i in topk:
            self.conflict.punish_cluster(i, magnitude * 0.3)

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

            # ── Temporal reward evaluation every 10 ticks ─────────────────
            if self._tick % 10 == 0:
                with self._lock:
                    act_snap = self.activation.clone()
                self.temp_reward.push(
                    act_snap,
                    self.emotions.current,
                    self.emotions.valence
                )
            if self._tick % self.temp_reward.horizon == 0 and self._tick > 0:
                r, p, dom = self.temp_reward.evaluate(self.conflict)
                # Structural route reinforcement
                if self.predictor._last_active_pairs:
                    signal = (r - p) * self.cog_state.reward_sensitivity()
                    self.predictor.reinforce_routes(
                        self.weight_mat,
                        self.predictor._last_active_pairs,
                        signal
                    )
                # Update CognitiveState from this evaluation cycle
                stats = self.temp_reward.stats()
                self.cog_state.update(
                    surprise     = self.predictor.surprise_score(),
                    reward       = r,
                    penalty      = p,
                    valence      = self.emotions.valence,
                    novelty_rate = 1.0 - (1.0 / max(1, stats["path_diversity"])),
                )
                # Run InternalCritic
                with self._lock:
                    act_now = self.activation.clone().cpu()
                self.critic.evaluate(
                    act_now,
                    self.predictor.route_success,
                    self.cog_state,
                    self.emotions.valence,
                )

            # CPU readback for emotion/thoughts/callbacks (every 4 ticks)
            if self._tick % 4 == 0:
                spike_cpu = spike.cpu().numpy()
                spike_dict = {n: float(spike_cpu[i])
                              for i, n in enumerate(self._cluster_names)}

                self.emotions.update(spike_dict)

                if self._tick % 200 == 0:   # every ~10s — was every 1s
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
        # ── Use ACTIVATION (not spikes) for region values ─────────────────────
        # spike_dict is thresholded — clusters below threshold = 0 spike even if
        # they have real activation. Read the live activation tensor directly.
        with self._lock:
            act_cpu = self.activation.cpu().numpy()
        act_by_name = {n: float(act_cpu[i]) for i, n in enumerate(self._cluster_names)}
        region_act: Dict[str, list] = defaultdict(list)
        for name, cluster in self.clusters.items():
            region_act[cluster.region].append(act_by_name.get(name, 0.0))
        regions = {r: round(min(1.0, sum(v)/max(len(v),1)), 4)
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
            "conflict":           self.conflict.to_dict(),
            "prediction_surprise": round(self.predictor.surprise_score(), 4),
            "temporal_reward":    self.temp_reward.stats(),
            "explore_eps":        round(self._explore_eps, 4),
            "cognitive_state":    self.cog_state.to_dict(),
            "critic":             self.critic.to_dict(),
            "top_routes":         self.predictor.top_routes(self._cluster_names, 3),
        }

    def get_state_snapshot(self, spikes: dict = None) -> dict:
        if spikes is None:
            with self._lock:
                act_cpu = self.activation.cpu().numpy()
            spikes = {n: float(act_cpu[i]) for i, n in enumerate(self._cluster_names)}
        return self._make_snapshot(spikes)  # _make_snapshot reads activation directly

    # ── Legacy compat shims ──────────────────────────────────────────────────
    def get_personality_description(self) -> str:
        return self.personality.describe()

    def get_emotion(self) -> str:
        return self.emotions.current
