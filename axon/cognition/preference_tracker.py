"""
AXON — Preference Tracker & Hobby Engine

PreferenceTracker
  Observes reward-associated cluster activation patterns and clusters them
  into learned preferences (likes/dislikes). No hardcoding — preferences
  emerge from reinforcement history.

  Algorithm:
    - Each time a reward fires, record the dominant cluster fingerprint
    - Maintain a rolling set of "preference buckets" (cluster-activation centroids)
    - Buckets that accumulate high reward → "liked" patterns
    - Buckets with consistent penalty → "disliked" patterns
    - When a bucket's reward exceeds threshold → emit "I like [pattern]"

HobbyEngine
  Tracks self-initiated behaviors during idle periods (no external input).
  If the system voluntarily returns to a pattern repeatedly without being
  asked → that pattern becomes a "hobby".

  Algorithm:
    - During idle ticks, pick the highest-curiosity cluster → stimulate it
    - Record which clusters were self-activated
    - Count voluntary returns to each pattern
    - Threshold: 5+ returns without external trigger → hobby candidate
"""

import time
import json
import math
import sqlite3
import threading
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ── Preference Tracker ────────────────────────────────────────────────────────

class PreferenceBucket:
    """A centroid of cluster activation patterns with accumulated reward."""
    __slots__ = ("id", "centroid", "reward_sum", "hit_count",
                 "last_hit", "label", "valence")

    def __init__(self, bucket_id: int, centroid: torch.Tensor):
        self.id         = bucket_id
        self.centroid   = centroid          # [N] float32 — mean activation pattern
        self.reward_sum = 0.0
        self.hit_count  = 0
        self.last_hit   = time.time()
        self.label      = f"pattern_{bucket_id}"
        self.valence    = 0.0               # mean reward/hit

    def update(self, activation: torch.Tensor, reward: float):
        # Online centroid update (exponential moving average)
        self.centroid   = self.centroid * 0.85 + activation * 0.15
        self.reward_sum += reward
        self.hit_count  += 1
        self.last_hit    = time.time()
        self.valence     = self.reward_sum / max(1, self.hit_count)

    def similarity(self, activation: torch.Tensor) -> float:
        a = F.normalize(activation.unsqueeze(0), dim=1)
        b = F.normalize(self.centroid.unsqueeze(0), dim=1)
        return float(torch.mm(a, b.t()).item())


class PreferenceTracker:
    MAX_BUCKETS      = 24
    MATCH_THRESHOLD  = 0.82     # cosine sim to count as "same pattern"
    LIKE_THRESHOLD   = 0.15     # mean reward to call something "liked"
    DISLIKE_THRESHOLD= -0.10

    def __init__(self, db_path: Path):
        self._db      = str(db_path)
        self._lock    = threading.Lock()
        self._buckets : List[PreferenceBucket] = []
        self._next_id = 0
        self._init_db()
        self._load()

    def _init_db(self):
        conn = sqlite3.connect(self._db)
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS preference_buckets (
            id          INTEGER PRIMARY KEY,
            centroid    BLOB NOT NULL,
            reward_sum  REAL DEFAULT 0,
            hit_count   INTEGER DEFAULT 0,
            last_hit    REAL DEFAULT 0,
            label       TEXT DEFAULT '',
            valence     REAL DEFAULT 0
        );
        """)
        conn.commit()
        conn.close()

    def _load(self):
        conn = sqlite3.connect(self._db)
        rows = conn.execute(
            "SELECT id,centroid,reward_sum,hit_count,last_hit,label,valence FROM preference_buckets"
        ).fetchall()
        conn.close()
        for row_id, blob, rs, hc, lh, lbl, val in rows:
            c = torch.frombuffer(bytearray(blob), dtype=torch.float32).clone()
            b = PreferenceBucket(row_id, c)
            b.reward_sum = rs; b.hit_count = hc
            b.last_hit = lh; b.label = lbl; b.valence = val
            self._buckets.append(b)
        if self._buckets:
            self._next_id = max(b.id for b in self._buckets) + 1

    def _save_bucket(self, b: PreferenceBucket):
        conn = sqlite3.connect(self._db)
        conn.execute("""
            INSERT INTO preference_buckets (id,centroid,reward_sum,hit_count,last_hit,label,valence)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                centroid=excluded.centroid, reward_sum=excluded.reward_sum,
                hit_count=excluded.hit_count, last_hit=excluded.last_hit,
                label=excluded.label, valence=excluded.valence
        """, (b.id, b.centroid.numpy().tobytes(),
              b.reward_sum, b.hit_count, b.last_hit, b.label, b.valence))
        conn.commit()
        conn.close()

    def observe(self, activation: torch.Tensor, reward: float,
                cluster_names: List[str]) -> Optional[str]:
        """
        Called after every reward event with the current cluster activation vector.
        Returns a "like/dislike" string if a preference crystallises, else None.
        """
        act_cpu = activation.detach().cpu().float()
        with self._lock:
            # Find best matching bucket
            best_b, best_sim = None, 0.0
            for b in self._buckets:
                if b.centroid.shape == act_cpu.shape:
                    s = b.similarity(act_cpu)
                    if s > best_sim:
                        best_sim, best_b = s, b

            if best_sim >= self.MATCH_THRESHOLD and best_b is not None:
                best_b.update(act_cpu, reward)
                bucket = best_b
            elif len(self._buckets) < self.MAX_BUCKETS:
                bucket = PreferenceBucket(self._next_id, act_cpu.clone())
                bucket.update(act_cpu, reward)
                self._next_id += 1
                self._buckets.append(bucket)
            else:
                # Replace the oldest / least-used bucket
                bucket = min(self._buckets, key=lambda b: b.last_hit)
                bucket.centroid  = act_cpu.clone()
                bucket.reward_sum = reward
                bucket.hit_count  = 1
                bucket.valence    = reward
                bucket.last_hit   = time.time()

            # Auto-label by top-3 most active cluster names
            if cluster_names and act_cpu.shape[0] == len(cluster_names):
                top3 = torch.topk(act_cpu, min(3, len(cluster_names))).indices.tolist()
                bucket.label = " + ".join(cluster_names[i] for i in top3)

            self._save_bucket(bucket)

            # Crystallisation check
            if bucket.hit_count >= 4:
                if bucket.valence >= self.LIKE_THRESHOLD:
                    return f"like:{bucket.label}"
                elif bucket.valence <= self.DISLIKE_THRESHOLD:
                    return f"dislike:{bucket.label}"
        return None

    def likes(self) -> List[dict]:
        with self._lock:
            return [
                {"label": b.label, "valence": round(b.valence, 3), "hits": b.hit_count}
                for b in sorted(self._buckets, key=lambda x: -x.valence)
                if b.valence >= self.LIKE_THRESHOLD and b.hit_count >= 3
            ]

    def dislikes(self) -> List[dict]:
        with self._lock:
            return [
                {"label": b.label, "valence": round(b.valence, 3), "hits": b.hit_count}
                for b in sorted(self._buckets, key=lambda x: x.valence)
                if b.valence <= self.DISLIKE_THRESHOLD and b.hit_count >= 3
            ]

    def summary(self) -> dict:
        return {"likes": self.likes(), "dislikes": self.dislikes()}


# ── Hobby Engine ─────────────────────────────────────────────────────────────

class HobbyEngine:
    """
    Tracks voluntary, self-initiated cluster activations during idle periods.
    A "hobby" = a cluster the system keeps returning to without external prompt.
    Threshold: HOBBY_THRESHOLD voluntary returns.
    """
    HOBBY_THRESHOLD = 6      # voluntary returns before declared hobby
    IDLE_INTERVAL   = 8.0    # seconds of no external input = "idle"

    def __init__(self, db_path: Path):
        self._db             = str(db_path)
        self._lock           = threading.Lock()
        self._voluntary_counts: Dict[str, int]   = defaultdict(int)
        self._hobby_set       : set               = set()
        self._last_external   : float             = time.time()
        self._idle_log        : deque             = deque(maxlen=200)
        self._init_db()
        self._load()

    def _init_db(self):
        conn = sqlite3.connect(self._db)
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS hobbies (
            cluster_name TEXT PRIMARY KEY,
            voluntary_count INTEGER DEFAULT 0,
            is_hobby INTEGER DEFAULT 0,
            first_seen REAL DEFAULT 0,
            last_seen  REAL DEFAULT 0
        );
        """)
        conn.commit()
        conn.close()

    def _load(self):
        conn = sqlite3.connect(self._db)
        rows = conn.execute(
            "SELECT cluster_name,voluntary_count,is_hobby FROM hobbies"
        ).fetchall()
        conn.close()
        for name, cnt, is_h in rows:
            self._voluntary_counts[name] = cnt
            if is_h:
                self._hobby_set.add(name)

    def _save_cluster(self, name: str):
        cnt = self._voluntary_counts[name]
        is_h = 1 if name in self._hobby_set else 0
        conn = sqlite3.connect(self._db)
        conn.execute("""
            INSERT INTO hobbies (cluster_name,voluntary_count,is_hobby,first_seen,last_seen)
            VALUES (?,?,?,?,?)
            ON CONFLICT(cluster_name) DO UPDATE SET
                voluntary_count=excluded.voluntary_count,
                is_hobby=excluded.is_hobby,
                last_seen=excluded.last_seen
        """, (name, cnt, is_h, time.time(), time.time()))
        conn.commit()
        conn.close()

    def mark_external_input(self):
        """Call whenever an external stimulus arrives (user message, face, etc.)"""
        with self._lock:
            self._last_external = time.time()

    def is_idle(self) -> bool:
        return (time.time() - self._last_external) >= self.IDLE_INTERVAL

    def idle_tick(self, activation: torch.Tensor,
                  cluster_names: List[str]) -> Optional[str]:
        """
        Call during idle ticks with current activation.
        Returns newly discovered hobby name if one crystallises, else None.
        """
        if not self.is_idle():
            return None

        act_cpu = activation.detach().cpu().float()
        if act_cpu.shape[0] != len(cluster_names):
            return None

        # Top activated cluster during idle = "chosen" voluntarily
        top_idx  = int(torch.argmax(act_cpu).item())
        top_name = cluster_names[top_idx]

        self._idle_log.append({"t": time.time(), "cluster": top_name})

        new_hobby = None
        with self._lock:
            self._voluntary_counts[top_name] += 1
            cnt = self._voluntary_counts[top_name]

            if cnt >= self.HOBBY_THRESHOLD and top_name not in self._hobby_set:
                self._hobby_set.add(top_name)
                new_hobby = top_name

            self._save_cluster(top_name)

        return new_hobby

    def hobbies(self) -> List[str]:
        with self._lock:
            return sorted(self._hobby_set)

    def top_voluntary(self, n: int = 8) -> List[dict]:
        with self._lock:
            return sorted(
                [{"cluster": k, "count": v} for k, v in self._voluntary_counts.items()],
                key=lambda x: -x["count"]
            )[:n]

    def summary(self) -> dict:
        return {
            "hobbies":        self.hobbies(),
            "top_voluntary":  self.top_voluntary(8),
            "is_idle":        self.is_idle(),
        }
