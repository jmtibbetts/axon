"""
AXON — Hierarchical Memory System
Splits memory into 4 distinct tiers with separate access, decay, and influence rules.

  Episodic    — what happened (events, conversations, observations)
  Semantic    — what it means (concepts, facts, ingested knowledge)
  Value       — what was good/bad (reward-tagged outcomes)
  Identity    — what I am like (self-model conclusions, personality anchors)

Only Identity memory feeds personality drift.
Only Value memory shapes reward expectations.
Episodic feeds reflection (not raw decision-making).
Semantic feeds knowledge retrieval + belief challenge.

Storage: SQLite table 'memory_hierarchy' in axon.db
"""

import time
import json
import threading
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any


TIERS = ("episodic", "semantic", "value", "identity")


class MemoryRecord:
    __slots__ = ("id", "tier", "content", "salience",
                 "valence", "created_at", "last_accessed",
                 "access_count", "decay_rate", "tags")

    def __init__(self, tier: str, content: str,
                 salience: float = 0.5, valence: float = 0.0,
                 tags: List[str] = None, decay_rate: float = 0.001):
        self.id            = None
        self.tier          = tier
        self.content       = content
        self.salience      = float(salience)
        self.valence       = float(valence)
        self.created_at    = time.time()
        self.last_accessed = time.time()
        self.access_count  = 0
        self.decay_rate    = float(decay_rate)
        self.tags          = tags or []

    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "tier":         self.tier,
            "content":      self.content[:200],
            "salience":     round(self.salience, 3),
            "valence":      round(self.valence,  3),
            "created_at":   self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "tags":         self.tags,
        }


class MemoryHierarchy:
    """
    Tiered memory store. Each tier has different defaults:
      Episodic:  salience decays quickly, moderate capacity
      Semantic:  low decay, large capacity
      Value:     moderate decay, feeds reward expectations
      Identity:  very low decay, small capacity, high salience
    """

    TIER_DEFAULTS = {
        "episodic":  {"decay_rate": 0.003, "capacity": 500},
        "semantic":  {"decay_rate": 0.0005,"capacity": 2000},
        "value":     {"decay_rate": 0.002, "capacity": 300},
        "identity":  {"decay_rate": 0.0001,"capacity": 100},
    }

    def __init__(self, db_path: Path):
        self._db   = str(db_path)
        self._lock = threading.Lock()
        self._init_db()

    # ── Schema ─────────────────────────────────────────────────────────────

    def _init_db(self):
        conn = sqlite3.connect(self._db)
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS memory_hierarchy (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            tier          TEXT    NOT NULL,
            content       TEXT    NOT NULL,
            salience      REAL    DEFAULT 0.5,
            valence       REAL    DEFAULT 0.0,
            created_at    REAL    DEFAULT 0,
            last_accessed REAL    DEFAULT 0,
            access_count  INTEGER DEFAULT 0,
            decay_rate    REAL    DEFAULT 0.001,
            tags          TEXT    DEFAULT '[]'
        );
        CREATE INDEX IF NOT EXISTS idx_mh_tier    ON memory_hierarchy(tier);
        CREATE INDEX IF NOT EXISTS idx_mh_sal     ON memory_hierarchy(salience DESC);
        """)
        conn.commit()
        conn.close()

    # ── Write ──────────────────────────────────────────────────────────────

    def store(self, tier: str, content: str,
              salience: float = None, valence: float = 0.0,
              tags: List[str] = None) -> int:
        """Store a memory in the given tier. Returns the row id."""
        assert tier in TIERS, f"Unknown tier: {tier}"
        defaults = self.TIER_DEFAULTS[tier]
        if salience is None:
            salience = 0.7 if tier == "identity" else 0.5
        decay = defaults["decay_rate"]
        now   = time.time()

        # Enforce capacity — prune lowest salience if over limit
        self._maybe_prune(tier, defaults["capacity"])

        conn = sqlite3.connect(self._db)
        cur  = conn.execute(
            "INSERT INTO memory_hierarchy (tier,content,salience,valence,"
            "created_at,last_accessed,decay_rate,tags) VALUES (?,?,?,?,?,?,?,?)",
            (tier, content, salience, valence, now, now, decay,
             json.dumps(tags or []))
        )
        row_id = cur.lastrowid
        conn.commit()
        conn.close()
        return row_id

    def _maybe_prune(self, tier: str, capacity: int):
        conn = sqlite3.connect(self._db)
        count = conn.execute(
            "SELECT COUNT(*) FROM memory_hierarchy WHERE tier=?", (tier,)
        ).fetchone()[0]
        if count >= capacity:
            # Delete bottom 10% by salience
            to_delete = max(1, capacity // 10)
            conn.execute(
                "DELETE FROM memory_hierarchy WHERE id IN "
                "(SELECT id FROM memory_hierarchy WHERE tier=? ORDER BY salience ASC LIMIT ?)",
                (tier, to_delete)
            )
            conn.commit()
        conn.close()

    # ── Read ───────────────────────────────────────────────────────────────

    def recall(self, tier: str, n: int = 10,
               min_salience: float = 0.0,
               tag: str = None) -> List[dict]:
        """Return top-N records from a tier, sorted by salience."""
        query = "SELECT id,tier,content,salience,valence,created_at,last_accessed,access_count,tags FROM memory_hierarchy WHERE tier=?"
        params: list = [tier]
        if min_salience > 0:
            query += " AND salience >= ?"
            params.append(min_salience)
        if tag:
            query += " AND tags LIKE ?"
            params.append(f'%"{tag}"%')
        query += " ORDER BY salience DESC LIMIT ?"
        params.append(n)

        conn = sqlite3.connect(self._db)
        rows = conn.execute(query, params).fetchall()
        conn.close()

        records = []
        for row in rows:
            records.append({
                "id": row[0], "tier": row[1], "content": row[2],
                "salience": row[3], "valence": row[4],
                "created_at": row[5], "last_accessed": row[6],
                "access_count": row[7],
                "tags": json.loads(row[8] or "[]"),
            })
        return records

    def value_summary(self) -> dict:
        """Aggregate value memory into behavioral priors for reward shaping."""
        records = self.recall("value", n=50, min_salience=0.3)
        if not records:
            return {"mean_valence": 0.0, "high_reward_tags": []}
        valences   = [r["valence"] for r in records]
        mean_val   = sum(valences) / len(valences)
        # Extract most common positive tags
        tag_counts: Dict[str, int] = {}
        for r in records:
            if r["valence"] > 0.3:
                for t in r["tags"]:
                    tag_counts[t] = tag_counts.get(t, 0) + 1
        top_tags = sorted(tag_counts, key=tag_counts.get, reverse=True)[:5]
        return {"mean_valence": round(mean_val, 3), "high_reward_tags": top_tags}

    def identity_summary(self) -> List[str]:
        """Return high-salience identity memory contents (for self-model injection)."""
        records = self.recall("identity", n=10, min_salience=0.5)
        return [r["content"] for r in records]

    # ── Decay ──────────────────────────────────────────────────────────────

    def decay_tick(self, tiers: List[str] = None):
        """Apply salience decay across one or all tiers. Call periodically."""
        if tiers is None:
            tiers = list(TIERS)
        conn = sqlite3.connect(self._db)
        for tier in tiers:
            conn.execute(
                "UPDATE memory_hierarchy SET salience = MAX(0.01, salience - decay_rate) WHERE tier=?",
                (tier,)
            )
        conn.commit()
        conn.close()

    # ── Stats ───────────────────────────────────────────────────────────────

    def tier_stats(self) -> Dict[str, dict]:
        conn = sqlite3.connect(self._db)
        out = {}
        for tier in TIERS:
            row = conn.execute(
                "SELECT COUNT(*), AVG(salience), AVG(valence) "
                "FROM memory_hierarchy WHERE tier=?", (tier,)
            ).fetchone()
            out[tier] = {
                "count":       row[0] or 0,
                "avg_salience": round(row[1] or 0.0, 3),
                "avg_valence":  round(row[2] or 0.0, 3),
            }
        conn.close()
        return out
