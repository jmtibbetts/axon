"""
AXON — Memory System
Episodic, semantic, and Hebbian pathway memory. SQLite-backed.

Design principles:
  - Episodic: stores USER events only (not Axon's own output — that loops)
  - Semantic: key/value facts about the world and the user, with timestamps
  - Hebbian: co-activation weights between concept pairs (real learning)
  - Forgetting: Ebbinghaus decay — weight decays as w *= e^(-t/tau)
  - Strengthening: repeated activation raises weight, caps at 1.0
  - Context: build_context_string gives LLM ONLY what it needs — no noise
"""

import sqlite3
import time
import math
import json
import re
from pathlib import Path
from typing import Optional, List

DB_PATH   = Path("data/memory/axon.db")
DECAY_TAU = 60 * 60 * 24 * 3   # 3-day half-life
DELTA     = 0.15                 # strength added per co-activation
MAX_IMPORTANCE = 1.0

# Topic words that map to brain region concepts (for real pathway learning)
TOPIC_CONCEPTS = {
    "coding":          ["prefrontal_cortex", "language_processing", "problem_solving"],
    "programming":     ["prefrontal_cortex", "language_processing", "problem_solving"],
    "python":          ["prefrontal_cortex", "language_processing"],
    "ai":              ["prefrontal_cortex", "pattern_recognition", "working_memory"],
    "machine learning":["prefrontal_cortex", "pattern_recognition", "hippocampus"],
    "neural":          ["prefrontal_cortex", "pattern_recognition"],
    "music":           ["auditory_cortex",   "emotional_core", "reward_system"],
    "gaming":          ["reward_system",     "visual_cortex",  "motor_cortex"],
    "fitness":         ["motor_cortex",      "reward_system"],
    "health":          ["emotional_core",    "reward_system"],
    "finance":         ["prefrontal_cortex", "reward_system"],
    "trading":         ["prefrontal_cortex", "reward_system", "amygdala"],
    "family":          ["emotional_core",    "social_cognition"],
    "work":            ["prefrontal_cortex", "working_memory"],
    "art":             ["visual_cortex",     "emotional_core"],
    "travel":          ["hippocampus",       "spatial_navigation"],
    "reading":         ["language_processing","working_memory"],
    "science":         ["prefrontal_cortex", "hippocampus"],
}


class MemorySystem:
    def __init__(self, db_path: Path = DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = __import__('threading').Lock()
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS episodic (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  REAL NOT NULL,
            modality   TEXT NOT NULL,
            content    TEXT NOT NULL,
            emotion    TEXT,
            importance REAL DEFAULT 0.5,
            topics     TEXT DEFAULT '[]'
        );
        CREATE TABLE IF NOT EXISTS semantic (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated    REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS hebbian (
            neuron_a   TEXT NOT NULL,
            neuron_b   TEXT NOT NULL,
            weight     REAL DEFAULT 0.1,
            last_fired REAL NOT NULL,
            fire_count INTEGER DEFAULT 1,
            PRIMARY KEY(neuron_a, neuron_b)
        );
        CREATE TABLE IF NOT EXISTS topics (
            name       TEXT PRIMARY KEY,
            count      INTEGER DEFAULT 1,
            last_seen  REAL NOT NULL,
            first_seen REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ep_time ON episodic(timestamp);
        CREATE INDEX IF NOT EXISTS ep_mod  ON episodic(modality);
        CREATE INDEX IF NOT EXISTS ep_imp  ON episodic(importance);
        """)
        self.conn.commit()
        self._migrate()

    def _migrate(self):
        """Add new columns to existing DBs without breaking old installs."""
        cur = self.conn.execute("PRAGMA table_info(semantic)")
        cols = {row[1] for row in cur.fetchall()}
        if "confidence" not in cols:
            self.conn.execute("ALTER TABLE semantic ADD COLUMN confidence REAL DEFAULT 1.0")
        if "source" not in cols:
            self.conn.execute("ALTER TABLE semantic ADD COLUMN source TEXT DEFAULT 'inferred'")

        cur2 = self.conn.execute("PRAGMA table_info(episodic)")
        ep_cols = {row[1] for row in cur2.fetchall()}
        if "topics" not in ep_cols:
            self.conn.execute("ALTER TABLE episodic ADD COLUMN topics TEXT DEFAULT '[]'")

        self.conn.commit()

    # ── Episodic memory ──────────────────────────────────────────────────────

    def store_episode(self, modality: str, content: dict,
                      emotion: str = None, importance: float = 0.5,
                      topics: list = None) -> int:
        """Store a user event. Call this ONLY for user events — not Axon's output."""
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO episodic(timestamp,modality,content,emotion,importance,topics) VALUES(?,?,?,?,?,?)",
                (time.time(), modality, json.dumps(content), emotion,
                 importance, json.dumps(topics or []))
            )
            self.conn.commit()
        return cur.lastrowid

    def recall_recent(self, n: int = 20, modality: str = None) -> list:
        if modality:
            rows = self.conn.execute(
                "SELECT timestamp,modality,content,emotion,importance,topics FROM episodic "
                "WHERE modality=? ORDER BY timestamp DESC LIMIT ?",
                (modality, n)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT timestamp,modality,content,emotion,importance,topics FROM episodic "
                "ORDER BY timestamp DESC LIMIT ?", (n,)
            ).fetchall()
        return [{"time":r[0],"modality":r[1],"content":json.loads(r[2]),
                 "emotion":r[3],"importance":r[4],
                 "topics":json.loads(r[5] or "[]")} for r in rows]

    def recall_important(self, n: int = 10) -> list:
        """Retrieve highest-importance episodes (emotionally salient moments)."""
        rows = self.conn.execute(
            "SELECT timestamp,modality,content,emotion,importance FROM episodic "
            "ORDER BY importance DESC LIMIT ?", (n,)
        ).fetchall()
        return [{"time":r[0],"modality":r[1],"content":json.loads(r[2]),
                 "emotion":r[3],"importance":r[4]} for r in rows]

    def count_episodes(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM episodic").fetchone()[0]

    # ── Semantic memory ──────────────────────────────────────────────────────

    def learn(self, key: str, value: str, confidence: float = 1.0, source: str = "inferred"):
        """Store or update a semantic fact."""
        with self._lock:
            self.conn.execute(
                "INSERT INTO semantic(key,value,confidence,updated,source) VALUES(?,?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "confidence=excluded.confidence, updated=excluded.updated, source=excluded.source",
                (key, value, confidence, time.time(), source)
            )
            self.conn.commit()

    def recall(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM semantic WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def all_facts(self) -> dict:
        rows = self.conn.execute(
            "SELECT key,value FROM semantic ORDER BY updated DESC"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ── Topic tracking ───────────────────────────────────────────────────────

    def record_topic(self, topic: str):
        """Increment topic frequency. Also fires Hebbian pathways for this topic."""
        now = time.time()
        with self._lock:
            self.conn.execute(
                "INSERT INTO topics(name,count,last_seen,first_seen) VALUES(?,1,?,?) "
                "ON CONFLICT(name) DO UPDATE SET count=count+1, last_seen=excluded.last_seen",
                (topic, now, now)
            )
            self.conn.commit()
        # Fire real Hebbian pathways for this topic
        regions = TOPIC_CONCEPTS.get(topic, [])
        for i, region_a in enumerate(regions):
            for region_b in regions[i+1:]:
                self.coactivate(region_a, region_b)
        # Also connect topic → episodic_memory
        self.coactivate(f"topic_{topic}", "episodic_memory")

    def top_topics(self, n: int = 10) -> list:
        rows = self.conn.execute(
            "SELECT name,count,last_seen FROM topics ORDER BY count DESC LIMIT ?", (n,)
        ).fetchall()
        return [{"topic":r[0],"count":r[1],"last_seen":r[2]} for r in rows]

    def topic_counts(self) -> dict:
        rows = self.conn.execute("SELECT name,count FROM topics").fetchall()
        return {r[0]:r[1] for r in rows}

    # ── Hebbian pathway learning ─────────────────────────────────────────────

    def coactivate(self, neuron_a: str, neuron_b: str) -> dict:
        """Fire together → wire together. Returns {is_new, weight, fires}."""
        if neuron_a == neuron_b:
            return {"is_new": False, "weight": 0, "fires": 0}
        # Canonical ordering so (a,b) and (b,a) are the same link
        if neuron_a > neuron_b:
            neuron_a, neuron_b = neuron_b, neuron_a
        now = time.time()
        with self._lock:
            row = self.conn.execute(
                "SELECT weight,last_fired,fire_count FROM hebbian WHERE neuron_a=? AND neuron_b=?",
                (neuron_a, neuron_b)
            ).fetchone()
            if row:
                w, last, cnt = row
                elapsed   = now - last
                w_decayed = w * math.exp(-elapsed / DECAY_TAU)
                w_new     = min(MAX_IMPORTANCE, w_decayed + DELTA)
                self.conn.execute(
                    "UPDATE hebbian SET weight=?,last_fired=?,fire_count=? WHERE neuron_a=? AND neuron_b=?",
                    (w_new, now, cnt+1, neuron_a, neuron_b)
                )
                self.conn.commit()
                return {"is_new": False, "weight": round(w_new, 4), "fires": cnt+1, "a": neuron_a, "b": neuron_b}
            else:
                self.conn.execute(
                    "INSERT INTO hebbian(neuron_a,neuron_b,weight,last_fired,fire_count) VALUES(?,?,?,?,1)",
                    (neuron_a, neuron_b, DELTA, now)
                )
                self.conn.commit()
                return {"is_new": True, "weight": round(DELTA, 4), "fires": 1, "a": neuron_a, "b": neuron_b}

    def get_weight(self, neuron_a: str, neuron_b: str) -> float:
        if neuron_a > neuron_b:
            neuron_a, neuron_b = neuron_b, neuron_a
        row = self.conn.execute(
            "SELECT weight,last_fired FROM hebbian WHERE neuron_a=? AND neuron_b=?",
            (neuron_a, neuron_b)
        ).fetchone()
        if not row:
            return 0.0
        w, last = row
        return w * math.exp(-(time.time()-last) / DECAY_TAU)

    def top_connections(self, n: int = 20) -> list:
        rows = self.conn.execute(
            "SELECT neuron_a,neuron_b,weight,last_fired,fire_count FROM hebbian "
            "ORDER BY weight DESC LIMIT ?", (n,)
        ).fetchall()
        now = time.time()
        results = []
        for r in rows:
            w = r[2] * math.exp(-(now-r[3]) / DECAY_TAU)
            if w > 0.01:
                results.append({"a":r[0],"b":r[1],"weight":round(w,4),"fires":r[4]})
        return results

    # ── Context builder (LLM prompt injection) ───────────────────────────────

    def build_context_string(self) -> str:
        """
        Build a focused memory context for the LLM.
        Rules:
          1. NO Axon output (modality='language') — causes topic looping
          2. Recent user speech only — what they actually said
          3. Key facts (user profile) — names, job, projects
          4. Top topics — what they care about most
        """
        import datetime
        facts  = self.all_facts()
        recent = self.recall_recent(20)  # fetch more, then filter
        lines  = []

        # ── Facts (skip internal profile blob — too noisy) ────────────────
        skip_keys = {"_user_profile_v1"}
        important_facts = {k:v for k,v in facts.items()
                           if k not in skip_keys
                           and not k.startswith("_")
                           and len(str(v)) < 200}
        if important_facts:
            lines.append("[WHAT I KNOW ABOUT YOU]")
            for k, v in list(important_facts.items())[:8]:
                label = k.replace("user_","").replace("_"," ")
                lines.append(f"  {label}: {v}")

        # ── Top topics ────────────────────────────────────────────────────
        top = self.top_topics(5)
        if top:
            names = [t["topic"] for t in top]
            lines.append(f"[TOPICS YOU CARE ABOUT] {', '.join(names)}")

        # ── Recent USER speech only ───────────────────────────────────────
        user_speech = [ep for ep in recent
                       if ep["modality"] == "auditory"
                       and ep["content"].get("text","").strip()]
        if user_speech:
            lines.append("[RECENT THINGS YOU SAID]")
            for ep in user_speech[:5]:
                ts  = datetime.datetime.fromtimestamp(ep["time"]).strftime("%H:%M")
                txt = ep["content"]["text"].strip()
                em  = f" ({ep['emotion']})" if ep.get("emotion") else ""
                lines.append(f"  {ts}: \"{txt}\"{em}")

        return "\n".join(lines) if lines else ""

    # ── Summary / diagnostics ────────────────────────────────────────────────

    def memory_summary(self) -> dict:
        return {
            "episodes":        self.count_episodes(),
            "facts":           self.all_facts(),
            "top_topics":      self.top_topics(10),
            "top_connections": self.top_connections(10),
        }

    def close(self):
        self.conn.close()
