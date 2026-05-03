"""
AXON — Memory System
Human-like memory: episodic, semantic, procedural + Hebbian weight store.
All persisted to SQLite so Axon remembers across sessions.

Forgetting curve: Ebbinghaus — weight decays as w *= e^(-t/tau)
Strengthening:    each re-activation adds +delta, capped at 1.0
"""

import sqlite3
import time
import math
import json
from pathlib import Path
from typing import Optional


DB_PATH  = Path("data/memory/axon.db")
DECAY_TAU = 60 * 60 * 24 * 3   # half-life ~3 days (seconds)
DELTA     = 0.15                 # strength added per co-activation


class MemorySystem:
    def __init__(self, db_path: Path = DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        c = self.conn
        c.executescript("""
        CREATE TABLE IF NOT EXISTS episodic (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  REAL NOT NULL,
            modality   TEXT NOT NULL,   -- 'visual','auditory','language','internal'
            content    TEXT NOT NULL,   -- JSON blob
            emotion    TEXT,            -- detected emotion at time of event
            importance REAL DEFAULT 0.5
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
        CREATE INDEX IF NOT EXISTS ep_time ON episodic(timestamp);
        CREATE INDEX IF NOT EXISTS ep_mod  ON episodic(modality);
        """)
        c.commit()

    # ── Episodic ────────────────────────────────────────────────

    def store_episode(self, modality: str, content: dict,
                      emotion: str = None, importance: float = 0.5) -> int:
        cur = self.conn.execute(
            "INSERT INTO episodic(timestamp,modality,content,emotion,importance) VALUES(?,?,?,?,?)",
            (time.time(), modality, json.dumps(content), emotion, importance)
        )
        self.conn.commit()
        return cur.lastrowid

    def recall_recent(self, n: int = 20, modality: str = None) -> list:
        if modality:
            rows = self.conn.execute(
                "SELECT timestamp,modality,content,emotion FROM episodic WHERE modality=? ORDER BY timestamp DESC LIMIT ?",
                (modality, n)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT timestamp,modality,content,emotion FROM episodic ORDER BY timestamp DESC LIMIT ?",
                (n,)
            ).fetchall()
        return [{"time":r[0],"modality":r[1],"content":json.loads(r[2]),"emotion":r[3]} for r in rows]

    def recall_by_emotion(self, emotion: str, n: int = 10) -> list:
        rows = self.conn.execute(
            "SELECT timestamp,modality,content,emotion FROM episodic WHERE emotion=? ORDER BY importance DESC LIMIT ?",
            (emotion, n)
        ).fetchall()
        return [{"time":r[0],"modality":r[1],"content":json.loads(r[2]),"emotion":r[3]} for r in rows]

    def count_episodes(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM episodic").fetchone()[0]

    # ── Semantic ─────────────────────────────────────────────────

    def learn(self, key: str, value: str):
        self.conn.execute(
            "INSERT INTO semantic(key,value,updated) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated=excluded.updated",
            (key, value, time.time())
        )
        self.conn.commit()

    def recall(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM semantic WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def all_facts(self) -> dict:
        rows = self.conn.execute("SELECT key,value FROM semantic").fetchall()
        return {r[0]: r[1] for r in rows}

    # ── Hebbian weights ──────────────────────────────────────────

    def coactivate(self, neuron_a: str, neuron_b: str):
        """Strengthen connection between two neurons that fired together."""
        now = time.time()
        row = self.conn.execute(
            "SELECT weight,last_fired,fire_count FROM hebbian WHERE neuron_a=? AND neuron_b=?",
            (neuron_a, neuron_b)
        ).fetchone()
        if row:
            w, last, cnt = row
            # Apply forgetting since last activation
            elapsed = now - last
            w_decayed = w * math.exp(-elapsed / DECAY_TAU)
            w_new = min(1.0, w_decayed + DELTA)
            self.conn.execute(
                "UPDATE hebbian SET weight=?,last_fired=?,fire_count=? WHERE neuron_a=? AND neuron_b=?",
                (w_new, now, cnt+1, neuron_a, neuron_b)
            )
        else:
            self.conn.execute(
                "INSERT INTO hebbian(neuron_a,neuron_b,weight,last_fired,fire_count) VALUES(?,?,?,?,1)",
                (neuron_a, neuron_b, DELTA, now)
            )
        self.conn.commit()

    def get_weight(self, neuron_a: str, neuron_b: str) -> float:
        row = self.conn.execute(
            "SELECT weight,last_fired FROM hebbian WHERE neuron_a=? AND neuron_b=?",
            (neuron_a, neuron_b)
        ).fetchone()
        if not row:
            return 0.0
        w, last = row
        elapsed = time.time() - last
        return w * math.exp(-elapsed / DECAY_TAU)  # apply decay

    def top_connections(self, n: int = 20) -> list:
        """Return strongest active connections (for visualization)."""
        rows = self.conn.execute(
            "SELECT neuron_a,neuron_b,weight,last_fired,fire_count FROM hebbian ORDER BY weight DESC LIMIT ?",
            (n,)
        ).fetchall()
        now = time.time()
        results = []
        for r in rows:
            elapsed = now - r[3]
            w = r[2] * math.exp(-elapsed / DECAY_TAU)
            if w > 0.01:
                results.append({"a":r[0],"b":r[1],"weight":round(w,4),"fires":r[4]})
        return results

    def memory_summary(self) -> dict:
        facts = self.all_facts()
        recent = self.recall_recent(5)
        top_conn = self.top_connections(10)
        return {
            "episodes": self.count_episodes(),
            "facts": facts,
            "recent": recent,
            "top_connections": top_conn,
        }

    def build_context_string(self) -> str:
        """Build a natural language memory context for the LLM."""
        facts   = self.all_facts()
        recent  = self.recall_recent(8)
        lines   = ["[AXON MEMORY]"]
        if facts:
            lines.append("Known facts:")
            for k,v in list(facts.items())[:12]:
                lines.append(f"  {k}: {v}")
        if recent:
            lines.append("Recent experiences:")
            for ep in recent[:5]:
                import datetime
                ts = datetime.datetime.fromtimestamp(ep['time']).strftime('%H:%M:%S')
                c  = ep['content']
                em = f" [{ep['emotion']}]" if ep['emotion'] else ""
                if ep['modality'] == 'auditory':
                    lines.append(f"  {ts} heard: \"{c.get('text','')}\" {em}")
                elif ep['modality'] == 'visual':
                    lines.append(f"  {ts} saw: {c.get('description','')} {em}")
                elif ep['modality'] == 'language':
                    lines.append(f"  {ts} said: \"{c.get('text','')}\"")
        return "\n".join(lines)

    def close(self):
        self.conn.close()
