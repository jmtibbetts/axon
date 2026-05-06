"""
AXON — Face Identity System
Recognises returning faces and builds relationship profiles.

Pipeline:
  1. Detect face via YOLO (optic.py)
  2. Extract 128-d face embedding (face_recognition / dlib)
  3. Compare against known embeddings (cosine similarity, threshold 0.50)
  4. If match  → update last_seen, increment visit count
  5. If no match → assign temp_id, trigger "who are you?" prompt
  6. When user provides a name → bind name to embedding permanently

Storage: SQLite table 'people' in the same axon.db
Relationship profile: JSON blob per person with:
  - name, first_seen, last_seen, visit_count
  - emotion_history (last 20 emotions observed)
  - known_facts (free-form key/value from conversation)
  - notes (running narrative added by Axon)
"""

import json
import time
import threading
import numpy as np
from pathlib import Path
from typing  import Optional
import sqlite3

# ── Embedding backend ─────────────────────────────────────────────────────────
_FR_OK = False
try:
    import face_recognition as _fr
    _FR_OK = True
    print("  [FaceID] face_recognition loaded (dlib embeddings)")
except ImportError:
    print("  [FaceID] face_recognition not available — pip install face_recognition")

SIMILARITY_THRESHOLD = 0.50   # lower = stricter match (cosine distance)
REIDENTIFY_COOLDOWN  = 10.0   # seconds between re-ID attempts for same tracklet
UNKNOWN_PROMPT_DELAY = 3.0    # seconds to wait before asking "who are you?"


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Returns 0 (identical) to 2 (opposite). < 0.5 is a confident match."""
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(1.0 - np.dot(a, b))


class FaceIdentitySystem:
    def __init__(self, db_path: Path, on_new_face=None, on_known_face=None):
        """
        db_path      — path to axon.db (shared with MemorySystem)
        on_new_face  — callback(temp_id, face_img) when unknown face seen
        on_known_face— callback(person_dict) when known face recognised
        """
        self._db_path       = db_path
        self.on_new_face    = on_new_face    # callable
        self.on_known_face  = on_known_face  # callable
        self._lock          = threading.Lock()
        self._embeddings    = {}   # person_id → np.ndarray (128-d)
        self._people        = {}   # person_id → dict

        # Tracklet state — who is currently in frame
        self._current_person_id: Optional[str] = None
        self._last_seen_time:    float          = 0.0
        self._unknown_timer:     Optional[threading.Timer] = None
        self._pending_unknown:   Optional[str]  = None  # temp_id awaiting name

        self._init_db()
        self._load_all()
        print(f"  [FaceID] Loaded {len(self._people)} known people")

    # ── DB setup ──────────────────────────────────────────────────────────────

    def _init_db(self):
        conn = sqlite3.connect(str(self._db_path))
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS people (
            person_id   TEXT PRIMARY KEY,
            name        TEXT,
            embedding   BLOB NOT NULL,
            first_seen  REAL NOT NULL,
            last_seen   REAL NOT NULL,
            visit_count INTEGER DEFAULT 1,
            profile     TEXT DEFAULT '{}'
        );
        """)
        conn.commit()
        conn.close()

    def _load_all(self):
        conn = sqlite3.connect(str(self._db_path))
        rows = conn.execute(
            "SELECT person_id, name, embedding, first_seen, last_seen, visit_count, profile FROM people"
        ).fetchall()
        conn.close()
        for row in rows:
            pid, name, emb_blob, first_seen, last_seen, visits, profile_json = row
            emb = np.frombuffer(emb_blob, dtype=np.float64) if emb_blob else np.array([], dtype=np.float64)
            if emb.shape[0] != 128:
                # __owner__ is intentionally stored without an embedding (text-only profile).
                # Other mismatches get a debug note. Skip the noisy startup warning.
                if pid != "__owner__":
                    print(f"  [FaceID] Skipping person {pid!r} ({name!r}) — bad embedding shape {emb.shape}, will re-learn on next sighting.")
                # Still load the profile so the name is known, just don't add to embeddings
                self._people[pid] = {
                    "person_id":   pid,
                    "name":        name or "Unknown",
                    "first_seen":  first_seen,
                    "last_seen":   last_seen,
                    "visit_count": visits,
                    "profile":     json.loads(profile_json or "{}"),
                }
                continue
            self._embeddings[pid] = emb
            self._people[pid] = {
                "person_id":   pid,
                "name":        name or "Unknown",
                "first_seen":  first_seen,
                "last_seen":   last_seen,
                "visit_count": visits,
                "profile":     json.loads(profile_json or "{}"),
            }

    def _save_person(self, pid: str, embedding: np.ndarray):
        """Upsert person record."""
        p = self._people[pid]
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("""
            INSERT INTO people (person_id, name, embedding, first_seen, last_seen, visit_count, profile)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(person_id) DO UPDATE SET
                name        = excluded.name,
                last_seen   = excluded.last_seen,
                visit_count = excluded.visit_count,
                profile     = excluded.profile
        """, (
            pid,
            p.get("name", "Unknown"),
            embedding.tobytes(),
            p["first_seen"],
            p["last_seen"],
            p["visit_count"],
            json.dumps(p.get("profile", {})),
        ))
        conn.commit()
        conn.close()

    # ── Public API ────────────────────────────────────────────────────────────

    def process_face(self, face_bgr) -> Optional[dict]:
        """
        Called with a cropped face image (BGR numpy array).
        Returns person dict if identified, or {'temp_id': ..., 'unknown': True}
        if this is a new face.
        """
        if not _FR_OK or face_bgr is None or face_bgr.size == 0:
            return None

        import cv2
        # face_recognition needs RGB
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        # Ensure minimum size for dlib
        h, w = face_rgb.shape[:2]
        if h < 40 or w < 40:
            face_rgb = cv2.resize(face_rgb, (96, 96))

        try:
            encodings = _fr.face_encodings(face_rgb)
            if not encodings:
                return None
            emb = np.array(encodings[0], dtype=np.float64)
        except Exception as e:
            print(f"  [FaceID] Encoding error: {e}")
            return None

        with self._lock:
            matched_pid, best_dist = self._find_match(emb)

            if matched_pid:
                return self._update_known(matched_pid, emb)
            else:
                return self._register_unknown(emb)

    def _find_match(self, emb: np.ndarray):
        best_pid  = None
        best_dist = 9.0
        for pid, known_emb in self._embeddings.items():
            if known_emb.shape[0] != 128:
                continue   # corrupted embedding — skip silently
            d = _cosine_dist(emb, known_emb)
            if d < best_dist:
                best_dist = d
                best_pid  = pid
        if best_dist <= SIMILARITY_THRESHOLD:
            return best_pid, best_dist
        return None, best_dist

    def _update_known(self, pid: str, emb: np.ndarray) -> dict:
        p = self._people[pid]
        now = time.time()

        # Update embedding as running average (gradual drift compensation)
        old = self._embeddings[pid]
        self._embeddings[pid] = (old * 0.85 + emb * 0.15)

        # Track visit as new if gap > 5 minutes
        if now - p["last_seen"] > 300:
            p["visit_count"] += 1

        p["last_seen"] = now

        # Decide if we should fire callback (cooldown prevents spam)
        if pid != self._current_person_id or (now - self._last_seen_time) > REIDENTIFY_COOLDOWN:
            self._current_person_id = pid
            self._last_seen_time    = now
            if self.on_known_face:
                threading.Thread(
                    target=self.on_known_face, args=(dict(p),), daemon=True
                ).start()

        self._save_person(pid, self._embeddings[pid])
        return {**p, "matched": True}

    def _register_unknown(self, emb: np.ndarray) -> dict:
        import uuid
        now     = time.time()
        temp_id = "person_" + uuid.uuid4().hex[:8]

        self._embeddings[temp_id] = emb
        self._people[temp_id] = {
            "person_id":   temp_id,
            "name":        "Unknown",
            "first_seen":  now,
            "last_seen":   now,
            "visit_count": 1,
            "profile":     {"emotion_history": [], "known_facts": {}, "notes": ""},
        }
        self._save_person(temp_id, emb)

        # Only fire the callback if we're not already waiting for a name
        if self._pending_unknown is None:
            self._pending_unknown       = temp_id
            self._current_person_id     = temp_id
            self._last_seen_time        = now

            if self.on_new_face:
                # Slight delay — don't ask immediately, let the face stabilise
                timer = threading.Timer(
                    UNKNOWN_PROMPT_DELAY,
                    self.on_new_face,
                    args=(temp_id,),
                )
                timer.daemon = True
                self._unknown_timer = timer
                timer.start()

        return {**self._people[temp_id], "matched": False, "unknown": True}

    def name_person(self, person_id: str, name: str, extra_facts: dict = None):
        """
        Bind a name (and optional extra facts) to a person_id.
        Called when the user tells Axon who someone is.
        """
        with self._lock:
            if person_id not in self._people:
                print(f"  [FaceID] name_person: unknown id {person_id}")
                return
            self._people[person_id]["name"] = name
            if extra_facts:
                self._people[person_id]["profile"].setdefault("known_facts", {}).update(extra_facts)
            self._pending_unknown = None
            self._save_person(person_id, self._embeddings[person_id])
            print(f"  [FaceID] Named {person_id} → '{name}'")

    def update_emotion_for_current(self, emotion: str, conf: float):
        """Record an emotion observation for whoever is currently in frame."""
        pid = self._current_person_id
        if not pid or pid not in self._people:
            return
        hist = self._people[pid]["profile"].setdefault("emotion_history", [])
        hist.append({"emotion": emotion, "conf": round(conf, 3), "t": round(time.time())})
        if len(hist) > 30:
            hist[:] = hist[-30:]
        # Save periodically (every 5 observations)
        if len(hist) % 5 == 0:
            self._save_person(pid, self._embeddings[pid])

    def add_note(self, person_id: str, note: str):
        """Append a note to a person's profile (called by language core)."""
        if person_id not in self._people:
            return
        p = self._people[person_id]
        existing = p["profile"].get("notes", "")
        p["profile"]["notes"] = (existing + "\n" + note).strip()
        self._save_person(person_id, self._embeddings[person_id])

    def get_person(self, person_id: str) -> Optional[dict]:
        return self._people.get(person_id)

    def get_current_person(self) -> Optional[dict]:
        return self._people.get(self._current_person_id)

    def all_people(self) -> list:
        return list(self._people.values())

    def rename_person(self, old_name: str, new_name: str) -> bool:
        """Find by name and rename."""
        with self._lock:
            for pid, p in self._people.items():
                if p["name"].lower() == old_name.lower():
                    p["name"] = new_name
                    self._save_person(pid, self._embeddings[pid])
                    return True
        return False

    def forget_person(self, person_id: str):
        """Remove a person entirely."""
        with self._lock:
            if person_id in self._people:
                del self._people[person_id]
            if person_id in self._embeddings:
                del self._embeddings[person_id]
            conn = sqlite3.connect(str(self._db_path))
            conn.execute("DELETE FROM people WHERE person_id=?", (person_id,))
            conn.commit()
            conn.close()

    def get_summary(self) -> dict:
        return {
            "total_people": len(self._people),
            "named_people": sum(1 for p in self._people.values() if p["name"] != "Unknown"),
            "current_person": self.get_current_person(),
            "all": [
                {k: v for k, v in p.items() if k != "profile"}
                for p in self._people.values()
            ],
        }

    # ── Per-face profile helpers (called by UserModel) ────────────────────────

    def get_person_profile(self, person_id: str) -> Optional[dict]:
        """Return the profile dict for a person_id, or None if not found."""
        if person_id == "__owner__":
            conn = sqlite3.connect(str(self._db_path))
            row = conn.execute(
                "SELECT profile FROM people WHERE person_id=?", (person_id,)
            ).fetchone()
            conn.close()
            if row:
                try:
                    return json.loads(row[0] or "{}")
                except:
                    pass
            return None
        p = self._people.get(person_id)
        return dict(p["profile"]) if p else None

    def save_owner_profile(self, profile: dict):
        """Persist the owner profile record (person_id='__owner__', no embedding)."""
        conn = sqlite3.connect(str(self._db_path))
        now  = time.time()
        conn.execute("""
            INSERT INTO people (person_id, name, embedding, first_seen, last_seen, visit_count, profile)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(person_id) DO UPDATE SET
                profile   = excluded.profile,
                last_seen = excluded.last_seen
        """, (
            "__owner__",
            profile.get("identity", {}).get("full_name") or
            profile.get("identity", {}).get("name", "Owner"),
            b"",   # no embedding for owner
            now, now, 0,
            json.dumps(profile),
        ))
        conn.commit()
        conn.close()

    def update_person_profile(self, person_id: str, profile: dict):
        """Update the profile blob for an existing person."""
        if person_id not in self._people:
            return
        self._people[person_id]["profile"] = profile
        self._save_person(person_id, self._embeddings[person_id])
