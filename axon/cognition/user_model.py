"""
AXON -- Per-Face User Model
Each identified person gets their own persistent profile stored inside
FaceIdentitySystem's 'profile' blob in the people table.

At startup: profiles load from DB automatically via face_identity.py.
Never wiped on reboot — only reset_memory.py can clear them.
"""

import json
import re
import time
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from axon.cognition.face_identity import FaceIdentitySystem

# ── Extraction patterns ──────────────────────────────────────────────────────

PATTERNS = {
    "user_name": [
        r"(?:my name is|i'm called|call me|i am)\s+([A-Z][a-z]{1,20})",
        r"^(?:i'm|im)\s+([A-Z][a-z]{1,20})\b",
    ],
    "user_age": [
        r"i(?:'m| am)\s+(\d{1,2})\s+years? old",
    ],
    "user_job_title": [
        r"i(?:'m| am) a(?:n)?\s+([\w\s]{3,40}?)(?:\.|,|\s+at|\s+for|\s+who)",
        r"i work as a(?:n)?\s+([\w\s]{3,30})",
        r"my job is\s+([\w\s]{3,30})",
    ],
    "user_employer": [
        r"i work (?:at|for)\s+([\w\s&,\.]{2,40}?)(?:\.|,|$)",
        r"(?:my company|my employer|my firm) is\s+([\w\s&,\.]{2,40}?)(?:\.|,|$)",
    ],
    "user_industry": [
        r"i(?:'m| am) in (?:the\s+)?([\w\s]{3,30}?) (?:industry|sector|field|space)",
        r"i work in (?:the\s+)?([\w\s]{3,30}?) (?:industry|sector|field|space)",
    ],
    "user_location": [
        r"i(?:'m| am) (?:in|from|based in)\s+([\w\s,]{3,40}?)(?:\.|,|$)",
        r"i live in\s+([\w\s,]{3,40}?)(?:\.|,|$)",
    ],
    "user_timezone": [
        r"i(?:'m| am) in\s+([\w\s/]{3,30})\s+time",
        r"(?:my timezone|my tz) is\s+([\w/+\-]{3,20})",
    ],
    "user_wake_time": [
        r"i (?:wake up|get up|start) (?:at|around)\s+(\d{1,2}(?::\d{2})?\s*[ap]m?)",
    ],
    "user_sleep_time": [
        r"i (?:go to sleep|sleep|go to bed) (?:at|around)\s+(\d{1,2}(?::\d{2})?\s*[ap]m?)",
    ],
}

TOPIC_PATTERNS = {
    "interests": [
        r"i (?:love|like|enjoy|am into|am passionate about|obsessed with)\s+([\w\s,]{3,50}?)(?:\.|,|$|and\b)",
    ],
    "dislikes": [
        r"i (?:hate|dislike|can't stand|don't like)\s+([\w\s,]{3,50}?)(?:\.|,|$)",
    ],
    "goals": [
        r"i (?:want to|am trying to|hope to|plan to|need to)\s+([\w\s,]{5,80}?)(?:\.|,|$)",
    ],
    "relations": [
        r"my (?:wife|husband|partner|girlfriend|boyfriend|spouse|kid|son|daughter|dog|cat|pet|friend|boss|coworker)\s+(?:is|named|called)?\s*([\w\s]{1,25}?)(?:\.|,|$)",
    ],
    "skills": [
        r"i (?:know how to|can|am good at|specialize in)\s+([\w\s,]{3,50}?)(?:\.|,|$)",
    ],
    "projects": [
        r"i(?:'m| am) (?:working on|building|developing|making)\s+([\w\s,]{3,60}?)(?:\.|,|$)",
    ],
}

DOMAIN_TOPICS = [
    "coding", "programming", "python", "javascript", "machine learning",
    "ai", "gaming", "music", "finance", "trading", "stocks", "crypto",
    "fitness", "health", "cooking", "travel", "reading", "writing",
    "design", "art", "photography", "video", "sports", "family",
    "work", "business", "startup", "neural", "research", "science",
    "news", "politics", "movies", "anime", "tech", "hardware",
]

FIELD_MAP = {
    "user_name":       ("identity", "name"),
    "user_age":        ("identity", "age"),
    "user_job_title":  ("work",     "job_title"),
    "user_employer":   ("work",     "employer"),
    "user_industry":   ("work",     "industry"),
    "user_location":   ("location", "city"),
    "user_timezone":   ("location", "timezone"),
    "user_wake_time":  ("location", "wake_time"),
    "user_sleep_time": ("location", "sleep_time"),
}


def _blank_profile() -> dict:
    return {
        "identity":       {},   # name, age
        "work":           {},   # job_title, employer, industry
        "location":       {},   # city, timezone, wake/sleep
        "interests":      [],
        "dislikes":       [],
        "goals":          [],
        "skills":         [],
        "projects":       [],
        "relations":      [],
        "style":          {},   # communication style
        "topics":         {},   # topic frequency map
        "activity_hours": [],
        "peak_hours":     [],
        "total_turns":    0,
        "sessions":       0,
    }


class UserModel:
    """
    Per-face user profile.  Profile data lives inside face_identity's
    person record (people.profile JSON).  Switching faces swaps the
    active profile without losing anyone's data.

    When no face is recognised (text-only interaction), falls back to
    the 'owner' profile which is seeded once with known owner info and
    never overwritten on reboot.
    """

    OWNER_KEY = "__owner__"   # person_id used when no face is in frame

    def __init__(self, face_id_system: "FaceIdentitySystem"):
        self._fid          = face_id_system
        self._person_id    = self.OWNER_KEY   # start as owner
        self._profile      = self._get_or_create(self.OWNER_KEY)
        self._turn_count   = 0
        self._msg_lengths  = []

    # ── Profile load/save (all storage via FaceIdentitySystem) ───────────────

    def _get_or_create(self, person_id: str) -> dict:
        """Load profile from FaceID DB or create blank."""
        if person_id == self.OWNER_KEY:
            # Owner stored as a special people record with no embedding
            return self._fid.get_person_profile(person_id) or _blank_profile()
        person = self._fid.get_person(person_id)
        if person:
            p = person.get("profile", {})
            # Ensure all keys present (schema migration)
            blank = _blank_profile()
            for k, v in blank.items():
                if k not in p:
                    p[k] = v
            return p
        return _blank_profile()

    def _save(self):
        """Flush current profile back to FaceID storage."""
        if self._person_id == self.OWNER_KEY:
            self._fid.save_owner_profile(self._profile)
        else:
            self._fid.update_person_profile(self._person_id, self._profile)

    # ── Face switching ────────────────────────────────────────────────────────

    def switch_to_person(self, person_id: str):
        """Called by engine when a face is recognised. Swaps active profile."""
        if person_id == self._person_id:
            return
        # Save current before switching
        self._save()
        self._person_id   = person_id
        self._profile     = self._get_or_create(person_id)
        self._turn_count  = 0
        self._msg_lengths = []
        name = self._profile.get("identity", {}).get("name", person_id)
        print(f"  [UserModel] Switched profile → {name} ({person_id})")

    def switch_to_owner(self):
        """Revert to the owner profile (no face in frame)."""
        self.switch_to_person(self.OWNER_KEY)

    # ── Seed (owner only, run once, never overwrites) ────────────────────────

    def seed_owner(self, name: str, full_name: str = None, city: str = None,
                   timezone: str = None, job_title: str = None,
                   projects: list = None, skills: list = None):
        """
        Pre-populate the owner profile with known info.
        ONLY fills blank fields — never overwrites anything learned from conversation.
        Safe to call on every startup.
        """
        if self._person_id != self.OWNER_KEY:
            self.switch_to_owner()

        p = self._profile
        id_ = p.setdefault("identity", {})
        loc = p.setdefault("location", {})
        wrk = p.setdefault("work", {})

        if not id_.get("name"):       id_["name"]       = name
        if not id_.get("full_name") and full_name:
            id_["full_name"] = full_name
        if not loc.get("city") and city:
            loc["city"] = city
        if not loc.get("timezone") and timezone:
            loc["timezone"] = timezone
        if not wrk.get("job_title") and job_title:
            wrk["job_title"] = job_title

        if projects and not p.get("projects"):
            p["projects"] = list(projects)
        if skills and not p.get("skills"):
            p["skills"] = list(skills)

        self._save()
        print(f"  [UserModel] Owner profile seeded: {id_['name']}")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, user_text: str):
        """Call on every user message. Extracts facts silently."""
        self._turn_count  += 1
        self._profile["total_turns"] = self._profile.get("total_turns", 0) + 1
        self._msg_lengths.append(len(user_text.split()))

        # Single-value patterns
        for field, patterns in PATTERNS.items():
            for pat in patterns:
                m = re.search(pat, user_text, re.IGNORECASE)
                if m:
                    val = m.group(1).strip().rstrip(".,!?")
                    if val:
                        section, key = FIELD_MAP.get(field, ("identity", field))
                        self._profile.setdefault(section, {})[key] = val
                    break

        # List patterns (additive)
        for field, patterns in TOPIC_PATTERNS.items():
            lst = self._profile.setdefault(field, [])
            for pat in patterns:
                for m in re.finditer(pat, user_text, re.IGNORECASE):
                    val = m.group(1).strip().rstrip(".,!? ")
                    if val and len(val) > 2 and val not in lst:
                        lst.append(val)

        # Topic frequency
        low = user_text.lower()
        topics = self._profile.setdefault("topics", {})
        for d in DOMAIN_TOPICS:
            if d in low:
                topics[d] = topics.get(d, 0) + 1

        # Communication style inference
        self._infer_style(user_text)

        # Activity-time tracking
        import datetime
        hour = datetime.datetime.now().hour
        hours = self._profile.setdefault("activity_hours", [])
        hours.append(hour)
        if len(hours) > 100:
            self._profile["activity_hours"] = hours[-100:]
        if len(hours) >= 10:
            from collections import Counter
            top = Counter(hours).most_common(3)
            self._profile["peak_hours"] = [h for h, _ in top]

        # Save every 3 turns
        if self._turn_count % 3 == 0:
            self._save()

    def _infer_style(self, text: str):
        style = self._profile.setdefault("style", {})
        lengths = self._msg_lengths
        if len(lengths) >= 10:
            avg = sum(lengths) / len(lengths)
            if avg < 6:
                style["message_length"] = "brief"
            elif avg < 20:
                style["message_length"] = "moderate"
            else:
                style["message_length"] = "detailed"
        casual = ["lol","lmao","tbh","idk","ngl","fr","gonna","wanna","yeah","yep","nah","btw"]
        formal = ["therefore","however","furthermore","consequently","regarding","please","could you"]
        low = text.lower()
        if sum(1 for m in casual if m in low) > sum(1 for m in formal if m in low):
            style["formality"] = "casual"
        elif sum(1 for m in formal if m in low) > sum(1 for m in casual if m in low):
            style["formality"] = "formal"

    # ── Profile rendering for LLM ─────────────────────────────────────────────

    def describe(self) -> str:
        p  = self._profile
        id_ = p.get("identity", {})
        work = p.get("work", {})
        loc  = p.get("location", {})

        name = id_.get("full_name") or id_.get("name")
        if not name and self._person_id == self.OWNER_KEY:
            return ""

        parts = []

        # Who
        intro = f"The person I am talking to is {name}" if name else "I am talking to someone"
        if id_.get("age"):
            intro += f", age {id_['age']}"
        if loc.get("city"):
            intro += f", based in {loc['city']}"
        if loc.get("timezone"):
            intro += f" ({loc['timezone']})"
        intro += "."
        parts.append(intro)

        # Work
        if work.get("job_title"):
            job_line = f"They work as a {work['job_title']}"
            if work.get("employer"):
                job_line += f" at {work['employer']}"
            job_line += "."
            parts.append(job_line)

        # Projects / skills / interests / goals / relations
        for key, label in [
            ("projects",  "Current projects"),
            ("skills",    "Technical skills"),
            ("interests", "Interests"),
            ("goals",     "Goals"),
            ("relations", "Mentioned relationships"),
        ]:
            items = p.get(key, [])
            if items:
                parts.append(f"{label}: {', '.join(items[:5])}.")

        # Top topics
        topics = p.get("topics", {})
        if topics:
            top = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:4]
            parts.append(f"Frequently discusses: {', '.join(t for t, _ in top)}.")

        # Style
        style = p.get("style", {})
        sp = []
        if style.get("message_length"): sp.append(f"{style['message_length']} messages")
        if style.get("formality"):      sp.append(f"{style['formality']} tone")
        if sp:
            parts.append(f"Communication style: {', '.join(sp)}.")

        if not parts:
            return ""

        header = "[WHAT I KNOW ABOUT THIS PERSON — personalise every response using this]"
        return header + "\n" + " ".join(parts)

    def get_profile(self) -> dict:
        return {**self._profile, "_person_id": self._person_id}

    def get_name(self) -> Optional[str]:
        return self._profile.get("identity", {}).get("name")

    def set(self, section: str, key: str, value):
        """Manually set a profile field."""
        self._profile.setdefault(section, {})[key] = value
        self._save()

    def increment_sessions(self):
        self._profile["sessions"] = self._profile.get("sessions", 0) + 1
        self._save()
