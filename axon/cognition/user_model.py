"""
AXON -- User Model
Passively learns who the user is from natural conversation.
Persists to SQLite alongside the memory system.
Continuously refines: job, habits, schedule, preferences, communication style,
interests, relationships, goals, and personal context.
"""

import json
import re
import time
import sqlite3
from pathlib import Path
from typing import Optional

# ── Extraction patterns ──────────────────────────────────────────────────────

PATTERNS = {
    # Identity
    "user_name": [
        r"(?:my name is|i'm called|call me|i am)\s+([A-Z][a-z]{1,20})",
        r"^(?:i'm|im)\s+([A-Z][a-z]{1,20})\b",
    ],
    "user_age": [
        r"i(?:'m| am)\s+(\d{1,2})\s+years? old",
        r"i(?:'m| am)\s+(\d{1,2})\b",
    ],
    # Work
    "user_job_title": [
        r"i(?:'m| am) a(?:n)?\s+([\w\s]{3,40}?)(?:\.|,|\s+at|\s+for|\s+who)",
        r"i work as a(?:n)?\s+([\w\s]{3,30})",
        r"my job is\s+([\w\s]{3,30})",
        r"i(?:'m| am) (?:the|a)\s+([\w\s]{3,30}?)\s+(?:at|for|of)\b",
    ],
    "user_employer": [
        r"i work (?:at|for)\s+([\w\s&,\.]{2,40}?)(?:\.|,|$)",
        r"(?:my company|my employer|my firm) is\s+([\w\s&,\.]{2,40}?)(?:\.|,|$)",
    ],
    "user_industry": [
        r"i(?:'m| am) in (?:the\s+)?([\w\s]{3,30}?) (?:industry|sector|field|space)",
        r"i work in (?:the\s+)?([\w\s]{3,30}?) (?:industry|sector|field|space)",
    ],
    # Location & time
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
    # Preferences
    "user_language_pref": [
        r"i prefer (?:to use|using|speaking)\s+([\w\s]{2,20})",
        r"my preferred (?:language|tongue) is\s+([\w]{2,15})",
    ],
    "user_communication_style": [],  # inferred, not pattern-matched
}

# Topics extracted as lists (additive, not overwrite)
TOPIC_PATTERNS = {
    "user_interests": [
        r"i (?:love|like|enjoy|am into|am passionate about|obsessed with)\s+([\w\s,]{3,50}?)(?:\.|,|$|and\b)",
    ],
    "user_dislikes": [
        r"i (?:hate|dislike|can't stand|don't like)\s+([\w\s,]{3,50}?)(?:\.|,|$)",
    ],
    "user_goals": [
        r"i (?:want to|am trying to|hope to|plan to|need to)\s+([\w\s,]{5,80}?)(?:\.|,|$)",
    ],
    "user_relationships": [
        r"my (?:wife|husband|partner|girlfriend|boyfriend|spouse|kid|son|daughter|dog|cat|pet|friend|boss|coworker)\s+(?:is|named|called)?\s*([\w\s]{1,25}?)(?:\.|,|$)",
    ],
    "user_skills": [
        r"i (?:know how to|can|am good at|specialize in)\s+([\w\s,]{3,50}?)(?:\.|,|$)",
    ],
    "user_projects": [
        r"i(?:'m| am) (?:working on|building|developing|making)\s+([\w\s,]{3,60}?)(?:\.|,|$)",
    ],
}


class UserModel:
    """Learns and stores a rich user profile from natural conversation."""

    PROFILE_KEY = "_user_profile_v1"

    def __init__(self, memory_system):
        self.memory = memory_system
        self._profile = self._load_profile()
        self._turn_count = 0
        self._message_lengths = []

    # ── Persistence ────────────────────────────────────────────

    def _load_profile(self) -> dict:
        raw = self.memory.recall(self.PROFILE_KEY)
        if raw:
            try:
                return json.loads(raw)
            except:
                pass
        return {
            "identity":   {},   # name, age
            "work":       {},   # job, employer, industry
            "location":   {},   # city, timezone, wake/sleep
            "interests":  [],   # hobbies, passions
            "dislikes":   [],
            "goals":      [],
            "skills":     [],
            "projects":   [],
            "relations":  [],   # family, pets, colleagues
            "style":      {},   # communication style observations
            "topics":     {},   # topic frequency map {topic: count}
            "sessions":   0,
            "total_turns":0,
        }

    def _save_profile(self):
        """Save full profile blob + individual semantic keys for context retrieval."""
        self.memory.learn(self.PROFILE_KEY, json.dumps(self._profile), source="profile_blob")
        p = self._profile
        # Write individual facts so build_context_string can surface them cleanly
        ident = p.get("identity", {})
        if ident.get("name"):
            self.memory.learn("user_name", ident["name"], source="user_model")
        if ident.get("age"):
            self.memory.learn("user_age", str(ident["age"]), source="user_model")
        work = p.get("work", {})
        if work.get("job_title"):
            self.memory.learn("user_job", work["job_title"], source="user_model")
        if work.get("employer"):
            self.memory.learn("user_employer", work["employer"], source="user_model")
        loc = p.get("location", {})
        if loc.get("city"):
            self.memory.learn("user_location", loc["city"], source="user_model")
        interests = p.get("interests", [])
        if interests:
            self.memory.learn("user_interests", ", ".join(interests[:6]), source="user_model")
        projects = p.get("projects", [])
        if projects:
            self.memory.learn("user_projects", ", ".join(projects[:4]), source="user_model")
        goals = p.get("goals", [])
        if goals:
            self.memory.learn("user_goals", ", ".join(goals[:3]), source="user_model")

    # ── Extraction ─────────────────────────────────────────────

    def ingest(self, user_text: str):
        """Call this on every user message. Extracts facts silently."""
        self._turn_count += 1
        self._profile["total_turns"] = self._profile.get("total_turns", 0) + 1
        self._message_lengths.append(len(user_text.split()))

        # Run all single-value patterns
        for field, patterns in PATTERNS.items():
            if not patterns:
                continue
            for pat in patterns:
                m = re.search(pat, user_text, re.IGNORECASE)
                if m:
                    val = m.group(1).strip().rstrip(".,!?")
                    if val:
                        section, key = self._route_field(field)
                        self._profile[section][key] = val
                        break

        # Run list-additive topic patterns
        for field, patterns in TOPIC_PATTERNS.items():
            section, key = self._route_list_field(field)
            for pat in patterns:
                for m in re.finditer(pat, user_text, re.IGNORECASE):
                    val = m.group(1).strip().rstrip(".,!? ")
                    if val and len(val) > 2 and val not in self._profile[section]:
                        self._profile[section].append(val)

        # Topic frequency tracking (rough keyword extraction)
        self._track_topics(user_text)

        # Communication style inference
        self._infer_style(user_text)

        # Time-of-day habit tracking
        self._track_activity_time()

        # Save every 3 turns (keep profile fresh)
        if self._turn_count % 3 == 0:
            self._save_profile()

    def _route_field(self, field: str):
        mapping = {
            "user_name":               ("identity", "name"),
            "user_age":                ("identity", "age"),
            "user_job_title":          ("work",     "job_title"),
            "user_employer":           ("work",     "employer"),
            "user_industry":           ("work",     "industry"),
            "user_location":           ("location", "city"),
            "user_timezone":           ("location", "timezone"),
            "user_wake_time":          ("location", "wake_time"),
            "user_sleep_time":         ("location", "sleep_time"),
            "user_language_pref":      ("style",    "language_pref"),
            "user_communication_style":("style",    "communication_style"),
        }
        return mapping.get(field, ("identity", field))

    def _route_list_field(self, field: str):
        mapping = {
            "user_interests":    ("interests",),
            "user_dislikes":     ("dislikes",),
            "user_goals":        ("goals",),
            "user_relationships":("relations",),
            "user_skills":       ("skills",),
            "user_projects":     ("projects",),
        }
        parts = mapping.get(field, (field,))
        return (parts[0], None)  # list fields don't have sub-key

    def _track_topics(self, text: str):
        """Rough topic frequency: track nouns/concepts mentioned often."""
        topics = self._profile.setdefault("topics", {})
        # Common domain words to track
        domains = [
            "coding", "programming", "python", "javascript", "machine learning",
            "ai", "gaming", "music", "finance", "trading", "stocks", "crypto",
            "fitness", "health", "cooking", "travel", "reading", "writing",
            "design", "art", "photography", "video", "sports", "family",
            "work", "business", "startup", "neural", "research", "science",
            "news", "politics", "movies", "anime", "tech", "hardware",
        ]
        low = text.lower()
        for d in domains:
            if d in low:
                topics[d] = topics.get(d, 0) + 1

    def _infer_style(self, text: str):
        """Infer communication style from message patterns."""
        style = self._profile.setdefault("style", {})
        words = text.split()
        lengths = self._message_lengths

        # Message length style
        if len(lengths) >= 10:
            avg = sum(lengths) / len(lengths)
            if avg < 6:
                style["message_length"] = "brief"
            elif avg < 20:
                style["message_length"] = "moderate"
            else:
                style["message_length"] = "detailed"

        # Formality
        casual_markers = ["lol", "lmao", "tbh", "idk", "ngl", "fr", "gonna", "wanna", "kinda", "yeah", "yep", "nah", "btw"]
        formal_markers = ["therefore", "however", "furthermore", "consequently", "regarding", "please", "could you"]
        low = text.lower()
        casual = sum(1 for m in casual_markers if m in low)
        formal = sum(1 for m in formal_markers if m in low)
        if casual > formal:
            style["formality"] = "casual"
        elif formal > casual:
            style["formality"] = "formal"

        # Question vs command ratio
        if "?" in text:
            style["interaction_type"] = style.get("interaction_type", "mixed")

    def _track_activity_time(self):
        """Track what time of day the user is typically active."""
        import datetime
        hour = datetime.datetime.now().hour
        times = self._profile.setdefault("activity_hours", [])
        times.append(hour)
        if len(times) > 100:
            self._profile["activity_hours"] = times[-100:]
        # Derive peak hours
        if len(times) >= 10:
            from collections import Counter
            common = Counter(times).most_common(3)
            self._profile["peak_hours"] = [h for h, _ in common]

    # ── Profile rendering ──────────────────────────────────────

    def describe(self) -> str:
        """Render the user profile as a natural-language context block for the LLM."""
        p = self._profile
        lines = ["[USER PROFILE — what I know about you]"]

        id_ = p.get("identity", {})
        if id_.get("name"):
            lines.append(f"Name: {id_['name']}")
        if id_.get("age"):
            lines.append(f"Age: {id_['age']}")

        work = p.get("work", {})
        if work.get("job_title"):
            job_line = f"Job: {work['job_title']}"
            if work.get("employer"):
                job_line += f" at {work['employer']}"
            if work.get("industry"):
                job_line += f" ({work['industry']} industry)"
            lines.append(job_line)

        loc = p.get("location", {})
        if loc.get("city"):
            lines.append(f"Location: {loc['city']}")
        if loc.get("timezone"):
            lines.append(f"Timezone: {loc['timezone']}")
        if loc.get("wake_time") or loc.get("sleep_time"):
            sched = []
            if loc.get("wake_time"):  sched.append(f"up at {loc['wake_time']}")
            if loc.get("sleep_time"): sched.append(f"sleeps at {loc['sleep_time']}")
            lines.append(f"Schedule: {', '.join(sched)}")

        interests = p.get("interests", [])
        if interests:
            lines.append(f"Interests: {', '.join(interests[:8])}")

        skills = p.get("skills", [])
        if skills:
            lines.append(f"Skills: {', '.join(skills[:6])}")

        projects = p.get("projects", [])
        if projects:
            lines.append(f"Current projects: {', '.join(projects[:4])}")

        goals = p.get("goals", [])
        if goals:
            lines.append(f"Goals: {', '.join(goals[:4])}")

        relations = p.get("relations", [])
        if relations:
            lines.append(f"People/pets: {', '.join(relations[:5])}")

        dislikes = p.get("dislikes", [])
        if dislikes:
            lines.append(f"Dislikes: {', '.join(dislikes[:4])}")

        # Top topics
        topics = p.get("topics", {})
        if topics:
            top = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]
            lines.append(f"Talks about most: {', '.join(t for t,_ in top)}")

        # Style
        style = p.get("style", {})
        style_parts = []
        if style.get("message_length"):
            style_parts.append(f"{style['message_length']} messages")
        if style.get("formality"):
            style_parts.append(f"{style['formality']} tone")
        if style_parts:
            lines.append(f"Communication style: {', '.join(style_parts)}")

        # Peak hours
        peak = p.get("peak_hours", [])
        if peak:
            def fmt_hour(h):
                return f"{'12' if h==12 else h%12}{'am' if h<12 else 'pm'}"
            lines.append(f"Usually active around: {', '.join(fmt_hour(h) for h in peak[:3])}")

        lines.append(f"Total conversations: {p.get('total_turns',0)} turns")

        return "\n".join(lines) if len(lines) > 2 else ""

    def get_profile(self) -> dict:
        return self._profile

    def get_name(self) -> Optional[str]:
        return self._profile.get("identity", {}).get("name")

    def set(self, section: str, key: str, value):
        """Manually set a profile field (e.g. from UI)."""
        if section not in self._profile:
            self._profile[section] = {}
        if isinstance(self._profile[section], dict):
            self._profile[section][key] = value
        self._save_profile()

    def increment_sessions(self):
        self._profile["sessions"] = self._profile.get("sessions", 0) + 1
        self._save_profile()
