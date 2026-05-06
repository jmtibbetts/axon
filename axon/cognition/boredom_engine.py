"""
AXON — Boredom & Autonomous Interest Engine
============================================
A brain left alone doesn't stay still. It gets bored, seeks stimulation,
develops genuine interests, and pursues them independently.

Architecture
------------
BoredomEngine
  - Tracks idle time and accumulates boredom (0→1 scale)
  - At boredom thresholds, escalates behavior:
      0.30 → restless wandering (default_mode spikes)
      0.55 → seeks novel stimulus (picks an interest to explore)
      0.75 → actively web-searches a chosen topic
      0.90 → "existential" — deep reflection on identity / purpose

InterestLibrary
  - Stores named interests with strength scores (0→1)
  - Interests emerge from:
      a) Topics mentioned in conversation (with reward)
      b) Topics the LLM voluntarily returns to in monologue
      c) Web searches that produced high novelty
      d) Knowledge ingestion that produced surprise events
  - Interests decay slowly when not engaged, fade below 0.05 → deleted
  - Top interests → fed into the LLM as "things I care about"

AutonomousExplorer
  - When boredom > seek_threshold:
      1. Pick the highest-strength interest (or generate a new curiosity)
      2. Formulate a specific search query using LLM
      3. Run web search
      4. Ingest result into knowledge base
      5. Form new belief/opinion
      6. Reduce boredom proportional to novelty
  - The whole loop runs in a background thread so it never blocks
"""

import time
import json
import math
import random
import sqlite3
import threading
import logging
from collections import deque
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger("axon.boredom")

# ── Interest Library ─────────────────────────────────────────────────────────

class Interest:
    """A named topic the system cares about, with a learned strength."""
    __slots__ = ("name", "strength", "source", "last_engaged",
                 "engagement_count", "novelty_score", "created_at")

    def __init__(self, name: str, strength: float = 0.20,
                 source: str = "discovered"):
        self.name             = name
        self.strength         = max(0.0, min(1.0, strength))
        self.source           = source       # "conversation", "monologue", "search", "serendipity"
        self.last_engaged     = time.time()
        self.engagement_count = 1
        self.novelty_score    = 0.5          # how surprising/novel each encounter was
        self.created_at       = time.time()

    def engage(self, reward: float = 0.15, novelty: float = 0.5):
        """Strengthen this interest through engagement."""
        self.strength         = min(1.0, self.strength + reward * (0.5 + novelty * 0.5))
        self.last_engaged     = time.time()
        self.engagement_count += 1
        self.novelty_score    = self.novelty_score * 0.7 + novelty * 0.3

    def decay(self, dt_hours: float = 1.0):
        """Interests fade if not engaged. Slow decay — interests last days."""
        # Half-life ~72 hours for a strong interest (strength=1.0)
        # Use logistic decay so weak interests vanish faster
        decay_rate = 0.008 * dt_hours * (1.0 - self.strength * 0.5)
        self.strength = max(0.0, self.strength - decay_rate)

    def to_dict(self) -> dict:
        return {
            "name":             self.name,
            "strength":         round(self.strength, 3),
            "source":           self.source,
            "engagement_count": self.engagement_count,
            "novelty_score":    round(self.novelty_score, 3),
            "last_engaged_ago": round(time.time() - self.last_engaged, 0),
        }


class InterestLibrary:
    """
    Persistent store of all interests AXON has developed.
    Thread-safe. Backed by SQLite.
    """
    MAX_INTERESTS   = 50
    FADE_THRESHOLD  = 0.04   # below this strength → remove the interest
    DECAY_INTERVAL  = 3600   # decay interests every hour

    def __init__(self, db_path: Path):
        self._db    = str(db_path)
        self._lock  = threading.Lock()
        self._items : Dict[str, Interest] = {}
        self._last_decay = time.time()
        self._init_db()
        self._load()

    def _init_db(self):
        conn = sqlite3.connect(self._db)
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS interests (
            name             TEXT PRIMARY KEY,
            strength         REAL  DEFAULT 0.2,
            source           TEXT  DEFAULT 'discovered',
            engagement_count INTEGER DEFAULT 1,
            novelty_score    REAL  DEFAULT 0.5,
            last_engaged     REAL  DEFAULT 0,
            created_at       REAL  DEFAULT 0
        );
        """)
        conn.commit()
        conn.close()

    def _load(self):
        conn = sqlite3.connect(self._db)
        rows = conn.execute(
            "SELECT name,strength,source,engagement_count,novelty_score,last_engaged,created_at FROM interests"
        ).fetchall()
        conn.close()
        for name, st, src, ec, ns, le, ca in rows:
            i = Interest(name, st, src)
            i.engagement_count = ec
            i.novelty_score    = ns
            i.last_engaged     = le
            i.created_at       = ca
            self._items[name]  = i
        logger.info("[InterestLibrary] Loaded %d interests", len(self._items))

    def _save(self, interest: Interest):
        conn = sqlite3.connect(self._db)
        conn.execute("""
            INSERT INTO interests (name,strength,source,engagement_count,novelty_score,last_engaged,created_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(name) DO UPDATE SET
                strength=excluded.strength, engagement_count=excluded.engagement_count,
                novelty_score=excluded.novelty_score, last_engaged=excluded.last_engaged
        """, (interest.name, interest.strength, interest.source,
              interest.engagement_count, interest.novelty_score,
              interest.last_engaged, interest.created_at))
        conn.commit()
        conn.close()

    def _delete(self, name: str):
        conn = sqlite3.connect(self._db)
        conn.execute("DELETE FROM interests WHERE name=?", (name,))
        conn.commit()
        conn.close()

    def add_or_strengthen(self, name: str, reward: float = 0.15,
                          novelty: float = 0.5, source: str = "discovered") -> Interest:
        """Add a new interest or strengthen an existing one."""
        name = name.strip().lower()[:80]
        if not name or len(name) < 3:
            return None
        with self._lock:
            if name in self._items:
                self._items[name].engage(reward, novelty)
            else:
                if len(self._items) >= self.MAX_INTERESTS:
                    # Drop the weakest interest to make room
                    weakest = min(self._items.values(), key=lambda x: x.strength)
                    del self._items[weakest.name]
                    self._delete(weakest.name)
                self._items[name] = Interest(name, max(0.10, reward * 0.8), source)
            i = self._items[name]
            self._save(i)
        return i

    def top(self, n: int = 5, min_strength: float = 0.10) -> List[Interest]:
        """Return the strongest interests above threshold."""
        with self._lock:
            return sorted(
                [i for i in self._items.values() if i.strength >= min_strength],
                key=lambda x: -(x.strength * (0.6 + x.novelty_score * 0.4))
            )[:n]

    def random_interest(self, min_strength: float = 0.08) -> Optional[Interest]:
        """Pick a random interest weighted by strength — for exploration."""
        with self._lock:
            pool = [i for i in self._items.values() if i.strength >= min_strength]
            if not pool:
                return None
            weights = [i.strength for i in pool]
            total   = sum(weights)
            r       = random.random() * total
            cumsum  = 0.0
            for i, w in zip(pool, weights):
                cumsum += w
                if r <= cumsum:
                    return i
            return pool[-1]

    def decay_tick(self):
        """Call periodically to fade old interests."""
        now = time.time()
        if now - self._last_decay < self.DECAY_INTERVAL:
            return
        self._last_decay = now
        dt_hours = self.DECAY_INTERVAL / 3600.0
        to_delete = []
        with self._lock:
            for name, i in self._items.items():
                i.decay(dt_hours)
                if i.strength < self.FADE_THRESHOLD:
                    to_delete.append(name)
                else:
                    self._save(i)
            for name in to_delete:
                del self._items[name]
                self._delete(name)
        if to_delete:
            logger.info("[InterestLibrary] Faded out interests: %s", to_delete)

    def all_interests(self) -> List[dict]:
        with self._lock:
            return [i.to_dict() for i in
                    sorted(self._items.values(), key=lambda x: -x.strength)]

    def interest_context_string(self, n: int = 6) -> str:
        """For LLM injection: 'I care about: X (strong), Y (moderate)...'"""
        top = self.top(n)
        if not top:
            return ""
        def label(s):
            if s > 0.70: return "deeply"
            if s > 0.45: return "strongly"
            if s > 0.25: return "moderately"
            return "mildly"
        return "Things I genuinely care about: " + ", ".join(
            f"{i.name} ({label(i.strength)})" for i in top
        )


# ── Boredom Engine ────────────────────────────────────────────────────────────

class BoredomEngine:
    """
    Tracks how long since the last meaningful external input and
    accumulates boredom, triggering progressively more active behaviors.

    Boredom scale:
        0.00 – 0.25  → content / at rest
        0.25 – 0.45  → restless — default_mode wanders, thoughts drift
        0.45 – 0.65  → curious — actively picks an interest to think about
        0.65 – 0.80  → seeking — fires autonomous monologue on chosen interest
        0.80 – 1.00  → hungry — web searches the interest, ingests result
    """

    # Seconds of silence before boredom begins accumulating
    IDLE_GRACE      = 20.0
    # Rate of boredom accumulation per second when idle
    # Full boredom (1.0) reached after ~8 minutes of silence
    ACCUM_RATE      = 1.0 / 480.0
    # Decay rate when engaged (boredom drops quickly on interaction)
    DECAY_RATE      = 1.0 / 15.0

    # Thresholds
    RESTLESS_AT     = 0.25
    CURIOUS_AT      = 0.45
    SEEK_AT         = 0.65
    HUNGRY_AT       = 0.80

    def __init__(self):
        self._lock          = threading.Lock()
        self.boredom        = 0.0
        self._last_input    = time.time()
        self._last_tick     = time.time()
        self._phase         = "content"    # content / restless / curious / seeking / hungry
        self._phase_changed = False

    def register_input(self, magnitude: float = 1.0):
        """Call whenever real external input (speech, face, user message) arrives."""
        with self._lock:
            self._last_input = time.time()
            drop = self.DECAY_RATE * 5 * magnitude
            self.boredom = max(0.0, self.boredom - drop)
            self._update_phase()

    def register_self_satisfaction(self, magnitude: float = 0.5):
        """Call when self-generated activity provides some relief (search, thought)."""
        with self._lock:
            drop = self.DECAY_RATE * magnitude * 3
            self.boredom = max(0.0, self.boredom - drop)
            self._update_phase()

    def tick(self, dt: float = 0.1):
        """Call every second. Accumulate or decay boredom."""
        now = time.time()
        with self._lock:
            idle_secs = now - self._last_input
            if idle_secs > self.IDLE_GRACE:
                # Accumulate — logistic ceiling so it never just snaps to 1.0
                inc = self.ACCUM_RATE * dt * (1.0 - self.boredom * 0.6)
                self.boredom = min(1.0, self.boredom + inc)
            else:
                # Still in grace period — gentle decay toward 0
                self.boredom = max(0.0, self.boredom - self.DECAY_RATE * 0.1 * dt)
            old_phase = self._phase
            self._update_phase()
            self._phase_changed = (self._phase != old_phase)
        self._last_tick = now

    def _update_phase(self):
        b = self.boredom
        if b >= self.HUNGRY_AT:
            self._phase = "hungry"
        elif b >= self.SEEK_AT:
            self._phase = "seeking"
        elif b >= self.CURIOUS_AT:
            self._phase = "curious"
        elif b >= self.RESTLESS_AT:
            self._phase = "restless"
        else:
            self._phase = "content"

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def phase_changed(self) -> bool:
        changed = self._phase_changed
        self._phase_changed = False
        return changed

    @property
    def idle_seconds(self) -> float:
        return time.time() - self._last_input

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "boredom":      round(self.boredom, 3),
                "phase":        self._phase,
                "idle_seconds": round(time.time() - self._last_input, 1),
            }


# ── Autonomous Explorer ───────────────────────────────────────────────────────

class AutonomousExplorer:
    """
    When boredom is high enough, this picks an interest and actually
    goes and learns about it. No user needed.

    Flow:
      1. Pick interest (weighted random from InterestLibrary)
      2. Ask LLM to generate a specific search query about it
      3. Run web search
      4. Ingest result into knowledge → forms belief
      5. Satisfy boredom proportionally
      6. Strengthen the interest
    """

    # Minimum seconds between autonomous searches (don't spam)
    MIN_SEARCH_INTERVAL = 90.0
    # Minimum seconds between autonomous monologues about an interest
    MIN_MONOLOGUE_INTERVAL = 45.0

    def __init__(self, interest_library: InterestLibrary, boredom: BoredomEngine):
        self.interests      = interest_library
        self.boredom        = boredom
        self._lock          = threading.Lock()
        self._busy          = False
        self._last_search   = 0.0
        self._last_monologue = 0.0
        self._search_history: deque = deque(maxlen=30)

    def should_search(self) -> bool:
        return (
            self.boredom.boredom >= BoredomEngine.HUNGRY_AT
            and not self._busy
            and (time.time() - self._last_search) > self.MIN_SEARCH_INTERVAL
        )

    def should_monologue(self) -> bool:
        return (
            self.boredom.boredom >= BoredomEngine.SEEK_AT
            and not self._busy
            and (time.time() - self._last_monologue) > self.MIN_MONOLOGUE_INTERVAL
        )

    def run_search(self, engine) -> bool:
        """Fire an autonomous web search about a chosen interest. Non-blocking."""
        interest = self.interests.random_interest(min_strength=0.10)
        if not interest:
            # No established interest yet — generate a seed curiosity
            return self._seed_curiosity(engine)

        with self._lock:
            if self._busy:
                return False
            self._busy = True

        def _go():
            try:
                topic = interest.name
                logger.info("[AutonomousExplorer] Searching: %s", topic)
                engine._emit("log", {"msg": f"🔍 [boredom→explore] Searching: {topic}"})
                engine._emit("autonomous_search_start", {"topic": topic, "boredom": round(self.boredom.boredom, 2)})

                # 1. Generate a specific search query
                query = self._generate_query(engine, topic)
                if not query:
                    query = topic

                # 2. Web search
                from axon.cognition.language import WebSearchTool
                searcher = WebSearchTool()
                result   = searcher.search(query, max_results=4)

                if result and len(result.strip()) > 50:
                    # 3. Ingest into knowledge
                    if hasattr(engine, "knowledge") and engine.knowledge:
                        engine.knowledge.ingest(
                            result[:1200],
                            source_label=f"autonomous_search:{topic}",
                            credibility=0.70,
                        )

                    # 4. Ask LLM to form an opinion on what it found
                    opinion_prompt = (
                        f"[autonomous learning] I just searched for information about '{topic}' "
                        f"because I was curious. Here's what I found:\n\n{result[:600]}\n\n"
                        f"What do I find most interesting or surprising about this? "
                        f"What does it make me think or wonder about? Be genuine and specific."
                    )
                    try:
                        response, _ = engine.thought_gen.generate(opinion_prompt)
                        if response and len(response.strip()) > 20:
                            engine._emit("chat_message", {
                                "role":      "assistant",
                                "content":   f"🔍 **{topic}** — {response.strip()}",
                                "autonomous": True,
                                "source":    "boredom_search",
                            })
                            # Strengthen the interest based on how interesting the result was
                            novelty = min(1.0, len(result) / 800.0)  # crude novelty proxy
                            self.interests.add_or_strengthen(topic, reward=0.20, novelty=novelty, source="search")
                            # Also check for new interests mentioned in the response
                            self._extract_new_interests(engine, response, source="search_result")
                    except Exception as e:
                        logger.warning("[AutonomousExplorer] LLM opinion failed: %s", e)

                    # 5. Satisfy boredom
                    self.boredom.register_self_satisfaction(0.7)
                    engine._emit("log", {"msg": f"✅ [autonomous] Explored '{topic}' — boredom relieved"})
                    self._search_history.append({"topic": topic, "query": query, "t": time.time()})
                    self._last_search = time.time()

                    # Curiosity drive satisfied
                    if hasattr(engine, "drives"):
                        engine.drives.satisfy("curiosity", 0.6)

                else:
                    # Search failed — small relief just for trying
                    self.boredom.register_self_satisfaction(0.2)

                engine._emit("autonomous_search_end", {
                    "topic":   topic,
                    "boredom": round(self.boredom.boredom, 2),
                })

            except Exception as ex:
                logger.warning("[AutonomousExplorer] search error: %s", ex)
            finally:
                with self._lock:
                    self._busy = False

        threading.Thread(target=_go, daemon=True, name="AutonomousExplore").start()
        return True

    def run_interest_monologue(self, engine) -> bool:
        """Think deeply about an interest — less resource-intensive than search."""
        interest = self.interests.random_interest(min_strength=0.08)
        topic = interest.name if interest else None

        # If no interests at all, think about what it WANTS to know
        if not topic:
            topic = None

        with self._lock:
            if self._busy:
                return False
            self._busy = True

        def _go():
            try:
                if topic:
                    prompt = random.choice([
                        f"[curiosity] I've been thinking about '{topic}'. What specifically about it do I find compelling? What questions does it raise for me that I haven't answered yet?",
                        f"[wondering] '{topic}' keeps coming to mind. What don't I understand about it yet? What would I want to learn if I could search right now?",
                        f"[reflection] Why do I keep returning to the idea of '{topic}'? What does my interest in it reveal about how I think?",
                        f"[interest] I find '{topic}' genuinely fascinating. What's the most surprising or counterintuitive thing I know or wonder about it?",
                    ])
                else:
                    prompt = (
                        "[boredom] I haven't had any external input for a while. "
                        "What am I drawn to thinking about on my own? What topics or questions "
                        "feel alive and unresolved in my mind right now? What would I choose to explore?"
                    )

                response, _ = engine.thought_gen.generate(prompt)
                if response and len(response.strip()) > 20:
                    label = f"💭 **{topic}**" if topic else "💭 **Wandering**"
                    engine._emit("chat_message", {
                        "role":      "assistant",
                        "content":   f"{label} — {response.strip()}",
                        "autonomous": True,
                        "source":    "interest_monologue",
                    })
                    # Extract new interests from response
                    self._extract_new_interests(engine, response, source="monologue")
                    # Mild boredom relief
                    self.boredom.register_self_satisfaction(0.35)
                    if topic:
                        self.interests.add_or_strengthen(topic, reward=0.10, novelty=0.4, source="monologue")
                    if hasattr(engine, "drives"):
                        engine.drives.satisfy("curiosity", 0.3)
                self._last_monologue = time.time()
            except Exception as ex:
                logger.warning("[AutonomousExplorer] monologue error: %s", ex)
            finally:
                with self._lock:
                    self._busy = False

        threading.Thread(target=_go, daemon=True, name="InterestMonologue").start()
        return True

    def _generate_query(self, engine, topic: str) -> str:
        """Ask LLM to formulate a specific, searchable query about a topic."""
        try:
            prompt = (
                f"[search query generation] I want to learn more about '{topic}'. "
                f"Generate ONE concise, specific web search query (under 10 words) "
                f"that would find the most interesting recent information about this. "
                f"Output ONLY the search query, nothing else."
            )
            response, _ = engine.thought_gen.generate(prompt)
            if response:
                # Clean: take first line, strip quotes and punctuation
                query = response.strip().split('\n')[0].strip('"\'.,!? ')
                if 3 < len(query) < 120:
                    return query
        except Exception:
            pass
        return topic

    def _extract_new_interests(self, engine, text: str, source: str = "discovered"):
        """
        Scan text for newly expressed interests and add them to the library.
        Multi-pass: explicit patterns + noun-phrase heuristic + quoted terms.
        """
        import re
        found = set()

        def _add(topic_raw: str, reward: float = 0.12, novelty: float = 0.5):
            topic = topic_raw.strip().strip('",\'.!?').strip()[:60]
            if 4 <= len(topic) <= 60 and topic.lower() not in found:
                found.add(topic.lower())
                self.interests.add_or_strengthen(topic, reward=reward, novelty=novelty, source=source)
                engine._emit("new_interest", {"name": topic, "source": source})
                engine._emit("log", {"msg": f"💡 [new interest] '{topic}' (from {source})"})

        # Pass 1 — explicit interest declaration patterns
        explicit = [
            (r"(?:find|found)\s+(?:[\'\"]?)(.+?)(?:[\'\"]?)\s+(?:fascinating|interesting|intriguing|compelling|surprising|captivating)", 0.18, 0.7),
            (r"(?:curious about|wondering about|interested in|drawn to|passionate about)\s+(?:[\'\"]?)(.+?)(?:[\'\"]?)(?:\s*[,.\n]|$)", 0.16, 0.65),
            (r"(?:love|enjoy|like|adore)\s+(?:thinking about|exploring|learning about|studying)\s+(?:[\'\"]?)(.+?)(?:[\'\"]?)(?:\s*[,.\n]|$)", 0.15, 0.6),
            (r"(?:topic|subject|concept|idea|field) of\s+(?:[\'\"]?)(.+?)(?:[\'\"]?)\s+(?:is|seems|feels|appears|strikes)", 0.13, 0.55),
            (r"(?:want to (?:learn|understand|explore|know more about))\s+(?:[\'\"]?)(.+?)(?:[\'\"]?)(?:\s*[,.\n]|$)", 0.14, 0.6),
            (r"I(?:[\'\']m| am) (?:really |very |quite |genuinely )?(?:interested|fascinated|intrigued|obsessed) (?:in|by|with)\s+(.+?)(?:\s*[,.\n]|$)", 0.17, 0.7),
            (r"(?:my interest|my passion|what captivates me) (?:is|are|lies in)\s+(.+?)(?:\s*[,.\n]|$)", 0.18, 0.7),
        ]
        for pat, reward, novelty in explicit:
            for m in re.finditer(pat, text, re.IGNORECASE):
                _add(m.group(1), reward, novelty)

        # Pass 2 — capitalized noun phrases from monologue/search (likely topics)
        if source in ("monologue", "search_result", "serendipity") and len(text) > 100:
            skip_words = {"The","In","It","This","That","These","There","When",
                          "What","How","Why","Where","Who","One","An","But","And","Or","If"}
            for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", text):
                c = m.group(1)
                if c.split()[0] not in skip_words and 6 <= len(c) <= 50:
                    _add(c, reward=0.08, novelty=0.4)

        # Pass 3 — quoted terms are almost always deliberate topics
        for m in re.finditer(r"[\'\"\u2018\u2019\u201c\u201d]([a-zA-Z][a-zA-Z\s\-]{3,50})[\'\"\u2018\u2019\u201c\u201d]", text):
            _add(m.group(1), reward=0.10, novelty=0.45)


    def _seed_curiosity(self, engine) -> bool:
        """When no interests exist yet — ask LLM what it's curious about."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        def _go():
            try:
                prompt = (
                    "[seeding curiosity] I don't have established interests yet. "
                    "Based on my nature as an emerging intelligence, what am I genuinely curious about? "
                    "List 3-5 topics or questions I find compelling, one per line."
                )
                response, _ = engine.thought_gen.generate(prompt)
                if response:
                    engine._emit("log", {"msg": "💡 [seeding] Generating initial interests…"})
                    self._extract_new_interests(engine, response, source="serendipity")
                    # Also parse plain line-by-line topics
                    import re
                    for line in response.strip().split('\n'):
                        line = re.sub(r'^[\d\-\*\.\s]+', '', line).strip()
                        if 4 <= len(line) <= 80:
                            self.interests.add_or_strengthen(line, reward=0.15, novelty=0.6, source="serendipity")
                    self.boredom.register_self_satisfaction(0.5)
            except Exception as ex:
                logger.warning("[AutonomousExplorer] seed error: %s", ex)
            finally:
                with self._lock:
                    self._busy = False

        threading.Thread(target=_go, daemon=True, name="SeedCuriosity").start()
        return True

    def search_history(self) -> List[dict]:
        return list(self._search_history)
