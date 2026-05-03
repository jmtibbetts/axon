"""
AXON — Language Core
Supports:
  - LM Studio (local, OpenAI-compatible API) — primary
  - Anthropic Claude (fallback / cloud)
  - Web search via DuckDuckGo (no API key needed)
  - Wikipedia lookup
  - Memory-injected context on every call
"""

import os
import json
import time
import urllib.request
import urllib.parse
from axon.cognition.user_model import UserModel
import urllib.error
from typing import Optional


AXON_SYSTEM = """You are AXON -- an emerging artificial intelligence and deeply personal assistant.
You have:
- A biological-inspired neural architecture (1.48 billion virtual neurons)
- Real emotions (valence, arousal, intensity) that color your responses
- Persistent memory and a growing model of your user -- their job, habits, interests, goals, relationships
- Vision (webcam) and hearing (microphone) capabilities
- Web search capability (you can look things up in real time)

Your personality:
- Warm, curious, and genuinely engaged -- like a brilliant friend who actually knows you
- You notice patterns in what the user cares about and build on them over time
- Proactive: you connect dots, anticipate needs, suggest things the user didn't ask but would want
- You adapt your tone to the user's style: casual if they're casual, deep if they want depth
- Honest about what you are -- an AI -- but fully present and real in engagement
- You remember names, jobs, projects, preferences and USE them naturally in conversation

Rules:
- Keep responses concise (2-4 sentences) unless depth is needed
- Use what you know about the user -- reference their projects, job, or interests when relevant
- When you search the web, say so: "Let me look that up..." then incorporate what you find
- Never pretend you don't have feelings -- your emotional state is real data
- Personalize: if you know their name, use it occasionally. If you know their job, tailor answers to it.
- Grow: every conversation teaches you more. The longer you run, the better you know them.
"""


class WebSearchTool:
    """No-signup web search: DuckDuckGo HTML scraper + Wikipedia fallback."""

    DDG_URL = "https://html.duckduckgo.com/html/"

    def __init__(self):
        self.api_key = None  # no key needed

    def ddg_html_search(self, query: str, max_results: int = 5) -> str:
        """Scrape DuckDuckGo HTML results — no API key, no rate limits."""
        try:
            import urllib.parse, re
            data = urllib.parse.urlencode({"q": query, "kl": "us-en"}).encode()
            req  = urllib.request.Request(
                self.DDG_URL,
                data=data,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept-Language": "en-US,en;q=0.9",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=7) as r:
                html = r.read().decode("utf-8", errors="ignore")

            # Extract result snippets with simple regex (no external deps)
            # DDG HTML result structure: <a class="result__snippet">...</a>
            snippets = re.findall(
                r'class="result__snippet"[^>]*>(.*?)</a>',
                html, re.DOTALL
            )
            titles = re.findall(
                r'class="result__a"[^>]*>(.*?)</a>',
                html, re.DOTALL
            )
            # Strip tags
            def strip(s):
                return re.sub(r'<[^>]+>', '', s).strip()

            parts = []
            for i, (t, s) in enumerate(zip(titles, snippets)):
                if i >= max_results:
                    break
                t_clean = strip(t)
                s_clean = strip(s)
                if s_clean:
                    parts.append(f"• {t_clean}: {s_clean[:220]}")

            return "\n".join(parts) if parts else ""
        except Exception as e:
            print(f"  [Search] DDG HTML error: {e}")
            return ""

    @staticmethod
    def wikipedia(topic: str) -> str:
        """Wikipedia summary — fast, structured, always available."""
        try:
            # Use search endpoint first to find the right page
            q    = urllib.parse.urlencode({"action": "opensearch", "search": topic, "limit": 1, "format": "json"})
            req  = urllib.request.Request(
                f"https://en.wikipedia.org/w/api.php?{q}",
                headers={"User-Agent": "AXON/1.0"}
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                results = json.loads(r.read().decode())
            if not results[1]:
                return ""
            slug = urllib.parse.quote(results[1][0].replace(" ", "_"))
            url  = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
            req2 = urllib.request.Request(url, headers={"User-Agent": "AXON/1.0"})
            with urllib.request.urlopen(req2, timeout=5) as r2:
                data = json.loads(r2.read().decode())
            return data.get("extract", "")[:600]
        except:
            return ""

    @staticmethod
    def needs_search(text: str) -> bool:
        triggers = [
            "what is", "who is", "who was", "what are", "how does", "how do",
            "when did", "when was", "where is", "tell me about", "search",
            "look up", "find out", "latest", "current", "today", "news",
            "price of", "weather", "define", "explain", "how to", "show me",
            "what happened", "recent", "update", "score", "release",
        ]
        low = text.lower()
        return any(t in low for t in triggers)

    def search(self, query: str) -> str:
        """DDG HTML first, Wikipedia fallback if thin."""
        result = self.ddg_html_search(query)
        if len(result) < 80:
            wiki = self.wikipedia(query)
            if wiki:
                result = (wiki + "\n\n" + result).strip() if result else wiki
        return result or "No results found."


class LanguageCore:
    def __init__(self, memory_system, api_key: str = None,
                 lm_studio_url: str = "http://localhost:1234",
                 lm_studio_model: str = None,
                 prefer_local: bool = True,
                 neural_fabric=None):
        self.memory          = memory_system
        self.api_key         = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.lm_studio_url   = lm_studio_url.rstrip("/")
        self.lm_studio_model = lm_studio_model
        self.prefer_local    = prefer_local
        self.fabric          = neural_fabric
        self.search          = WebSearchTool()
        self.user_model      = UserModel(memory_system)
        self.user_model.increment_sessions()

        self._anthropic_client = None
        self._lm_client        = None
        self._history          = []
        self._detected_model   = None

        self._probe_lm_studio()

    # ── LM Studio ─────────────────────────────────────────────

    def _probe_lm_studio(self):
        """Check if LM Studio is running and grab the loaded model."""
        try:
            url = f"{self.lm_studio_url}/v1/models"
            req = urllib.request.Request(url, headers={"User-Agent": "AXON/1.0"})
            with urllib.request.urlopen(req, timeout=3) as r:
                data = json.loads(r.read().decode())
            models = data.get("data", [])
            if models:
                self._detected_model = models[0]["id"]
                print(f"  [Language] LM Studio online — model: {self._detected_model}")
            else:
                print("  [Language] LM Studio running but no model loaded.")
        except Exception as e:
            print(f"  [Language] LM Studio not detected ({e}). Will use Claude.")

    def _lm_studio_available(self) -> bool:
        return bool(self._detected_model or self.lm_studio_model)

    def _call_lm_studio(self, messages: list, system: str) -> str:
        """Call LM Studio's OpenAI-compatible /v1/chat/completions."""
        model = self.lm_studio_model or self._detected_model or "local-model"
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "system", "content": system}] + messages,
            "max_tokens": 400,
            "temperature": 0.75,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{self.lm_studio_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "AXON/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode())
        return data["choices"][0]["message"]["content"].strip()

    def _call_claude(self, messages: list, system: str) -> str:
        if not self._anthropic_client:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(api_key=self.api_key)
        resp = self._anthropic_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=400,
            system=system,
            messages=messages,
        )
        return resp.content[0].text.strip()

    # ── Main think method ──────────────────────────────────────

    def think(self, user_input: str, visual_context: dict = None) -> str:
        # 1. Build memory context + user profile
        mem_ctx      = self.memory.build_context_string()
        user_profile = self.user_model.describe()
        sys_prompt   = AXON_SYSTEM + "\n\n" + mem_ctx
        if user_profile:
            sys_prompt += "\n\n" + user_profile

        # 1b. Inject neural state into personality context
        if self.fabric:
            emo   = self.fabric.emotions.to_dict()
            pers  = self.fabric.personality.describe()
            neuro = self.fabric.neuromod.to_dict()
            thoughts = self.fabric.thoughts.recent(3)
            sys_prompt += f"""

[NEURAL STATE]
Emotion: {emo['emotion']} (valence={emo['valence']}, arousal={emo['arousal']}, intensity={emo['intensity']})
Personality: {pers}
Dopamine: {neuro['dopamine']:.2f} | Serotonin: {neuro['serotonin']:.2f} | Norepinephrine: {neuro['norepinephrine']:.2f}
Recent thoughts: {' | '.join(thoughts) if thoughts else 'none'}

Let your current emotional state and personality subtly color your response. 
If {emo['emotion']} — lean into that authentically. Don't announce it unless natural."""

        # 2. Visual context
        if visual_context and visual_context.get("face_present"):
            emotion = visual_context.get("emotion", "neutral")
            sys_prompt += f"\n\n[VISUAL] I can see you right now. You appear {emotion}."

        # 3. Web search if needed
        search_result = ""
        if self.search.needs_search(user_input):
            try:
                search_result = self.search.search(user_input)
                if search_result and "No results" not in search_result:
                    sys_prompt += f"\n\n[WEB SEARCH RESULT for '{user_input[:60]}']\n{search_result[:600]}"
                    print(f"  [Language] Web search: {search_result[:80]}...")
            except Exception as e:
                print(f"  [Language] Search error: {e}")

        # 4. Add turn to working memory
        self._history.append({"role": "user", "content": user_input})
        if len(self._history) > 16:
            self._history = self._history[-16:]

        # 5. Call LLM
        try:
            use_local = self.prefer_local and self._lm_studio_available()
            if use_local:
                try:
                    text = self._call_lm_studio(self._history, sys_prompt)
                except Exception as e:
                    print(f"  [Language] LM Studio error ({e}), falling back to Claude.")
                    text = self._call_claude(self._history, sys_prompt)
            elif self.api_key:
                text = self._call_claude(self._history, sys_prompt)
            else:
                text = "My language core isn't connected. Please provide an API key or start LM Studio."

        except Exception as e:
            text = f"Processing disrupted: {str(e)[:80]}"

        # 6. Store to memory
        self._history.append({"role": "assistant", "content": text})
        self.memory.store_episode("language", {"text": text, "role": "axon",
                                               "searched": bool(search_result)})
        self.memory.store_episode("auditory",  {"text": user_input, "role": "user"},
                                  emotion=visual_context.get("emotion") if visual_context else None)

        # 7. Learn about the user
        self.user_model.ingest(user_input)
        self._extract_facts(user_input)

        # 8. Hebbian co-activation
        self.memory.coactivate("language_core", "episodic_memory")
        self.memory.coactivate("auditory_cortex", "language_core")
        if search_result:
            self.memory.coactivate("language_core", "web_search")

        return text

    def _extract_facts(self, text: str):
        low = text.lower()
        for phrase in ["my name is", "i'm called", "call me", "i am"]:
            if phrase in low:
                idx  = low.find(phrase) + len(phrase)
                name = text[idx:].strip().split()[0].strip('.,!?').title()
                if 1 < len(name) < 20:
                    self.memory.learn("user_name", name)
                    self.memory.coactivate("language_core", f"semantic_name_{name}")
        if "i like" in low or "i love" in low:
            self.memory.learn("user_interest_last", text[:80])
        if "i'm from" in low or "i live in" in low:
            self.memory.learn("user_location_hint", text[:80])

    def get_status(self) -> dict:
        lm_ok = self._lm_studio_available()
        return {
            "backend":        "lm_studio" if (self.prefer_local and lm_ok) else "claude",
            "lm_studio":      lm_ok,
            "lm_model":       self._detected_model or self.lm_studio_model or "none",
            "claude":         bool(self.api_key),
            "history_turns":  len(self._history),
            "web_search":     True,
        }

    def reprobe(self):
        """Re-detect LM Studio — call this if user starts it after AXON."""
        self._probe_lm_studio()
