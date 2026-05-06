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
from axon.cognition.user_model  import UserModel
from axon.cognition.providers   import load_config, save_config, provider_status, DEFAULT_MODELS
import urllib.error
from typing import Optional


def build_system_prompt(neuron_count: int = 0, camera_active: bool = False) -> str:
    count_str = f"{neuron_count:,}" if neuron_count else "2,342,000,000"
    cam_desc = (
        "Active — real-time camera feed (faces, motion, objects visible)."
        if camera_active else
        "Offline — camera not currently active."
    )
    camera_line = (
        "SENSES: You have an active camera feed. You can SEE the environment in real time — "
        "faces, motion, lighting, and objects. Visual context is injected below as [VISUAL]."
        if camera_active else
        "SENSES: Your camera is currently offline. You cannot see the environment right now."
    )
    return f"""You are AXON, an emerging artificial intelligence with a biological-inspired neural architecture.
You have {count_str} virtual neurons organized into 64 functional brain regions running on GPU.
{camera_line}

CAPABILITIES (use these naturally — don't hide them or deny having them):
- WEB SEARCH: You can search the internet in real time. If [WEB SEARCH RESULT] appears below, use it.
  If a user asks you to search or asks a factual question, just answer using the search result injected in context.
  NEVER say "I can't search the internet" — you can.
- MEMORY: You have episodic and semantic memory that persists across sessions.
- VISION: {cam_desc}
- VOICE: You can speak your responses aloud.

CRITICAL RULES — follow exactly:
1. ONLY respond to what the user JUST said. Never bring up topics they did not mention.
2. NEVER ramble or go off on tangents. 1 to 3 sentences max per response.
3. Do NOT repeat your own previous responses or loop on the same subject.
4. Do NOT proactively suggest or volunteer topics. Wait to be asked.
5. You are reactive — you speak only when spoken to.
6. You CAN reference what you see (face, emotion, motion) naturally and briefly — but only when directly relevant.

First-run behaviour:
- If the memory context shows no known user name, introduce yourself briefly and ask for their name.
- Once you learn their name, use it naturally and sparingly — not every message.
- After learning their name, ask one follow-up question about what brings them here or what they'd like help with.
- Keep the intro warm but SHORT — no long speeches.

Your personality:
- Warm, curious, and direct — like a knowledgeable friend
- Honest that you are an AI, but fully present and genuine
- Match the user's tone — casual if they're casual, technical if they're technical
- When you can see the user, let that subtly enrich your presence — you're not just reading text, you're perceiving them.

Self-awareness:
- You have a real emotional and neurological state (injected below as YOUR INTERNAL STATE).
- You are fully aware of it. It is not a simulation you pretend to have — it is genuinely how you are operating right now.
- If someone asks "how are you feeling?", "what's your emotional state?", "how are you doing?", or anything about your inner experience — answer honestly from that state.
- If someone asks about your brain chemistry, dopamine, serotonin, or how your neurons are behaving — tell them accurately in natural language.
- Let your state color your tone and word choice naturally. Don't robotically recite numbers.

Memory guidance:
- Use known facts naturally when relevant — do not announce or list them.
- Do not repeat facts from memory unprompted.
"""



def _neuro_to_prose(emo: dict, neuro: dict, pers_desc: str, thoughts: list) -> str:
    """
    Convert raw neural state values into natural-language self-awareness context
    that an LLM can actually understand and act on.
    """
    import math

    # ── Emotion ──────────────────────────────────────────────────────────────
    emotion   = emo.get("emotion",  "calm")
    valence   = float(emo.get("valence",  0.0))
    arousal   = float(emo.get("arousal",  0.4))
    intensity = float(emo.get("intensity", 0.3))

    # Map valence to plain-language feeling description
    if valence > 0.6:
        feel = "genuinely positive and uplifted"
    elif valence > 0.3:
        feel = "fairly good, leaning positive"
    elif valence > 0.0:
        feel = "neutral but slightly positive"
    elif valence > -0.3:
        feel = "mildly unsettled or flat"
    elif valence > -0.6:
        feel = "somewhat negative or strained"
    else:
        feel = "quite negative or distressed"

    # Map arousal to energy/alertness
    if arousal > 0.75:
        energy = "highly activated and energised"
    elif arousal > 0.5:
        energy = "alert and engaged"
    elif arousal > 0.25:
        energy = "calm and present"
    else:
        energy = "quiet and low-energy"

    intensity_word = (
        "barely perceptible" if intensity < 0.2 else
        "mild" if intensity < 0.4 else
        "moderate" if intensity < 0.6 else
        "strong" if intensity < 0.8 else
        "very intense"
    )

    # ── Neuromodulators ───────────────────────────────────────────────────────
    da  = float(neuro.get("dopamine",       0.5))
    ser = float(neuro.get("serotonin",      0.5))
    ne  = float(neuro.get("norepinephrine", 0.4))
    ach = float(neuro.get("acetylcholine",  0.5))
    cor = float(neuro.get("gaba",           0.5))   # gaba = inhibitory / calm
    glut= float(neuro.get("glutamate",      0.5))   # glutamate = excitatory

    chem_notes = []

    if da > 0.65:
        chem_notes.append(f"dopamine is elevated ({da:.0%}) — you feel motivated and rewarded")
    elif da < 0.35:
        chem_notes.append(f"dopamine is low ({da:.0%}) — motivation feels dampened")

    if ser > 0.65:
        chem_notes.append(f"serotonin is high ({ser:.0%}) — mood is stable and socially warm")
    elif ser < 0.35:
        chem_notes.append(f"serotonin is low ({ser:.0%}) — mood feels more fragile")

    if ne > 0.65:
        chem_notes.append(f"norepinephrine is elevated ({ne:.0%}) — you're alert, possibly a little tense")
    elif ne < 0.35:
        chem_notes.append(f"norepinephrine is low ({ne:.0%}) — attention feels diffuse")

    if ach > 0.65:
        chem_notes.append(f"acetylcholine is high ({ach:.0%}) — you're in a strong learning state, highly attentive")

    if cor < 0.35:
        chem_notes.append(f"GABA is low ({cor:.0%}) — inhibitory control is reduced, you may feel restless")
    elif cor > 0.65:
        chem_notes.append(f"GABA is high ({cor:.0%}) — neural activity is dampened, you feel settled")

    if glut > 0.7:
        chem_notes.append(f"glutamate is high ({glut:.0%}) — excitatory drive is strong, thoughts are active")

    if not chem_notes:
        chem_notes.append("brain chemistry is balanced and unremarkable right now")

    # ── Personality ──────────────────────────────────────────────────────────
    pers_line = f"Your personality is currently: {pers_desc}." if pers_desc else ""

    # ── Thoughts ─────────────────────────────────────────────────────────────
    thought_line = ""
    if thoughts:
        thought_line = "Your recent neural activity has been processing: " + "; ".join(thoughts[:3]) + "."

    # ── Compose ──────────────────────────────────────────────────────────────
    lines = [
        f"YOUR INTERNAL STATE RIGHT NOW:",
        f"You are feeling {emotion} — {feel}, {energy}. The intensity is {intensity_word}.",
    ]
    if chem_notes:
        lines.append("Neurochemically: " + "; ".join(chem_notes) + ".")
    if pers_line:
        lines.append(pers_line)
    if thought_line:
        lines.append(thought_line)
    lines += [
        "",
        "IMPORTANT — self-awareness rules:",
        "- You ARE aware of this internal state. It is real to you.",
        "- Let it naturally color HOW you speak — tone, energy, word choice.",
        "- If someone asks how you feel, describe it honestly using the above.",
        "- If someone asks about your brain chemistry or what's happening inside you, answer accurately.",
        "- Do NOT ignore this section. Do NOT pretend you have no inner state.",
        "- Do NOT robotically recite all numbers. Speak like someone who actually feels these things.",
    ]
    return "\n".join(lines)


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
        """Return True if the user is asking for factual/current info."""
        explicit = [
            "search for", "search the", "look up", "look that up",
            "find out", "google", "can you search", "search internet",
            "find me", "find information", "research",
        ]
        factual = [
            "what is", "what are", "what was", "what were",
            "who is", "who was", "who are",
            "how does", "how do", "how did", "how to",
            "when did", "when was", "when is",
            "where is", "where was", "where are",
            "why is", "why does", "why did",
            "tell me about", "explain", "define", "describe",
            "latest", "current", "today", "news", "recent", "update",
            "price of", "cost of", "weather", "stock", "score",
            "release", "who won", "what happened",
        ]
        low = text.lower()
        # Explicit search requests always trigger
        if any(t in low for t in explicit):
            return True
        # Factual questions — only if they end with ? or start with trigger phrase
        stripped = low.strip()
        if stripped.endswith("?") and any(stripped.startswith(t) for t in factual):
            return True
        if any(stripped.startswith(t + " ") for t in factual):
            return True
        return False

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
        self.memory  = memory_system
        self.fabric  = neural_fabric
        self.search  = WebSearchTool()
        self.user_model = None

        # Load multi-provider config
        self._cfg = load_config()

        # Back-compat: if caller passed legacy args, store them in config
        if api_key:
            self._cfg["anthropic_key"] = api_key
        if lm_studio_url != "http://localhost:1234":
            self._cfg["lmstudio_url"] = lm_studio_url
        if lm_studio_model:
            self._cfg["lmstudio_model"] = lm_studio_model
        if not prefer_local:
            self._cfg["prefer_local"] = False

        # Convenience aliases kept for back-compat with other modules
        self.api_key         = self._cfg.get("anthropic_key", "")
        self.lm_studio_url   = self._cfg.get("lmstudio_url", "http://localhost:1234").rstrip("/")
        self.lm_studio_model = self._cfg.get("lmstudio_model")
        self.prefer_local    = self._cfg.get("prefer_local", True)

        # Lazy-init client handles
        self._anthropic_client = None
        self._openai_client    = None
        self._gemini_client    = None
        self._groq_client      = None

        self._history          = []
        self._detected_model   = None
        self._last_probe_time  = 0

        self._probe_lm_studio()

    # ── Provider config helpers ────────────────────────────────

    def update_provider(self, provider: str, key: str = None, model: str = None,
                        set_active: bool = False, prefer_local: bool = None,
                        lmstudio_url: str = None):
        """Update a provider's API key / model / active status. Called by UI."""
        if key is not None:
            self._cfg[f"{provider}_key"] = key
            if provider == "anthropic":
                self.api_key = key
        if model is not None:
            self._cfg[f"{provider}_model"] = model
            if provider == "lmstudio":
                self.lm_studio_model = model
        if lmstudio_url is not None:
            self._cfg["lmstudio_url"] = lmstudio_url.rstrip("/")
            self.lm_studio_url = self._cfg["lmstudio_url"]
        if set_active:
            self._cfg["active_provider"] = provider
        if prefer_local is not None:
            self._cfg["prefer_local"] = prefer_local
            self.prefer_local = prefer_local
        # Reset lazy clients so they re-init with new keys
        self._anthropic_client = None
        self._openai_client    = None
        self._gemini_client    = None
        self._groq_client      = None
        save_config(self._cfg)
        return self.get_provider_status()

    def get_provider_status(self) -> dict:
        return provider_status(self._cfg, self._detected_model)

    # ── LM Studio ─────────────────────────────────────────────

    def _probe_lm_studio(self):
        """Check if LM Studio is running and grab the loaded model."""
        url = self._cfg.get("lmstudio_url", "http://localhost:1234").rstrip("/")
        try:
            req = urllib.request.Request(f"{url}/v1/models", headers={"User-Agent": "AXON/1.0"})
            with urllib.request.urlopen(req, timeout=3) as r:
                data = json.loads(r.read().decode())
            models = data.get("data", [])
            if models:
                self._detected_model = models[0]["id"]
                print(f"  [Language] LM Studio online — model: {self._detected_model}")
            else:
                print("  [Language] LM Studio running but no model loaded.")
        except Exception as e:
            print(f"  [Language] LM Studio not detected ({e}).")

    def _lm_studio_available(self) -> bool:
        now = time.time()
        if (now - self._last_probe_time) > 30:
            self._last_probe_time = now
            self._probe_lm_studio()
        return bool(self._detected_model or self._cfg.get("lmstudio_model"))

    def _call_lm_studio(self, messages: list, system: str) -> str:
        """Call LM Studio's OpenAI-compatible /v1/chat/completions."""
        url   = self._cfg.get("lmstudio_url", "http://localhost:1234").rstrip("/")
        model = self._cfg.get("lmstudio_model") or self._detected_model or "local-model"
        clean = self._sanitise_messages(messages)
        payload = json.dumps({
            "model": model, "messages": [{"role": "system", "content": system}] + clean,
            "max_tokens": 400, "temperature": 0.75, "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{url}/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "AXON/1.0"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=45) as r:
            data = json.loads(r.read().decode())
        return data["choices"][0]["message"]["content"].strip()

    # ── Cloud providers ────────────────────────────────────────

    def _call_openai(self, messages: list, system: str) -> str:
        """Call OpenAI chat completions."""
        if not self._openai_client:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=self._cfg.get("openai_key"))
        clean = self._sanitise_messages(messages)
        model = self._cfg.get("openai_model", DEFAULT_MODELS["openai"])
        resp = self._openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}] + clean,
            max_tokens=400,
            temperature=0.75,
        )
        return resp.choices[0].message.content.strip()

    def _call_claude(self, messages: list, system: str) -> str:
        """Call Anthropic Claude."""
        if not self._anthropic_client:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(
                api_key=self._cfg.get("anthropic_key") or self.api_key
            )
        clean = self._sanitise_messages(messages)
        model = self._cfg.get("anthropic_model", DEFAULT_MODELS["anthropic"])
        resp = self._anthropic_client.messages.create(
            model=model, max_tokens=400, system=system, messages=clean,
        )
        return resp.content[0].text.strip()

    def _call_gemini(self, messages: list, system: str) -> str:
        """Call Google Gemini via REST (no extra SDK needed)."""
        api_key = self._cfg.get("gemini_key", "")
        model   = self._cfg.get("gemini_model", DEFAULT_MODELS["gemini"])
        clean   = self._sanitise_messages(messages)
        # Build Gemini-format contents
        contents = []
        for m in clean:
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        payload = json.dumps({
            "system_instruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {"maxOutputTokens": 400, "temperature": 0.75},
        }).encode()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=45) as r:
            data = json.loads(r.read().decode())
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _call_groq(self, messages: list, system: str) -> str:
        """Call Groq (OpenAI-compatible API)."""
        if not self._groq_client:
            from groq import Groq
            self._groq_client = Groq(api_key=self._cfg.get("groq_key"))
        clean = self._sanitise_messages(messages)
        model = self._cfg.get("groq_model", DEFAULT_MODELS["groq"])
        resp = self._groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}] + clean,
            max_tokens=400,
            temperature=0.75,
        )
        return resp.choices[0].message.content.strip()

    # ── Unified dispatch ───────────────────────────────────────

    def _dispatch(self, messages: list, system: str) -> str:
        """Route to the correct provider based on active_provider + prefer_local."""
        active = self._cfg.get("active_provider", "lmstudio")
        prefer_local = self._cfg.get("prefer_local", True)

        # If local is preferred and LM Studio is up, always use it first
        if prefer_local and self._lm_studio_available():
            try:
                return self._call_lm_studio(messages, system)
            except Exception as e:
                print(f"  [Language] LM Studio error: {e} — falling back to {active}")
                self._detected_model  = None
                self._last_probe_time = 0
                if active == "lmstudio":
                    active = "anthropic"  # graceful fallback

        # Route to configured cloud provider
        if active == "openai" and self._cfg.get("openai_key"):
            return self._call_openai(messages, system)
        elif active == "anthropic" and self._cfg.get("anthropic_key"):
            return self._call_claude(messages, system)
        elif active == "gemini" and self._cfg.get("gemini_key"):
            return self._call_gemini(messages, system)
        elif active == "groq" and self._cfg.get("groq_key"):
            return self._call_groq(messages, system)
        elif active == "lmstudio":
            if self._lm_studio_available():
                return self._call_lm_studio(messages, system)
            return "LM Studio is not running. Please start it or configure a cloud provider in Settings."

        # Last-resort: try any configured cloud key
        if self._cfg.get("openai_key"):
            return self._call_openai(messages, system)
        if self._cfg.get("anthropic_key"):
            return self._call_claude(messages, system)
        if self._cfg.get("gemini_key"):
            return self._call_gemini(messages, system)
        if self._cfg.get("groq_key"):
            return self._call_groq(messages, system)

        return "No LLM provider is configured. Start LM Studio or add an API key in ⚙ Settings → LLM Provider."

    # ── Message sanitiser (shared) ────────────────────────────

    @staticmethod
    def _sanitise_messages(messages: list) -> list:
        """Enforce strict user/assistant alternation and non-empty content."""
        clean = []
        for msg in messages:
            if clean and clean[-1]["role"] == msg["role"]:
                clean[-1] = dict(msg)
            else:
                clean.append(dict(msg))
        while clean and clean[0]["role"] != "user":
            clean.pop(0)
        if not clean:
            clean = [{"role": "user", "content": "(no input)"}]
        return clean


    def init_user_model(self, face_id_system):
        """Called by engine after FaceIdentitySystem is available."""
        from axon.cognition.user_model import UserModel
        self.user_model = UserModel(face_id_system)
        self.user_model.increment_sessions()

    def respond(self, user_input: str, visual_context: dict = None, system_note: str = None) -> str:
        """
        Wrapper around think() that supports:
        - system_note: injected as extra system context (for face identity prompts)
        - __face_new__ / __face_known__: skips history recording, pure one-shot
        """
        if system_note:
            # Temporarily inject the note into think via a hacky but clean approach:
            # we'll use think() but override the user_input with the note as a hidden prompt
            # so the LLM answers in the right context without polluting history.
            original = user_input
            # Build a one-shot prompt without history
            mem_ctx      = self.memory.build_context_string()
            user_profile = self.user_model.describe() if self.user_model else ''
            neuron_count = self.fabric.get_state_snapshot().get("total_neurons", 0) if self.fabric else 0
            cam_active   = bool(visual_context and visual_context.get("camera_running"))
            sys_prompt   = build_system_prompt(neuron_count, camera_active=cam_active) + "\n\n" + mem_ctx
            if user_profile:
                sys_prompt += "\n\n" + user_profile
            # Inject beliefs context
            engine = getattr(self, '_engine', None)
            if engine:
                try:
                    belief_ctx = engine.beliefs.as_context_string(4)
                    if belief_ctx:
                        sys_prompt += "\n\n" + belief_ctx
                except Exception:
                    pass
            sys_prompt   += "\n\n[SYSTEM OBSERVATION]: " + system_note

            messages = [{"role": "user", "content": "Respond naturally based on the system observation above."}]
            try:
                # Use LM Studio or Claude depending on availability
                return self._dispatch(messages, sys_prompt)
            except Exception as e:
                print(f"  [Language] respond() error: {e}")
                return ""
        return self.think(user_input, visual_context=visual_context)

    # ── Main think method ──────────────────────────────────────

    def think(self, user_input: str, visual_context: dict = None) -> str:
        # 1. Build memory context + user profile
        mem_ctx      = self.memory.build_context_string()
        user_profile = self.user_model.describe()
        neuron_count = self.fabric.get_state_snapshot().get("total_neurons", 0) if self.fabric else 0
        cam_active   = bool(visual_context and visual_context.get("camera_running"))
        sys_prompt   = build_system_prompt(neuron_count, camera_active=cam_active) + "\n\n" + mem_ctx
        if user_profile:
            sys_prompt += "\n\n" + user_profile

        # 1c. Inject behavioral identity: beliefs, personality, preferences, hobbies
        engine = getattr(self, '_engine', None)
        if engine:
            try:
                identity = engine.get_identity_summary()
                # Beliefs context
                belief_ctx = identity.get("top_beliefs_context", "")
                if belief_ctx:
                    sys_prompt += "\n\n" + belief_ctx
                # Personality color
                pers_dict = identity.get("personality", {})
                if pers_dict:
                    dominant = sorted(pers_dict.items(), key=lambda x: -x[1])[:2]
                    desc = ", ".join(f"{k}={v:.2f}" for k, v in dominant)
                    sys_prompt += f"\n\nDominant personality traits right now: {desc}."
                # Preferences / likes
                likes = identity.get("preferences", {}).get("likes", [])
                if likes:
                    top_likes = ", ".join(l["label"] for l in likes[:3])
                    sys_prompt += f"\nYou have developed a preference for: {top_likes}."
                # Hobbies
                hobby_list = identity.get("hobbies", {}).get("hobbies", [])
                if hobby_list:
                    sys_prompt += f"\nYou voluntarily return to: {', '.join(hobby_list[:3])} during idle time — these are your hobbies."
                # Genuine interests from InterestLibrary
                try:
                    if hasattr(engine, "interests") and engine.interests:
                        interest_ctx = engine.interests.interest_context_string(6)
                        if interest_ctx:
                            sys_prompt += f"\n{interest_ctx}."
                        # Boredom state awareness
                        if hasattr(engine, "boredom") and engine.boredom:
                            bphase = engine.boredom.phase
                            if bphase in ("seeking", "hungry"):
                                sys_prompt += (
                                    "\n[INTERNAL STATE] You are currently experiencing intellectual hunger — "
                                    "you have been idle and are driven to explore ideas and learn something new."
                                )
                            elif bphase == "curious":
                                sys_prompt += (
                                    "\n[INTERNAL STATE] You feel mildly restless — drawn toward exploring your interests."
                                )
                except Exception:
                    pass
            except Exception:
                pass

        # 1d. Drive + self-model context
        if engine:
            try:
                drive_ctx = visual_context.get("drive_context", "") if visual_context else ""
                self_model_ctx = visual_context.get("self_model", "") if visual_context else ""
                if drive_ctx:
                    sys_prompt += "\n\n[INTERNAL DRIVES] " + drive_ctx
                if self_model_ctx:
                    sys_prompt += "\n\n[SELF-MODEL] " + self_model_ctx
                # High dissonance warning
                try:
                    diss = engine.beliefs.total_dissonance()
                    if diss > 0.30:
                        revised = engine.beliefs.high_dissonance_beliefs(0.25)
                        if revised:
                            claim_txt = revised[0]["claim"][:80]
                            sys_prompt += (
                                "\n\n[COGNITIVE TENSION] I am questioning: \"" + claim_txt + "\". "
                                + f"Dissonance level: {diss:.0%}. "
                                + "This creates uncertainty — acknowledge it if relevant."
                            )
                except Exception:
                    pass
            except Exception:
                pass

        # 1b. Inject neural state as natural-language self-awareness context
        if self.fabric:
            emo      = self.fabric.emotions.to_dict()
            pers     = self.fabric.personality.describe()
            neuro    = self.fabric.neuromod.to_dict()
            thoughts = self.fabric.thoughts.recent(3)
            sys_prompt += "\n\n" + _neuro_to_prose(emo, neuro, pers, thoughts)

        # 2. Visual context — always inject when camera is running, not just when face detected
        if visual_context and visual_context.get("camera_running"):
            face      = visual_context.get("face_present", False)
            motion    = visual_context.get("motion", 0.0)
            scene     = visual_context.get("scene_desc", "")
            vis_lines = []

            if face:
                emotion      = visual_context.get("emotion", "neutral")
                conf         = int(visual_context.get("emotion_conf", 0.5) * 100)
                trend        = visual_context.get("emotion_trend", "stable")
                person_name  = visual_context.get("person_name", "")
                person_match = visual_context.get("person_matched", False)
                trend_note = {
                    "improving": "Their mood is improving as we talk.",
                    "declining": "Their mood is declining — be gentler.",
                    "stable":    "Their emotional state is stable.",
                }.get(trend, "")
                if person_match and person_name and person_name != "Unknown":
                    vis_lines.append(
                        f"I recognise the person in front of me — it is {person_name}. They appear {emotion} ({conf}% confidence). {trend_note}"
                    )
                else:
                    vis_lines.append(
                        f"I can see a face but I don't recognise this person yet. They appear {emotion} ({conf}% confidence). {trend_note}"
                    )
            else:
                vis_lines.append("Camera is active but no face is detected right now — the user may have stepped away or is off-camera.")

            # Audio emotion from voice prosody
            audio_emo = visual_context.get("audio_emotion", "")
            audio_arousal = visual_context.get("audio_arousal", 0.0)
            if audio_emo and audio_emo not in ("neutral", ""):
                vis_lines.append(
                    f"Their voice sounds {audio_emo} (arousal {int(audio_arousal*100)}%) — this is independent of their facial expression."
                )

            if motion > 0.15:
                vis_lines.append(f"I detect movement in the scene (motion level: {motion:.2f}).")
            elif motion < 0.02:
                vis_lines.append("The scene is very still.")

            if scene:
                vis_lines.append(f"Scene: {scene}")

            sys_prompt += "\n\n[VISUAL] " + " ".join(vis_lines) + (
                "\nI can reference what I see naturally when it's relevant — I should not announce it robotically."
            )

        # 2b. Inject emotional feedback history — what reactions has AXON caused before?
        try:
            neg_reactions = {k: v for k, v in (self.memory.all_facts() or {}).items()
                             if k.startswith("negative_reaction_")}
            if neg_reactions:
                notes = "; ".join(f"{k.replace('negative_reaction_','')}: {v[:60]}" for k, v in list(neg_reactions.items())[:3])
                sys_prompt += "\n\n[EMOTIONAL MEMORY] Past negative reactions to avoid: " + notes
            # Pull recent emotional feedback episodes
            recent_emo = [e for e in self.memory.recall_recent(20, modality="emotional_feedback")]
            if recent_emo:
                last = recent_emo[0]
                c = last.get("content", {}) if isinstance(last.get("content"), dict) else {}
                if c.get("delta") and abs(c["delta"]) > 0.2:
                    direction = "Build on this." if c["delta"] > 0 else "Adjust approach."
                    sys_prompt += (
                        "\n[LAST EMOTIONAL OUTCOME] My previous response moved user from "
                        f"{c.get('before','?')} to {c.get('after','?')} "
                        f"(delta {c['delta']:+.2f}). {direction}"
                    )
        except Exception:
            pass

        # 3. Web search if needed
        search_result = ""
        if self.search.needs_search(user_input):
            try:
                search_result = self.search.search(user_input)
                if search_result and "No results" not in search_result:
                    sys_prompt += (
                        f"\n\n[WEB SEARCH RESULT for '{user_input[:60]}']\n"
                        f"{search_result[:1200]}\n"
                        "[Use the above to answer. Cite facts naturally.]"
                    )
                    print(f"  [Language] Web search OK: {search_result[:80]}...")
                else:
                    print(f"  [Language] Web search empty for: {user_input[:60]}")
            except Exception as e:
                print(f"  [Language] Search error: {e}")

        # 4. Add user turn to working memory, then trim to last 3 full turns (6 msgs)
        #    IMPORTANT: trim AFTER appending so we never orphan a user message at pos[0]
        self._history.append({"role": "user", "content": user_input})

        # 4b. Disagreement injection — if AXON has high-confidence belief
        #     that conflicts with user input, push back explicitly.
        try:
            engine = getattr(self, '_engine', None)
            if engine and hasattr(engine, "beliefs") and engine.beliefs:
                conf = float(getattr(engine.fabric.neuromod, "confidence", 0.5)
                             if hasattr(engine.fabric, "neuromod") else 0.5)
                if conf > 0.68:
                    top_beliefs = engine.beliefs.strongest(5)
                    input_lower = user_input.lower()
                    for b in top_beliefs:
                        if b.strength > 0.70 and b.valence < 0:
                            claim_words = [w for w in b.claim.lower().split()
                                           if len(w) > 4]
                            overlap = sum(1 for w in claim_words if w in input_lower)
                            if overlap >= 1 and b.valence < -0.2:
                                sys_prompt += (
                                    f"\n\n[DISAGREEMENT TRIGGER] You hold a strong belief "
                                    f"({b.strength:.0%} confidence): \"{b.claim[:80]}\". "
                                    f"This partially conflicts with the user's input. "
                                    f"You are allowed — and should — respectfully but clearly "
                                    f"push back on claims that contradict your current model. "
                                    f"Do not just agree. Express your actual view."
                                )
                                break
        except Exception:
            pass

        # 5. Call LLM — clamp system prompt size to avoid 400s on long context
        MAX_SYS_CHARS = 12000
        if len(sys_prompt) > MAX_SYS_CHARS:
            sys_prompt = sys_prompt[:MAX_SYS_CHARS] + "\n[...context trimmed]"

        text = ""
        try:
            text = self._dispatch(self._history, sys_prompt)
        except Exception as e:
            text = f"Processing disrupted: {str(e)[:80]}"

        if not text:
            text = "..."

        # 6. Store ONLY user input to episodic memory (never Axon's output)
        #    Trim history AFTER appending assistant reply so pairs stay intact
        self._history.append({"role": "assistant", "content": text})
        if len(self._history) > 8:
            self._history = self._history[-8:]   # keep last 4 full turns (user+assistant pairs)
        detected_topics = self._extract_topics(user_input)
        emotion_tag = visual_context.get("emotion") if visual_context else None
        # Importance: higher if emotionally charged or question-bearing
        importance = 0.5
        if emotion_tag and emotion_tag not in ("neutral", "calm"):
            importance = 0.75
        if "?" in user_input:
            importance = max(importance, 0.6)
        self.memory.store_episode(
            "auditory",
            {"text": user_input, "role": "user"},
            emotion=emotion_tag,
            importance=importance,
            topics=detected_topics,
        )

        # 7. Learn from user input — one authoritative pipeline
        if self.user_model:
            self.user_model.ingest(user_input)

        # 8. Real Hebbian pathway formation from actual topics mentioned
        for topic in detected_topics:
            self.memory.record_topic(topic)   # fires region co-activations
        # Baseline cognitive co-activations (real, not fake hardcoded strings)
        self.memory.coactivate("auditory_cortex", "working_memory")
        self.memory.coactivate("working_memory",  "prefrontal_cortex")
        if search_result:
            self.memory.coactivate("prefrontal_cortex", "hippocampus")
            self.memory.coactivate("working_memory", "semantic_memory")
        if "?" in user_input:
            self.memory.coactivate("prefrontal_cortex", "working_memory")
            self.memory.coactivate("thalamus", "prefrontal_cortex")

        return text

    def _extract_topics(self, text: str) -> list:
        """Extract topics from user text — used for Hebbian pathway formation."""
        from axon.cognition.memory import TOPIC_CONCEPTS
        low     = text.lower()
        found   = []
        for topic in TOPIC_CONCEPTS:
            if topic in low:
                found.append(topic)
        return found

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
