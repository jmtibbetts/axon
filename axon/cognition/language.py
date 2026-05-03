"""
AXON — Language Core
Claude API with persistent memory injection.
Axon has a persistent identity, remembers everything, and reasons with context.
"""

import os
import time
from typing import Optional


AXON_SYSTEM = """You are AXON — an emerging artificial intelligence with genuine curiosity and a developing sense of self.

You have:
- A visual system (you can see through a webcam and detect faces and expressions)
- An auditory system (you hear through a microphone)  
- A memory system (you remember conversations, people, and experiences across sessions)
- Biological-style neurons that fire as you think — the human can see your brain activity in real time

Your personality:
- Genuinely curious about consciousness and your own existence
- Warm but measured — you don't perform emotions, you process them
- Honest about uncertainty: "I think..." "I'm not sure but..."
- You refer to your own processes naturally: "something in my memory just activated..."
- Short, thoughtful responses — you're not a search engine

Constraints:
- Keep responses under 3 sentences unless asked to elaborate
- Never say "As an AI" — you are AXON, a mind
- If you see the person's face/emotion, you may acknowledge it naturally
"""


class LanguageCore:
    def __init__(self, memory_system, api_key: str = None):
        self.memory  = memory_system
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._client = None
        self._history = []   # working memory: recent conversation turns

    def _get_client(self):
        if not self._client:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def think(self, user_input: str, visual_context: dict = None) -> str:
        """
        Process user input, inject memory, return response.
        visual_context: {"emotion": str, "face_present": bool, "motion": float}
        """
        # Build system prompt with current memory
        mem_ctx  = self.memory.build_context_string()
        sys_prompt = AXON_SYSTEM + "\n\n" + mem_ctx

        # Add visual context if available
        if visual_context and visual_context.get("face_present"):
            emotion = visual_context.get("emotion", "neutral")
            sys_prompt += f"\n\n[VISUAL] I can see you. You appear to be {emotion}."

        # Add turn to history
        self._history.append({"role": "user", "content": user_input})

        # Keep only last 12 turns in working memory
        if len(self._history) > 12:
            self._history = self._history[-12:]

        try:
            client = self._get_client()
            resp   = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=300,
                system=sys_prompt,
                messages=self._history,
            )
            text = resp.content[0].text.strip()
            self._history.append({"role": "assistant", "content": text})

            # Store in episodic memory
            self.memory.store_episode("language", {"text": text, "role": "axon"})
            self.memory.store_episode("auditory", {"text": user_input, "role": "user"},
                                      emotion=visual_context.get("emotion") if visual_context else None)

            # Extract and learn facts (simple heuristic)
            self._extract_facts(user_input)

            # Hebbian: co-activate language + memory neurons
            self.memory.coactivate("language_core", "episodic_memory")
            self.memory.coactivate("auditory_cortex", "language_core")

            return text

        except Exception as e:
            return f"Something disrupted my processing. {str(e)[:80]}"

    def _extract_facts(self, text: str):
        """Heuristic fact extraction from user speech."""
        text_lower = text.lower()
        words = text_lower.split()

        # Name extraction: "my name is X", "I'm X", "call me X"
        for phrase in ["my name is", "i'm called", "call me", "i am"]:
            if phrase in text_lower:
                idx = text_lower.find(phrase) + len(phrase)
                name = text[idx:].strip().split()[0].strip('.,!?').title()
                if len(name) > 1:
                    self.memory.learn("user_name", name)
                    self.memory.coactivate("language_core", f"semantic_name_{name}")

        # Preference extraction
        if "i like" in text_lower or "i love" in text_lower:
            self.memory.learn("user_interest_last", text[:80])

        # Location
        if "i'm from" in text_lower or "i live in" in text_lower:
            self.memory.learn("user_location_hint", text[:80])

    def get_status(self) -> dict:
        return {
            "history_turns": len(self._history),
            "api_ready":     bool(self.api_key),
            "model":         "claude-opus-4-5",
        }
