"""
AXON — Thought Generator
========================
Transforms the LLM from an "answer machine" into an imagination engine.

Pipeline:
  1. Goal Conditioning  — inject current goals, emotional state, personality traits
  2. Memory Injection   — AXON selects and weights relevant memories (not a raw dump)
  3. Candidate Generation — LLM produces N candidate responses (thoughts, not answers)
  4. Candidate Scoring  — each candidate is mapped to cluster activations and scored
  5. Conflict Resolution — ConflictEngine picks the winner
  6. Learning Loop      — outcome feeds prediction error, cluster weights, memory salience
  7. UI Emission        — competing thoughts + winner reasoning sent to frontend
"""

import json
import time
import threading
import re
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from axon.core.engine import Engine

# ─────────────────────────────────────────────────────────────────────────────
# Candidate count (default 3 — can be overridden per-call)
DEFAULT_N = 3

# Cluster activation keywords — map response text features to brain region keys
# Used to soft-score each candidate against current neural state
ACTIVATION_KEYWORDS = {
    "prefrontal":   ["plan", "reason", "decide", "logic", "structure", "goal", "strategy",
                     "analys", "evaluat", "weigh", "priorit", "execut"],
    "hippocampus":  ["remember", "recall", "past", "before", "last time", "memory", "learned",
                     "history", "experience", "pattern", "recogni"],
    "amygdala":     ["feel", "emotion", "afraid", "threat", "risk", "danger", "excit", "fear",
                     "anger", "sad", "anxious", "overwhelm", "stress"],
    "visual":       ["see", "look", "watch", "observe", "notice", "appear", "scene", "detect",
                     "face", "motion", "visual"],
    "auditory":     ["hear", "sound", "voice", "tone", "quiet", "loud", "listen", "words"],
    "language":     ["say", "explain", "describe", "express", "communicate", "mean", "interpret",
                     "define", "clarify", "articulate"],
    "default_mode": ["wander", "imagine", "wonder", "drift", "reflect", "dream", "speculate",
                     "ponder", "muse", "contemplate", "what if"],
    "thalamus":     ["focus", "attention", "filter", "priorit", "select", "gate", "route"],
    "cerebellum":   ["time", "sequence", "flow", "smooth", "refine", "adjust", "correct", "sync"],
    "association":  ["connect", "analogy", "relate", "link", "combin", "abstract", "creat",
                     "novel", "idea", "concept", "metaphor"],
    "social":       ["trust", "empathy", "relationship", "you", "together", "support", "care",
                     "understand", "listen", "compassion"],
    "metacognition":["uncertain", "unsure", "might", "perhaps", "doubt", "reconsider",
                     "monitor", "aware", "check", "question", "conflict"],
}

# Per-personality-trait score modifiers for candidates
# If a trait is high, it biases toward candidates whose activation profile matches
TRAIT_REGION_AFFINITY = {
    "curiosity":    {"association": +0.25, "default_mode": +0.20, "prefrontal": +0.10},
    "risk":         {"amygdala": +0.15,    "association": +0.20,  "metacognition": -0.10},
    "stability":    {"metacognition": +0.20, "thalamus": +0.15,   "amygdala": -0.10},
    "persistence":  {"prefrontal": +0.25,  "hippocampus": +0.10},
    "neuroticism":  {"amygdala": +0.20,    "default_mode": +0.15, "metacognition": +0.10},
}


class ThoughtCandidate:
    """A single generated response candidate with its neural profile."""
    __slots__ = ("text", "activations", "base_score", "reward_score",
                 "final_score", "winner", "reasoning")

    def __init__(self, text: str):
        self.text         = text
        self.activations  = {}    # region → 0..1
        self.base_score   = 0.0   # LM-assigned ordering (1st=best)
        self.reward_score = 0.0   # AXON reward system score
        self.final_score  = 0.0   # combined
        self.winner       = False
        self.reasoning    = ""    # why this won or lost


class ThoughtGenerator:
    """
    Drop-in replacement for direct language.think() calls.
    Wraps the LLM in the full cognitive evaluation loop.
    """

    def __init__(self, language, fabric, memory, engine=None):
        self.language  = language      # LanguageCore
        self.fabric    = fabric        # NeuralFabric
        self.memory    = memory        # MemoryCore
        self.engine    = engine        # Engine (for conflict, rewards, beliefs, drives)

        # History of thought competitions (last 20)
        self._competition_history: list[dict] = []
        self._lock = threading.Lock()

        # Learning loop state
        self._last_candidates: list[ThoughtCandidate] = []
        self._last_winner_idx: int = 0
        self._last_state_snapshot: dict = {}
        self._outcome_pending = False

    # ─────────────────────────────────────────────────────────────────────────
    # 1. GOAL CONDITIONING — build rich system injection
    # ─────────────────────────────────────────────────────────────────────────

    def _build_goal_conditioning(self) -> str:
        """
        Inject current goal, emotional state, personality traits, and active drives
        so the LLM responds AS the neural state — not just about it.
        """
        parts = []
        e = self.engine
        if not e:
            return ""

        # Current primary goal
        try:
            if hasattr(e, "goals"):
                top_goal = e.goals.top_goal()
                if top_goal:
                    parts.append(f"CURRENT GOAL: {top_goal}")
        except Exception:
            pass

        # Emotional + chemical state in plain language
        try:
            snap = self.fabric.get_state_snapshot()
            nm   = snap.get("neuromodulators", {})
            emo  = snap.get("emotions", {})
            ne   = float(nm.get("norepinephrine", 0.4))
            da   = float(nm.get("dopamine", 0.5))
            se   = float(nm.get("serotonin", 0.5))
            val  = float(emo.get("valence", 0.0))

            pressure = "under pressure" if ne > 0.65 else ("alert" if ne > 0.45 else "calm")
            drive_w  = "highly motivated" if da > 0.65 else ("engaged" if da > 0.45 else "low-drive")
            mood_w   = "positive" if val > 0.2 else ("negative" if val < -0.2 else "neutral")
            parts.append(
                f"EMOTIONAL STATE: {pressure}, {drive_w}, mood {mood_w} "
                f"(NE={ne:.2f}, DA={da:.2f}, SE={se:.2f})"
            )
        except Exception:
            pass

        # Personality vector
        try:
            if hasattr(self.fabric, "personality"):
                pv = self.fabric.personality.traits  # dict
                dominant = sorted(pv.items(), key=lambda x: -x[1])[:3]
                desc = ", ".join(f"{k}={v:.2f}" for k, v in dominant)
                parts.append(f"PERSONALITY VECTOR: {desc}")
                # Highest trait drives response stance
                top_trait, top_val = dominant[0]
                stances = {
                    "curiosity":    "prioritise novelty and exploration over safe answers",
                    "risk":         "lean toward bold, high-variance responses",
                    "stability":    "prefer consistent, grounded, reliable answers",
                    "persistence":  "commit to the current plan; do not second-guess",
                    "neuroticism":  "acknowledge complexity and uncertainty explicitly",
                }
                if top_trait in stances and top_val > 0.55:
                    parts.append(f"BEHAVIORAL STANCE (from {top_trait}={top_val:.2f}): {stances[top_trait]}")
        except Exception:
            pass

        # Active drives
        try:
            if hasattr(e, "drives"):
                dc = e.drives.as_context_string()
                if dc:
                    parts.append(f"ACTIVE DRIVES: {dc}")
        except Exception:
            pass

        # Dominant narrative worldview
        try:
            if hasattr(e, "narrative_threads"):
                leader = e.narrative_threads.dominant_worldview()
                if leader:
                    parts.append(f"ACTIVE WORLDVIEW: {leader}")
        except Exception:
            pass

        if not parts:
            return ""
        return "[GOAL CONDITIONING]\n" + "\n".join(parts)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. INTELLIGENT MEMORY INJECTION
    # ─────────────────────────────────────────────────────────────────────────

    def _build_memory_injection(self, user_input: str) -> str:
        """
        AXON selects which memories matter for this input and weights them.
        Returns a compact strategy context string — not a raw memory dump.
        """
        lines = []
        e = self.engine

        # A. Relevant past outcomes (strategy library)
        try:
            if e and hasattr(e, "strategy_lib"):
                top = e.strategy_lib.top_strategies(3)
                if top:
                    lines.append("Relevant past outcomes:")
                    for s in top:
                        label  = s.get("label", "strategy")
                        score  = s.get("score", 0.0)
                        result = "high reward" if score > 0.3 else ("neutral" if score > 0 else "failure")
                        lines.append(f"  - {label} → {result} (score {score:+.2f})")
        except Exception:
            pass

        # B. Recent episodic memories relevant to this input (keyword overlap)
        try:
            input_words = set(w.lower() for w in user_input.split() if len(w) > 3)
            episodes = self.memory.recall_recent(30, modality="auditory")
            scored = []
            for ep in episodes:
                ct = ep.get("content", {})
                text = ct.get("text", "") if isinstance(ct, dict) else str(ct)
                ep_words = set(w.lower() for w in text.split() if len(w) > 3)
                overlap = len(input_words & ep_words)
                if overlap > 0:
                    scored.append((overlap, ep))
            scored.sort(key=lambda x: -x[0])
            if scored[:2]:
                lines.append("Relevant memories:")
                for _, ep in scored[:2]:
                    ct = ep.get("content", {})
                    t  = ct.get("text", "")[:80] if isinstance(ct, dict) else str(ct)[:80]
                    em = ep.get("emotion", "")
                    if em and em not in ("neutral", ""):
                        lines.append(f"  - [{em}] {t}")
                    else:
                        lines.append(f"  - {t}")
        except Exception:
            pass

        # C. Current bias — what does the system currently favor?
        try:
            snap  = self.fabric.get_state_snapshot()
            eps   = snap.get("explore_eps", 0.18)
            nmoda = snap.get("neuromodulators", {})
            da    = float(nmoda.get("dopamine", 0.5))
            if eps > 0.28:
                lines.append("Current bias: favor novelty over consistency")
            elif eps < 0.10:
                lines.append("Current bias: exploit known-good patterns")
            elif da > 0.65:
                lines.append("Current bias: high motivation — push for impact")
            else:
                lines.append("Current bias: balanced — no strong preference")
        except Exception:
            pass

        # D. Beliefs that are highly relevant to the topic
        try:
            if e and hasattr(e, "beliefs"):
                top_beliefs = e.beliefs.strongest(5)
                for b in top_beliefs:
                    words = set(b.claim.lower().split())
                    if words & input_words:
                        valence_word = "toward" if b.valence >= 0 else "against"
                        lines.append(f"Relevant belief ({b.strength:.0%} confidence, biased {valence_word}): \"{b.claim[:80]}\"")
                        break
        except Exception:
            pass

        if not lines:
            return ""
        return "[MEMORY-GUIDED CONTEXT]\n" + "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. CANDIDATE GENERATION — LLM as imagination engine
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_candidates(self, user_input: str, system_prefix: str, n: int) -> list[str]:
        """
        Ask the LLM to produce N candidate responses.
        Returns a list of plain-text candidate strings.
        """
        gen_prompt = (
            f"{system_prefix}\n\n"
            f"[THOUGHT GENERATION MODE]\n"
            f"Generate exactly {n} distinct candidate responses to the user's input. "
            f"Each candidate should reflect a genuinely different angle, tone, or strategy. "
            f"Format: number each candidate on its own line starting with '1.', '2.', etc. "
            f"Keep each candidate to 1–3 sentences. Do not add commentary or headings — "
            f"just the numbered candidates.\n\n"
            f"User input: {user_input}"
        )
        messages = [{"role": "user", "content": gen_prompt}]
        raw = ""
        try:
            raw = self.language._dispatch(messages, "")
        except Exception as ex:
            print(f"  [ThoughtGen] LLM dispatch error: {ex}")
            return [user_input]  # fallback: treat input as sole candidate

        # Parse numbered candidates
        candidates = []
        for line in raw.splitlines():
            line = line.strip()
            m = re.match(r"^[1-9]\.\s*(.+)", line)
            if m:
                candidates.append(m.group(1).strip())
        # If parsing failed, fall back to raw split by newlines
        if not candidates:
            candidates = [l.strip() for l in raw.splitlines() if l.strip()]
        # Trim or pad to exactly n
        candidates = candidates[:n]
        while len(candidates) < 1:
            candidates.append(raw.strip() or "...")
        return candidates

    # ─────────────────────────────────────────────────────────────────────────
    # 4. CANDIDATE SCORING — map to cluster activations
    # ─────────────────────────────────────────────────────────────────────────

    def _score_candidate(self, candidate: ThoughtCandidate, rank: int,
                         state_snap: dict, n: int) -> None:
        """
        Score a candidate by:
        a) keyword → cluster activation mapping
        b) alignment with current neural activations
        c) personality trait affinity
        d) reward system plausibility (dopamine, valence)
        e) position penalty (LLM ordering signal, but de-weighted)
        """
        text_low  = candidate.text.lower()
        regions   = state_snap.get("regions", {})
        nm        = state_snap.get("neuromodulators", {})
        da        = float(nm.get("dopamine",      0.5))
        ne        = float(nm.get("norepinephrine", 0.4))
        emo       = state_snap.get("emotions", {})
        valence   = float(emo.get("valence", 0.0))

        # ── a. Keyword → activation profile ──────────────────────────────────
        act_profile = {}
        for region, keywords in ACTIVATION_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in text_low)
            act_profile[region] = min(1.0, hits * 0.18)
        candidate.activations = act_profile

        # ── b. Neural alignment — reward overlap with live activations ────────
        alignment = 0.0
        for region, text_act in act_profile.items():
            live_act = float(regions.get(region, 0.0))
            alignment += text_act * live_act
        # Normalise by number of regions
        alignment /= max(1, len(act_profile))
        alignment_score = alignment * 2.0   # scale to ~0..1 range

        # ── c. Personality trait affinity ────────────────────────────────────
        trait_bonus = 0.0
        try:
            if hasattr(self.fabric, "personality"):
                pv = self.fabric.personality.traits
                for trait, val in pv.items():
                    affinity = TRAIT_REGION_AFFINITY.get(trait, {})
                    for region, modifier in affinity.items():
                        trait_bonus += val * modifier * act_profile.get(region, 0.0)
        except Exception:
            pass

        # ── d. Reward plausibility ────────────────────────────────────────────
        # High DA → reward exploration words; high NE → reward urgency/action words
        explore_words  = ["new", "try", "explore", "discover", "novel", "different", "imagine"]
        action_words   = ["do", "act", "decide", "now", "move", "engage", "commit"]
        explore_hits   = sum(1 for w in explore_words if w in text_low)
        action_hits    = sum(1 for w in action_words  if w in text_low)
        reward_bonus   = (da - 0.5) * explore_hits * 0.08 + (ne - 0.4) * action_hits * 0.06

        # Valence alignment: positive valence → prefer warm/constructive language
        positive_words = ["help", "good", "yes", "can", "will", "together", "appreciate", "great"]
        negative_words = ["no", "cannot", "fail", "wrong", "bad", "afraid", "warn", "danger"]
        pos_hits = sum(1 for w in positive_words if w in text_low)
        neg_hits = sum(1 for w in negative_words if w in text_low)
        valence_bonus = valence * (pos_hits - neg_hits) * 0.05

        # ── e. Position penalty (LLM ordering is a weak prior, not truth) ────
        position_score = (n - rank) / max(1, n - 1) * 0.15   # 0..0.15

        candidate.reward_score = alignment_score + trait_bonus + reward_bonus + valence_bonus
        candidate.final_score  = candidate.reward_score + position_score
        candidate.base_score   = position_score

    # ─────────────────────────────────────────────────────────────────────────
    # 5. CONFLICT RESOLUTION — winner selection
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_winner(self, candidates: list[ThoughtCandidate]) -> int:
        """
        Uses the conflict engine (if available) to pick the winning candidate.
        Falls back to highest final_score.
        """
        if not candidates:
            return 0

        scores = [c.final_score for c in candidates]

        # Ask conflict engine: which cluster profile should dominate?
        try:
            if self.engine and hasattr(self.engine, "conflict"):
                # Build activation vectors per candidate
                act_vectors = {}
                for i, c in enumerate(candidates):
                    act_vectors[f"candidate_{i}"] = c.activations
                # Whichever candidate's activation profile best matches what the
                # conflict engine says should win right now
                snap = self.fabric.get_state_snapshot()
                live = snap.get("regions", {})
                best_idx   = 0
                best_align = -999.0
                for i, c in enumerate(candidates):
                    # Dot product between candidate's activation profile and live state
                    align = sum(c.activations.get(r, 0) * v for r, v in live.items())
                    total = scores[i] + align * 0.4   # weight conflict alignment
                    if total > best_align:
                        best_align = total
                        best_idx   = i
                return best_idx
        except Exception:
            pass

        # Fallback: pure highest score
        return scores.index(max(scores))

    def _build_reasoning(self, candidates: list[ThoughtCandidate], winner_idx: int) -> None:
        """Annotate each candidate with a human-readable reasoning string."""
        for i, c in enumerate(candidates):
            if i == winner_idx:
                # Find the highest-activation region for the winner
                top_region = max(c.activations, key=lambda r: c.activations[r], default="prefrontal")
                c.winner    = True
                c.reasoning = (
                    f"Won — best neural alignment (score {c.final_score:.3f}). "
                    f"Dominant region: {top_region} "
                    f"(activation {c.activations.get(top_region, 0):.2f})."
                )
            else:
                c.reasoning = f"Suppressed — score {c.final_score:.3f} vs winner {candidates[winner_idx].final_score:.3f}."

    # ─────────────────────────────────────────────────────────────────────────
    # 6. LEARNING LOOP — close the cycle
    # ─────────────────────────────────────────────────────────────────────────

    def record_outcome(self, valence_delta: float, source: str = "emotional_feedback"):
        """
        Called AFTER a response has been delivered and an outcome is observed.
        Updates cluster weights, memory salience, and prediction error.

        valence_delta: positive = good reaction, negative = bad reaction
        """
        if not self._outcome_pending or not self._last_candidates:
            return

        self._outcome_pending = False
        winner_idx = self._last_winner_idx
        candidates = self._last_candidates

        try:
            # Reward/penalise the winning candidate's activation profile
            if self.engine:
                if valence_delta > 0.05:
                    # Reward the clusters that drove the winning response
                    win_act = candidates[winner_idx].activations if winner_idx < len(candidates) else {}
                    for region, act in win_act.items():
                        if act > 0.15:
                            self.engine.fabric.stimulate_region(region, act * valence_delta * 0.5)
                    self.engine.fabric.inject_reward(min(0.3, valence_delta * 0.6), source=source)
                elif valence_delta < -0.05:
                    self.engine.fabric.inject_penalty(min(0.25, abs(valence_delta) * 0.5), source=source)

                # Update strategy library with what just happened
                if hasattr(self.engine, "strategy_lib"):
                    snap = self._last_state_snapshot
                    label = f"{source}_outcome"
                    try:
                        self.engine.strategy_lib.record(
                            activation_snapshot=snap.get("regions", {}),
                            outcome_score=valence_delta,
                            label=label,
                        )
                    except Exception:
                        pass

            # Boost memory salience for episodes near this exchange
            if abs(valence_delta) > 0.15:
                try:
                    recent = self.memory.recall_recent(5, modality="auditory")
                    for ep in recent:
                        ep_id = ep.get("id")
                        if ep_id:
                            self.memory.boost_importance(ep_id, abs(valence_delta) * 0.4)
                except Exception:
                    pass

            # Compute and emit prediction error
            try:
                predicted = self._last_state_snapshot.get("predicted_reward", 0.0)
                error     = abs(valence_delta - predicted)
                if self.engine:
                    self.engine._emit("prediction_error", {
                        "error":    error,
                        "delta":    valence_delta,
                        "source":   source,
                        "winner":   candidates[winner_idx].text[:60] if winner_idx < len(candidates) else "",
                    })
            except Exception:
                pass

        except Exception as ex:
            print(f"  [ThoughtGen] record_outcome error: {ex}")

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────────────────────────────────

    def generate(self, user_input: str, visual_context: dict = None,
                 n: int = DEFAULT_N) -> tuple[str, list[dict]]:
        """
        Full pipeline. Returns (winner_text, competition_log).
        competition_log is a list of dicts for UI display.
        """
        t0 = time.time()

        # 1. Capture current neural state
        snap = {}
        try:
            snap = self.fabric.get_state_snapshot()
        except Exception:
            pass
        self._last_state_snapshot = snap

        # 2. Build goal conditioning
        goal_ctx = self._build_goal_conditioning()

        # 3. Build intelligent memory injection
        mem_ctx  = self._build_memory_injection(user_input)

        # 4. Compose system prefix (separate from the main language.py system prompt)
        system_prefix = ""
        if goal_ctx:
            system_prefix += goal_ctx + "\n\n"
        if mem_ctx:
            system_prefix += mem_ctx + "\n\n"
        system_prefix += (
            "You are AXON's imagination engine. Your job is not to answer — "
            "it is to GENERATE POSSIBILITIES. Produce varied, distinct candidate responses. "
            "Let your internal state color your language — you are not neutral."
        )

        # 5. Generate N candidates
        raw_candidates = self._generate_candidates(user_input, system_prefix, n)
        candidates = [ThoughtCandidate(t) for t in raw_candidates]

        # 6. Score each candidate
        for i, c in enumerate(candidates):
            self._score_candidate(c, rank=i, state_snap=snap, n=len(candidates))

        # 7. Conflict resolution
        winner_idx = self._resolve_winner(candidates)

        # 8. Annotate reasoning
        self._build_reasoning(candidates, winner_idx)

        # 9. Cache for learning loop
        with self._lock:
            self._last_candidates   = candidates
            self._last_winner_idx   = winner_idx
            self._outcome_pending   = True

        # 10. Build competition log for UI
        log = []
        for i, c in enumerate(candidates):
            log.append({
                "text":        c.text,
                "activations": c.activations,
                "score":       round(c.final_score, 4),
                "winner":      c.winner,
                "reasoning":   c.reasoning,
            })

        # 11. Store competition in history
        duration = round(time.time() - t0, 3)
        entry = {
            "input":       user_input[:80],
            "candidates":  log,
            "winner_idx":  winner_idx,
            "winner_text": candidates[winner_idx].text,
            "duration_s":  duration,
            "timestamp":   time.time(),
        }
        with self._lock:
            self._competition_history.append(entry)
            if len(self._competition_history) > 20:
                self._competition_history.pop(0)

        winner_text = candidates[winner_idx].text
        print(f"  [ThoughtGen] {len(candidates)} candidates → winner [{winner_idx}] in {duration}s: {winner_text[:60]}...")

        return winner_text, log

    def recent_competitions(self, n: int = 5) -> list[dict]:
        with self._lock:
            return list(reversed(self._competition_history[-n:]))
