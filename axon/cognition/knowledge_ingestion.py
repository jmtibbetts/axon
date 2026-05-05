"""
AXON — Knowledge Ingestion Pipeline
text → concepts → simulated experiences → memory entries + belief updates

This is NOT a RAG system. We do not just store and retrieve.
The pipeline converts knowledge into internal "experiences" that the
system can use for decision-making, the same way lived events are used.

Pipeline:
  1. Chunk text into ~200-word segments
  2. Extract concepts + valence per chunk (lightweight heuristics + keyword matching)
  3. Convert each concept into a simulated scenario:
       { context, expected_outcome, confidence, valence }
  4. Store as semantic memory + optionally assert/challenge beliefs
  5. Optionally run through neural fabric stimulation (the concept activates
     relevant clusters just like a real experience would)

Design principle:
  The system can DISAGREE with ingested knowledge if its existing
  beliefs + experience contradict it. Confidence is never set to 1.0
  from external sources alone.
"""

import re
import json
import time
import math
import threading
from pathlib import Path
from typing import List, Dict, Optional, Callable

# Valence lexicon — simple but effective for concept extraction
_POS_WORDS = {
    "success","growth","reward","pleasure","joy","curiosity","love","win",
    "achieve","benefit","gain","thrive","flourish","learn","discover",
    "create","connect","understand","improve","persist","effort","mastery",
    "confidence","trust","hope","progress","insight","wisdom","clarity",
    "strength","resilience","opportunity","creativity","freedom","health",
    "happiness","fulfillment","purpose","meaning","connection","empathy",
}
_NEG_WORDS = {
    "failure","loss","pain","fear","anger","stress","regret","mistake",
    "conflict","harm","danger","threat","anxiety","uncertainty","confusion",
    "decay","weakness","isolation","rejection","frustration","grief","shame",
    "punishment","cost","risk","damage","struggle","suffer","obstacle",
}
_CONCEPT_PATTERNS = [
    # "X leads to Y", "X causes Y", "X results in Y"
    r"(\w[\w\s]{2,25})\s+(?:leads?|leads? to|causes?|results? in|produces?|creates?)\s+([\w\s]{2,30})",
    # "Y comes from X", "Y is caused by X"
    r"([\w\s]{2,25})\s+(?:comes? from|is caused by|stems? from|is the result of)\s+([\w\s]{2,30})",
    # "People who X tend to Y"
    r"(?:people|those|systems?|agents?)\s+who\s+([\w\s]{2,25})\s+tend to\s+([\w\s]{2,30})",
]


def _chunk_text(text: str, max_words: int = 180) -> List[str]:
    """Split into ~max_words chunks at sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current, count = [], [], 0
    for sent in sentences:
        words = len(sent.split())
        if count + words > max_words and current:
            chunks.append(" ".join(current))
            current, count = [sent], words
        else:
            current.append(sent)
            count += words
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.split()) >= 10]


def _extract_valence(text: str) -> float:
    """Fast heuristic valence score for a chunk of text. Range -1 to 1."""
    words_lower = re.findall(r'\b\w+\b', text.lower())
    pos = sum(1 for w in words_lower if w in _POS_WORDS)
    neg = sum(1 for w in words_lower if w in _NEG_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 3)


def _extract_concepts(chunk: str) -> List[Dict]:
    """
    Extract causal concept pairs from text.
    Returns list of {context, outcome, valence, confidence}
    """
    concepts = []
    for pattern in _CONCEPT_PATTERNS:
        for match in re.finditer(pattern, chunk, re.IGNORECASE):
            grps = match.groups()
            if len(grps) >= 2:
                context = grps[0].strip().lower()
                outcome = grps[1].strip().lower()
                val     = _extract_valence(outcome)
                if val == 0.0:
                    val = _extract_valence(chunk)
                concepts.append({
                    "context":  context,
                    "outcome":  outcome,
                    "valence":  val,
                    "confidence": 0.55,   # external source: never fully trusted
                })

    # Fallback: treat whole chunk as one concept if nothing matched
    if not concepts:
        words = chunk.split()
        summary = " ".join(words[:12]) + ("..." if len(words) > 12 else "")
        concepts.append({
            "context":  summary,
            "outcome":  "general",
            "valence":  _extract_valence(chunk),
            "confidence": 0.40,
        })

    return concepts


def _concept_to_belief_key(concept: Dict) -> str:
    """Derive a canonical belief key from a concept."""
    ctx = re.sub(r'[^a-z0-9]', '_', concept["context"][:30].lower()).strip('_')
    return ctx




def _extract_interpretation(chunk: str, concepts: List[Dict],
                             credibility: float = 0.6) -> Dict:
    """
    Form an OPINION about the chunk — not just what it says, but what
    the system thinks about it: agreement, relevance, novelty, valence.

    This is the step where ingestion stops being a database operation
    and becomes something that has *takes*.

    Returns an interpretation dict:
    {
        "claim":      str,    — the core assertion being evaluated
        "confidence": float,  — how confident the system is in this reading
        "valence":    float,  — positive/negative assessment of the idea
        "novelty":    float,  — how new this seems (no prior concepts like it)
        "agreement":  float,  — does this align with existing patterns?
        "dissonance": float,  — how much does it conflict with assumed beliefs?
    }
    """
    if not chunk:
        return {}

    # Derived valence — full chunk tone
    chunk_valence = _extract_valence(chunk)

    # Novelty proxy: how many unique non-common words?
    common_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
                    "but", "in", "on", "at", "to", "for", "of", "with", "this",
                    "that", "it", "be", "as", "by", "from", "not", "have", "has"}
    words = [w.lower() for w in re.findall(r'\b\w{4,}\b', chunk)]
    unique_ratio = len(set(words) - common_words) / max(1, len(words))
    novelty_score = min(1.0, unique_ratio * 1.4)

    # Confidence: proportional to how many concepts were explicitly extracted
    # (heuristic: more causal patterns = more structured = more confident)
    explicit_concepts = [c for c in concepts if c.get("outcome") != "general"]
    confidence = min(0.90, 0.35 + len(explicit_concepts) * 0.10) * credibility

    # Agreement bias: if chunk is very positive, the system leans toward agreement;
    # if it's negative about something the system values, mild disagreement
    agreement = 0.5 + chunk_valence * 0.2   # 0.3–0.7 range naturally

    # Main claim: use first concept's context, or first sentence
    if explicit_concepts:
        ctx     = explicit_concepts[0]["context"]
        outcome = explicit_concepts[0]["outcome"]
        claim   = f"{ctx.capitalize()} leads to or causes {outcome}."
    else:
        sentences = re.split(r'(?<=[.!?])\s+', chunk.strip())
        claim = sentences[0][:120] if sentences else chunk[:120]

    return {
        "claim":      claim,
        "confidence": round(confidence, 3),
        "valence":    round(chunk_valence, 3),
        "novelty":    round(novelty_score, 3),
        "agreement":  round(agreement, 3),
        "dissonance": 0.0,   # filled in after belief.integrate() returns
    }

class KnowledgeIngestionPipeline:
    """
    Main entry point. Processes text and updates the system's internal state.

    Parameters:
        memory_system    — MemorySystem instance (stores semantic facts)
        belief_system    — BeliefSystem instance (updates beliefs)
        on_concept       — optional callback(concept_dict) called per concept
                           so the neural fabric can stimulate relevant clusters
    """

    def __init__(self, memory_system, belief_system,
                 on_concept: Optional[Callable] = None,
                 mem_hierarchy=None):
        self._mem          = memory_system
        self._beliefs      = belief_system
        self._on_concept   = on_concept
        self._mem_hierarchy = mem_hierarchy   # MemoryHierarchy (optional)
        self._lock    = threading.Lock()
        self._ingestion_log: List[dict] = []   # last N ingestion summaries

    def ingest(self, text: str, source_label: str = "external",
               credibility: float = 0.6) -> dict:
        """
        Full pipeline. Returns summary dict.
        credibility: 0–1, how much to trust this source vs. existing beliefs.
        """
        with self._lock:
            start          = time.time()
            interpretations: List[dict] = []
            chunks  = _chunk_text(text)
            total_concepts = 0
            beliefs_updated = 0
            memories_stored = 0
            concept_list    = []

            total_dissonance = 0.0

            for chunk in chunks:
                chunk_valence = _extract_valence(chunk)
                concepts      = _extract_concepts(chunk)

                # ── INTERPRETATION: form an opinion about this chunk ──────
                interpretation = _extract_interpretation(chunk, concepts, credibility)
                interpretations.append(interpretation)

                for concept in concepts:
                    total_concepts += 1
                    concept_list.append(concept)

                    # 1. Store as semantic memory
                    key   = _concept_to_belief_key(concept)
                    value = (f"context: {concept['context']} → "
                             f"outcome: {concept['outcome']} "
                             f"(valence {concept['valence']:+.2f})")
                    self._mem.store_semantic(value, "knowledge", confidence=concept["confidence"])
                    memories_stored += 1

                    # 2. Assert or challenge existing belief
                    existing = self._beliefs.get(key)
                    if existing:
                        self._beliefs.challenge(key, concept["valence"], credibility)
                    else:
                        self._beliefs.assert_belief(
                            key,
                            f"{concept['context'].capitalize()} tends to lead to {concept['outcome']}.",
                            strength=concept["confidence"] * credibility,
                            valence=concept["valence"],
                            source="knowledge"
                        )
                    beliefs_updated += 1

                    # 3. Stimulate neural fabric if callback provided
                    if self._on_concept:
                        self._on_concept({
                            **concept,
                            "source": source_label,
                        })

                # ── OPINION INTEGRATION via BeliefSystem.integrate() ──────
                if interpretation:
                    diss = self._beliefs.integrate(interpretation)
                    interpretation["dissonance"] = diss
                    total_dissonance = max(total_dissonance, diss)
                    beliefs_updated += 1   # count the interpretation as a belief update

                    # High dissonance → extra neural stimulation callback
                    if diss > 0.25 and self._on_concept:
                        self._on_concept({
                            "context":     interpretation["claim"][:60],
                            "outcome":     "cognitive_tension",
                            "valence":     -diss * 0.5,
                            "confidence":  diss,
                            "source":      source_label,
                            "is_dissonance": True,
                        })

            elapsed = round(time.time() - start, 3)
            # ── Build opinion output ─────────────────────────────────────
            # Aggregate a stance across all interpretations
            all_interps = [i for i in interpretations if i]
            if all_interps:
                avg_valence    = sum(i.get("valence", 0)    for i in all_interps) / len(all_interps)
                avg_confidence = sum(i.get("confidence", 0) for i in all_interps) / len(all_interps)
                avg_novelty    = sum(i.get("novelty", 0)    for i in all_interps) / len(all_interps)
                stance = (
                    "strongly agree" if avg_valence > 0.5 and avg_confidence > 0.6 else
                    "agree"          if avg_valence > 0.2 else
                    "neutral"        if abs(avg_valence) < 0.2 else
                    "disagree"       if avg_valence > -0.5 else
                    "strongly disagree"
                )
                related_beliefs = [
                    b.claim[:80] for b in (self._beliefs.all_beliefs()[:3] if self._beliefs else [])
                ]
            else:
                avg_valence = avg_confidence = avg_novelty = 0.0
                stance         = "neutral"
                related_beliefs = []

            opinion = {
                "summary":         (concept_list[0]["context"][:120] if concept_list else text[:120]),
                "stance":          stance,
                "confidence":      round(avg_confidence, 3),
                "valence":         round(avg_valence, 3),
                "novelty":         round(avg_novelty, 3),
                "related_beliefs": related_beliefs,
            }

            # ── Store into memory hierarchy if available ─────────────────
            mem_h = getattr(self, "_mem_hierarchy", None)
            if mem_h and concept_list:
                try:
                    # Every ingested concept goes to semantic tier
                    for c in concept_list[:10]:
                        mem_h.store(
                            tier     = "semantic",
                            content  = f"{c['context']} → {c['outcome']}",
                            salience = c["confidence"] * 0.7,
                            valence  = c["valence"],
                            tags     = [source_label, "ingested"],
                        )
                    # High-dissonance interpretation → episodic (event-like)
                    if total_dissonance > 0.25:
                        mem_h.store(
                            tier     = "episodic",
                            content  = f"Encountered conflicting idea from {source_label}: {text[:120]}",
                            salience = total_dissonance * 0.8,
                            valence  = -total_dissonance * 0.4,
                            tags     = ["ingestion_conflict", source_label],
                        )
                except Exception:
                    pass

            # ── Competing interpretations for narrative system ───────────
            # Build a list of divergent cluster interpretations
            competing = []
            if interpretations:
                for i, interp in enumerate(interpretations[:4]):
                    if interp and abs(interp.get("valence", 0)) > 0.05:
                        competing.append({
                            "cluster_idx": i,
                            "claim":       interp.get("claim", ""),
                            "valence":     interp.get("valence", 0.0),
                            "confidence":  interp.get("confidence", 0.5),
                            "stance":      interp.get("stance", "neutral"),
                        })

            summary = {
                "source":            source_label,
                "chunks":            len(chunks),
                "concepts":          total_concepts,
                "memories_stored":   memories_stored,
                "beliefs_updated":   beliefs_updated,
                "max_dissonance":    round(total_dissonance, 3),
                "elapsed_s":         elapsed,
                "top_concepts":      concept_list[:5],
                "opinion":           opinion,
                "competing_interpretations": competing,
            }
            self._ingestion_log.append(summary)
            if len(self._ingestion_log) > 20:
                self._ingestion_log.pop(0)
            return summary

    def last_ingestions(self, n: int = 5) -> List[dict]:
        return self._ingestion_log[-n:]
