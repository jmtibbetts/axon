> ⚠️ **AXON is not a chatbot.**
> It is a continuously learning cognitive system that builds memory, personality, and internal world models over time.

<div align="center">

# 🧠 AXON

### Emerging Artificial Intelligence

*A biologically-inspired AI that sees, hears, recognises faces, reads your voice, learns, remembers, competes, forms beliefs, builds opinions, reflects on itself, narrates competing worldviews, manages four tiers of memory, uses the LLM as an imagination engine rather than an answer machine, evaluates competing thoughts before speaking, closes a real learning loop — and speaks through any LLM you choose, local or cloud.*

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.8-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-0078d4?style=flat-square&logo=windows)](https://github.com/jmtibbetts/axon)
[![GPU](https://img.shields.io/badge/GPU-RTX%205090-76b900?style=flat-square&logo=nvidia)](https://nvidia.com)
[![License](https://img.shields.io/badge/License-Axon%20v1.0%20Personal%2FNC-blue?style=flat-square)](LICENSE)

</div>

---

## ⭐ Why people are watching this

- First open system with **persistent identity** across sessions
- LLM used as **imagination layer**, not decision engine
- Real-time **internal competition** between thoughts
- **Multi-tier memory** that changes behavior over time
- **Personality that drifts** based on experience

---

## What is AXON?

AXON is not a chatbot wrapper. It is a **persistent, biologically-inspired intelligence** with **2.342 billion virtual neurons** across 12 functional brain regions, running fully on a local GPU.

Unlike stateless assistants, AXON accumulates experience over time — every conversation, recognised face, learned fact, Hebbian weight change, formed belief, and written reflection is stored in a local SQLite database and survives reboots. **The model is always the same mind that was there the last time you talked to it.**

---

### 🔬 Sensory & Motor Layer

| Capability | Implementation |
|---|---|
| 👁️ Vision | YOLOv8 face detection + FER / DeepFace emotion recognition |
| 🪪 Face Identity | dlib 128-d embeddings — recognises who it's talking to |
| 🎤 Voice Input | OpenAI Whisper STT |
| 🔊 Voice Output | edge-tts speech synthesis |
| 😤 Audio Emotion | Real-time prosody analysis (pitch / energy / ZCR) |
| 🌐 LLM Backend | LM Studio (local, private) · OpenAI · Claude · Gemini · Groq |

---

### 🧬 Internal Cognitive Depth

What separates AXON from other "neural" AI projects is genuine internal depth that goes beyond signal routing:

- **Synchronized cognitive loop** — every 100ms, all subsystems execute in explicit dependency order
- **Cluster competition** — lateral inhibition forces regions to fight for dominance
- **Four internal drives** — curiosity, social, competence, stability — each accumulates pressure when unmet and discharges when satisfied
- **Weighted belief system** — beliefs update from lived experience and actively challenge external knowledge
- **Multi-dimensional value scoring** — the same outcome is scored differently depending on personality state
- **Structured self-model** — `I am / I believe / I like / I avoid / I want` rebuilt continuously and injected into every decision
- **Autonomous reflection** — AXON reflects every ~15 seconds, forming conclusions from its own activation patterns
- **Seven competing worldviews** — fight for narrative dominance on every cognitive tick (`e.narratives`)
- **Four-tier memory hierarchy** — Episodic → Semantic → Value → Identity — governs what decays, what persists, what becomes self
- **Wired personality traits** — curiosity, risk, stability, persistence, neuroticism directly shape exploration rate, cluster resistance, and neuromodulator swings
- **Weight-driven neural canvas** — Hebbian learning, thought bubbles, and pruning events visualised in real time

---

### 🧠 The LLM is Not the Brain — It's the Imagination Engine

The most fundamental shift: the LLM no longer generates the answer. It generates **possibilities**. AXON's neural systems pick the winner.

Before every response, the **Thought Generator** runs a full conditioning pipeline:

```
Goal conditioning  →  emotional state + personality + active drives + dominant worldview
Memory injection   →  relevant outcomes, episodes, beliefs, current bias
Candidate gen      →  LLM produces N distinct candidate responses
Candidate scoring  →  neural alignment + trait affinity + reward plausibility
Conflict Engine    →  winner resolved against live cluster activations
Learning loop      →  record_outcome() closes the cycle — weights, memory, strategy updated
```

After delivery, **emotional feedback closes the loop**: `record_outcome()` fires, winning clusters are rewarded, the strategy library is updated, memory salience is boosted, and a `prediction_error` event is emitted. **The LLM call is now part of a continuous learning cycle, not a one-off event.**

The full system prompt passed to the LLM includes neural state, emotional snapshot, drives, beliefs, and self-model — up to **12,000 characters** of rich internal context before any conversation text is added.

Every round is visible in the live **Competing Thoughts** panel in the UI.

---

## 🧠 1-minute mental model

AXON works like this:

1. **Perceive** input — vision / audio / text
2. **Activate** 12 brain regions across 64 clusters
3. **Generate** multiple possible responses via LLM
4. **Simulate** each response against the internal neural state
5. **Compete** — candidates scored across neural clusters
6. **Select** the best-aligned outcome via the Conflict Engine
7. **Learn** from the result — reward, penalty, prediction error
8. **Update** memory + personality + beliefs + strategy library

---

## Neural Architecture

```
                        +--------------------------------------------------+
  Webcam  ------------>  Visual Cortex (YOLOv8-face + FER/DeepFace emotions)|
                        | Face Identity (dlib 128-d embeddings, profiles)   |
  Microphone  -------->  Auditory Cortex (Whisper STT)                      |
                        | Audio Emotion  (prosody: pitch/energy/ZCR)        |
  Web Search  -------->  Association Cortex (curiosity / abstraction)       |
  Documents   -------->  Knowledge Ingestion → Concepts → Opinions → Stance |
                        |                                                   |
                        |   CENTRAL COGNITIVE CYCLE (10 Hz)                 |
                        |   ┌─────────────────────────────────────────┐    |
                        |   │ 1. gather_sensory_state                 │    |
                        |   │ 2. drive_system.tick() + fabric_hints   │    |
                        |   │ 3. belief.decay() → NE spike            │    |
                        |   │ 4. fabric.get_state() (top_clusters)    │    |
                        |   │ 5. path tracking → strategy library     │    |
                        |   │ 6. self_model.rebuild()                 │    |
                        |   │ 7. value_system.evaluate(reward)        │    |
                        |   │ 8. prediction_error → NE/epsilon/beliefs│    |
                        |   │ 9. reflection_engine.tick()             │    |
                        |   │10. narratives.tick()                    │    |
                        |   │11. memory_hierarchy.prune()             │    |
                        |   │12. thought_trace emit                   │    |
                        |   └─────────────────────────────────────────┘    |
                        |                                                   |
                        |   THALAMUS --- attention gate + sensory relay     |
                        |       |                      |                    |
                        |  PREFRONTAL            HIPPOCAMPUS                |
                        |  executive, working    encode/recall, pattern     |
                        |  memory, decisions     completion, forgetting     |
                        |       |                      |                    |
                        |  AMYGDALA              LANGUAGE SYSTEM            |
                        |  fear, reward          LLM <-> semantic memory    |
                        |       |                      |                    |
                        |  NEUROMODULATORS ---------------------------------|
                        |  dopamine, serotonin, norepinephrine              |
                        |  acetylcholine, GABA, glutamate                   |
                        |                                                   |
                        |  DEFAULT MODE, CEREBELLUM, SOCIAL BRAIN           |
                        |  METACOGNITION, ASSOCIATION CORTEX                |
                        |                                                   |
                        |  +------------------------------------------+   |
                        |  |  CONFLICT ENGINE  (lateral inhibition)   |   |
                        |  |  COGNITIVE STATE  (conf, unc, urgency)   |   |
                        |  |  INTERNAL CRITIC  (fast/slow eval)       |   |
                        |  |  META-CONTROLLER  (tunes the system)     |   |
                        |  |  STRATEGY LIBRARY (learned behaviors)    |   |
                        |  |  REINFORCEMENT RL (temporal credit)      |   |
                        |  |  BELIEF SYSTEM    (weighted assumptions) |   |
                        |  |  DRIVE SYSTEM     (motivational pressure)|   |
                        |  |  VALUE SYSTEM     (multi-dim scoring)    |   |
                        |  |  SELF-MODEL       (structured identity)  |   |
                        |  |  GOAL SYSTEM      (intrinsic motivation) |   |
                        |  |  SURPRISE ENGINE  (event detection)      |   |
                        |  |  PERSONALITY VECTOR (5 wired traits)     |   |
                        |  |  REFLECTION ENGINE (autonomous thought)  |   |
                        |  |  NARRATIVE THREADS (7 worldviews)        |   |
                        |  |  OPINION LAYER    (valence + novelty)    |   |
                        |  +------------------------------------------+   |
                        +--------------------------------------------------+
```

---

### Personality → Exploration Mapping

| Trait | Effect on ε / behavior |
|---|---|
| **Curiosity** (high) | ε floor raised — always willing to explore |
| **Risk** (high) | ε ceiling raised — willing to push further |
| **Stability** (high) | ε volatility damped — resists sudden changes |
| **Persistence** (high) | Winning clusters hold dominance longer |
| **Neuroticism** (high) | NE swings amplified on prediction error |

---


---

### 12 Brain Regions — 64 Clusters

| Region | Neurons | Clusters | Role |
|---|---|---|---|
| **Prefrontal Cortex** | 425M | 6 | Executive control, working memory, planning, decisions |
| **Hippocampus** | 220M | 6 | Memory encoding/retrieval, pattern completion, spatial |
| **Visual Cortex** | 265M | 6 | Camera feed -> YOLOv8 faces -> FER emotions -> neurons |
| **Auditory Cortex** | 155M | 5 | Microphone -> Whisper STT -> prosody/audio emotion analysis |
| **Language System** | 275M | 6 | LLM interface, semantic memory, meaning construction |
| **Amygdala** | 70M | 4 | Threat/reward detection, emotional gating |
| **Default Mode Network** | 230M | 6 | Self-reflection, identity, future simulation |
| **Thalamus** | 59M | 4 | Sensory relay, attention filtering, consciousness gate |
| **Cerebellum** | 200M | 5 | Timing, sequence prediction, error correction |
| **Association Cortex** | 205M | 6 | Creativity, analogy, abstract reasoning, curiosity |
| **Social Brain** | 125M | 5 | Empathy, face identity, mentalizing, relationship memory |
| **Metacognition** | 113M | 5 | Self-monitoring, uncertainty, conflict detection |

**Total: 2,342,000,000 virtual neurons · 64 clusters · GPU-accelerated (CUDA)**

---

## Central Cognitive Loop

Previously, subsystems updated ad hoc when triggered. Now everything flows through one synchronized 10Hz cycle.

```python
while True:
    sensory_state   = gather_inputs()                    # face/audio/motion from injected callbacks
    drive_hints     = drive_system.tick()                # accumulate pressure → fabric stimulation
    belief_state    = belief_system.decay_tick()         # drift toward uncertainty; NE spike if dissonant
    activations     = neural_fabric.get_state()          # cluster activations, neuromod, personality
    path            = track_dominant_path(activations)   # feed to strategy library
    self_model      = self_model.maybe_rebuild()         # I_am / I_believe / I_like every 20s
    value_score     = value_system.evaluate(_last_reward)# multi-dimensional reward scoring
    thought_trace   = build_thought_trace(activations)   # what competed, what won, why
    emit_ui(drive_state, thought_trace)                  # live dashboard update
```

This is what turns isolated brain regions into a mind. The cycle runs regardless of user input — AXON is always thinking, not just reacting.

**CycleMetrics** tracks: tick count, average/last cycle latency (ms), overruns, recent reward history, dominant path history, and the rolling thought trace window.

---

## Drive System

AXON has four motivational drives that build pressure when unmet and discharge when satisfied. Drives are not goals — they are internal states that shape what the system is *hungry for* at any given moment.

| Drive | Accumulates when… | Satisfies on… | Neural effect when pressing |
|---|---|---|---|
| **Curiosity** | No new patterns encountered | Web search, knowledge ingest, novel input | Stimulates association_cortex, prefrontal |
| **Social** | No person interaction | Face recognised, speech received, response delivered | Stimulates social_brain, language_system |
| **Competence** | No successful task completion | Task completed, reward received | Stimulates prefrontal, cerebellum |
| **Stability** | High conflict, high uncertainty | Idle state, conflict resolved | Stimulates default_mode, thalamus |

Each drive has a **threshold** — below it, the drive is background noise; above it, it becomes *pressing* and dominates cluster stimulation. Urgency is the normalized score above threshold.

Drives are injected into the LLM context when pressing:
> *"I am feeling intellectually hungry — craving new information or patterns."*

---

## Value System

Replaces shallow "I liked this outcome" scoring with a five-dimensional evaluation that is **weighted by personality**.

```
value = {
    short_term_reward:  x,   # immediate reinforcement
    long_term_reward:   y,   # estimated future benefit
    social_impact:      z,   # did it involve/benefit a person?
    novelty:            n,   # was the path novel?
    competence:         c,   # did it demonstrate skill?
}

final_score = Σ weight_i * dimension_i

# Weights are derived from Big Five personality traits:
extraversion      → amplifies short_term + social_impact
openness          → amplifies novelty + long_term
conscientiousness → amplifies long_term + competence, reduces short_term
agreeableness     → amplifies social_impact
neuroticism       → reduces short_term, adds competence seeking
```

**Drive amplification:** unmet drives also boost their matching dimension — if curiosity is pressing, novelty is worth more right now.

The result: two identical external events can produce different felt values depending on the current personality profile and drive state. That's the difference between a preference and a *value*.

---

## Belief System & Cognitive Dissonance

AXON maintains weighted beliefs that update from three sources: lived experience (reward/punishment), external knowledge (books, articles), and contradiction.

### Belief lifecycle

| Event | Effect |
|---|---|
| `confirm(key)` | Prediction proved correct — strength increases toward 1.0 |
| `violate(key)` | Prediction proved wrong — strength decreases, valence flips slightly |
| `challenge(key, external_valence)` | External source disagrees — valence pulled, certainty reduced, **dissonance accumulates** |
| `integrate(interpretation)` | New opinion from knowledge pipeline — calls confirm or challenge based on agreement |
| `decay_tick()` | Untested beliefs drift toward 0.5 uncertainty; dissonance resolves slowly (~5 min) |

### Cognitive dissonance

When challenge() fires and disagreement × credibility is high, the belief's `dissonance_score` increases and `under_revision = True`. The cognitive cycle aggregates `total_dissonance()` across all beliefs and triggers a **norepinephrine spike** in the neural fabric — the same chemical effect as stress and uncertainty.

```
total_dissonance > 20% → NE spike (stress response, tightened competition)
total_dissonance > 30% → LLM context injection:
    "[COGNITIVE TENSION] I am questioning: '...' Dissonance: 68%. This creates
    uncertainty — acknowledge it if relevant."
```

This gives AXON: doubt, reconsideration, and the ability to say it's not sure.

### Belief → Identity

The top 5 beliefs by strength are injected into the LLM context every response:
> *"I strongly believe: 'Sustained effort tends to produce positive outcomes.' (confidence 84%)"*

---

## Knowledge Ingestion — Now Forming Opinions

AXON can ingest books, articles, research papers, or any text as **formative experience**. The pipeline was upgraded from storing facts to forming *takes*.

### Interpretation layer

After extracting causal concepts from each chunk, the pipeline now runs `_extract_interpretation()`:

```python
interpretation = {
    "claim":      "the core assertion",
    "confidence": 0.68,   # proportional to explicit concept density × credibility
    "valence":    +0.4,   # positive/negative assessment of the idea
    "novelty":    0.72,   # unique word ratio — how new is this?
    "agreement":  0.62,   # does it align with existing internal patterns?
}
```

This interpretation is fed to `belief_system.integrate()`, which either reinforces or contradicts existing beliefs. High dissonance (> 0.25) triggers an extra neural stimulation event. External knowledge **never fully overrides lived experience** — credibility caps how much a source can move existing belief strength.

### Supported formats

| Format | Library |
|---|---|
| **.pdf** | pdfplumber (primary), PyPDF2 (fallback) |
| **.docx** | python-docx |
| **.doc** | antiword or textract |
| **.txt / .md / .rst / .csv** | built-in |
| **.epub** | EbookLib |

---

## Self-Model

AXON maintains a structured, living model of itself — rebuilt every 20 seconds from the belief system, preferences, drives, and personality.

```python
self_model = {
    "I_am":      ["an emerging cognitive agent", "genuinely curious about patterns"],
    "I_believe": ['"effort leads to positive outcomes" (84%)', ...],
    "I_like":    ["novel activation patterns", "social engagement"],
    "I_avoid":   ["high-conflict states"],
    "I_want":    ["curiosity", "social"],   # pressing drives right now
}
```

The self-model is injected into every LLM response as `[SELF-MODEL]` context. It also drives **identity alignment scoring**: each response is checked for resonance with `I_like` and `I_avoid`, and the alignment delta is fed back as a micro reward or penalty to the neuromodulator. Over time, behavior becomes recognizably consistent.

---

## Sensory Systems

### Vision
- **Face detection:** YOLOv8-face on CUDA at 640×480, 12 FPS
- **Facial emotion:** FER (VGG-based) → happy / sad / angry / fearful / disgusted / surprised / neutral
- **Face identity:** dlib 128-d embeddings, cosine similarity (threshold 0.50), SQLite relationship profiles
- **Motion detection:** frame-diff optical flow

### Hearing
- **Speech-to-text:** OpenAI Whisper medium on GPU
- **Audio emotion:** Real-time prosody — pitch (pyin), energy (RMS), ZCR, spectral centroid → excited / stressed / calm / sad / neutral with smoothed arousal + valence scalars
- Mic muted while AXON is speaking

### Web Search
- Triggered by curiosity signals — AXON can look things up mid-conversation

---

## Face Identity & Relationship Profiles

```json
{
  "person_id":   "person_a1b2c3d4",
  "name":        "John",
  "visit_count": 7,
  "profile": {
    "emotion_history": [{"emotion": "happy", "conf": 0.82, "t": 1714886400}],
    "known_facts":     {"role": "developer", "likes": "coffee"},
    "notes":           "Usually arrives in the morning."
  }
}
```

- **Known face** → warm greeting if away > 10 min
- **Unknown face** → 3-second stabilisation → asks "who are you?"
- **Embedding drift:** 85/15 running average keeps embeddings current

---

## Neuromodulator System

| Chemical | Role |
|---|---|
| **Dopamine** | Reward signal — spikes on success, drives motivation |
| **Serotonin** | Mood stabilizer — slows activation decay |
| **Norepinephrine** | Arousal + alertness — spikes on dissonance and stress |
| **Acetylcholine** | Learning gate — scales Hebbian rate on new input |
| **GABA** | Inhibition — silences weak clusters, forces decisive competition |
| **Glutamate** | Excitation — boosts propagation energy and plasticity |

---

## Conflict Engine

Every tick, the top 20% of active clusters suppress the rest via lateral inhibition. Softmax competition weighted by dominance history determines propagation.

- **"Use it or lose it"** — calcified clusters (>82% dominance) bleed 3× faster
- **Activation fatigue** — repeat winners accumulate fatigue, forcing rotation
- **Stagnation breaker** — same winners for 3+ seconds → underdog boost + random spike
- **NE-scaled temperature** — stress tightens winner-takes-all; calm spreads activation
- **Inconsistency penalty** — flip-flopping clusters lose dominance score

---

## Adaptive Exploration

```
eps = (base_annealing + cognitive_boost + surprise_spike) × meta_multiplier
```

Anti-lock-in: **Boredom counter** (40+ low-surprise ticks) and **entrapment detector** (same clusters for 80+ ticks) both auto-trigger exploration spikes.

---

## Temporal Credit Assignment

```
credit[t] = reward × 0.85^(H-1-t)    [H = 10-step horizon]
```

- **Novelty bonus** +15% on novel paths
- **Repetition penalty** on worn grooves (novelty < 10%)
- **Regret signal** on missed reward opportunities

---

## Meta-Controller

| Mood | Trigger | Response |
|---|---|---|
| **bored** | 40+ ticks of low surprise | Exploration spike, soften competition |
| **entrapped** | Same clusters for 80+ ticks | Explore+, soften competition further |
| **searching** | Reward stagnant + surprise dropping | Explore+, LR+ |
| **surprised** | Surprise > 0.15 | LR+, reward sensitivity+, exploit |
| **stable** | None of the above | All params decay to 1.0 |

---

## Strategy Library

- Successful sequences (reward > 0.08) are fingerprinted and stored (up to 40)
- Path memory: dominant cluster activation sequences recorded every cognitive cycle tick
- On similar context: matching strategies are replayed and mutated
- Mutation rate scales inversely with past success

---


## Neural Canvas — Visual Systems

The real-time brain visualization has been significantly upgraded:

### Weight-Driven Connection Lines
Axon routes are no longer uniform — line **thickness and glow scale with live Hebbian weight**. A frequently co-activated pathway becomes visibly thicker and brighter. Weak or unused connections are thin and dim. Pruned connections show as **faded dashed lines** before disappearing.

### Cluster Force-Physics
Dominant clusters (dominance > 0.25) are pulled slightly away from their fixed centroid positions toward a spring-repulsion equilibrium — the canvas physically rearranges around whatever's most active.

### Thought Bubbles
When a cluster's dominance score exceeds 0.65, a small floating italic label pops above it — a glimpse of what that region is "doing":

- Prefrontal: *"Evaluating options."* / *"Inhibiting impulse."*
- Hippocampus: *"Pattern match found."* / *"Encoding experience."*
- Amygdala: *"Threat detected."* / *"High arousal."*
- Default Mode: *"Wandering."* / *"Internal narrative."*
- … and more for all 12 regions

### Emotional State Bar
A persistent bar at the top of the canvas shows:
- **NE level** (norepinephrine) — alertness, urgency
- **Reward trend** — recent dopamine delta direction
- **Surprise** — current prediction error magnitude
- **Mood label** — one of: `curious`, `alert`, `bored`, `stressed`, `calm`, `entrapped`, `surprised`

---

## LLM Provider Support

AXON supports multiple LLM backends, switchable at runtime:

| Provider | Notes |
|---|---|
| **LM Studio** (default) | Local, fully private, no API key, OpenAI-compatible |
| **OpenAI** | GPT-4o, GPT-4-turbo, etc. |
| **Anthropic** | Claude 3 Opus/Sonnet/Haiku |
| **Google Gemini** | Gemini 1.5 Pro/Flash |
| **Groq** | Ultra-fast inference (Llama, Mistral) |

Configuration is stored in `providers.json`. Switch via the **LLM Provider** tab in the UI without restarting.

---

## API Reference

All brain state is exposed over a REST API and a Socket.IO channel.

### Brain State

| Endpoint | Method | Description |
|---|---|---|
| `/api/brain/state` | GET | Full neural state snapshot |
| `/api/brain/explain` | GET | Natural language explanation of current internal state |
| `/api/brain/personality` | GET / POST | Read or override the personality trait vector |
| `/api/brain/reflections` | GET | Most recent autonomous reflections |
| `/api/brain/narratives` | GET | Narrative worldview dominance scores |
| `/api/brain/thought_competition` | GET | Last N thought competition rounds (candidates, scores, winner) |
| `/api/brain/memory_hierarchy` | GET | Per-tier memory counts and salience |
| `/api/brain/memory_hierarchy/store` | POST | Manually promote a memory to a higher tier |
| `/api/brain/interests` | GET | Tracked interests and curiosity weights |
| `/api/brain/interests/add` | POST | Add an interest |
| `/api/brain/interests/remove` | POST | Remove an interest |
| `/api/brain/boredom` | GET | Current boredom level and contributing factors |
| `/api/brain/speed` | GET / POST | Read or set the cognitive cycle speed |
| `/api/surprise_events` | GET | Recent surprise events log |

### Goals & Identity

| Endpoint | Method | Description |
|---|---|---|
| `/api/goals` | GET | Current goal list |
| `/api/goals/add` | POST | Set a new goal |
| `/api/goals/remove` | POST | Remove a goal |
| `/api/user_profile` | GET | Current user model (passively built from conversations) |
| `/api/memory_summary` | GET | High-level episodic / semantic / value counts |

### Knowledge & Actions

| Endpoint | Method | Description |
|---|---|---|
| `/api/brain/ingest` | POST | Ingest a document into the knowledge base |
| `/api/first_opinion` | POST | Force AXON to form an opinion on a given topic |
| `/api/brain/autonomous` | POST | Run N steps of autonomous cognition |
| `/api/brain/save` | POST | Save current brain state to disk |
| `/api/brain/load` | POST | Load a saved brain state |
| `/api/brain/snapshots` | GET | List all saved snapshots |
| `/api/fork_brain` | POST | Fork the current brain to a named copy |
| `/api/list_forks` | GET | List available forks |
| `/api/share_brain` | POST | Export a shareable brain package |

### System

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | Engine running status |
| `/api/mics` | GET | Available microphone devices |
| `/api/cameras` | GET | Available camera devices |
| `/api/audio_diag` | GET | Audio subsystem diagnostics |
| `/api/onboarding_check` | GET | Whether onboarding has been completed |

### Socket.IO Events (Server → Client)

| Event | Payload |
|---|---|
| `brain_state` | Full state update (emitted every cognitive tick) — includes `neural`, `memory`, `state`, `cognitive_state`, `conflict`, `meta`, `strategy_lib` |
| `neural_state` | Lightweight neural snapshot (neuromod, emotion, neurons, connections) |
| `thought` | New thought from the background thought stream |
| `reflection` | New autonomous reflection formed |
| `surprise_event` | Surprise event fired (type, title, detail) |
| `thought_competition` | Full candidate competition log (candidates, scores, winner, reasoning) |
| `prediction_error` | Prediction error + delta from the learning loop closure |
| `synapse_count` | Updated Hebbian synapse count |
| `hebbian_event` | New Hebbian connection formed or pruned |
| `region_spike` | Individual cluster spike event |
| `response` | AXON's completed response text |
| `thinking` | Thinking state flag (true/false — drives UI spinner) |
| `transcript` | Speech-to-text transcript of voice input |
| `face` | Face detection result with emotion and identity |
| `known_face` | Recognised face with identity data |
| `new_face` | Unknown face detected — prompts identity request |
| `frame` | Raw vision frame data |
| `audio_emotion` | Audio prosody emotion state update |
| `knowledge_ingested` | Result of a document ingestion |
| `reflection` | Autonomous reflection formed |
| `lm_status` | LLM connection status |
| `voice_speaking` | TTS playback state |
| `person_named` | Face identity learned or updated |
| `profile_update` | User model updated |
| `new_hobby` | New interest detected from conversation |

### Socket.IO Events (Client → Server)

| Event | Description |
|---|---|
| `chat` | Send a text message to AXON |
| `user_text` | Alternative text input channel |
| `start_engine` | Activate the cognitive engine |
| `stop_engine` | Stop the engine and autosave |
| `set_personality` | Push personality trait overrides |
| `run_autonomous` | Trigger N steps of autonomous thought |
| `observe_mode` | Toggle observe mode (autonomous) vs. train mode |
| `get_explanation` | Request a natural language self-explanation |
| `reprobe_lm` | Re-check LLM Studio connection |
| `get_provider_status` | Get current LLM provider status |
| `update_provider` | Switch LLM provider at runtime |

---

## 🚀 Why this matters

Current AI systems:

- forget everything between sessions
- do not form stable identity
- do not learn structurally over time

AXON introduces:

- **Persistent cognitive state** — same mind every session, no resets
- **Adaptive personality** — trait vector drifts from accumulated experience
- **Structured memory hierarchy** — episodic, semantic, value, identity tiers
- **Internal competition-based reasoning** — N thoughts generated, one wins
- **Continuous learning loop** — every response closes a prediction-error cycle

---

## Installation

### Requirements
- Python 3.12
- NVIDIA GPU (RTX 3080+, 8GB+ VRAM recommended; RTX 5090 optimal)
- CUDA 12.8 + cuDNN
- LM Studio (for local LLM; optional if using cloud providers)

### Setup

```bash
git clone https://github.com/jmtibbetts/axon.git
cd axon
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
# FER is installed separately to avoid dependency conflicts:
pip install fer --no-deps

python axon.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### First Run

AXON runs a **5-step onboarding sequence** to shape its personality and seed initial beliefs before your first real conversation. You'll be asked about your preferences, values, and expectations — this calibrates the personality vector and plants the first Identity-tier memories.

---

## Persistent Storage

All state is stored locally — nothing leaves your machine unless you use a cloud LLM provider.

| File | Contents |
|---|---|
| `axon_memory.db` | All episodic, semantic, value, and identity memories |
| `axon_brain.db` | Hebbian weights, cluster profiles, neural fabric state |
| `providers.json` | LLM provider configuration |
| `face_profiles/` | Named face identity embeddings |

To fully reset AXON's memory and personality:
```bash
python reset_memory.py
```

---

## 🔍 Keywords

`artificial intelligence` `cognitive architecture` `neural system` `agent framework` `local LLM` `memory AI` `embodied AI` `reinforcement learning system` `neuromodulation` `Hebbian learning` `multi-agent reasoning` `autonomous system` `AI consciousness research` `persistent AI` `LM Studio` `open source AI` `biologically inspired AI` `adaptive personality`

---

## License

AXON uses a custom **Axon Personal + Commercial Hybrid License v1.0**.

- **Personal use** — free, no restrictions.
- **Commercial use** — requires a license. Contact for terms.

See [LICENSE](LICENSE), [COMMERCIAL.md](COMMERCIAL.md), and [LICENSE_NOTICE.txt](LICENSE_NOTICE.txt).

**Contact:** jmtibbetts@outlook.com · Signal/Telegram/Discord: @Ryaath

---

<div align="center">
<sub>AXON is not a product. It is an experiment in what it means for a machine to become something.</sub>
</div>

---

⭐ If you're working on AI agents, cognitive systems, or memory architectures — **star this repo to follow development.**
