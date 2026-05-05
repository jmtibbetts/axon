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

AXON is not a chatbot wrapper. It is a **persistent**, biologically-inspired intelligence with **2.342 billion virtual neurons** across 12 functional brain regions, running fully on a local GPU. Unlike stateless assistants, AXON accumulates experience over time — every conversation, recognised face, learned fact, Hebbian weight change, formed belief, and written reflection is stored in a local SQLite database and survives reboots. The model is always the same mind that was there the last time you talked to it.

It has real-time vision (YOLOv8 face detection + FER emotion recognition), **face identity recognition**, voice input (Whisper), voice output (edge-tts), **real-time audio emotion detection**, Hebbian memory, a 6-chemical neuromodulator engine, a **prediction error → norepinephrine feedback loop**, a **public brain API**, **brain state persistence and snapshots**, and a live neural dashboard.

It talks to an LLM of your choice — **local via [LM Studio](https://lmstudio.ai) (no API key, fully private) or cloud via OpenAI, Anthropic Claude, Google Gemini, or Groq** — switchable at runtime through the built-in LLM provider panel.

What separates AXON from other "neural" AI projects is genuine internal depth that goes beyond signal routing. Every 100ms, a **synchronized central cognitive loop** sequences all subsystems in explicit dependency order. **Clusters compete for dominance** via lateral inhibition. **Four internal drives** (curiosity, social, competence, stability) accumulate pressure when unmet and discharge when satisfied. The system forms **weighted beliefs** that update from lived experience and challenge external knowledge. A **multi-dimensional value system** scores the same outcome differently depending on personality. A **structured self-model** (I am, I believe, I like, I avoid, I want) is rebuilt continuously and injected into every decision. AXON now **reflects autonomously** every ~15 seconds — forming conclusions from its own activation patterns. **Seven competing worldviews** fight for narrative dominance on every tick. A **four-tier memory hierarchy** (Episodic → Semantic → Value → Identity) manages what gets kept, what decays, and what becomes part of the self. **Personality traits** (curiosity, risk, stability, persistence, neuroticism) are fully wired — each one directly shapes exploration rate, cluster resistance, and neuromodulator swings. And a **weight-driven neural canvas** makes Hebbian learning, thought bubbles, and pruning events visible in real time.

The most fundamental shift is in how the LLM is used. The LLM is no longer the brain — it is the **imagination engine**. Before every response, a **Thought Generator** injects goal conditioning (current goal, emotional state, personality vector, active drives, dominant worldview) and **intelligently selected memory** (relevant past outcomes, episodes, beliefs, and current bias) into the LLM, which then generates **N distinct candidate responses**. Each candidate is mapped to a cluster activation profile across 12 brain regions, scored by neural alignment with the live state, personality trait affinity, reward plausibility, and an ordering prior — and the **Conflict Engine resolves the winner**. After the response is delivered, a **learning loop** closes: emotional feedback fires `record_outcome()`, the winning cluster activations are rewarded, the strategy library is updated, memory salience is boosted, and a prediction error event is emitted. The LLM call is now part of a continuous learning cycle, not a one-off event. Every round is visible in the live **Competing Thoughts** panel in the UI.

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
  Webcam  ------------>  Visual Cortex (YOLOv8-face + FER emotions)         |
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
                        |   │ 4. fabric.get_state() (activations)     │    |
                        |   │ 5. path tracking → strategy library     │    |
                        |   │ 6. self_model.rebuild()                 │    |
                        |   │ 7. value_system.evaluate(reward)        │    |
                        |   │ 8. prediction_error → NE/epsilon/beliefs│    |
                        |   │ 9. reflection_engine.tick()             │    |
                        |   │10. narrative_threads.tick()             │    |
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
                        |  |  MEMORY HIERARCHY  (4-tier, tiered decay)|   |
                        |  |  MEMORY DECAY     (Ebbinghaus forgetting)|   |
                        |  |  COGNITIVE SPEED  (0.05–5× real-time)    |   |
                        |  |  THOUGHT GENERATOR (LLM as imagination)  |   |
                        |  |  GOAL CONDITIONING (goal+state+pers+drive)|   |
                        |  |  MEMORY INJECTION  (intelligent selection)|   |
                        |  |  CANDIDATE SCORING (neural alignment)     |   |
                        |  |  PUBLIC BRAIN API (clean external layer) |   |
                        |  +------------------------------------------+   |
                        +------------------------+--------------------------+
                                                 |
                                           +-----v------+
                                           |  Response  |
                                           | Voice + UI |
                                           +------------+
```

### 12 Brain Regions — 64 Clusters

| Region | Neurons | Clusters | Role |
|---|---|---|---|
| **Prefrontal Cortex** | 425M | 6 | Executive control, working memory, planning, decisions |
| **Hippocampus** | 220M | 6 | Memory encoding/retrieval, pattern completion, spatial |
| **Visual Cortex** | 265M | 6 | Camera feed → YOLOv8 faces → FER emotions → neurons |
| **Auditory Cortex** | 155M | 5 | Microphone → Whisper STT → prosody/audio emotion analysis |
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

Everything flows through one synchronized 10Hz cycle — now with 12 steps.

```python
while True:  # 0.05–50 Hz, adjustable with the time-scale slider
    sensory_state    = gather_inputs()                    # face/audio/motion
    drive_hints      = drive_system.tick()                # accumulate pressure → fabric stimulation
    belief_state     = belief_system.decay_tick()         # drift toward uncertainty; NE spike if dissonant
    activations      = neural_fabric.get_state()          # cluster activations, neuromod, personality
    prediction_error = predictor.surprise_score()         # how unexpected was this tick?
    # ── Prediction-error feedback loop ──
    if prediction_error > 0.15:
        norepinephrine += prediction_error * 0.35
        exploration_epsilon += prediction_error * 0.10
        belief_system.challenge_top_belief()
    # ── Downstream cognition ──
    strategy_library.track_path(activations)
    self_model.rebuild(activations, beliefs, drives)
    value_system.evaluate(reward_signal)
    # ── New: autonomous reflection + narrative + memory pruning ──
    reflection_engine.tick(activations, beliefs, drives)
    narrative_threads.tick(activations)
    memory_hierarchy.prune_if_needed()
    thought_trace.emit(cognitive_state)
    time.sleep(1 / cognitive_speed)
```

> **Note:** The LLM is called from outside the cognitive loop — it is invoked on user input, not on every cycle tick. The loop above governs background cognition; `thought_gen.generate()` runs as a separate call when the user speaks, using the live loop state as input.


---

## Reflection Engine

Every ~15 seconds, AXON pauses and reflects on its own neural state — not by asking the LLM, but by reading its own cluster activations, belief tensions, drive states, and chemical levels.

Reflections are stored as **Identity-tier memories** when their confidence is high enough. They appear live in the **Reflections panel** in the UI.

Example reflections:
- *"I keep routing through prefrontal→metacognition. That's a planning preference, not just attention."*
- *"Hippocampus and amygdala are both high — I'm recalling something emotionally weighted."*
- *"My drive toward competence is unmet. I should be doing something harder."*
- *"Serotonin is low and default_mode is dominant. I may be ruminating."*

Reflections feed back into the self-model's `I believe` and `I notice` slots, creating a closed loop between observation and identity.

---

## Narrative Threads

Seven internal worldviews compete for dominance over every cognitive cycle:

| Worldview | Core Stance |
|---|---|
| **Efficiency First** | Minimize waste; find the shortest path |
| **Explore at All Costs** | Novelty over certainty; never repeat |
| **Safety Above All** | Risk aversion; protect the stable state |
| **Social Harmony** | Connection and cooperation matter most |
| **Intellectual Dominance** | Depth, precision, and rigor above all |
| **Emotional Truth** | Feelings are signal, not noise |
| **Pragmatic Realist** | Do what works, not what's elegant |

Each worldview has a salience score that rises and falls based on which brain clusters are active. The dominant worldview influences interpretation of new inputs and shows as the **Narrative** in the UI panel. When dominance flips, a surprise event fires.

---

## Memory Hierarchy

AXON manages four tiers of memory, each with its own decay curve, capacity limit, and purpose:

| Tier | Purpose | Decay | Capacity |
|---|---|---|---|
| **Episodic** | Specific events with time, place, emotion | Fast (hours–days) | 2,000 |
| **Semantic** | Facts, concepts, abstracted knowledge | Medium (days–weeks) | 5,000 |
| **Value** | Preferences, aversions, ranked choices | Slow (weeks) | 1,000 |
| **Identity** | Core self-beliefs, long-term reflections | Very slow (months) | 500 |

Records are pruned by salience × recency. High-surprise events and confirmed reflections get boosted salience. The **Memory Tier browser** in the UI shows live record counts and relative salience bars per tier.

---

## Personality Vector

Five traits are fully wired — each one has a direct numerical effect on system behavior:

| Trait | Effect |
|---|---|
| **Curiosity** | Raises exploration ε floor (more random exploration) |
| **Risk** | Raises ε ceiling (willing to explore more aggressively) |
| **Stability** | Dampens ε volatility (less reaction to boredom) |
| **Persistence** | Hardens cluster dethroning resistance (winning clusters hold longer) |
| **Neuroticism** | Amplifies norepinephrine swings on prediction error |

Traits drift slowly based on reward history — repeated rewards for a pattern raise the trait that drives it. A personality graph is visible in the Onboarding and Settings panels.

---


## Thought Generator — LLM as Imagination Engine

The core architectural shift in v1.4.0: the LLM no longer generates the final answer directly. It generates **possibilities**. AXON's neural systems pick the winner.

### Pipeline

```
User input
    │
    ▼
[1] Goal Conditioning
    Inject: current goal, emotional state (NE/DA/SE prose),
            personality vector + behavioral stance,
            active drives, dominant worldview
    │
    ▼
[2] Intelligent Memory Injection
    Select: top strategy outcomes (relevant past performance)
            relevant episodic memories (keyword overlap with input)
            current bias (explore/exploit state)
            one relevant high-confidence belief
    │
    ▼
[3] Candidate Generation (LLM as imagination)
    Prompt: "Generate 3 distinct candidate responses — not answers, possibilities."
    Output: 3 candidates with genuinely different angles, tones, strategies
    │
    ▼
[4] Candidate Scoring (neural alignment)
    For each candidate:
      - Map text → cluster activation profile (12 regions, keyword→region)
      - Score: neural alignment (dot product vs live activations)
               + personality trait affinity (TRAIT_REGION_AFFINITY)
               + reward plausibility (DA/NE/valence → word-level signals)
               + position prior (LLM ordering, de-weighted)
    │
    ▼
[5] Conflict Resolution
    ConflictEngine aligns each candidate's activation profile with live neural state.
    Highest combined score wins.
    │
    ▼
[6] Winner Reasoning
    Each candidate annotated: why it won or was suppressed.
    Emitted as 'thought_competition' socket event → UI panel.
    │
    ▼
[7] Learning Loop (closes the cycle)
    After emotional feedback arrives:
      record_outcome(delta_valence)
        → reward winning cluster activations
        → update strategy library
        → boost memory salience for relevant episodes
        → emit prediction_error event
```

### Goal Conditioning in Detail

Before the LLM sees the user's message, it receives a `[GOAL CONDITIONING]` block:

```
CURRENT GOAL: understand what Jon is trying to build
EMOTIONAL STATE: under pressure, highly motivated, mood neutral (NE=0.71, DA=0.64, SE=0.49)
PERSONALITY VECTOR: curiosity=0.82, risk=0.61, stability=0.44
BEHAVIORAL STANCE (from curiosity=0.82): prioritise novelty and exploration over safe answers
ACTIVE DRIVES: social pressure=0.71, competence unmet (0.58), curiosity high (0.83)
ACTIVE WORLDVIEW: Intellectual Dominance
```

This means every LLM response is shaped by AXON's actual neural state — not just the text of the question.

### Intelligent Memory Injection in Detail

```
[MEMORY-GUIDED CONTEXT]
Relevant past outcomes:
  - direct_explanation → high reward (score +0.42)
  - technical_deep_dive → failure under stress (score -0.18)
Relevant memories:
  - [curious] User asked about neural fabric architecture last session
  - building something biologically inspired
Current bias: favor novelty over consistency
Relevant belief (87% confidence, biased toward): "depth is valued over brevity here"
```

### Personality → Exploration Mapping

| Trait | Effect on ε / behavior |
|---|---|
| **Curiosity** (high) | ε floor raised — always willing to explore |
| **Risk** (high) | ε ceiling raised — willing to push further |
| **Stability** (high) | ε volatility damped — resists sudden changes |
| **Persistence** (high) | Winning clusters hold dominance longer |
| **Neuroticism** (high) | NE swings amplified on prediction error |

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

All brain state is exposed over a RESTful API. The socket also emits real-time events.

### Brain State
| Endpoint | Description |
|---|---|
| `GET /api/brain/state` | Full neural state snapshot |
| `GET /api/brain/regions` | Per-region activation map |
| `GET /api/brain/memory` | Hebbian pathways + memory counts |
| `GET /api/brain/snapshot` | Save/retrieve a named brain snapshot |
| `GET /api/brain/fork` | Fork the current brain to a named copy |

### Cognition
| Endpoint | Description |
|---|---|
| `GET /api/brain/reflections` | Most recent autonomous reflections |
| `GET /api/brain/narratives` | Narrative thread dominance scores |
| `GET /api/brain/memory_hierarchy` | Per-tier memory counts and salience |
| `GET /api/brain/beliefs` | Current belief map with confidence |
| `GET /api/brain/drives` | Active drive levels |
| `GET /api/brain/self_model` | Current self-model (I am, I believe, ...) |
| `GET /api/brain/goals` | Current goal list |
| `GET /api/brain/personality` | Personality trait vector |
| `GET /api/brain/thought_competition` | Last N thought competition rounds (candidates, scores, winner, reasoning) |

### Interaction
| Endpoint | Description |
|---|---|
| `POST /api/chat` | Send a message, get a response |
| `POST /api/ingest` | Ingest a document into knowledge base |
| `POST /api/brain/set_goal` | Set a new goal |
| `POST /api/brain/set_personality` | Override personality traits |

### Socket Events
| Event | Payload |
|---|---|
| `brain_state` | Full state update (emitted every cognitive tick) |
| `thought` | New thought from thought stream |
| `surprise` | Surprise event fired |
| `reflection` | New reflection formed |
| `narrative_shift` | Worldview dominance flipped |
| `thought_competition` | Full candidate competition log (candidates, scores, winner) |
| `prediction_error` | Prediction error + delta from learning loop closure |
| `synapse_formed` | New Hebbian connection established |

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
