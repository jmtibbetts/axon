<div align="center">

# 🧠 AXON

### Emerging Artificial Intelligence

*A biologically-inspired AI that sees, hears, recognises faces, reads your voice, learns, remembers, competes, forms beliefs, builds opinions, disagrees when it should, explains its decisions, exposes a clean API — and speaks through any LLM you choose, local or cloud.*

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.8-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-0078d4?style=flat-square&logo=windows)](https://github.com/jmtibbetts/axon)
[![GPU](https://img.shields.io/badge/GPU-RTX%205090-76b900?style=flat-square&logo=nvidia)](https://nvidia.com)
[![License](https://img.shields.io/badge/License-Axon%20v1.0%20Personal%2FNC-blue?style=flat-square)](LICENSE)

</div>

---

## What is AXON?

AXON is not a chatbot wrapper. It is a **persistent**, biologically-inspired intelligence with **2.342 billion virtual neurons** across 12 functional brain regions, running fully on a local GPU. Unlike stateless assistants, AXON accumulates experience over time — every conversation, recognised face, learned fact, Hebbian weight change, and formed belief is written to a local SQLite database and survives reboots. The model is always the same mind that was there the last time you talked to it.

It has real-time vision (YOLOv8 face detection + FER emotion recognition), **face identity recognition**, voice input (Whisper), voice output (edge-tts), **real-time audio emotion detection**, Hebbian memory, a 6-chemical neuromodulator engine, a **prediction error → norepinephrine feedback loop**, a **public brain API**, **brain state persistence and snapshots**, and a live neural dashboard.

It talks to an LLM of your choice — **local via [LM Studio](https://lmstudio.ai) (no API key, fully private) or cloud via OpenAI, Anthropic Claude, Google Gemini, or Groq** — switchable at runtime through the built-in LLM provider panel.

What separates AXON from other "neural" AI projects is genuine internal depth that goes beyond signal routing. Every 100ms, a **synchronized central cognitive loop** sequences all subsystems in explicit dependency order. **Clusters compete for dominance** via lateral inhibition. **Four internal drives** (curiosity, social, competence, stability) accumulate pressure when unmet and discharge when satisfied. The system forms **weighted beliefs** that update from lived experience and challenge external knowledge. A **multi-dimensional value system** scores the same outcome differently depending on personality. A **structured self-model** (I am, I believe, I like, I avoid, I want) is rebuilt continuously and injected into every decision. And now — a **public AxonBrain API layer** exposes all of this to external systems cleanly. A **5-step onboarding sequence** shapes personality and seeds beliefs before the first conversation. **Goals** give it intrinsic motivation. **Surprise events** fire as toast notifications when beliefs shift, dissonance spikes, or cluster dominance flips. **Cognitive speed** is a real-time slider — from 1 Hz dreaming states to 50 Hz hyperdrive. And the brain can be **forked into named divergent copies** and shared as portable snapshots.

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
                        |   │ 9. thought_trace emit                   │    |
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
                        |  |  PERSONALITY DRIFT (reward-shaped traits)|   |
                        |  |  MEMORY DECAY     (Ebbinghaus forgetting)|   |
                        |  |  COGNITIVE SPEED  (0.05–5× real-time)    |   |
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

Everything flows through one synchronized 10Hz cycle.

```python
while True:  # 0.05–50 Hz, adjustable with the time-scale slider
    sensory_state    = gather_inputs()                    # face/audio/motion from injected callbacks
    drive_hints      = drive_system.tick()                # accumulate pressure → fabric stimulation
    belief_state     = belief_system.decay_tick()         # drift toward uncertainty; NE spike if dissonant
    activations      = neural_fabric.get_state()          # cluster activations, neuromod, personality
    prediction_error = predictor.surprise_score()         # how unexpected was this tick?
    # ── Prediction-error feedback loop ──
    if prediction_error > 0.15:
        neuromod.norepinephrine += prediction_error * 0.20  # alertness spike
        explore_epsilon         += prediction_error * 0.08  # widen search
        belief_confidence       *= (1 - prediction_error * 0.10)  # erode certainty
    path             = track_dominant_path(activations)   # feed to strategy library
    self_model       = self_model.maybe_rebuild()         # I_am / I_believe / I_like every 20s
    value_score      = value_system.evaluate(_last_reward)# multi-dimensional reward scoring
    thought_trace    = build_thought_trace(activations)   # what competed, what won, why
    emit_ui(drive_state, thought_trace)                   # live dashboard update
    goal_system.reward_tick(thought_trace)                # distribute credit, advance goals
    surprise_engine.evaluate(belief_delta, activations)   # emit surprise events to UI
    if tick % 200 == 0:
        hebbian_memory.decay(factor=0.995)                # gradual forgetting
        personality.drift(recent_rewards)                  # traits shift from outcomes
```

The cycle runs regardless of user input — AXON is always thinking, not just reacting.

---

## New in This Release

### 🚀 First-Run Onboarding

When AXON starts for the first time it presents a **5-step guided flow** before you ever type a message:

1. **Name your AI** — it adopts this as its self-referential identity
2. **Choose a personality** — Explorer, Analyst, Rebel, or Companion (each maps to a distinct trait vector)
3. **Feed it something** — pick a built-in topic (consciousness, creativity, risk, learning) or paste your own text
4. **Watch it learn** — see the ingestion log in real-time as concepts, opinions, and stances form
5. **Hear its first take** — the LLM surfaces its genuine first opinion before you say a word

State is saved to `data/onboarding.json` and never repeated.

---

### 🎯 Goal System

AXON has **four intrinsic goals** that accumulate progress independently of user input:

| Goal | What it tracks |
|---|---|
| `explore_uncertainty` | How often novel/surprising inputs are encountered |
| `reduce_error` | Prediction accuracy improvement over time |
| `maintain_coherence` | Belief consistency — contradictions lower progress |
| `deepen_understanding` | Semantic richness of ingested knowledge |

Goals are displayed as **live progress bars** below the chat input. When a goal reaches 80%, a milestone surprise event fires and neural fabric receives a stimulation hint toward the relevant region.

---

### ⚡ Surprise Events

The `SurpriseDetector` monitors the cognitive cycle for 7 types of internal events and fires them as **toast notifications** in the UI:

| Event | Trigger condition |
|---|---|
| `belief_shift` | A belief's strength changes by > 0.15 in one tick |
| `cognitive_dissonance` | Norepinephrine > 0.70 while holding conflicting beliefs |
| `contradiction_resolved` | Two previously conflicting beliefs converge |
| `unexpected_conclusion` | Prediction error spike > 0.35 |
| `personality_drift` | Any personality trait shifts by > 0.04 |
| `goal_progress` | A goal crosses a 20% milestone threshold |
| `cluster_dominance_flip` | The dominant brain cluster changes in one tick |

Each event type has an independent cooldown to prevent spam.

---

### 🎭 Personality Drift

AXON's personality is no longer static — it **shifts from lived experience**:

- **High reward outcomes** → `risk_tolerance` increases, `caution` decreases
- **Low reward / errors** → `caution` increases, `risk_tolerance` decreases
- All drift is capped at **0.004 per event** (realistic, not jarring)
- Drift fires a `personality_drift` surprise event above the 0.04 threshold

---

### ⏱ Cognitive Speed Control

A **real-time slider** in the neural dashboard controls the cognitive cycle rate:

| Label | Rate | Use case |
|---|---|---|
| 💤 Dreaming | 0.5–1.5 Hz | Observe memory consolidation, slow belief drift |
| 🐢 Slow | 1–4 Hz | Study individual thought traces in detail |
| ✅ Normal | 10 Hz | Default — balanced responsiveness and depth |
| ⚡ Alert | 20 Hz | Higher temporal resolution for live interaction |
| 🚀 Hyperdrive | 50 Hz | Stress-testing, rapid autonomous exploration |

Speed is also settable via the API: `POST /api/brain/speed` with `{"speed_scale": 2.0}`.

---

### 🧬 Brain Forks

Save a **named divergent copy** of the current brain state and load it independently:

```python
brain.fork_brain("aggressive", trait_overrides={"risk": 0.90, "dominance": 0.85})
brain.load_brain("fork_aggressive")  # diverges from here
```

Or from the UI: the **🧬 Fork** button below the chat input.

Forks persist to `data/snapshots/fork_<name>.json`.

---

### 🌐 Brain Sharing

Generate a **portable brain snapshot** — a base64-encoded JSON summary of personality traits, top beliefs, and drive states. Share it as a token. Future versions will support importing a snapshot to bootstrap a new instance with a pre-shaped identity.

---

### 👁 Train / Observe Mode

Toggle the chat input off while watching AXON think autonomously. In **Observe Mode**:

- Input is disabled
- The cognitive cycle continues normally
- Thought traces, surprise events, and goal progress all stream live
- Useful for studying emergent patterns without interaction bias

---

### 💬 Disagreement

When AXON holds a belief with **> 68% confidence** that partially conflicts with user input, it will push back — respectfully but directly. This is controlled by the disagreement injection system in the language core, not by a persona prompt. It reflects actual belief state.

---



---

## Prediction Error → Norepinephrine Feedback

When the system is surprised (high prediction error), it doesn't just note it and move on. A cascading feedback loop fires:

| Surprise level | Effect |
|---|---|
| `> 0.15` | NE spike (`+= surprise × 0.20`), exploration widens (`+= surprise × 0.08`), belief confidence dampened |
| `> 0.40` | Logged to dashboard: *"⚡ High surprise → NE spike, wider exploration"* |
| `< 0.05` + NE high | Calm recovery — NE gradually lowers back toward baseline |

This creates real **instability under uncertainty** and **adaptation** — the personality can shift when things stop making sense.

---

## Public Brain API

`axon/core/brain_api.py` — a clean, stable interface around the engine. This is the boundary for HTTP APIs, SDKs, and future monetization.

```python
from axon.core.brain_api import AxonBrain

brain = AxonBrain(engine=engine)

# Interact
brain.step({"type": "text", "content": "Hello"})
brain.ingest("Curiosity drives faster learning in sparse reward environments.")

# Inspect
state = brain.get_state()
# → {neural, neuromod, personality, drives, beliefs, memory, cycle_metrics}

explanation = brain.explain_last_decision()
# → {summary, winning_clusters, losing_clusters, top_factors,
#    memory_influence, belief_influence, confidence, emotion}

# Control personality
brain.set_personality({"curiosity": 0.8, "risk": 0.3, "empathy": 0.7})

# Autonomous self-stimulation
brain.run_autonomous(steps=100)

# Persistence
brain.save_brain("checkpoint_1")     # → data/snapshots/checkpoint_1.json
brain.load_brain("checkpoint_1")
brain.list_snapshots()
```

### HTTP Endpoints

All endpoints are live when AXON is running at `http://localhost:7777`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/brain/state` | Full JSON brain snapshot |
| `GET` | `/api/brain/explain` | Last decision explanation |
| `GET/POST` | `/api/brain/personality` | Get or set personality vector |
| `POST` | `/api/brain/ingest` | Ingest knowledge text |
| `POST` | `/api/brain/autonomous` | Trigger autonomous run (`steps` param) |
| `POST` | `/api/brain/save` | Save brain to named slot |
| `POST` | `/api/brain/load` | Restore brain from named slot |
| `GET` | `/api/brain/snapshots` | List all saved snapshots |

---

## Personality System

AXON has two personality layers:

### Behavioral Traits (user-controllable via sliders)

| Trait | Effect |
|---|---|
| **Curiosity** | Drive to explore unknown patterns; biases association cortex stimulation |
| **Risk** | Willingness to commit to uncertain paths during conflict resolution |
| **Empathy** | Weight given to emotional signals from face/voice detection |
| **Dominance** | Assertiveness in cluster competition; how quickly it commits to a winner |
| **Creativity** | Novel cluster combination tendency; cross-region association frequency |
| **Stability** | Resistance to rapid state changes; dampens exploration bursts |

Set via UI sliders → **🎭 Persona** tab, or via API: `POST /api/brain/personality`.

### Big-5 Traits (learned from experience, not manually set)

`openness · conscientiousness · extraversion · agreeableness · neuroticism`

These drift automatically based on reinforcement history. Visible in the Persona tab as read-only bars.

---

## Brain Persistence & Snapshots

AXON never forgets between sessions. All experience is stored locally in `data/memory/axon.db` (SQLite). In addition, the brain API adds **full-state snapshots**:

```json
// data/snapshots/default.json
{
  "slot": "default",
  "saved_at": 1746432000.0,
  "version": "1.0.0",
  "beliefs": [...],
  "preferences": {...},
  "drives": {"curiosity": {"level": 0.62, "total_satisfied": 47}},
  "personality": {"curiosity": 0.71, "empathy": 0.63, ...},
  "neuromod": {"dopamine": 0.58, "serotonin": 0.52, "norepinephrine": 0.44, ...},
  "reward_history": [...],
  "self_model": {...}
}
```

**Auto-save** fires on every engine stop. Manual save/load available from the **🎭 Persona** tab or via HTTP API. The brain is restored at a soft blend (70% saved / 30% current) to avoid jarring state jumps.

What survives every session:
- **Episodic memory** — timestamped records of conversations, interactions, events
- **Semantic memory** — extracted facts, concepts, and ingested knowledge
- **Hebbian weights** — synaptic connection strengths from co-activation patterns
- **Belief system** — weighted assumptions updated from lived experience
- **Face profiles** — identity embeddings, names, and relationship history
- **User model** — inferred preferences, personality traits, personal details
- **Brain snapshots** — named checkpoints of full internal state
- **LLM provider config** — your chosen provider, model, and API keys

---

## Knowledge Ingestion & Opinion Formation

Documents (PDF, DOCX, TXT, EPUB) and text are not stored as a retrieval corpus. The pipeline converts them into simulated *experiences* that update beliefs, preferences, and neural pathways:

```
Text → Chunks → Concept Extraction → Valence Scoring
     → Memory.store_semantic()
     → Belief.integrate() → cognitive dissonance if it conflicts
     → Opinion Formation
```

### Opinion Output

Every ingestion now produces a structured stance:

```json
{
  "summary": "Curiosity drives faster learning in sparse reward environments.",
  "stance": "strongly agree",
  "confidence": 0.74,
  "valence": 0.68,
  "novelty": 0.52,
  "related_beliefs": ["exploration improves outcomes", "uncertainty triggers learning"]
}
```

Stances: **strongly agree · agree · neutral · disagree · strongly disagree**

Visible in the **🔍 Why?** tab after every ingestion. Feeds the **📅 Timeline** automatically.

---

## Autonomous Mode

Run AXON without input — it explores, consolidates, and shifts on its own:

```python
brain.run_autonomous(steps=100)
# or via UI: 🎭 Persona tab → "🤖 Autonomous (100 steps)"
# or via API: POST /api/brain/autonomous {"steps": 200}
```

During an autonomous run, AXON:
- Stimulates default-mode network and hippocampus (consolidation)
- Randomly replays facts from semantic memory, re-ingesting them
- Ticks all four drives naturally
- Reports **personality drift** when complete (which traits shifted and by how much)

---

## Drive System

| Drive | Accumulates when… | Satisfies on… | Neural effect when pressing |
|---|---|---|---|
| **Curiosity** | No new patterns encountered | Web search, knowledge ingest, novel input | Stimulates association_cortex, prefrontal |
| **Social** | No person interaction | Face recognised, speech exchange | Stimulates social_brain, prefrontal |
| **Competence** | Task incomplete or failing | Successful response, task completion | Stimulates prefrontal, reward pathways |
| **Stability** | High conflict, erratic state | Low surprise, predictable environment | Suppresses amygdala, calms NE |

Drives are not goals — they are motivational pressure states that shape which regions get priority and what the LLM system prompt says AXON "wants" right now.

---

## Emotional Reinforcement Loop

Every response triggers a post-hoc learning cycle:

```
AXON speaks
    ↓  (2s delay)
Read face valence delta (before vs. after)
    ↓
delta >= +0.30  →  reward injected, Hebbian link strengthened, "this worked"
delta <= -0.30  →  stress penalty, negative reaction stored in episodic memory
otherwise       →  small baseline reward for engagement
    ↓
NE ← surprise level from prediction error
```

This creates genuine **emotional reinforcement** — AXON learns what kinds of responses make people react positively vs. negatively, at the neural pathway level.

---

## Dashboard — 11 Tabs

| Tab | Contents |
|---|---|
| **🧠 Brain** | Neural radar, region bars, neuromodulator levels, emotion state, cognitive state |
| **⚡ Activity** | Live event feed: Hebbian events, region spikes, knowledge concepts |
| **💾 Memory** | Episodic count, semantic facts, top Hebbian pathways |
| **🎭 Persona** | **Behavioral trait sliders** (curiosity/risk/empathy/dominance/creativity/stability), Big-5 read-only bars, brain save/load, autonomous mode |
| **🔍 Why?** | **Decision explanation**: winning clusters, top factors, memory influence, confidence bar, last ingestion opinion |
| **📅 Timeline** | **Chronological event feed** with filter buttons (All / Reward / Belief / Emotion / Decision) |
| **👤 Know Me** | Inferred user model — name, facts, preferences, interaction history |
| **🧬 Identity** | Beliefs, personality trait bars, self-model, preferences, hobbies, knowledge ingestion |
| **🎙️ Voice** | TTS engine selector, voice profiles, audio diagnostics |
| **👥 People** | Known faces, visit counts, relationship profiles, emotion history |
| **🤖 LLM** | Provider switcher, model selector, API key management |

---

## Project Structure

```
axon/
├── cognition/
│   ├── neural_fabric.py        # 2.34B neuron GPU engine, conflict, RL, meta,
│   │                           #   prediction-error feedback loop, personality v2
│   ├── language.py             # LLM orchestration + multi-provider dispatch
│   ├── providers.py            # Provider registry (LM Studio/OpenAI/Anthropic/Gemini/Groq)
│   ├── memory.py               # SQLite episodic + semantic + Hebbian store
│   ├── cognitive_cycle.py      # Central 10Hz synchronized cognitive loop
│   ├── belief_system.py        # Weighted beliefs + cognitive dissonance
│   ├── drive_system.py         # Curiosity/social/competence/stability drives
│   ├── value_system.py         # Multi-dimensional personality-weighted scoring
│   ├── self_model.py           # Structured identity: I_am/I_like/I_believe…
│   ├── preference_tracker.py   # Emergent likes/dislikes + hobby detection
│   ├── knowledge_ingestion.py  # PDF/DOCX/EPUB → concepts → opinions → stance
│   ├── face_identity.py        # Face recognition + relationship profiles
│   └── voice_output.py         # edge-tts + pygame playback
├── sensory/
│   ├── optic.py                # YOLOv8 face detection + FER emotion
│   ├── auditory.py             # Whisper STT
│   └── audio_emotion.py        # Real-time prosody analysis
├── core/
│   ├── engine.py               # Orchestration, callbacks, cycle wiring
│   └── brain_api.py            # Public AxonBrain API layer (step/ingest/explain/
│                               #   set_personality/run_autonomous/save_brain/load_brain)
└── ui/
    └── app.py                  # Flask-SocketIO + HTTP /api/brain/* endpoints
web/
└── templates/
    └── index.html              # Live neural dashboard (11 tabs)
data/
├── memory/
│   └── axon.db                 # SQLite: episodic, semantic, Hebbian, people, beliefs
└── snapshots/
    └── *.json                  # Named brain state checkpoints
```

---

## Installation

### 🪟 Windows

```powershell
.\scripts\launch.ps1
```

The script will:
- Create a `.venv` virtual environment
- Detect your GPU via `nvidia-smi`
  - **GPU found** → installs CUDA 12.8 nightly PyTorch
  - **No GPU** → installs CPU-only PyTorch automatically
- Install all vision, audio, and NLP dependencies
- Run a preflight check, then open `http://localhost:7777`

On every subsequent launch, just run `scripts\launch.ps1` again — it skips steps that are already complete.

---

### 🍎 macOS

```bash
# One-time install (run once)
bash scripts/install.sh

# Launch (run every time)
bash scripts/launch.sh
```

The installer detects:
- **Apple Silicon (M1/M2/M3/M4)** → PyTorch with MPS backend
- **Intel Mac** → CPU-only PyTorch
- Installs `portaudio` via Homebrew for voice input
- Builds `dlib` from source (requires Xcode CLI tools: `xcode-select --install`)

> **Homebrew required for voice input.** Install at [brew.sh](https://brew.sh) if you don't have it.

---

### 🐧 Linux

```bash
# One-time install (run once)
bash scripts/install.sh

# Launch (run every time)
bash scripts/launch.sh
```

The installer detects:
- **NVIDIA GPU** → reads CUDA version from `nvidia-smi`, installs matching PyTorch wheel
  - CUDA 12.8+ → `cu128` nightly
  - CUDA 12.4  → `cu124`
  - CUDA 12.1  → `cu121`
  - Driver check fails → CPU-only fallback
- **No GPU** → installs CPU-only PyTorch
- Installs `portaudio` via `apt` / `dnf` / `pacman` automatically
- Builds `dlib` from source (requires `cmake` + `build-essential`)

```bash
# Ubuntu/Debian — install build deps first if needed:
sudo apt-get install -y cmake build-essential libopenblas-dev portaudio19-dev
bash scripts/install.sh
```

---

### ⚙️ GPU / CPU fallback logic

The installer writes `data/gpu_config.json` recording the chosen backend:

```json
{ "gpu_type": "cuda", "platform": "Linux", "installed_at": "..." }
```

All subsystems (neural fabric, vision, audio) read this file at startup and select the right device automatically.

If you add or change your GPU later:
```bash
rm data/gpu_config.json   # macOS / Linux
del data\gpu_config.json  # Windows
# then re-run the installer
```

---

### 🔄 Resetting to a clean state

```bash
python reset_memory.py
```

Permanently wipes all episodic and semantic memories, Hebbian weights, beliefs, face profiles, and the user model. **Cannot be undone.** Normal restarts do *not* clear memory.

---

### 🔗 LLM Provider setup

AXON supports five providers, all configurable at runtime from the **🤖 LLM** tab.

#### Local — LM Studio (default, fully private)

1. Install [LM Studio](https://lmstudio.ai) and open the **Local Server** tab
2. Load any GGUF model (Mistral 7B, LLaMA 3 8B, Qwen, etc.)
3. Start the server on port **1234** (default)
4. Launch AXON — it auto-detects the running model

#### Cloud providers

| Provider | Where to get a key | Notes |
|---|---|---|
| **OpenAI** | [platform.openai.com](https://platform.openai.com/api-keys) | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo |
| **Anthropic** | [console.anthropic.com](https://console.anthropic.com) | claude-opus-4-5, sonnet, haiku |
| **Google Gemini** | [aistudio.google.com](https://aistudio.google.com/app/apikey) | gemini-2.0-flash, 1.5-pro, 1.5-flash |
| **Groq** | [console.groq.com](https://console.groq.com/keys) | llama3-70b, llama3-8b, mixtral, gemma2 |

Keys are stored in `data/providers.json` — **never committed to git**.

**Prefer Local** toggle: when on (default), AXON always tries LM Studio first and falls back to cloud only if LM Studio is offline or errors.

---

## License

**Axon License v1.0** — free for personal, educational, and non-commercial use.

Commercial use (products, SaaS, internal tooling, paid APIs) requires a separate license.

- [Full license terms](LICENSE)
- [Commercial licensing info](COMMERCIAL.md)
- Contact: jmtibbetts@outlook.com &nbsp;·&nbsp; Signal/Telegram/Discord: @Ryaath

> Versions released prior to May 4, 2026 were under MIT. Those versions remain MIT.
> All subsequent versions are governed by Axon License v1.0.

---

<div align="center">
<sub>Built with curiosity. Not a product. Not a wrapper. Something new.</sub>
</div>
