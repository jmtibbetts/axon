<div align="center">

# 🧠 AXON

### Emerging Artificial Intelligence

*A biologically-inspired AI that sees, hears, recognises faces, reads your voice, learns, remembers, competes, forms beliefs, grows drives, thinks in a synchronized cognitive cycle — and speaks through any LLM you choose, local or cloud.*

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.8-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-0078d4?style=flat-square&logo=windows)](https://github.com/jmtibbetts/axon)
[![GPU](https://img.shields.io/badge/GPU-RTX%205090-76b900?style=flat-square&logo=nvidia)](https://nvidia.com)
[![License](https://img.shields.io/badge/License-Axon%20v1.0%20Personal%2FNC-blue?style=flat-square)](LICENSE)

</div>

---

## What is AXON?

AXON is not a chatbot wrapper. It is a **persistent**, biologically-inspired intelligence with **2.342 billion virtual neurons** across 12 functional brain regions, running fully on a local GPU. Unlike stateless assistants, AXON accumulates experience over time — every conversation, recognised face, learned fact, Hebbian weight change, and formed belief is written to a local SQLite database and survives reboots. The model is always the same mind that was there the last time you talked to it. It has real-time vision (YOLOv8 face detection + FER emotion recognition), **face identity recognition**, voice input (Whisper), voice output (edge-tts), **real-time audio emotion detection**, Hebbian memory, a 6-chemical neuromodulator engine, and a live neural dashboard.

It talks to an LLM of your choice — **local via [LM Studio](https://lmstudio.ai) (no API key, fully private) or cloud via OpenAI, Anthropic Claude, Google Gemini, or Groq** — switchable at runtime through the built-in LLM provider panel.

What separates AXON from other "neural" AI projects is genuine internal depth that goes beyond signal routing. Every 100ms, a **synchronized central cognitive loop** sequences all subsystems in explicit dependency order. **Clusters compete for dominance** via lateral inhibition. **Four internal drives** (curiosity, social, competence, stability) accumulate pressure when unmet and discharge when satisfied — shaping which brain regions get priority. The system forms **weighted beliefs** that update from lived experience and challenge external knowledge. A **multi-dimensional value system** scores the same outcome differently depending on personality. And a **structured self-model** (I am, I believe, I like, I avoid, I want) is rebuilt continuously and injected into every decision.

---

## Neural Architecture

```
                        +--------------------------------------------------+
  Webcam  ------------>  Visual Cortex (YOLOv8-face + FER emotions)         |
                        | Face Identity (dlib 128-d embeddings, profiles)   |
  Microphone  -------->  Auditory Cortex (Whisper STT)                      |
                        | Audio Emotion  (prosody: pitch/energy/ZCR)        |
  Web Search  -------->  Association Cortex (curiosity / abstraction)       |
  Documents   -------->  Knowledge Ingestion → Concepts → Opinions          |
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
                        |   │ 8. thought_trace emit                   │    |
                        |   │ 9. drive_state UI emit                  │    |
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

## Live Neural Dashboard

Real-time web interface at `http://localhost:5000` — **7 tabs**:

| Tab | Contents |
|---|---|
| **🧠 Brain** | 64-cluster heatmap, valence/arousal scatter, radial region chart, Hebbian arc animations |
| **⚡ Activity** | **Drive meters** (curiosity/social/competence/stability with threshold lines + PRESSING badges) · **Thought Trace** (per-cycle snapshot: winner, contenders, conflict, drive pressure, belief under revision) · Live neural event feed |
| **💾 Memory** | Episodic + semantic counts, top topics, Hebbian connection weights |
| **👤 Know Me** | User model — passively extracted preferences, traits, personal details |
| **🧬 Identity** | Personality trait bars · Belief list with **cognitive dissonance bar** + contested claims · **Self-Model panel** (I am / I believe / I like / I avoid / I want) · Identity alignment ratio · Emergent preferences · Discovered hobbies · Knowledge ingestion panel |
| **🎙️ Voice** | TTS voice selector, speed/pitch sliders, playback backend |
| **👥 People** | Current person in frame (name + visit #), audio emotion panel, known-people list |

---

## LLM Provider Panel

The **🤖 LLM** tab in the right dashboard column lets you configure the language brain at runtime:

- **Active provider selector** — click to switch between LM Studio, OpenAI, Anthropic, Gemini, and Groq
- **Prefer Local toggle** — always attempt LM Studio first, fall back to the cloud provider on failure
- **Per-provider cards** — live status badge (ONLINE / KEY SET / NOT SET), API key input (masked), model dropdown
- **LM Studio card** — editable server URL, ↻ Probe button to re-detect a model without restarting

Settings are saved immediately to `data/providers.json`. Switching providers takes effect on the next message — no restart required.

---

## Diagnostic Mode

Say **"diagnostic mode"** or click the Diagnostics button. AXON responds in natural language covering:
- Neural architecture (regions, clusters, total neuron count)
- Memory status (episodic, semantic, Hebbian)
- Neuromodulator levels
- Top active brain regions + dominant drives
- Emotional valence / arousal
- Face identity DB
- Audio emotion state

---

## Requirements

| | Windows | macOS | Linux |
|---|---|---|---|
| **Python** | 3.12 | 3.12 | 3.12 |
| **GPU (optional)** | NVIDIA + CUDA 12.x | Apple Silicon (MPS) | NVIDIA + CUDA 12.x |
| **CPU fallback** | ✅ auto | ✅ auto | ✅ auto |
| **LM Studio** | Optional¹ | Optional¹ | Optional¹ |
| **Cloud LLM** | Optional² | Optional² | Optional² |

> ¹ LM Studio is the default provider and requires no API key. If it is offline, AXON falls back to the configured cloud provider.
> ² OpenAI, Anthropic, Gemini, and Groq API keys can be entered at runtime — no restart required.

---

## Installation & Quick Start

> **Step 0 — Prerequisites (all platforms)**
> 1. Install [Python 3.12](https://python.org/downloads/)
> 2. Install [LM Studio](https://lmstudio.ai) and load any GGUF model
> 3. Clone the repo:
> ```bash
> git clone https://github.com/jmtibbetts/axon.git
> cd axon
> ```

---

### 🪟 Windows

```powershell
# Install everything + launch (right-click → Run with PowerShell, or):
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

All subsystems (neural fabric, vision, audio) read this file at startup and select the right device automatically. **You never need to set anything manually.**

If you add or change your GPU later:
```bash
rm data/gpu_config.json   # macOS / Linux
del data\gpu_config.json  # Windows
# then re-run the installer
```

---

### 💾 Persistent storage

AXON never forgets between sessions unless you explicitly ask it to. Every piece of experience is written to `data/memory/axon.db`, a local SQLite database that persists across reboots, reinstalls, and restarts.

What survives every session:
- **Episodic memory** — timestamped records of conversations, interactions, and events
- **Semantic memory** — extracted facts, concepts, and knowledge ingested from documents
- **Hebbian weights** — synaptic connection strengths learned from co-activation patterns
- **Belief system** — weighted assumptions updated from lived experience
- **Face profiles** — identity embeddings, names, and relationship history for every recognised person
- **User model** — inferred preferences, personality traits, and personal details passively built over time
- **LLM provider config** — your chosen provider, model, and API keys (`data/providers.json`)

> The database is local and never leaves your machine.

### 🔄 Resetting to a clean state

If you want to wipe AXON's accumulated experience and start fresh — new personality baseline, no memories, no known faces — run:

```bash
python reset_memory.py
```

This permanently deletes:
- All episodic and semantic memories
- All Hebbian connection weights
- All formed beliefs and preferences
- All face identity profiles and relationship data
- The inferred user model

**This cannot be undone.** Normal reboots, restarts, and even reinstalls do *not* clear memory — this script is the only way to reset the model to its default state.

---

### 🔗 LLM Provider setup

AXON supports five provider options, all configurable at runtime from the **🤖 LLM** tab in the dashboard.

#### Local — LM Studio (default, fully private)

1. Install [LM Studio](https://lmstudio.ai) and open the **Local Server** tab
2. Load any GGUF model (Mistral 7B, LLaMA 3 8B, Qwen, etc.)
3. Start the server on port **1234** (default)
4. Launch AXON — it auto-detects the running model

#### Cloud providers (optional)

Open `http://localhost:7777` → click the **🤖 LLM** tab → enter your key and click **Save**. No restart needed.

| Provider | Where to get a key | Notes |
|---|---|---|
| **OpenAI** | [platform.openai.com](https://platform.openai.com/api-keys) | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo |
| **Anthropic** | [console.anthropic.com](https://console.anthropic.com) | claude-opus-4-5, sonnet, haiku |
| **Google Gemini** | [aistudio.google.com](https://aistudio.google.com/app/apikey) | gemini-2.0-flash, 1.5-pro, 1.5-flash |
| **Groq** | [console.groq.com](https://console.groq.com/keys) | llama3-70b, llama3-8b, mixtral, gemma2 |

Keys are stored in `data/providers.json` — **never committed to git**.

**Prefer Local** toggle (on by default): when checked, AXON always tries LM Studio first and only calls the selected cloud provider if LM Studio is offline or returns an error. Uncheck it to force cloud-only mode.

Open the dashboard: `http://localhost:7777`

---

## Project Structure

```
axon/
├── cognition/
│   ├── neural_fabric.py        # 2.34B neuron GPU engine, conflict, RL, meta
│   ├── language.py             # LLM orchestration + multi-provider dispatch
│   ├── providers.py            # Provider registry (LM Studio/OpenAI/Anthropic/Gemini/Groq)
│   ├── memory.py               # SQLite episodic + semantic + Hebbian store
│   ├── cognitive_cycle.py      # Central 10Hz synchronized cognitive loop
│   ├── belief_system.py        # Weighted beliefs + cognitive dissonance
│   ├── drive_system.py         # Curiosity/social/competence/stability drives
│   ├── value_system.py         # Multi-dimensional personality-weighted scoring
│   ├── self_model.py           # Structured identity: I_am/I_like/I_believe…
│   ├── preference_tracker.py   # Emergent likes/dislikes + hobby detection
│   ├── knowledge_ingestion.py  # PDF/DOCX/EPUB → concepts → opinions → beliefs
│   ├── face_identity.py        # Face recognition + relationship profiles
│   └── voice_output.py         # edge-tts + pygame playback
├── sensory/
│   ├── optic.py                # YOLOv8 face detection + FER emotion
│   ├── auditory.py             # Whisper STT
│   └── audio_emotion.py        # Real-time prosody analysis
├── core/
│   └── engine.py               # Orchestration, callbacks, cycle wiring
└── ui/
    └── app.py                  # Flask-SocketIO + /upload_knowledge endpoint
web/
└── templates/
    └── index.html              # Live neural dashboard (8 tabs)
data/
└── memory/
    └── axon.db                 # SQLite: episodic, semantic, Hebbian, people, beliefs
```

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
