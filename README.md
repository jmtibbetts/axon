<div align="center">

# 🧠 AXON
### Emerging Artificial Intelligence

*A biologically-inspired AI that sees, hears, speaks, learns, remembers, competes, adapts, and tunes itself — running entirely on your own hardware.*

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.8-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Platform](https://img.shields.io/badge/Platform-Windows%2011-0078d4?style=flat-square&logo=windows)](https://microsoft.com)
[![GPU](https://img.shields.io/badge/GPU-RTX%205090-76b900?style=flat-square&logo=nvidia)](https://nvidia.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## What is AXON?

AXON is not a chatbot wrapper. It is a persistent, biologically-inspired intelligence with **2.342 billion virtual neurons** across 12 functional brain regions, running fully on a local GPU. It has real-time vision (YOLOv8 face detection + FER emotion recognition), voice input (Whisper), voice output (edge-tts), Hebbian memory, a 6-chemical neuromodulator engine, and a live neural dashboard.

It talks to a local LLM via [LM Studio](https://lmstudio.ai) — no cloud, no API keys.

What separates AXON from other "neural" AI projects is that the architecture has genuine internal depth: **clusters compete for dominance**, the system evaluates its own decisions with an Internal Critic, a Meta-Controller watches the system's own performance and tunes exploration/learning in real time, and a Strategy Library stores successful behavioral sequences for future replay and mutation.

---

## Neural Architecture

```
                        ┌─────────────────────────────────────────────────┐
  Webcam ──────────────▶│  Visual Cortex (YOLOv8-face + FER emotions)     │
  Microphone ──────────▶│  Auditory Cortex (Whisper STT)                  │
  Web Search ──────────▶│  Association Cortex (curiosity / abstraction)   │
                        │                                                 │
                        │   THALAMUS ──── attention gate + sensory relay  │
                        │       ↓                    ↓                    │
                        │  PREFRONTAL            HIPPOCAMPUS              │
                        │  executive · working   encode/recall · pattern  │
                        │  memory · decisions    completion · forgetting  │
                        │       ↓                    ↓                    │
                        │  AMYGDALA              LANGUAGE SYSTEM          │
                        │  fear · reward         LLM ↔ semantic memory   │
                        │       ↓                    ↓                    │
                        │  NEUROMODULATORS ──────────────────────────────▶│
                        │  dopamine · serotonin · norepinephrine          │
                        │  acetylcholine · GABA · glutamate               │
                        │                                                 │
                        │  DEFAULT MODE · CEREBELLUM · SOCIAL BRAIN       │
                        │  METACOGNITION · ASSOCIATION CORTEX             │
                        │                                                 │
                        │  ┌──────────────────────────────────────────┐  │
                        │  │  CONFLICT ENGINE  (lateral inhibition)   │  │
                        │  │  COGNITIVE STATE  (conf · unc · urgency) │  │
                        │  │  INTERNAL CRITIC  (fast/slow eval)       │  │
                        │  │  META-CONTROLLER  (tunes the system)     │  │
                        │  │  STRATEGY LIBRARY (learned behaviors)    │  │
                        │  │  REINFORCEMENT RL (temporal credit)      │  │
                        │  └──────────────────────────────────────────┘  │
                        └───────────────────────────┬─────────────────────┘
                                                    │
                                              ┌─────▼──────┐
                                              │  Response  │
                                              │ Voice + UI │
                                              └────────────┘
```

### 12 Brain Regions — 64 Clusters

| Region | Neurons | Clusters | Role |
|---|---|---|---|
| **Prefrontal Cortex** | 425M | 6 | Executive control, working memory, planning, decisions |
| **Hippocampus** | 220M | 6 | Memory encoding/retrieval, pattern completion, spatial |
| **Visual Cortex** | 265M | 6 | Camera feed → YOLOv8 faces → FER emotions → neurons |
| **Auditory Cortex** | 155M | 5 | Microphone → Whisper STT → phoneme/prosody analysis |
| **Language System** | 275M | 6 | LLM interface, semantic memory, meaning construction |
| **Amygdala** | 70M | 4 | Threat/reward detection, emotional gating |
| **Default Mode Network** | 230M | 6 | Self-reflection, identity, future simulation |
| **Thalamus** | 59M | 4 | Sensory relay, attention filtering, consciousness gate |
| **Cerebellum** | 200M | 5 | Timing, sequence prediction, error correction |
| **Association Cortex** | 205M | 6 | Creativity, analogy, abstract reasoning, curiosity |
| **Social Brain** | 125M | 5 | Empathy, face recognition, mentalizing |
| **Metacognition** | 113M | 5 | Self-monitoring, uncertainty, conflict detection |

**Total: 2,342,000,000 virtual neurons · 64 clusters · GPU-accelerated (CUDA)**

---

## Neuromodulator System

Six persistent chemical scalars that modulate the entire network in real time:

| Chemical | Role |
|---|---|
| **Dopamine** | Reward signal — spikes on success, drives motivation and curiosity |
| **Serotonin** | Mood stabilizer — slows activation decay, keeps tone calm |
| **Norepinephrine** | Arousal + alertness — sharpens competition under stress |
| **Acetylcholine** | Learning gate — surges on new input, scales Hebbian rate |
| **GABA** | Inhibition — silences weak clusters, forces decisive competition |
| **Glutamate** | Excitation — boosts propagation energy and plasticity |

---

## Conflict Engine

**Clusters don't cooperate — they compete.**

Every tick, the top 20% of active clusters suppress the rest via lateral inhibition. A softmax competition weighted by each cluster's dominance history and confidence track record determines which ones actually propagate.

Key behaviors:
- **"Use it or lose it"** — all clusters bleed dominance continuously; calcified ones (>82%) bleed 3× faster
- **Activation fatigue** — clusters that win repeatedly accumulate fatigue, reducing their effective activation and forcing rotation to other clusters
- **Stagnation breaker** — same winners for 3+ seconds → automatic underdog boost + random spike
- **NE-scaled temperature** — stress tightens the softmax (winner-takes-all); calm spreads it
- **Inconsistency penalty** — flip-flopping clusters lose dominance based on activation variance

---

## Adaptive Exploration

Epsilon is no longer a static schedule. It's driven by three compounding forces:

```
eps = (base_annealing + cognitive_boost + surprise_spike) × meta_multiplier
```

1. **CognitiveState** — uncertainty + urgency push exploration up when the system feels lost
2. **Surprise spike** — high prediction error temporarily opens exploration (up to +0.15)
3. **MetaController** — second-order multiplier based on system-wide performance trends

Anti-lock-in safeguards:
- **Boredom counter** — 40+ ticks of low surprise → exploration spike fires automatically
- **Entrapment detector** — same clusters dominating for 80+ ticks → competition temperature softened

---

## Temporal Credit Assignment

No instant gratification. Reward fires after a **10-step horizon**. Credit is assigned with temporal decay:

```
credit[t] = reward × 0.85^(H−1−t)
```

Clusters active at the outcome moment get full credit. Clusters that fired early get discounted credit. The system learns *which decision* actually mattered, not just who was in the room.

Additionally:
- **Novelty bonus** — path fingerprinted against last 20 sequences; novel paths earn +15% reward
- **Repetition penalty** — worn grooves (novelty < 10%) get penalized
- **Regret signal** — compares final valence to the best valence seen in the window. Leaving reward on the table adds a penalty proportional to the missed opportunity.

---

## Surprise as a First-Class Signal

High prediction error isn't just observed — it **drives the system**:

| Surprise Effect | Mechanism |
|---|---|
| Learning rate × 2.5 | `effective_lr = base_lr × (1 + surprise × 5.0)` |
| Hebbian trace boost | Trace update rate scales with surprise |
| Memory encoding boost | Episodic importance += `surprise_level × 0.4` |
| Exploration burst | eps += `min(0.15, surprise × 0.6)` when surprise > 0.10 |
| MetaController "surprised" mode | Tightens exploration to exploit the new signal |

Surprising moments are encoded stronger, learned faster, and explored more deeply.

---

## Meta-Controller

A second-order system that watches the system's own performance and tunes it in real time:

```
watches:  surprise trend · reward stagnation · dominance entropy
tunes:    explore_rate · reward_sensitivity · conflict_sharpness · lr_scale
```

| Mood | Trigger | Response |
|---|---|---|
| **bored** | 40+ ticks of low surprise | Exploration spike, soften competition |
| **entrapped** | Same clusters for 80+ ticks | Explore+, further soften competition |
| **searching** | Reward stagnant + surprise dropping | Explore+, learning rate+ |
| **surprised** | Surprise > 0.15 | LR+, reward sensitivity+, exploit |
| **stable** | None of the above | All params decay back to 1.0 |

All parameters bounded and decay toward 1.0 when the system is performing well.

---

## Strategy Library

AXON stores successful activation sequences as **reusable behavioral patterns**:

- Successful sequences (reward > 0.08) are fingerprinted and stored (up to 40)
- When context (emotion + cognitive state) matches a stored strategy (≥65% similarity), the network is biased toward that known-good cluster sequence
- High surprise blocks replay — new territory is explored, not exploited
- 5% chance per eval cycle to mutate a stored strategy (add noise → new variations)
- Weaker strategies are evicted when the library is full

This is the transition from *learned weights* to *learned behaviors*.

---

## Reinforcement Learning Loop

Facial emotion feedback closes the loop in real time:

```
camera → FER → detect emotion delta → dopamine reward or stress penalty
    → reinforce Hebbian pathways → adjust cluster dominance
```

The prediction engine also runs structural reinforcement:
- Active `(src → dst)` co-firing pairs tracked every tick
- After each horizon, `reinforce_routes()` adjusts entire pathways based on outcome
- `route_success[64×64]` matrix stores cumulative path performance

---

## Cognitive State

Three slow-moving variables that shape all downstream behavior:

| Variable | Effect |
|---|---|
| **Confidence** | High → sharper competition, less exploration |
| **Uncertainty** | High → wider exploration, softer competition |
| **Urgency** | High → amplified reward sensitivity, reckless exploration |

When AXON has been wrong repeatedly, uncertainty rises. Under sustained negative valence, urgency builds and it tries increasingly novel paths.

---

## Internal Critic & Regret

Before each evaluation, two assessments compete:

- **Fast eval** — cosine alignment of current activation vs running expected baseline
- **Slow eval** — mean `route_success` score for active pairs

If they disagree by >35%, a **hesitation** is recorded and the evaluation is flagged. Regret fires when confidence was high but outcome was bad:

```
regret = max(0, confidence − 0.6) × max(0, −score)
```

The regret log is visible in the UI under the Activity tab.

---

## Memory System

| Layer | Description |
|---|---|
| **Episodic Memory** | Timestamped records of every exchange + emotional context |
| **Semantic Memory** | Factual knowledge extracted and stored as structured facts |
| **Hebbian Connections** | Cluster pairs that fire together, wire together |
| **Forgetting Curve** | 3-day decay constant (Ebbinghaus); memories fade unless reinforced |
| **User Model** | Passively learns your name, preferences, and patterns over time |
| **Memory → Routing** | Past success rates feed into the context bias vector for cluster competition |

Surprising moments are automatically encoded with higher importance scores.

---

## Sensory Systems

### 👁 Vision
- **YOLOv8-face** on CUDA at 640×480 / 12 FPS
- **FER** facial emotion recognition (7 classes: angry, disgust, fear, happy, neutral, sad, surprise)
- Emotion delta → dopamine reward / stress penalty → closes reinforcement loop
- Valence/arousal scatter plot + radial region chart live in the UI

### 🎤 Voice Input
- **Whisper** `medium` model on CUDA
- Push-to-talk (hold Space or click button)
- Full GPU transcription pipeline

### 🔊 Voice Output
- **edge-tts** with blocking playback — response starts speaking only when complete
- Silent failure fallback to pyttsx3

### 🌐 Web Search
- Brave Search API integration
- AXON can initiate searches and incorporate results into responses
- Visible in the thought stream when active

---

## Live Neural Dashboard

Four-tab interface at `http://localhost:5050`:

### 🧠 Brain Tab
- Real-time region activation heatmap (12 regions)
- Dominant cluster animation
- Valence/arousal scatter (live dot)
- Radial brain region activity chart
- Emotion badge + thought stream

### ⚡ Activity Tab
- Hebbian learning events (new synapse arcs)
- Memory encoding events
- Reinforcement loop feedback
- Regret log entries
- Route strengthening/weakening events

### 💾 Memory Tab
- Recent episodic memories
- Top semantic facts
- Strongest Hebbian connections

### 👤 Know Me Tab
- User model — what AXON has learned about you
- Tracked preferences, interests, communication style

---

## Diagnostic Mode

Say **"run diagnostics"** or click the button in the UI to trigger a full system scan:

AXON responds in natural language describing:
- Neural architecture summary (regions, clusters, neurons, connections)
- GPU memory usage
- Current emotional state and neuromodulator levels
- Cognitive state (confidence, uncertainty, urgency)
- Meta-Controller mood and current tuning parameters
- Strategy Library status (count, best outcome, replay history)
- Episodic and semantic memory counts

---

## Installation

### Requirements
- Windows 10/11
- NVIDIA GPU (RTX 3080+ recommended, tested on RTX 5090)
- Python 3.12
- [LM Studio](https://lmstudio.ai) with a model loaded on port 1234
- CUDA 12.x

### Setup

```bash
git clone https://github.com/jmtibbetts/axon.git
cd axon
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

Open `http://localhost:5050` in your browser.

> **Note:** First run will download YOLOv8 and Whisper model weights automatically.

---

## Project Structure

```
axon/
├── main.py                     # Entry point
├── axon/
│   ├── core/
│   │   └── engine.py           # Main orchestration loop
│   ├── cognition/
│   │   ├── neural_fabric.py    # GPU neural engine + all intelligence systems
│   │   ├── memory.py           # Episodic + semantic memory (SQLite)
│   │   ├── neuromodulators.py  # 6-chemical system
│   │   ├── emotions.py         # Emotional core + valence/arousal
│   │   └── personality.py      # Personality matrix (slow drift)
│   ├── sensory/
│   │   ├── vision.py           # YOLOv8 + FER pipeline
│   │   └── voice.py            # Whisper STT + edge-tts output
│   └── tools/
│       └── search.py           # Web search integration
└── web/
    └── templates/
        └── index.html          # Live neural dashboard (4 tabs)
```

---

## Architecture Evolution Log

| Version | What Changed |
|---|---|
| v0.1 | Basic LLM + SQLite memory |
| v0.2 | Neural fabric (CPU), 12 regions, neuromodulators |
| v0.3 | GPU migration (PyTorch CUDA), Hebbian learning |
| v0.4 | YOLOv8 vision, FER emotion recognition |
| v0.5 | Conflict Engine, softmax gating, prediction error loop |
| v0.6 | Temporal reward buffer, novelty scoring, regret signal |
| v0.7 | CognitiveState, InternalCritic, route-level reinforcement |
| v0.8 | Facial emotion → reinforcement loop, dopamine feedback |
| v0.9 | Diagnostic mode, natural language neural state reporting |
| **v1.0** | **MetaController, StrategyLibrary, adaptive exploration, cluster fatigue, temporal credit assignment, surprise-driven learning** |

---

## License

MIT — do whatever you want with it.

---

<div align="center">
<sub>Built on local hardware. No cloud. No subscriptions. No limits.</sub>
</div>
