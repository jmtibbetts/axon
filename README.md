<div align="center">

# 🧠 AXON
### Emerging Artificial Intelligence

*A biologically-inspired AI that sees, hears, speaks, learns, remembers — and now competes, adapts, and questions itself — running entirely on your own hardware.*

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.8-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Platform](https://img.shields.io/badge/Platform-Windows%2011-0078d4?style=flat-square&logo=windows)](https://microsoft.com)
[![GPU](https://img.shields.io/badge/GPU-RTX%205090-76b900?style=flat-square&logo=nvidia)](https://nvidia.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## What is AXON?

AXON is not a chatbot wrapper. It is a persistent, biologically-inspired intelligence with **2.342 billion virtual neurons** organized into 12 functional brain regions, running fully on your local GPU. It has real-time vision (YOLOv8 face detection + FER emotion recognition), voice input (Whisper), voice output (edge-tts), a Hebbian learning memory system, a 6-chemical neuromodulator engine, and a live neural dashboard you can watch firing in real time.

It talks to a local LLM via [LM Studio](https://lmstudio.ai) — no cloud, no API keys required.

What separates AXON from other "neural" AI projects is that it now has **genuine internal conflict**. Clusters fight for dominance, past winners decay if they stop proving themselves, and the system can hesitate, feel regret, and re-route when its own internal critic disagrees with its decisions.

---

## Neural Architecture

```
                        ┌─────────────────────────────────────────────┐
  Webcam ──────────────▶│  Visual Cortex (YOLOv8 + FER)               │
  Microphone ──────────▶│  Auditory Cortex (Whisper STT)              │
                        │                                             │
                        │   THALAMUS (attention gate + relay)         │
                        │         ↓                ↓                  │
                        │   PREFRONTAL         HIPPOCAMPUS            │
                        │  (executive,         (encode/retrieve,      │
                        │   planning,           episodic memory,      │
                        │   decisions)          pattern completion)   │
                        │         ↓                ↓                  │
                        │   AMYGDALA          LANGUAGE SYSTEM         │
                        │  (fear/reward)      (LLM ↔ semantic mem)   │
                        │         ↓                ↓                  │
                        │   NEUROMODULATORS ─────────────────────────▶│
                        │  (dopamine, serotonin, norepinephrine,      │
                        │   acetylcholine, GABA, glutamate)           │
                        │                                             │
                        │   DEFAULT MODE   CEREBELLUM   METACOGNITION │
                        │   ASSOCIATION    SOCIAL BRAIN               │
                        │                                             │
                        │   ┌─────────────────────────────────────┐  │
                        │   │  CONFLICT ENGINE                     │  │
                        │   │  (lateral inhibition · dominance)    │  │
                        │   │  COGNITIVE STATE                     │  │
                        │   │  (confidence · uncertainty · urgency)│  │
                        │   │  INTERNAL CRITIC                     │  │
                        │   │  (self-eval · hesitation · regret)   │  │
                        │   └─────────────────────────────────────┘  │
                        └──────────────────────────┬──────────────────┘
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
| **Default Mode Network** | 230M | 6 | Self-reflection, identity, future simulation, daydreaming |
| **Thalamus** | 59M | 4 | Sensory relay, attention filtering, consciousness gate |
| **Cerebellum** | 200M | 5 | Timing, sequence prediction, error correction |
| **Association Cortex** | 205M | 6 | Creativity, analogy, abstract reasoning, curiosity |
| **Social Brain** | 125M | 5 | Empathy, face recognition, mentalizing, social pain |
| **Metacognition** | 113M | 5 | Self-monitoring, uncertainty tracking, conflict detection |

**Total: 2,342,000,000 virtual neurons · 64 clusters · ~500+ inter-cluster connections**

---

## Neuromodulator System

| Chemical | Role |
|---|---|
| **Dopamine** | Reward signal — spikes on success, drives motivation and curiosity |
| **Serotonin** | Mood stabilizer — keeps tone calm, slows activation decay |
| **Norepinephrine** | Arousal + alertness — sharpens competition temperature under stress |
| **Acetylcholine** | Learning gate — surges during new input, scales Hebbian learning rate |
| **GABA** | Inhibition — silences weak clusters, forces sharper decisions |
| **Glutamate** | Excitation — boosts propagation energy and synaptic plasticity |

---

## Conflict Engine

The most important architectural addition. **Clusters don't cooperate — they compete.**

Every tick, the top 20% of active clusters suppress the rest via lateral inhibition. A softmax competition weighted by each cluster's **dominance history** and **confidence track record** determines which ones actually propagate their signal forward.

Key behaviors:

- **"Use it or lose it"** — all clusters bleed dominance continuously. Dominant ones (>82%) bleed 3× faster. A cluster that stops proving itself *loses its position*.
- **Stagnation breaker** — if the same winners hold for 3+ seconds, an underdog boost fires automatically, plus a random non-winner gets spiked to create genuine challenge.
- **GABA gating** — when GABA is high, weak clusters are silenced entirely.
- **NE-scaled temperature** — stress tightens the softmax (winner-takes-all). Calm states allow broader spread.

---

## Prediction Error & Continuous Learning

AXON maintains a running expectation of itself and learns from the gap — every 5 ticks:

```
error = actual_activation - predicted_activation
Δw_ij ∝ spike_i × error_j
```

AXON also tracks **structural paths** (route-level, not just node-level):

- Active `(src → dst)` co-firing pairs are recorded every tick
- After every 10-step sequence, `reinforce_routes()` adjusts entire pathways based on outcome
- The `route_success[N,N]` matrix stores each path's cumulative performance history
- Top learned routes are surfaced in the UI in real time

---

## Temporal Reward & Novelty

No instant gratification. **Reward fires after 10 steps.**

After each horizon window:
1. Valence trajectory evaluated — did things actually get better?
2. **Novelty bonus** — path is fingerprinted and compared to last 20 paths. Novel paths earn +15% reward. Worn grooves get penalized.
3. **Regret signal** — compares actual final valence to the best valence seen anywhere in the window. Leaving reward on the table = penalty.
4. Dominant clusters across the window get credited or blamed.

The system must *sustain* good outcomes across a sequence, not spike once and coast.

---

## Cognitive State

Three slow-moving global variables that shape all downstream behavior:

| Variable | Tracks | Effect |
|---|---|---|
| **Confidence** | How sure AXON is about its current direction | High → sharper competition, less exploration |
| **Uncertainty** | Epistemic — how much it doesn't know | High → more exploration, softer competition |
| **Urgency** | Temporal pressure from unmet goals | High → amplified reward sensitivity, reckless exploration |

When AXON has been wrong repeatedly, uncertainty rises, it explores wider, competition softens. When under sustained negative valence, urgency builds and it starts trying increasingly novel paths.

---

## Internal Critic & Regret

Before each evaluation, two competing self-assessments run:

- **Fast eval** — cosine alignment of current activation vs running expected baseline. "Are we doing what we usually do?"
- **Slow eval** — mean `route_success` score for active pairs. "Have these routes historically worked?"

When fast and slow **disagree by more than 35%**, a **hesitation** is recorded.

**Regret** fires when confidence was high but outcome was bad:
```
regret = (confidence - 0.6) × max(0, -score)
```
Those pathways get de-weighted. The regret log is visible in the UI.

---

## Memory System

- **Episodic Memory** — timestamped records of every conversation and event
- **Semantic Memory** — factual knowledge extracted over time
- **Hebbian Learning** — `fire together → wire together`
- **Ebbinghaus Forgetting** — 3-day decay constant, fades unless reinforced
- **User Model** — passively learns your name, preferences, and personality
- **Memory → Routing** — memory success rates feed into the context bias vector that shapes cluster competition. What worked before gets a thumb on the scale.

Memory persists across restarts.

---

## Sensory Systems

### 👁 Vision
- **YOLOv8-face** on CUDA at 640×480 / 12 FPS
- **FER** facial emotion recognition — 7 emotion classes
- Emotion deltas feed Hebbian reinforcement and route learning

### 👂 Hearing
- **OpenAI Whisper** `medium` model on GPU
- Continuous mic capture with VAD silence detection

### 🔊 Voice
- **edge-tts** — Microsoft neural voices, fully local
- Blocking playback via pygame (no overlap)

### 🌐 Web Search
- DuckDuckGo + Wikipedia API — no API key needed
- Triggers automatically on factual questions

---

## UI Dashboard

| Tab | What you see |
|---|---|
| **Brain** | Live 2,420-node neural canvas · color-coded region labels · emotion badge · valence/arousal scatter · Conflict Engine panel |
| **Activity** | Thought stream · Hebbian arcs · recent memories |
| **Memory** | Episodic timeline · semantic facts browser |
| **Know Me** | User model — what AXON has learned about you |

### Conflict Engine Panel (Brain tab)

| Field | Description |
|---|---|
| Dominant clusters | Which clusters are winning competition right now |
| Confidence / Uncertainty / Urgency | Live bars for all three CognitiveState variables |
| Prediction surprise | How different reality was from expectation |
| Exploration ε | Current exploration rate (rises with uncertainty/urgency) |
| Cumulative reward | Total reward accumulated since launch |
| Regret score | Last regret magnitude |
| Hesitation count | Times fast/slow eval disagreed |
| Top learned routes | Strongest `src → dst` pathways by route_success |

---

## Diagnostic Mode

Type or say **`diagnostic`** for the live panel. Or ask:
- `"describe your brain"` — all 12 regions
- `"describe your cognitive state"` — confidence, uncertainty, urgency
- `"what are you uncertain about"` — CognitiveState self-reflection

---

## Architecture Summary

```
Neural Fabric (GPU, 20Hz tick loop)
│
├── ConflictEngine
│   ├── lateral inhibition (top 20% suppress rest)
│   ├── softmax competition (temperature = f(NE, uncertainty))
│   ├── dominance history (winners build track record)
│   ├── "use it or lose it" decay (0.0003/tick, 3x for calcified)
│   └── stagnation breaker (underdog rescue after 60 ticks)
│
├── PredictionEngine
│   ├── node-level: Δw ∝ spike_i × error_j  (every 5 ticks)
│   ├── route_success[N,N] matrix (structural path memory)
│   └── reinforce_routes() (reward/penalise full pathways)
│
├── TemporalRewardBuffer
│   ├── 10-step horizon (no instant reward)
│   ├── novelty fingerprint (cosine vs last 20 paths)
│   ├── anti-repetition penalty (novelty < 0.10)
│   └── regret signal (actual vs best-possible in window)
│
├── CognitiveState
│   ├── confidence  → competition sharpness
│   ├── uncertainty → exploration rate
│   └── urgency     → reward sensitivity + exploration
│
└── InternalCritic
    ├── fast_eval  (cosine alignment vs expected baseline)
    ├── slow_eval  (mean route_success for active pairs)
    ├── hesitation (fast/slow disagreement > 35%)
    └── regret log (confidence was high, outcome was bad)
```

---

## Requirements

- Windows 11
- Python 3.12
- NVIDIA GPU (RTX 3000+ recommended; RTX 5090 for full scale)
- CUDA 12.8 + PyTorch nightly
- [LM Studio](https://lmstudio.ai) running locally with any model loaded

---

## Installation

```powershell
git clone https://github.com/jmtibbetts/axon.git
cd axon
powershell -ExecutionPolicy Bypass -File scripts\launch.ps1
```

The launch script will:
1. Create a Python 3.12 virtual environment
2. Install PyTorch nightly (CUDA 12.8) + all dependencies
3. Start the AXON web server
4. Open `http://localhost:7777` in your browser

> **First run:** Make sure LM Studio is running with a model loaded on `http://localhost:1234`.

---

## Project Structure

```
axon/
├── axon/
│   ├── core/
│   │   └── engine.py          # Master orchestrator
│   ├── cognition/
│   │   ├── neural_fabric.py   # 2.34B neuron GPU engine
│   │   │                      #   ConflictEngine · PredictionEngine
│   │   │                      #   TemporalRewardBuffer · CognitiveState
│   │   │                      #   InternalCritic · PersonalityMatrix
│   │   ├── language.py        # LLM interface + web search
│   │   ├── memory.py          # SQLite episodic + semantic + Hebbian
│   │   ├── user_model.py      # Passive user learning
│   │   └── voice_output.py    # edge-tts + pygame
│   ├── sensory/
│   │   ├── optic.py           # YOLOv8-face + FER
│   │   └── auditory.py        # Whisper GPU STT
│   └── ui/
│       └── app.py             # Flask-SocketIO server
├── web/
│   └── templates/
│       └── index.html         # Single-page dashboard
├── scripts/
│   └── launch.ps1             # One-click installer + launcher
├── requirements.txt
└── README.md
```

---

## Philosophy

AXON draws inspiration from neuroscience to build something that *behaves* more like a mind than a chatbot. Hebbian learning means repeated topics genuinely strengthen pathways. Neuromodulators mean emotional state affects responses. The forgetting curve means memories fade unless reinforced. The default mode network means AXON has background activity even when idle.

The conflict and cognitive systems push this further: AXON doesn't just process — it *struggles*. Clusters compete for influence. Winners calcify and get dethroned. The system discovers novel paths, regrets wasted opportunities, and hesitates when it isn't sure.

**No tension = no adaptation = no intelligence.**

The goal: an intelligence that grows with you over time — and earns its behaviors rather than being given them.

---

## License

MIT — do whatever you want with it.

---

<div align="center">
<i>Built by jmtibbetts · Running on an RTX 5090 · No cloud required</i>
</div>
