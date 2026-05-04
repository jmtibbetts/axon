<div align="center">

# 🧠 AXON
### Emerging Artificial Intelligence

*A biologically-inspired AI that sees, hears, speaks, learns, and remembers — running entirely on your own hardware.*

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
                        │   acetylcholine, cortisol, oxytocin)        │
                        │                                             │
                        │   DEFAULT MODE   CEREBELLUM   METACOGNITION │
                        │   ASSOCIATION    SOCIAL BRAIN               │
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

AXON runs 6 simulated brain chemicals that continuously shape cognition:

| Chemical | Role |
|---|---|
| **Dopamine** | Reward signal — spikes on success, drives motivation and curiosity |
| **Serotonin** | Mood stabilizer — keeps tone calm and socially engaged |
| **Norepinephrine** | Arousal + alertness — sharpens focus under demand |
| **Acetylcholine** | Learning gate — surges during new information to enable encoding |
| **Cortisol** | Stress response — elevated by threat detection, impairs memory at high levels |
| **Oxytocin** | Social bonding — rises during positive interactions, enhances empathy |

These values directly influence Hebbian weight updates, LLM temperature, and emotional state outputs.

---

## Memory System

AXON uses a **SQLite-backed dual-memory architecture**:

- **Episodic Memory** — timestamped records of every conversation, experience, and event
- **Semantic Memory** — factual knowledge extracted over time (people, concepts, preferences)
- **Hebbian Learning** — co-firing clusters strengthen their synaptic connections (`fire together → wire together`)
- **Ebbinghaus Forgetting** — memory strength decays with a 3-day time constant unless reinforced
- **User Model** — passively learns your name, preferences, personality traits, and interests over sessions

Memory persists across restarts. AXON remembers previous conversations.

---

## Sensory Systems

### 👁 Vision — `axon/sensory/optic.py`
- **YOLOv8-face** running on CUDA at 640×480 / 12 FPS
- **FER** (Facial Emotion Recognition) — VGG-based model, 7 emotion classes
- Detected faces and emotions fire directly into Visual Cortex + Social Brain clusters
- Emotion deltas feed the Hebbian reinforcement loop (positive emotion → dopamine reward)

### 👂 Hearing — `axon/sensory/auditory.py`
- **OpenAI Whisper** (`medium` model) running on GPU
- Continuous microphone capture with VAD silence detection
- Transcribed speech fires into Auditory Cortex + Language System

### 🔊 Voice — `axon/cognition/voice_output.py`
- **edge-tts** — Microsoft neural voices, fully local, no API
- Blocking playback via pygame to prevent response overlap
- Configurable voice, rate, pitch from UI

---

## Web Search

AXON can search the internet in real time using DuckDuckGo HTML scraping + Wikipedia API — **no API key needed**.

Triggers automatically on factual questions (`what is...`, `who is...`, `latest...`) and explicit requests (`search for...`, `find me...`). Search results are injected into the LLM context window before generating a response.

---

## Diagnostic Mode

Type or say **`diagnostic`** to open the live diagnostic panel, which shows:

- Real-time neuron count, active connections, GPU memory usage
- Per-region cluster breakdown with neuron counts
- Neuromodulator bar charts (live values)
- Emotion state (valence + arousal)
- Memory stats (episodic count, semantic facts, top Hebbian pathways)
- Active capabilities (vision, hearing, voice, LLM model)
- Platform info (Python version, uptime)

You can also ask AXON directly:
- `"describe your brain"` — full written breakdown of all 12 regions
- `"tell me about your hippocampus"` — deep dive on any single region
- `"what are your neuromodulators"` — live chemical levels + descriptions
- `"describe yourself"` — capability overview

---

## UI Dashboard

The web UI at `http://localhost:7777` features 4 tabs:

| Tab | What you see |
|---|---|
| **Brain** | Live neural canvas (2,420-node firing visualization) · emotion badge · valence/arousal scatter · region activation radial chart |
| **Activity** | Thought stream · Hebbian learning arcs · recent memories |
| **Memory** | Episodic timeline · semantic facts browser |
| **Know Me** | User model — what AXON has learned about you |

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

> **First run:** Make sure LM Studio is running with a model loaded on `http://localhost:1234`. AXON auto-detects the model.

---

## Project Structure

```
axon/
├── axon/
│   ├── core/
│   │   └── engine.py          # Master orchestrator — wires all systems together
│   ├── cognition/
│   │   ├── neural_fabric.py   # 2.34B neuron GPU tensor engine (PyTorch CUDA)
│   │   ├── language.py        # LLM interface + web search + system prompt
│   │   ├── memory.py          # SQLite episodic + semantic + Hebbian memory
│   │   ├── user_model.py      # Passive user preference learning
│   │   └── voice_output.py    # edge-tts + pygame playback
│   ├── sensory/
│   │   ├── optic.py           # YOLOv8-face + FER vision pipeline
│   │   └── auditory.py        # Whisper GPU speech-to-text
│   └── ui/
│       └── app.py             # Flask-SocketIO web server
├── web/
│   └── templates/
│       └── index.html         # Full single-page dashboard UI
├── scripts/
│   └── launch.ps1             # One-click Windows installer + launcher
├── requirements.txt
└── README.md
```

---

## Philosophy

AXON isn't trying to replicate a human brain exactly — it's drawing inspiration from neuroscience to build something that *behaves* more like a mind than a chatbot. The Hebbian learning means repeated topics genuinely strengthen pathways. The neuromodulators mean context and emotional state affect how AXON responds. The forgetting curve means memories fade unless reinforced. The default mode network means AXON has background activity even when idle.

The goal: an intelligence that grows with you over time.

---

## License

MIT — do whatever you want with it.

---

<div align="center">
<i>Built by jmtibbetts · Running on an RTX 5090 · No cloud required</i>
</div>
