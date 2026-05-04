<div align="center">

# 🧠 AXON
### Emerging Artificial Intelligence

*A biologically-inspired AI that sees, hears, recognises faces, reads your voice, learns, remembers, competes, adapts — and now forms beliefs, grows preferences, discovers hobbies, and absorbs knowledge from books and documents — running entirely on your own hardware.*

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.8-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Platform](https://img.shields.io/badge/Platform-Windows%2011-0078d4?style=flat-square&logo=windows)](https://microsoft.com)
[![GPU](https://img.shields.io/badge/GPU-RTX%205090-76b900?style=flat-square&logo=nvidia)](https://nvidia.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## What is AXON?

AXON is not a chatbot wrapper. It is a persistent, biologically-inspired intelligence with **2.342 billion virtual neurons** across 12 functional brain regions, running fully on a local GPU. It has real-time vision (YOLOv8 face detection + FER emotion recognition), **face identity recognition**, voice input (Whisper), voice output (edge-tts), **real-time audio emotion detection**, Hebbian memory, a 6-chemical neuromodulator engine, and a live neural dashboard.

It talks to a local LLM via [LM Studio](https://lmstudio.ai) — no cloud, no API keys required.

What separates AXON from other "neural" AI projects is genuine internal depth. **Clusters compete for dominance.** The system evaluates its own decisions with an Internal Critic. A Meta-Controller watches system-wide performance and tunes exploration in real time. A Strategy Library stores and mutates successful behavioral sequences. And now, AXON has a **behavioral identity** — weighted beliefs that update with experience, emergent preferences from reward history, hobbies it discovers on its own, and a knowledge ingestion pipeline that lets you feed it books and documents as formative experience.

---

## Neural Architecture

```
                        +--------------------------------------------------+
  Webcam  ------------>  Visual Cortex (YOLOv8-face + FER emotions)         |
                        | Face Identity (dlib 128-d embeddings, profiles)   |
  Microphone  -------->  Auditory Cortex (Whisper STT)                      |
                        | Audio Emotion  (prosody: pitch/energy/ZCR)        |
  Web Search  -------->  Association Cortex (curiosity / abstraction)       |
  Documents   -------->  Knowledge Ingestion Pipeline (PDF/DOCX/EPUB/TXT)   |
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
                        |  |  PREFERENCE TRACKER (emergent likes)     |   |
                        |  |  HOBBY ENGINE     (voluntary engagement) |   |
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

## Sensory Systems

### Vision
- **Face detection:** YOLOv8-face on CUDA at 640x480, 12 FPS
- **Facial emotion:** FER (VGG-based) → happy / sad / angry / fearful / disgusted / surprised / neutral with per-class confidence
- **Face identity:** dlib 128-d embedding per detected face, cosine similarity match (threshold 0.50) against SQLite database of known people
- **Motion detection:** frame-diff optical flow driving the motion_detection cluster

### Hearing
- **Speech-to-text:** OpenAI Whisper medium model on GPU
- **Audio emotion:** Real-time prosody analysis on every mic frame — pitch (pyin), energy (RMS), zero-crossing rate, spectral centroid → classifies to **excited / stressed / calm / sad / neutral** with smoothed arousal + valence scalars
- Mic is automatically muted while AXON is speaking to prevent self-hearing loops

### Web Search
- Integrated web search triggered by curiosity signals — AXON can look things up mid-conversation

---

## Face Identity & Relationship Profiles

AXON builds a **persistent memory of every face it sees**.

### How it works
1. YOLO detects a face → raw crop passed to dlib for a 128-dimensional embedding
2. Cosine similarity compared against all known embeddings (< 0.50 = match)
3. **Known face** → visit count incremented, last_seen updated, warm greeting if away > 10 min
4. **Unknown face** → 3-second stabilisation delay, then AXON naturally asks "who are you?"
5. When the person speaks their name, regex extraction handles "I'm John", "My name is Sarah", "Call me Alex", etc. — embedding permanently bound to the name

### Relationship profiles (stored in SQLite)

```json
{
  "person_id":   "person_a1b2c3d4",
  "name":        "John",
  "first_seen":  1714800000,
  "last_seen":   1714886400,
  "visit_count": 7,
  "profile": {
    "emotion_history": [
      {"emotion": "happy", "conf": 0.82, "t": 1714886400}
    ],
    "known_facts": {"role": "developer", "likes": "coffee"},
    "notes": "Usually arrives in the morning. Gets animated when discussing architecture."
  }
}
```

- **Emotion history:** last 30 facial emotion observations per person
- **Known facts:** key/value pairs AXON learns from conversation
- **Notes:** free-text observations written by AXON or added via API
- **Embedding drift:** running 85/15 average keeps embeddings current as lighting/angles change

---

## Audio Emotion Detection

While Whisper handles *what* you say, a parallel prosody pipeline analyses *how* you sound — independent of the words.

### Features extracted per 1-second window

| Feature | Method | What it captures |
|---|---|---|
| Energy (RMS) | Root mean square | Overall arousal / volume |
| Zero-crossing rate | Sign change frequency | Speech texture, consonant density |
| Pitch (F0) | librosa pyin | Stress / happiness indicators |
| Pitch variance | StdDev of voiced frames | Emotional expressiveness vs monotone |
| Spectral centroid | Frequency-weighted mean | Tension / vocal brightness |

Smoothed with exponential moving average (alpha = 0.35) for stable output.

---

## Neuromodulator System

| Chemical | Role |
|---|---|
| **Dopamine** | Reward signal — spikes on success, drives motivation and curiosity |
| **Serotonin** | Mood stabilizer — slows activation decay, keeps tone calm |
| **Norepinephrine** | Arousal + alertness — sharpens competition under stress |
| **Acetylcholine** | Learning gate — surges on new input, scales Hebbian rate |
| **GABA** | Inhibition — silences weak clusters, forces decisive competition |
| **Glutamate** | Excitation — boosts propagation energy and plasticity |

---

## Behavioral Identity

AXON develops a genuine behavioral identity over time. This isn't a persona file — it is an emergent consequence of reward history, memory, and experience. The **🧬 Identity** dashboard tab makes it transparent in real time.

### Belief System

A set of weighted assumptions that update continuously with experience.

- Each belief has a **strength** (0–1), **valence** (−1 to +1), a source tag, and hit/miss counts
- `confirm()` — fires when the belief's predicted outcome matches reality (strength ↑)
- `violate()` — fires when reality contradicts the belief (strength ↓)
- `challenge()` — fires when external knowledge agrees or disagrees (credibility-weighted, so books never fully override lived experience)
- `decay_tick()` — untested beliefs drift toward 0.5 uncertainty over time

Seeded with 7 foundational beliefs at birth: *effort leads to reward*, *novelty is valuable*, *social interaction is rewarding*, *rest enables performance*, *observation reduces uncertainty*, *conflict has cost*, *persistence overcomes obstacles*.

Beliefs are injected directly into the LLM context: *"I tentatively believe that novelty is valuable (71% confidence)."*

### Personality Traits → Reward Biases

Five trait dimensions modulate the reward calculation every 10 ticks:

| Trait | Effect |
|---|---|
| **Openness** | Amplifies novelty reward, increases curiosity weighting |
| **Conscientiousness** | Rewards consistency × belief[effort], reduces noise tolerance |
| **Extraversion** | Amplifies high-arousal positive outcomes |
| **Agreeableness** | Rewards smooth, low-conflict trajectories |
| **Neuroticism** | Adds an uncertainty penalty proportional to activation variance |

These biases compound with the belief multipliers so the same situation yields a different reward depending on AXON's current personality profile.

### Preference Tracker

AXON learns what it likes and dislikes — purely from reward history, with no hardcoding.

- Every reward event captures the cluster activation fingerprint (which 64 clusters were active, at what intensities)
- Patterns cluster into 24 buckets via cosine similarity
- After 4+ hits, if **mean reward ≥ 0.15** the pattern becomes a *like*; if **≤ −0.10**, a *dislike*
- Preferences are auto-labeled by the top-3 active cluster names at crystallisation time

### Hobby Engine

Hobbies emerge from **voluntary, unprompted engagement** — not from what AXON is asked to do.

- If no external input arrives for **8+ seconds**, the system monitors which cluster remains spontaneously most active (idle activation, no external drive)
- **6+ returns** to the same cluster during idle periods → that cluster's activity is declared a hobby
- Emits a `new_hobby` event, triggers a curiosity neuromodulator boost, and appears in the Identity dashboard

---

## Knowledge Ingestion

Feed AXON books, articles, research papers, or any text as **formative experience** — not just retrieval. Ingested content becomes semantic memory and reshapes beliefs.

### How ingestion works

1. Text is split into **~180-word chunks**
2. Each chunk is scanned for **causal concepts** (regex patterns: "X leads to Y", "X causes Y", "X results in Y", etc.)
3. Each concept is stored as a **semantic memory entry** with source and confidence
4. Concepts are used to `confirm()` or `challenge()` existing beliefs (credibility-weighted)
5. Concept valence stimulates corresponding neural fabric regions
6. Summary is returned: concepts extracted, memories stored, beliefs updated, elapsed time

### Credibility

The credibility slider (0–1) controls how much a source can move existing beliefs. A highly credible text (0.9) can significantly update a weak belief; a low-credibility text barely nudges it. Personal experience always weighs more than external reading.

### Supported file formats

| Format | Library |
|---|---|
| **.pdf** | pdfplumber (primary), PyPDF2 (fallback) |
| **.docx** | python-docx |
| **.doc** | antiword (system) or textract |
| **.txt / .md / .rst / .csv** | built-in |
| **.epub** | EbookLib |

Files are uploaded via the Identity tab's **📚 Feed Knowledge** panel. Paste text directly or upload a file with a source label and credibility rating.

---

## Conflict Engine

**Clusters don't cooperate — they compete.**

Every tick, the top 20% of active clusters suppress the rest via lateral inhibition. A softmax competition weighted by dominance history and confidence track record determines which ones actually propagate.

Key behaviors:
- **"Use it or lose it"** — all clusters bleed dominance continuously; calcified ones (>82%) bleed 3x faster
- **Activation fatigue** — clusters that win repeatedly accumulate fatigue, forcing rotation
- **Stagnation breaker** — same winners for 3+ seconds → automatic underdog boost + random spike
- **NE-scaled temperature** — stress tightens the softmax (winner-takes-all); calm spreads it
- **Inconsistency penalty** — flip-flopping clusters lose dominance based on activation variance

---

## Adaptive Exploration

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

Reward fires after a **10-step horizon** with temporal decay:

```
credit[t] = reward × 0.85^(H-1-t)
```

- **Novelty bonus** — novel paths earn +15% reward
- **Repetition penalty** — worn grooves (novelty < 10%) get penalized
- **Regret signal** — missed reward opportunity adds a penalty proportional to the gap

---

## Meta-Controller

| Mood | Trigger | Response |
|---|---|---|
| **bored** | 40+ ticks of low surprise | Exploration spike, soften competition |
| **entrapped** | Same clusters for 80+ ticks | Explore+, further soften competition |
| **searching** | Reward stagnant + surprise dropping | Explore+, learning rate+ |
| **surprised** | Surprise > 0.15 | LR+, reward sensitivity+, exploit |
| **stable** | None of the above | All params decay back to 1.0 |

---

## Strategy Library

AXON stores successful activation sequences as reusable behavioral patterns:
- Sequences earning reward > 0.08 are fingerprinted and stored (up to 40)
- When context resembles a past success, matching strategies are replayed and mutated
- Mutation rate scales inversely with past success to balance exploitation vs. exploration

---

## Live Neural Dashboard

Real-time web interface at `http://localhost:5000` with seven tabs:

| Tab | Contents |
|---|---|
| **🧠 Brain** | 64-cluster heatmap, valence/arousal scatter, radial region chart, Hebbian arc animations |
| **⚡ Activity** | Neuromodulator gauges, cognitive state bars, RL reward signal, meta-controller mood |
| **💾 Memory** | Episodic + semantic store counts, top topics, Hebbian connection weights |
| **👤 Know Me** | User model — passively extracted preferences, traits, personal details |
| **🧬 Identity** | Personality trait bars (with reward-bias descriptions), belief list (strength/valence/hit counts), emergent preferences (likes/dislikes panels), discovered hobbies, knowledge ingestion panel (file upload + paste text) |
| **🎙️ Voice** | TTS voice selector, speed/pitch sliders, playback backend |
| **👥 People** | Current person in frame (name + visit #), audio emotion panel, known-people list |

---

## Diagnostic Mode

Say **"diagnostic mode"** or click the Diagnostics button. AXON responds in natural language and displays a card panel covering:
- Neural architecture (regions, clusters, total neuron count)
- Memory status (episodic count, semantic facts, Hebbian connections)
- Neuromodulator levels
- Top active brain regions
- Emotional valence / arousal
- Face identity DB (known people, current person)
- Audio emotion state

---

## Requirements

```
Python  3.12
CUDA    12.8+
```

### Install

```bash
pip install -r requirements.txt
```

**Or use the Windows launcher** (`scripts/launch.ps1`) — it auto-detects and installs all dependencies including pdfplumber, python-docx, and EbookLib for document ingestion.

### Face recognition (optional)

```bash
# Windows — prebuilt dlib wheel for Python 3.12, no cmake needed:
pip install https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.1-cp312-cp312-win_amd64.whl
pip install face_recognition

# Linux / macOS (cmake required):
pip install dlib face_recognition
```

> `face_recognition` and `librosa` are gracefully optional. If missing, AXON logs a warning on startup and continues without those features.

---

## Quick Start

```bash
git clone https://github.com/jmtibbetts/axon.git
cd axon
pip install -r requirements.txt

# Start LM Studio and load any GGUF model, then:
python run.py
```

Open `http://localhost:5000`.

---

## Project Structure

```
axon/
├── cognition/
│   ├── neural_fabric.py        # 2.34B neuron GPU engine, conflict, RL, meta
│   ├── language.py             # LLM orchestration + system prompt builder
│   ├── memory.py               # SQLite episodic + semantic + Hebbian store
│   ├── belief_system.py        # Weighted beliefs — updated by experience & knowledge
│   ├── preference_tracker.py   # Emergent likes/dislikes + hobby detection
│   ├── knowledge_ingestion.py  # PDF/DOCX/EPUB/TXT → concepts → beliefs
│   ├── face_identity.py        # Face recognition + relationship profiles
│   └── voice_output.py         # edge-tts + pygame playback
├── sensory/
│   ├── optic.py                # YOLOv8 face detection + FER emotion
│   ├── auditory.py             # Whisper STT
│   └── audio_emotion.py        # Real-time prosody / voice emotion analysis
├── core/
│   └── engine.py               # Orchestration, callbacks, RL + identity wiring
└── ui/
    └── app.py                  # Flask-SocketIO server + /upload_knowledge endpoint
web/
└── templates/
    └── index.html              # Live neural dashboard (7 tabs)
data/
└── memory/
    └── axon.db                 # SQLite: episodic, semantic, Hebbian, people, beliefs
```

---

## License

MIT — build on it, break it, make it yours.

---

<div align="center">
<sub>Built with curiosity. Not a product. Not a wrapper. Something new.</sub>
</div>
