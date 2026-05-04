<div align="center">

# 🧠 AXON
### Emerging Artificial Intelligence

*A biologically-inspired AI that sees, hears, recognises faces, reads your voice, learns, remembers, competes, forms beliefs, grows drives, and now thinks in a synchronized cognitive cycle — running entirely on your own hardware.*

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

```
Python  3.12
CUDA    12.8+
```

### Install

```bash
pip install -r requirements.txt
```

### Face recognition (optional)

```bash
# Windows — prebuilt dlib wheel, no cmake needed:
pip install https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.1-cp312-cp312-win_amd64.whl
pip install face_recognition

# Linux / macOS:
pip install dlib face_recognition
```

> `face_recognition` and `librosa` are gracefully optional.

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
