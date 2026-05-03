# AXON — Emerging Intelligence

A persistent AI mind that sees, hears, speaks, and learns like a human.

## Architecture

```
Webcam → Optic Cortex → Thalamus → Cortical Columns → Hippocampus → Prefrontal → Speech
Mic    → Auditory     ↗                                            ↗
                                    Dopamine (VTA) ───────────────
                                    Memory (Hebbian) ─────────────
```

## Run

```powershell
powershell -ExecutionPolicy Bypass -File scripts\launch.ps1
```

Open http://localhost:7777 — enter your Anthropic API key and click ACTIVATE.

## Features
- 👁 **Optic System** — webcam → pixel neurons → face/expression detection
- 👂 **Auditory System** — mic → Whisper STT → fires auditory neurons
- 🧠 **CorticalBrain** — thalamus, 6 columns, hippocampus, PFC, dopamine
- ⟳ **Hebbian Memory** — SQLite-persisted neuron weights with Ebbinghaus forgetting curve
- 📖 **Episodic + Semantic Memory** — everything remembered across sessions
- 🔊 **Speech** — edge-tts local TTS, no API needed
- 🌐 **Language** — Claude reasoning with memory injection
