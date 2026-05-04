# Changelog

All notable changes to AXON are documented here.

---

## [1.0.0] — 2026-05-04

### Added
- **MetaController** — second-order system that watches surprise trend, reward stagnation, and dominance entropy and dynamically tunes `explore_rate`, `reward_sensitivity`, `conflict_sharpness`, and `lr_scale`. Modes: `bored`, `entrapped`, `searching`, `surprised`, `stable`.
- **StrategyLibrary** — stores successful cluster activation sequences as fingerprinted behavioral patterns (up to 40). Context-aware replay biases the network toward known-good sequences when emotion + cognitive state match (≥65% cosine similarity). Includes mutation (random noise on 5% of eval cycles) and eviction by outcome score.
- **Cluster Activation Fatigue** — two-layer fatigue system. `ConflictEngine._activation_fatigue` accumulates per winner every tick (bleeds at 0.998/tick, ~10s to halve). `NeuralFabric._cluster_use_count` adds a second long-term wear penalty. Dominant clusters must rest — behavioral rotation is now structurally enforced.
- **Adaptive Exploration** — epsilon now driven by three compounding forces: `CognitiveState.explore_boost()`, `MetaController.explore_rate` multiplier, and a surprise spike (+0.15 when surprise > 0.10). Boredom counter (40 ticks) and entrapment detector (80 ticks) fire automatic exploration pressure.
- **Temporal Credit Assignment** — replaced flat `mean_act` credit with proper per-timestep decay: `credit[t] = reward × 0.85^(H−1−t)`. Earlier decisions get discounted credit vs the outcome moment. Normalized across horizon length.
- **Surprise as a First-Class Signal** — high prediction error now drives: learning rate (up to 2.5×), Hebbian trace boost, episodic memory importance (+0.4×), exploration burst, and MetaController "surprised" mode.
- **Strategy Library + Meta panels** in Diagnostic overlay (🎛️ Meta-Controller, 🧬 Strategy Library).
- **Diagnostic mode** now includes Meta-Controller mood, exploration multiplier, and strategy count/outcome in spoken summary.
- **README v1.0** — full rewrite covering all current systems with updated architecture diagram.

### Changed
- `TemporalRewardBuffer.evaluate()` now accepts `meta_sensitivity` parameter — reward/penalty magnitude is scaled by `MetaController.reward_sensitivity` before credit assignment.
- `ConflictEngine.compete()` — fatigue penalty applied to activation before competition begins; fatigue updated after winners determined.
- `NeuralFabric._gpu_tick()` — complete rewrite integrating all new systems; strategy replay bias blended 40/60 with memory context bias.
- `get_state_snapshot()` — now returns `meta`, `strategy_lib`, and `cluster_wear` fields.
- Episodic memory `importance` now includes `surprise_level × 0.4` boost.

---

## [0.9.0] — 2026-05-04

### Added
- **Diagnostic Mode** — trigger via "run diagnostics" command or UI button. Returns a 6-card panel and natural language spoken response covering neural architecture, memory status, GPU usage, neuromodulator levels, and regional activation.
- **`neural_state_to_prose()`** — natural language description of current valence, arousal, and chemical state injected into the LLM system prompt.
- **Dynamic system prompt** — AXON is kept aware of its camera vision and web search capabilities in the LLM context at all times.
- DeepFace as fallback backend for facial emotion recognition if FER fails.

### Fixed
- Script execution order bug that broke Diagnostic Mode on startup.
- CUDA/PyTorch dependency installation issues with YOLOv8 + FER.
- Environment dependency errors preventing GPU-accelerated FER.

---

## [0.8.0] — 2026-05-04

### Added
- **Reinforcement Learning Loop** — real-time facial emotion deltas trigger dopamine rewards or stress penalties that directly adjust Hebbian pathways and cluster dominance.
- **Valence/arousal scatter plot** — live dot on the Brain tab.
- **Radial brain region chart** — 12-region activation visualization on Brain tab.
- **Persistent feedback loop** — FER emotion tracking provides continuous dopamine/stress signal to the neural fabric between LLM turns.
- UI button for Diagnostic Mode.

### Changed
- Neural dashboard upgraded to 4-tab architecture: Brain, Activity, Memory, Know Me.

---

## [0.7.0] — 2026-05-03

### Added
- **CognitiveState** — three slow-moving global variables: confidence, uncertainty, urgency. Shapes exploration rate, competition temperature, and reward sensitivity.
- **InternalCritic** — fast eval (cosine alignment) + slow eval (route success). Disagreement >35% → hesitation recorded. Regret fires when high confidence meets bad outcome.
- **Route-level structural reinforcement** — active `(src→dst)` cluster pairs tracked; entire pathways reinforced or weakened after each horizon based on outcome.
- **`route_success[64×64]`** matrix for cumulative path performance history.
- Four-tab neural dashboard (Brain / Activity / Memory / Know Me).
- User Model ("Know Me" tab) — passively extracts personal details and preferences.

### Changed
- Conflict engine upgraded with stagnation breaker and underdog rescue.
- Competition temperature now driven by both NE and CognitiveState.

---

## [0.6.0] — 2026-05-03

### Added
- **TemporalRewardBuffer** — 10-step delayed reward horizon. Novelty fingerprinting (cosine similarity vs last 20 paths). Anti-repetition penalty. Regret signal (best possible vs actual final valence).
- **PredictionEngine** — continuous prediction error loop with node-level and route-level weight updates.
- **Novelty bonus** — novel activation paths earn +15% reward; worn paths penalized.

---

## [0.5.0] — 2026-05-03

### Added
- **Conflict Engine** — lateral inhibition, softmax gating, dominance memory, confidence track record.
- "Use it or lose it" dominance decay.
- Calcification prevention (>0.82 dominance bleeds 3×).
- GABA-mediated global inhibition.

---

## [0.4.0] — 2026-05-03

### Added
- **YOLOv8-face** on CUDA at 640×480 / 12 FPS (upgraded from CPU Haar cascades).
- **FER** facial emotion recognition (VGG-based, FER2013, 7 emotion classes).
- Emotion badge in UI. Emotion delta fed into neuromodulator system.

### Changed
- Vision frame resolution increased to 640×480.
- Emotion recognition runs on GPU via CUDA.

---

## [0.3.0] — 2026-05-03

### Added
- Full GPU migration — all neural fabric math runs on PyTorch CUDA tensors.
- Hebbian learning (`fire together → wire together`) on GPU via outer product updates.
- Hebbian synapse visualization — glowing arcs in the Brain tab when co-activation detected.
- Neural dashboard with Hebbian animation system.

### Changed
- Neural tick speed increased significantly due to GPU acceleration.
- Weight matrix: `[N×N] float16` on CUDA.

---

## [0.2.0] — 2026-05-03

### Added
- Neural fabric — 2.342B virtual neurons, 64 clusters, 12 brain regions (CPU).
- 6-chemical neuromodulator system (dopamine, serotonin, norepinephrine, acetylcholine, GABA, glutamate).
- EmotionalCore — valence/arousal state machine.
- PersonalityMatrix — slow-drift personality traits.
- ThoughtStream — cluster-driven thought generation.
- Episodic + semantic memory (SQLite, Ebbinghaus forgetting curve, 3-day decay).
- Live neural dashboard (basic).
- Whisper `medium` model on CUDA for STT.
- edge-tts for voice output with blocking playback.

---

## [0.1.0] — 2026-05-03

### Added
- Initial project structure.
- LM Studio integration (local LLM via OpenAI-compatible API on port 1234).
- Basic SQLite memory (episodic + semantic).
- Flask web server + basic UI.
- Push-to-talk voice input.
