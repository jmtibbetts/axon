# Changelog

All notable changes to AXON are documented here.

---

## [1.4.0] — 2026-05-05

### Added
- **Thought Generator** (`axon/cognition/thought_generator.py`) — the LLM is no longer the brain; it is the imagination engine. Before every response, a full 7-step pipeline runs:
  1. **Goal Conditioning** — injects current goal, emotional state prose (NE/DA/SE), personality vector + behavioral stance, active drives, and dominant worldview into a `[GOAL CONDITIONING]` system prefix seen by the LLM before generation.
  2. **Intelligent Memory Injection** — selects and weights relevant memories from: top strategy outcomes, keyword-overlapping episodic memories, current explore/exploit bias, and one relevant high-confidence belief. Never a raw dump.
  3. **Candidate Generation** — LLM generates `N=3` distinct candidate responses in "THOUGHT GENERATION MODE" (possibilities, not answers). Parsed by numbered list pattern.
  4. **Candidate Scoring** — each candidate is mapped to a cluster activation profile across 12 brain regions via `ACTIVATION_KEYWORDS`. Scored by: neural alignment (dot product vs live activations), personality trait affinity (`TRAIT_REGION_AFFINITY`), reward plausibility (DA/NE/valence → word-level signals), and a de-weighted position prior.
  5. **Conflict Resolution** — `ConflictEngine`-aligned winner: candidate whose activation profile best matches live neural state wins. Fallback to highest score.
  6. **Winner Reasoning** — each candidate annotated with dominant region, score delta vs winner, and reason for winning or suppression.
  7. **Learning Loop** — `record_outcome(delta_valence)` closes the cycle after emotional feedback: rewards winning cluster activations proportional to valence delta, updates strategy library, boosts memory salience for recent episodes, emits `prediction_error` socket event.
- **`ACTIVATION_KEYWORDS`** map — 12 brain regions → keyword lists used to build per-candidate activation profiles.
- **`TRAIT_REGION_AFFINITY`** map — 5 personality traits × region modifiers that bias scoring toward candidate profiles matching the dominant trait.
- **`ThoughtCandidate`** dataclass — `text`, `activations`, `base_score`, `reward_score`, `final_score`, `winner`, `reasoning`.
- **`competition_history`** ring buffer — last 20 thought competitions stored, accessible via `recent_competitions(n)`.
- **"Competing Thoughts" UI panel** — live panel above Autonomous Reflections showing each round: all candidates with score bars, dominant region annotation, ✓ WINNER badge, suppression reasoning. Glows amber when a new round arrives. Displays up to 5 most recent rounds.
- **`socket.on('thought_competition')`** — live prepend of each new competition round to the UI panel.
- **`GET /api/brain/thought_competition`** — REST endpoint returning last N competition rounds.
- **`socket emit 'prediction_error'`** — emitted from `record_outcome()` with error magnitude, delta, source, and winning candidate preview.

### Changed
- `_think()` in `engine.py` — routes through `thought_gen.generate()` instead of `language.think()` directly. History, memory storage, user model ingestion, and Hebbian co-activations preserved as direct calls post-generation. Falls back to `language.think()` if ThoughtGenerator errors.
- `engine.py` — emits `thought_competition` socket event with full candidate log after each response.
- `engine.py` — emotional feedback block now calls `thought_gen.record_outcome(delta_valence)` to close the learning loop.
- `ThoughtGenerator` initialized after `CognitiveCycle` starts — receives `language`, `fabric`, `memory`, and `engine` references.

---

## [1.3.0] — 2026-05-05

### Added
- **Reflection Engine** (`axon/cognition/reflection_engine.py`) — autonomous thought loop that fires every ~15 seconds. Reads its own cluster activations, belief tensions, drive levels, and chemical state to form first-person reflections (*"Hippocampus and amygdala are both high — I'm recalling something emotionally weighted."*). High-confidence reflections are written to Identity-tier memory and fed back into the self-model's `I notice` slot. Live **Reflections panel** added to the neural UI tab. API: `GET /api/brain/reflections`.
- **Narrative Threads** (`axon/cognition/narrative_threads.py`) — 7 competing worldviews (Efficiency First, Explore at All Costs, Safety Above All, Social Harmony, Intellectual Dominance, Emotional Truth, Pragmatic Realist) fight for narrative dominance based on live cluster activations. Dominance flips fire surprise events. **Worldviews panel** with live salience bars added to UI. API: `GET /api/brain/narratives`. Socket: `narrative_shift` event.
- **Memory Hierarchy** (`axon/cognition/memory_hierarchy.py`) — 4-tier memory architecture: Episodic (fast decay, 2000 cap), Semantic (medium, 5000), Value (slow, 1000), Identity (very slow, 500). Each tier has its own decay rate, salience weighting, and capacity pruning. High-surprise events and confirmed reflections receive salience boosts. **Memory Tier browser** with per-tier counts and salience bars added to UI. API: `GET /api/brain/memory_hierarchy`.
- **Personality Vector — fully wired** — all 5 traits (curiosity, risk, stability, persistence, neuroticism) now directly drive system behavior: curiosity raises ε floor, risk raises ε ceiling, stability dampens ε volatility, persistence hardens cluster dethroning resistance, neuroticism amplifies NE swings on prediction error.
- **Weight-driven neural canvas** — axon route lines now scale in thickness (0.5–4.5px) and glow intensity with live Hebbian weight. Strong Hebbian paths are visibly brighter and thicker. Glow halo added for connections with weight > 0.4.
- **Cluster force-physics** — dominant clusters (dominance > 0.25) smoothly drift from fixed centroids toward a spring-repulsion equilibrium position, making the canvas physically respond to neural state.
- **Thought bubbles** — when a cluster's dominance exceeds 0.65, a floating italic decision label pops above it (*"Evaluating options."*, *"Pattern match found."*, *"Threat detected."*, etc.). Labels float upward and fade. Defined for all 12 regions.
- **Pruning fade** — weakened or pruned connections render as faded dashed lines for 8 seconds before disappearing. Tracked in `_prunedConnections` set.
- **Emotional State bar** — persistent canvas overlay showing live NE level, reward trend direction, surprise magnitude, and a text mood label (curious / alert / bored / stressed / calm / entrapped / surprised).
- **`_hebbianWeights` live map** — fed by `brain_state` socket events and hydrated on load via `GET /api/brain/memory`. Used by axon route renderer for weight-driven visuals.

### Changed
- Central cognitive loop extended from 9 to 12 steps: added `reflection_engine.tick()`, `narrative_threads.tick()`, and `memory_hierarchy.prune_if_needed()`.
- Knowledge ingestion (`axon/cognition/knowledge_ingestion.py`) now writes to memory hierarchy tiers in addition to SQLite; emits `competing_interpretations` list per chunk.
- Architecture diagram updated in README to include all new subsystems.

---

## [1.2.0] — 2026-05-05

### Added
- **Multi-provider LLM support** (`axon/cognition/providers.py`) — switch between LM Studio (default), OpenAI, Anthropic Claude, Google Gemini, and Groq at runtime via a **LLM Provider** tab in the UI. Configuration persisted in `providers.json`.
- **DeepFace fallback** — if FER fails to load, vision pipeline falls back to DeepFace for emotion recognition automatically.
- Face processing module patched to handle zero-length embeddings without crashing the vision thread.

### Fixed
- `ValueError` on zero-length face embeddings in `_cosine_dist` — added safety check to skip or repair invalid identity data.

---

## [1.1.0] — 2026-05-04

### Added
- **Winner/Loser Dominance Visualization** — top 20% active regions classified as winners every 8 frames. Winners: enlarged nodes (+70% size), brighter halos, faster pulse rings, glowing label pills, ▲ crown marker. Losers: dimmed to 45% alpha, reduced size, faint cross-out.
- **Conflict Tension Arcs** — competing high-activation regions get dual jittering arcs between them showing visible tension. Each region's color fights the other; arcs wiggle via sin wave. White spark at midpoint. Auto-fade after 1.2s.
- **Learning Bursts** — ripple animations fire at region centroids on surprise (>0.12) and reward delta (>0.08). Types: surprise (white, 3 rings), reward (green), penalty (red), weight_update (purple). Rings expand and fade.
- **Activation Trails** — comet-tail trail dots left by neurons firing >0.55. Sampled every 12 frames, fade over ~2.5s. Gives visual continuity of thought.
- **Ghost Paths** — dashed arcs between top-2 active regions showing memory-biased learned routes. Midpoint dot, semi-transparent, fade over 2s. "This decision came from experience."
- **Cluster Fatigue Overlay** — fatigueMap updated from `cluster_wear`. Dims neurons proportional to overuse, adds fragmentation shimmer, ⚡ label suffix.
- **Exploration vs Exploitation Mode** — explore mode fires random loser flares; exploit mode draws thick arcs between winner regions. Mode badge shows live mood + epsilon % with color coding.
- **Decision Playback Mode** — ring buffer of last 10 activation frames. ⏮ PLAYBACK button + ◀ ▶ scrubber to step through. Shows which clusters dominated each step with burst highlights. EXIT restores live state.

---

## [1.0.0] — 2026-05-04

### Added
- **MetaController** — second-order system that watches surprise trend, reward stagnation, and dominance entropy and dynamically tunes `explore_rate`, `reward_sensitivity`, `conflict_sharpness`, and `lr_scale`. Modes: `bored`, `entrapped`, `searching`, `surprised`, `stable`.
- **StrategyLibrary** — stores successful cluster activation sequences as fingerprinted behavioral patterns (up to 40). Context-aware replay biases the network toward known-good sequences when emotion + cognitive state match (≥65% cosine similarity). Includes mutation (random noise on 5% of eval cycles) and eviction by outcome score.
- **Cluster Activation Fatigue** — two-layer fatigue system. `ConflictEngine._activation_fatigue` accumulates per winner every tick (bleeds at 0.998/tick, ~10s to halve). `NeuralFabric._cluster_use_count` adds a second long-term wear penalty. Dominant clusters must rest — behavioral rotation is now structurally enforced.
- **Adaptive Exploration** — epsilon now driven by three compounding forces: `CognitiveState.explore_boost()`, `MetaController.explore_rate` multiplier, and a surprise spike (+0.15 when surprise > 0.10). Boredom counter (40 ticks) and entrapment detector (80 ticks) fire automatic exploration pressure.
- **Temporal Credit Assignment** — replaced flat `mean_act` credit with proper per-timestep decay: `credit[t] = reward × 0.85^(H−1−t)`. Earlier decisions get discounted credit vs the outcome moment. Normalized across horizon length.
- **Surprise as a First-Class Signal** — high prediction error now drives: learning rate (up to 2.5×), Hebbian trace boost, episodic memory importance (+0.4×), exploration burst, and MetaController "surprised" mode.
- **Strategy Library + Meta panels** in Diagnostic overlay.
- **Diagnostic mode** now includes Meta-Controller mood, exploration multiplier, and strategy count/outcome in spoken summary.

### Changed
- `TemporalRewardBuffer.evaluate()` now accepts `meta_sensitivity` parameter.
- `ConflictEngine.compete()` — fatigue penalty applied before competition begins.
- `NeuralFabric._gpu_tick()` — complete rewrite integrating all new systems.
- `get_state_snapshot()` — now returns `meta`, `strategy_lib`, and `cluster_wear` fields.

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
