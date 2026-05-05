"""
AXON — Public Brain API Layer
==============================
AxonBrain wraps AxonEngine with a clean, stable public interface.

This is the boundary layer for:
  - backend HTTP APIs
  - SDK consumers
  - future monetization (rate limiting, feature gating, billing)
  - external scripting and automation

Usage:
    from axon.core.brain_api import AxonBrain

    brain = AxonBrain()
    brain.start()
    brain.step({"type": "text", "content": "Hello"})
    state = brain.get_state()
    brain.stop()
"""

from __future__ import annotations

import time
import threading
import json
import os
from typing import Optional, Any
from pathlib import Path


class AxonBrain:
    """
    Thin, stable public wrapper around AxonEngine.
    All external interaction should go through this class.
    """

    VERSION = "1.0.0"

    def __init__(self, engine=None, data_dir: str = None):
        """
        engine   : existing AxonEngine instance (wired by app.py)
        data_dir : path for persistence files (defaults to ./data)
        """
        self._engine   = engine
        self._data_dir = Path(data_dir or os.path.join(os.getcwd(), "data"))
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._lock     = threading.Lock()
        self._callbacks: list = []

    # ────────────────────────────────────────────────────────────────
    # Lifecycle
    # ────────────────────────────────────────────────────────────────

    def start(self, enable_camera: bool = True, enable_mic: bool = True,
              camera_index: int = -1, mic_index: int = None) -> dict:
        """Start all sensory and cognitive subsystems."""
        if self._engine is None:
            return {"ok": False, "error": "No engine attached"}
        try:
            self._engine.start(
                enable_camera=enable_camera,
                enable_mic=enable_mic,
                camera_index=camera_index,
                mic_index=mic_index,
            )
            return {"ok": True, "message": "AXON started"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def stop(self) -> dict:
        """Stop all subsystems."""
        if self._engine is None:
            return {"ok": False, "error": "No engine attached"}
        try:
            self._engine.stop()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def reset(self) -> dict:
        """Reset transient state (does NOT wipe persisted memory)."""
        if self._engine is None:
            return {"ok": False, "error": "No engine attached"}
        try:
            self._engine.fabric.neuromod.reset()
            self._engine._emotion_history.clear()
            self._engine._last_face_data = {}
            return {"ok": True, "message": "Transient state reset"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ────────────────────────────────────────────────────────────────
    # Core interaction
    # ────────────────────────────────────────────────────────────────

    def step(self, input_data: dict) -> dict:
        """
        Submit a single step of input to the brain.

        input_data keys:
            type    : "text" | "image" | "tick" (required)
            content : text string for type="text"
        """
        if self._engine is None:
            return {"ok": False, "error": "No engine attached"}
        itype = input_data.get("type", "text")
        try:
            if itype == "text":
                text = input_data.get("content", "").strip()
                if text:
                    self._engine._on_transcript(text)
                return {"ok": True, "type": "text"}
            elif itype == "tick":
                # Manual cognitive tick (for headless/autonomous mode)
                if hasattr(self._engine, "cycle") and self._engine.cycle:
                    self._engine.cycle._tick()
                return {"ok": True, "type": "tick"}
            else:
                return {"ok": False, "error": f"Unknown input type: {itype}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def ingest(self, text: str, source: str = "api",
               credibility: float = 0.7) -> dict:
        """
        Ingest knowledge text — creates beliefs, facts, opinions.
        Returns the ingestion summary.
        """
        if self._engine is None:
            return {"ok": False, "error": "No engine attached"}
        try:
            result = self._engine.ingest_knowledge(text, source=source,
                                                    credibility=credibility)
            return {"ok": True, **result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ────────────────────────────────────────────────────────────────
    # State
    # ────────────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        """
        Return a complete public snapshot of brain state.
        Safe to JSON-serialize and expose over HTTP.
        """
        if self._engine is None:
            return {"ok": False, "error": "No engine attached"}
        try:
            e = self._engine
            fabric_snap = e.fabric.get_state_snapshot()
            identity    = e.get_identity_summary()
            pers_vec    = e.fabric.personality.to_dict() if hasattr(e.fabric, "personality") else {}

            return {
                "ok": True,
                "version": self.VERSION,
                "timestamp": time.time(),
                "neural": {
                    "total_neurons":  fabric_snap.get("total_neurons", 0),
                    "active_regions": {
                        k: round(v, 3)
                        for k, v in (fabric_snap.get("regions") or {}).items()
                        if v > 0.05
                    },
                    "emotion":        fabric_snap.get("emotion", "neutral"),
                    "valence":        round(fabric_snap.get("valence", 0.0), 3),
                    "arousal":        round(fabric_snap.get("arousal", 0.0), 3),
                    "surprise":       round(getattr(e.fabric, "_last_surprise", 0.0), 3),
                },
                "neuromod": fabric_snap.get("neuromod", {}),
                "personality": pers_vec,
                "drives":      identity.get("drives", {}),
                "beliefs":     identity.get("beliefs", [])[:10],
                "preferences": identity.get("preferences", {}),
                "memory": {
                    "episodes": e.memory.count_episodes(),
                    "facts":    len(e.memory.all_facts() or {}),
                },
                "cycle_metrics": identity.get("cycle_metrics", {}),
            }
        except Exception as ex:
            return {"ok": False, "error": str(ex)}

    def explain_last_decision(self) -> dict:
        """
        Return a human-readable explanation of the last cognitive decision.
        Shows which clusters won, why, what memories/beliefs influenced it,
        and the system's confidence.
        """
        if self._engine is None:
            return {"ok": False, "error": "No engine attached"}
        try:
            e      = self._engine
            fabric = e.fabric
            snap   = fabric.get_state_snapshot()

            # Winning/losing clusters by activation
            regions  = snap.get("regions") or {}
            sorted_r = sorted(regions.items(), key=lambda x: x[1], reverse=True)
            winners  = [(n, round(v, 3)) for n, v in sorted_r[:5] if v > 0.05]
            losers   = [(n, round(v, 3)) for n, v in sorted_r[-5:] if v < 0.1]

            # Last surprise + prediction error
            surprise = round(getattr(fabric, "_last_surprise", 0.0), 3)

            # Top contributing memory pathways
            top_paths = e.memory.top_connections(5) or []
            mem_influence = [
                {"a": p.get("a"), "b": p.get("b"),
                 "weight": round(p.get("weight", 0), 3)}
                for p in top_paths[:3]
            ]

            # Top beliefs
            top_beliefs = []
            try:
                top_beliefs = [
                    {"claim": b.claim, "strength": round(b.strength, 2)}
                    for b in (e.beliefs.all_beliefs()[:3] if hasattr(e, "beliefs") else [])
                ]
            except Exception:
                pass

            # Neuromodulator state
            neuro = snap.get("neuromod", {})

            # Confidence: driven by low surprise + high serotonin
            confidence = round(
                (1.0 - surprise) * 0.6
                + float(neuro.get("serotonin", 0.5)) * 0.4,
                3
            )

            # Build natural language summary
            win_names = [w[0] for w in winners[:3]]
            summary = (
                f"Decision driven by: {', '.join(win_names)}. "
                f"Surprise level: {surprise:.2f}. "
                f"System confidence: {confidence:.0%}. "
                f"Dominant emotion: {snap.get('emotion', 'neutral')}."
            )

            return {
                "ok": True,
                "summary":          summary,
                "winning_clusters": winners,
                "losing_clusters":  losers,
                "top_factors": [
                    {"factor": "norepinephrine",  "value": round(float(neuro.get("norepinephrine", 0.5)), 3)},
                    {"factor": "surprise",        "value": surprise},
                    {"factor": "dopamine",        "value": round(float(neuro.get("dopamine", 0.5)), 3)},
                ],
                "memory_influence":  mem_influence,
                "belief_influence":  top_beliefs,
                "confidence":        confidence,
                "emotion":           snap.get("emotion", "neutral"),
            }
        except Exception as ex:
            return {"ok": False, "error": str(ex)}

    # ────────────────────────────────────────────────────────────────
    # Personality
    # ────────────────────────────────────────────────────────────────

    def set_personality(self, traits: dict) -> dict:
        """
        Set personality vector.
        traits: {curiosity, risk, empathy, dominance, creativity, stability}
        Each 0.0–1.0. Persisted immediately.
        """
        if self._engine is None:
            return {"ok": False, "error": "No engine attached"}

        VALID = {"curiosity", "risk", "empathy", "dominance", "creativity", "stability"}
        sanitized = {}
        for k, v in traits.items():
            if k in VALID:
                sanitized[k] = float(max(0.0, min(1.0, v)))

        try:
            p = self._engine.fabric.personality
            for k, v in sanitized.items():
                if hasattr(p, k):
                    setattr(p, k, v)
            p.save()
            # Immediately push to cognitive context
            self._engine.fabric.set_personality_context(
                p.to_dict(), {}
            )
            self._engine._emit("personality_update", {"traits": p.to_dict()})
            return {"ok": True, "traits": p.to_dict()}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_personality(self) -> dict:
        """Return current personality vector."""
        if self._engine is None:
            return {}
        try:
            p = self._engine.fabric.personality
            return p.to_dict() if hasattr(p, "to_dict") else {}
        except Exception:
            return {}

    # ────────────────────────────────────────────────────────────────
    # Autonomous mode
    # ────────────────────────────────────────────────────────────────

    def run_autonomous(self, steps: int = 100, interval_ms: int = 200) -> dict:
        """
        Run the brain autonomously for N cognitive steps without external input.
        The brain will:
          - self-stimulate idle regions
          - replay memory traces
          - reinforce beliefs
          - shift personality based on internal drives

        Returns a summary of what changed.
        """
        if self._engine is None:
            return {"ok": False, "error": "No engine attached"}

        def _run():
            e = self._engine
            start_snap  = e.fabric.get_state_snapshot()
            start_pers  = e.fabric.personality.to_dict() if hasattr(e.fabric, "personality") else {}
            reward_acc  = 0.0
            belief_changes = 0
            explored    = []

            for i in range(steps):
                # Stimulate default-mode + memory consolidation pathways
                e.fabric.stimulate_region("hippocampus",         0.06)
                e.fabric.stimulate_region("default_mode_network",0.08)
                e.fabric.stimulate_region("prefrontal_cortex",   0.05)
                e.fabric.stimulate_region("temporal_lobe",       0.04)

                # Occasionally replay a random memory
                if i % 20 == 0:
                    try:
                        facts = list((e.memory.all_facts() or {}).items())
                        if facts:
                            import random
                            k, v = random.choice(facts)
                            e.fabric.stimulate_for_input("memory", 0.15)
                            explored.append(f"Recalled: {k}")
                            e.knowledge.ingest(
                                str(v)[:200],
                                source_label="autonomous_replay",
                                credibility=0.5,
                            )
                            belief_changes += 1
                    except Exception:
                        pass

                # Drive ticks accumulate naturally
                if hasattr(e, "drives"):
                    e.drives.tick()

                # Cognitive cycle tick (if running, already ticking — else manual)
                if not (hasattr(e, "cycle") and e.cycle and e.cycle._running):
                    try:
                        e.fabric._gpu_tick(0.1)
                    except Exception:
                        pass

                reward_acc += getattr(e.fabric, "_last_surprise", 0.0)
                time.sleep(interval_ms / 1000.0)

            # Diff personality
            end_pers   = e.fabric.personality.to_dict() if hasattr(e.fabric, "personality") else {}
            pers_delta = {
                k: round(end_pers.get(k, 0) - start_pers.get(k, 0), 3)
                for k in end_pers
                if abs(end_pers.get(k, 0) - start_pers.get(k, 0)) > 0.01
            }

            summary = {
                "steps":           steps,
                "belief_updates":  belief_changes,
                "explored":        explored[:10],
                "personality_drift": pers_delta,
                "avg_surprise":    round(reward_acc / max(1, steps), 4),
            }

            e._emit("autonomous_run_complete", summary)
            e._emit("log", {"msg": f"🧠 Autonomous run complete — {steps} steps, {belief_changes} belief updates"})

        threading.Thread(target=_run, daemon=True).start()
        return {"ok": True, "message": f"Autonomous run started ({steps} steps)"}

    # ────────────────────────────────────────────────────────────────
    # Persistence
    # ────────────────────────────────────────────────────────────────

    def save_brain(self, slot: str = "default") -> dict:
        """
        Serialize and persist the current brain state to a JSON snapshot.
        Saved to data/snapshots/<slot>.json

        Persists:
          beliefs, preferences, drives, personality, reward_history,
          neuromod state, hebbian top pathways, self-model
        """
        if self._engine is None:
            return {"ok": False, "error": "No engine attached"}
        try:
            e       = self._engine
            snap_dir = self._data_dir / "snapshots"
            snap_dir.mkdir(exist_ok=True)
            path    = snap_dir / f"{slot}.json"

            # Collect serializable state
            beliefs_raw = []
            try:
                beliefs_raw = [
                    {"key": b.key, "claim": b.claim, "strength": b.strength,
                     "valence": b.valence, "source": b.source}
                    for b in (e.beliefs.all_beliefs() or [])
                ]
            except Exception:
                pass

            prefs_raw = {}
            try:
                prefs_raw = e.preferences.summary() if hasattr(e, "preferences") else {}
            except Exception:
                pass

            drives_raw = {}
            try:
                drives_raw = {
                    k: {"level": round(d.level, 4), "total_satisfied": d.total_satisfied}
                    for k, d in (e.drives._drives if hasattr(e, "drives") else {}).items()
                }
            except Exception:
                pass

            pers_raw = {}
            try:
                pers_raw = e.fabric.personality.to_dict() if hasattr(e.fabric, "personality") else {}
            except Exception:
                pass

            neuro_raw = {}
            try:
                nm = e.fabric.neuromod
                neuro_raw = {
                    "dopamine":       round(float(nm.dopamine), 4),
                    "serotonin":      round(float(nm.serotonin), 4),
                    "norepinephrine": round(float(nm.norepinephrine), 4),
                    "acetylcholine":  round(float(nm.acetylcholine), 4),
                    "gaba":           round(float(nm.gaba), 4),
                    "glutamate":      round(float(nm.glutamate), 4),
                }
            except Exception:
                pass

            reward_hist = []
            try:
                reward_hist = list(e.fabric._reward_history)[-50:]
            except Exception:
                pass

            self_model_raw = {}
            try:
                self_model_raw = e.self_model.to_dict() if hasattr(e, "self_model") else {}
            except Exception:
                pass

            snapshot = {
                "slot":          slot,
                "saved_at":      time.time(),
                "version":       self.VERSION,
                "beliefs":       beliefs_raw,
                "preferences":   prefs_raw,
                "drives":        drives_raw,
                "personality":   pers_raw,
                "neuromod":      neuro_raw,
                "reward_history":reward_hist,
                "self_model":    self_model_raw,
            }

            path.write_text(json.dumps(snapshot, indent=2, default=str))
            return {
                "ok":      True,
                "slot":    slot,
                "path":    str(path),
                "saved_at": snapshot["saved_at"],
                "beliefs":  len(beliefs_raw),
            }
        except Exception as ex:
            return {"ok": False, "error": str(ex)}

    def load_brain(self, slot: str = "default") -> dict:
        """
        Restore brain state from a previously saved snapshot.
        Restores: personality, neuromod baselines, drives.
        (Beliefs/preferences live in SQLite — no explicit reload needed.)
        """
        if self._engine is None:
            return {"ok": False, "error": "No engine attached"}
        snap_dir = self._data_dir / "snapshots"
        path     = snap_dir / f"{slot}.json"
        if not path.exists():
            return {"ok": False, "error": f"Snapshot '{slot}' not found"}
        try:
            snap = json.loads(path.read_text())
            e    = self._engine

            # Restore personality
            pers = snap.get("personality", {})
            if pers and hasattr(e.fabric, "personality"):
                p = e.fabric.personality
                for k, v in pers.items():
                    if hasattr(p, k):
                        setattr(p, k, float(v))
                p.save()

            # Restore neuromod baselines (soft — just nudge toward saved values)
            neuro = snap.get("neuromod", {})
            if neuro:
                nm = e.fabric.neuromod
                for chem in ["dopamine","serotonin","norepinephrine","acetylcholine","gaba","glutamate"]:
                    if chem in neuro and hasattr(nm, chem):
                        current = float(getattr(nm, chem))
                        target  = float(neuro[chem])
                        # Blend: 70% saved, 30% current (don't slam to saved value)
                        setattr(nm, chem, current * 0.3 + target * 0.7)

            # Restore drive levels
            drives = snap.get("drives", {})
            if drives and hasattr(e, "drives"):
                for k, d in drives.items():
                    try:
                        e.drives._drives[k].level = float(d.get("level", 0.15))
                    except Exception:
                        pass

            return {
                "ok":      True,
                "slot":    slot,
                "restored_at": time.time(),
                "personality": pers,
            }
        except Exception as ex:
            return {"ok": False, "error": str(ex)}

    def list_snapshots(self) -> list:
        """List all available brain snapshots."""
        snap_dir = self._data_dir / "snapshots"
        if not snap_dir.exists():
            return []
        result = []
        for p in snap_dir.glob("*.json"):
            try:
                meta = json.loads(p.read_text())
                result.append({
                    "slot":     p.stem,
                    "saved_at": meta.get("saved_at"),
                    "beliefs":  len(meta.get("beliefs", [])),
                    "version":  meta.get("version"),
                })
            except Exception:
                pass
        return sorted(result, key=lambda x: x.get("saved_at", 0), reverse=True)

    # ─── Onboarding API ───────────────────────────────────────────────────────

    def get_onboarding_state(self) -> dict:
        """Return current onboarding state for the frontend."""
        if not hasattr(self._engine, "onboarding"):
            return {"completed": True}
        return self._engine.onboarding.to_client()

    def onboarding_set_name(self, name: str) -> dict:
        e = self._engine
        if not hasattr(e, "onboarding"):
            return {"ok": False}
        e.onboarding.set_name(name)
        return {"ok": True, "name": name}

    def onboarding_set_preset(self, preset: str) -> dict:
        """Apply a personality preset and update neural fabric."""
        from axon.cognition.onboarding import PRESETS
        e = self._engine
        if not hasattr(e, "onboarding"):
            return {"ok": False}
        traits = e.onboarding.set_preset(preset)
        if not traits:
            return {"ok": False, "error": f"Unknown preset: {preset}"}
        try:
            p = e.fabric.personality
            for k, v in traits.items():
                if k != "description" and hasattr(p, k):
                    setattr(p, k, float(v))
            p.save()
        except Exception as ex:
            return {"ok": False, "error": str(ex)}
        return {"ok": True, "preset": preset}

    def onboarding_ingest_sample(self, sample_id: str) -> dict:
        """Ingest a built-in sample topic and return the first opinion."""
        from axon.cognition.onboarding import SAMPLE_TOPICS
        e = self._engine
        topic = next((s for s in SAMPLE_TOPICS if s["id"] == sample_id), None)
        if not topic:
            return {"ok": False, "error": "Unknown sample"}
        if hasattr(e, "onboarding"):
            e.onboarding.set_sample(sample_id)
        try:
            e.ingest(topic["text"], source=f"sample:{sample_id}", emit_events=True)
        except Exception as ex:
            return {"ok": False, "error": str(ex)}
        return {"ok": True, "sample_id": sample_id, "label": topic["label"]}

    def onboarding_complete(self) -> dict:
        e = self._engine
        if hasattr(e, "onboarding"):
            e.onboarding.complete()
        return {"ok": True}

    def onboarding_ingest_text(self, text: str) -> dict:
        """Ingest user-supplied text during onboarding."""
        e = self._engine
        try:
            e.ingest(text.strip(), source="onboarding_upload", emit_events=True)
        except Exception as ex:
            return {"ok": False, "error": str(ex)}
        return {"ok": True}

    # ─── Goal API ─────────────────────────────────────────────────────────────

    def get_goals(self) -> list:
        if not hasattr(self._engine, "goals"):
            return []
        return self._engine.goals.all_goals()

    def add_goal(self, name: str, description: str, priority: float = 0.5) -> dict:
        if not hasattr(self._engine, "goals"):
            return {"ok": False}
        g = self._engine.goals.add_goal(name, description, priority)
        return {"ok": True, "goal": g.to_dict()}

    def remove_goal(self, name: str) -> dict:
        if not hasattr(self._engine, "goals"):
            return {"ok": False}
        ok = self._engine.goals.remove_goal(name)
        return {"ok": ok}

    # ─── Surprise Events API ──────────────────────────────────────────────────

    def recent_surprise_events(self, n: int = 20) -> list:
        if not hasattr(self._engine, "surprise"):
            return []
        return self._engine.surprise.recent_events(n)

    # ─── Brain Fork API ───────────────────────────────────────────────────────

    def fork_brain(self, fork_name: str, trait_overrides: dict = None) -> dict:
        """
        Create a named fork of the current brain state.
        Saves current snapshot under fork_name, optionally applying trait overrides.
        """
        result = self.save_brain(slot=f"fork_{fork_name}")
        if not result.get("ok"):
            return result
        if trait_overrides:
            snap_dir = self._data_dir / "snapshots"
            path     = snap_dir / f"fork_{fork_name}.json"
            try:
                snap = json.loads(path.read_text())
                for k, v in trait_overrides.items():
                    if k != "description":
                        snap.setdefault("personality", {})[k] = float(v)
                snap["fork_name"]   = fork_name
                snap["parent_slot"] = "default"
                path.write_text(json.dumps(snap, indent=2, default=str))
            except Exception as ex:
                return {"ok": False, "error": str(ex)}
        return {
            "ok":        True,
            "fork_name": fork_name,
            "slot":      f"fork_{fork_name}",
            "message":   f"Fork created. Load with load_brain('fork_{fork_name}') to diverge.",
        }

    def list_forks(self) -> list:
        snap_dir = self._data_dir / "snapshots"
        if not snap_dir.exists():
            return []
        result = []
        for p in snap_dir.glob("fork_*.json"):
            try:
                meta = json.loads(p.read_text())
                result.append({
                    "fork_name":   meta.get("fork_name", p.stem),
                    "slot":        p.stem,
                    "saved_at":    meta.get("saved_at"),
                    "beliefs":     len(meta.get("beliefs", [])),
                    "personality": meta.get("personality", {}),
                })
            except Exception:
                pass
        return sorted(result, key=lambda x: x.get("saved_at", 0), reverse=True)

    def generate_share_link(self, slot: str = "default", label: str = "") -> dict:
        """
        Generate a shareable brain snapshot summary.
        Returns a compact JSON summary + base64 share token.
        """
        snap_dir = self._data_dir / "snapshots"
        path     = snap_dir / f"{slot}.json"
        if not path.exists():
            self.save_brain(slot=slot)
        try:
            snap        = json.loads(path.read_text())
            top_beliefs = snap.get("beliefs", [])[:5]
            pers        = snap.get("personality", {})
            summary = {
                "label":         label or f"Brain snapshot @ {slot}",
                "saved_at":      snap.get("saved_at"),
                "version":       snap.get("version"),
                "personality":   {k: round(float(v), 2) for k, v in pers.items()
                                   if not isinstance(v, str)},
                "top_beliefs":   [{"claim": b["claim"][:80],
                                    "strength": round(b.get("strength", 0), 2)}
                                   for b in top_beliefs],
                "beliefs_count": len(snap.get("beliefs", [])),
            }
            import base64, hashlib
            raw         = json.dumps(summary, sort_keys=True)
            token       = base64.urlsafe_b64encode(raw.encode()).decode()
            fingerprint = hashlib.md5(raw.encode()).hexdigest()[:8]
            return {
                "ok":          True,
                "slot":        slot,
                "fingerprint": fingerprint,
                "summary":     summary,
                "share_token": token,
                "share_label": label or slot,
            }
        except Exception as ex:
            return {"ok": False, "error": str(ex)}
