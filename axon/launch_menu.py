"""
AXON — Pre-Launch Console Menu
================================
Shown every time AXON starts (after first-run completes).
Lets you choose what state to carry into this session.

Exits cleanly so the caller (app.py or launch script) continues.
"""

import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path

# ── Colour helpers ────────────────────────────────────────────────────────────
R  = "\033[91m"
Y  = "\033[93m"
G  = "\033[92m"
C  = "\033[96m"
D  = "\033[2m"
B  = "\033[1m"
RS = "\033[0m"

def r(s):  return f"{R}{s}{RS}"
def g(s):  return f"{G}{s}{RS}"
def y(s):  return f"{Y}{s}{RS}"
def c(s):  return f"{C}{s}{RS}"
def d(s):  return f"{D}{s}{RS}"
def b(s):  return f"{B}{s}{RS}"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR      = Path("data")
ONBOARDING    = DATA_DIR / "onboarding.json"
DB_PATH       = DATA_DIR / "memory" / "axon.db"
NEURAL_DIR    = DATA_DIR / "neural"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
HEBBIAN_DIR   = DATA_DIR / "hebbian"
PERSONALITY   = DATA_DIR / "neural" / "personality.json"
FACE_DIR      = DATA_DIR / "faces"
USER_MODEL    = DATA_DIR / "user_model.json"
GOALS_JSON    = DATA_DIR / "goals.json"
BELIEFS_JSON  = DATA_DIR / "beliefs.json"

PERSONA_DEFAULTS = {
    "openness":          0.72,
    "conscientiousness": 0.65,
    "extraversion":      0.55,
    "agreeableness":     0.60,
    "neuroticism":       0.35,
    "curiosity":         0.75,
    "empathy":           0.60,
    "risk":              0.40,
    "dominance":         0.45,
    "creativity":        0.65,
    "stability":         0.70,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_onboarding() -> dict:
    if ONBOARDING.exists():
        try:
            return json.loads(ONBOARDING.read_text())
        except Exception:
            pass
    return {"completed": False, "ai_name": "", "preset": ""}


def _wipe_tables(tables: list[str]):
    if not DB_PATH.exists():
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing = {row[0] for row in cur.fetchall()}
        for t in tables:
            if t in existing:
                cur.execute(f"DELETE FROM {t}")
        conn.commit()
        conn.close()
    except Exception as e:
        print(r(f"  [WARN] DB wipe partial: {e}"))


def _remove(p: Path):
    try:
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    except Exception as e:
        print(r(f"  [WARN] Could not remove {p}: {e}"))


# ── Reset modes ───────────────────────────────────────────────────────────────

def reset_learned_memory(name: str):
    """Wipe experience & learned data — keep name and personality."""
    print(f"\n  {y('Wiping learned memory...')}")
    _wipe_tables(["episodic", "semantic", "working", "beliefs",
                  "preferences", "narrative_threads", "identity_snapshots",
                  "goals", "memory_hierarchy"])
    _remove(HEBBIAN_DIR)
    _remove(SNAPSHOTS_DIR)
    # Wipe beliefs/goals JSON files if they exist
    _remove(BELIEFS_JSON)
    _remove(GOALS_JSON)
    print(g(f"  [OK] Learned memory cleared — name and personality preserved."))
    if name:
        print(d(f"  Name retained: {name}"))


def reset_user_profiles():
    """Wipe face profiles and user model only."""
    print(f"\n  {y('Wiping user profiles...')}")
    _wipe_tables(["face_identities", "user_model"])
    _remove(FACE_DIR)
    _remove(USER_MODEL)
    print(g("  [OK] User profiles cleared — memory and personality preserved."))


def factory_reset():
    """Full wipe: everything + personality back to defaults."""
    print(f"\n  {r('Factory reset — wiping everything...')}")
    # All DB tables
    _wipe_tables(["episodic", "semantic", "working", "beliefs",
                  "preferences", "narrative_threads", "identity_snapshots",
                  "goals", "memory_hierarchy", "face_identities", "user_model"])
    # Files
    for p in [HEBBIAN_DIR, SNAPSHOTS_DIR, FACE_DIR, USER_MODEL, BELIEFS_JSON, GOALS_JSON]:
        _remove(p)
    # Reset personality to defaults
    PERSONALITY.parent.mkdir(parents=True, exist_ok=True)
    PERSONALITY.write_text(json.dumps(PERSONA_DEFAULTS, indent=2))
    # Reset onboarding so the wizard runs again in the browser
    ob = _load_onboarding()
    ob["completed"]    = False
    ob["step"]         = 0
    ob["preset"]       = ""
    ob["completed_at"] = None
    ONBOARDING.write_text(json.dumps(ob, indent=2))
    print(g("  [OK] Factory reset complete — onboarding will run on next page load."))


# ── Main entry ────────────────────────────────────────────────────────────────

def run():
    """
    Show the pre-launch menu if AXON has been configured before.
    Returns immediately if it's a first launch (let browser onboarding handle it).
    """
    ob = _load_onboarding()
    name  = ob.get("ai_name") or "AXON"
    preset = ob.get("preset") or "—"

    # First-ever launch: nothing to choose — browser onboarding will run
    if not ob.get("completed", False):
        print(d("  [First launch — browser onboarding will guide setup]"))
        return

    # Returning launch — show menu
    print()
    print(c("  ┌─────────────────────────────────────────┐"))
    print(c(f"  │  Welcome back,  {b(name):<24}{c('│')}"))
    print(c(f"  │  Personality:   {preset:<24}│"))
    print(c("  ├─────────────────────────────────────────┤"))
    print(c("  │  [1]  Continue (keep all memory)        │"))
    print(c("  │  [2]  Reset learned memory              │"))
    print(c("  │       (wipes experience, keeps name +   │"))
    print(c("  │        personality)                     │"))
    print(c("  │  [3]  Delete user profiles              │"))
    print(c("  │       (face data + user model only)     │"))
    print(c("  │  [4]  Factory reset                     │"))
    print(c("  │       (wipe everything + re-run wizard) │"))
    print(c("  └─────────────────────────────────────────┘"))
    print()

    try:
        choice = input("  Choose [1-4]:  ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Starting with current memory.")
        return

    if choice == "1" or choice == "":
        print(g("  Continuing with existing memory."))

    elif choice == "2":
        confirm = input(r(f"  ⚠  This will erase all learned memory for '{name}'. Type YES to confirm: ")).strip()
        if confirm == "YES":
            reset_learned_memory(name)
        else:
            print(y("  Cancelled — keeping existing memory."))

    elif choice == "3":
        confirm = input(r("  ⚠  This will remove all face profiles and user data. Type YES to confirm: ")).strip()
        if confirm == "YES":
            reset_user_profiles()
        else:
            print(y("  Cancelled."))

    elif choice == "4":
        confirm = input(r("  ⚠  FACTORY RESET — everything will be erased. Type YES to confirm: ")).strip()
        if confirm == "YES":
            factory_reset()
        else:
            print(y("  Cancelled."))

    else:
        print(y("  Unrecognised input — continuing with existing memory."))

    print()


if __name__ == "__main__":
    run()
