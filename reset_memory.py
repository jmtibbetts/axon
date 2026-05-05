#!/usr/bin/env python3
"""
AXON — Reset Utility
====================
Interactively wipe AXON's learned data.

Two reset modes:

  [1] Full Memory Wipe
      Clears all episodic/semantic/Hebbian memory, beliefs,
      preferences, narrative threads, goals, identity snapshots,
      and cached neural state. AXON wakes up as a blank slate —
      same personality baseline, no learned history.

  [2] User Data Deletion
      Removes only the data tied to a specific person:
      face profile, recognition data, user model, and any episodic
      memories tagged to that person. Other memory is untouched.

  [3] Factory Reset (Full Wipe + Personality Reset)
      Everything from [1] PLUS resets personality.json to default
      trait values. AXON returns to its initial compiled state —
      no experience, no drift, no preferences.

Run from the axon root directory:
    python reset_memory.py
"""

import os
import sys
import json
import shutil
import sqlite3
from pathlib import Path

# ── Colour helpers ────────────────────────────────────────────────────────────
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
DIM    = "\033[2m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def red(s):    return f"{RED}{s}{RESET}"
def green(s):  return f"{GREEN}{s}{RESET}"
def yellow(s): return f"{YELLOW}{s}{RESET}"
def cyan(s):   return f"{CYAN}{s}{RESET}"
def dim(s):    return f"{DIM}{s}{RESET}"
def bold(s):   return f"{BOLD}{s}{RESET}"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR      = Path("data")
DB_PATH       = DATA_DIR / "memory" / "axon.db"
NEURAL_DIR    = DATA_DIR / "neural"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
PERSONALITY   = DATA_DIR / "neural" / "personality.json"

PERSONA_DEFAULTS = {
    "openness":          0.72,
    "conscientiousness": 0.61,
    "extraversion":      0.48,
    "agreeableness":     0.65,
    "neuroticism":       0.38,
    "curiosity":         0.80,
    "risk_tolerance":    0.50,
    "stability":         0.60,
    "persistence":       0.65,
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def divider(char="─", width=50):
    print(f"  {dim(char * width)}")

def section(title):
    print()
    print(f"  {CYAN}{BOLD}{title}{RESET}")
    divider()

def confirm(prompt, default_no=True):
    hint = "[y/N]" if default_no else "[Y/n]"
    try:
        ans = input(f"  {prompt} {dim(hint)} ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        return False
    if default_no:
        return ans == "y"
    return ans != "n"

def wipe_db_tables(tables: list[str], label: str = ""):
    """Delete all rows from given SQLite tables. Returns total rows deleted."""
    if not DB_PATH.exists():
        print(f"  {dim('No database found — nothing to clear.')}")
        return 0

    total = 0
    conn = sqlite3.connect(str(DB_PATH))
    for t in tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            conn.execute(f"DELETE FROM {t}")
            total += count
            tag = f"  ({count} rows)" if count else dim("  (empty)")
            print(f"  {green('✓')} {t}{tag}")
        except sqlite3.OperationalError:
            print(f"  {dim(f'— {t}: table not found (skipped)')}")
    conn.commit()
    conn.close()
    return total

def remove_glob(directory: Path, pattern: str, label: str):
    """Remove files matching a glob pattern; return count removed."""
    if not directory.exists():
        return 0
    files = list(directory.glob(pattern))
    for f in files:
        try:
            f.unlink()
            print(f"  {green('✓')} {f.name}")
        except Exception as e:
            print(f"  {yellow('⚠')} Could not remove {f.name}: {e}")
    return len(files)

def remove_dir(path: Path, label: str):
    if path.exists():
        try:
            shutil.rmtree(path)
            print(f"  {green('✓')} {label} ({path})")
        except Exception as e:
            print(f"  {yellow('⚠')} {label}: {e}")

def remove_json(path: Path, label: str):
    if path.exists():
        try:
            path.unlink()
            print(f"  {green('✓')} {label} ({path.name})")
        except Exception as e:
            print(f"  {yellow('⚠')} {label}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Reset modes
# ─────────────────────────────────────────────────────────────────────────────

def full_memory_wipe():
    """Clear all learned memory — episodic, semantic, beliefs, etc."""
    section("Full Memory Wipe")

    # SQLite tables
    ALL_TABLES = [
        "episodic", "semantic", "hebbian", "topics", "people",
        "beliefs", "memory_hierarchy", "preference_buckets", "hobbies",
    ]
    print(f"  {bold('Database tables:')}")
    total_rows = wipe_db_tables(ALL_TABLES)

    # JSON state files
    print()
    print(f"  {bold('State files:')}")
    remove_json(DATA_DIR / "goals.json",       "Goal list")
    remove_json(DATA_DIR / "narratives.json",  "Narrative threads")
    remove_json(DATA_DIR / "onboarding.json",  "Onboarding state")

    # Neural state files (weights, snapshots) — keep personality unless factory reset
    print()
    print(f"  {bold('Neural cache:')}")
    if NEURAL_DIR.exists():
        for f in NEURAL_DIR.glob("*.pt"):
            try:   f.unlink(); print(f"  {green('✓')} {f.name}")
            except Exception as e: print(f"  {yellow('⚠')} {f.name}: {e}")
        for f in NEURAL_DIR.glob("brain_state*.json"):
            try:   f.unlink(); print(f"  {green('✓')} {f.name}")
            except Exception as e: print(f"  {yellow('⚠')} {f.name}: {e}")
    else:
        print(f"  {dim('No neural cache found.')}")

    # Brain snapshots
    print()
    print(f"  {bold('Brain snapshots:')}")
    if SNAPSHOTS_DIR.exists():
        snaps = list(SNAPSHOTS_DIR.glob("*.json"))
        if snaps:
            for f in snaps:
                try:   f.unlink(); print(f"  {green('✓')} {f.name}")
                except Exception as e: print(f"  {yellow('⚠')} {f.name}: {e}")
        else:
            print(f"  {dim('No snapshots found.')}")
    else:
        print(f"  {dim('No snapshots directory.')}")

    print()
    print(f"  {green('✅ Memory wiped.')} {dim(f'{total_rows} total rows cleared.')}")
    print(f"  {dim('Personality, providers, and GPU config are untouched.')}")
    print(f"  {dim('Start normally:  python run.py')}")


def user_data_deletion():
    """Remove data for a specific person by name."""
    section("User Data Deletion")

    if not DB_PATH.exists():
        print(f"  {red('No database found.')} Nothing to delete.")
        return

    # List known people
    conn = sqlite3.connect(str(DB_PATH))
    try:
        rows = conn.execute("SELECT id, name, seen_count, last_seen FROM people ORDER BY seen_count DESC").fetchall()
    except sqlite3.OperationalError:
        rows = []

    if not rows:
        print(f"  {dim('No person profiles found in the database.')}")
        conn.close()
        return

    print(f"  {bold('Known people:')}")
    for i, (pid, name, seen, last) in enumerate(rows, 1):
        print(f"  {cyan(str(i))}.  {name}  {dim(f'(seen {seen}x, last: {last})')}")
    print(f"  {cyan('0')}.  Cancel")
    print()

    try:
        choice = input(f"  Select person to delete [0-{len(rows)}]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return

    if choice == "0" or not choice:
        print(f"  {dim('Cancelled.')}")
        conn.close()
        return

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(rows):
            raise ValueError
    except ValueError:
        print(f"  {red('Invalid selection.')} Cancelled.")
        conn.close()
        return

    person_id, person_name = rows[idx][0], rows[idx][1]
    print()
    print(f"  {yellow('⚠  You are about to delete all data for:')}  {bold(person_name)}")
    print(f"  {dim('This includes: face profile, recognition data, episodic memories tagged to this person.')}")
    print()

    if not confirm(f"Delete all data for '{person_name}'?"):
        print(f"  {dim('Cancelled.')}")
        conn.close()
        return

    deleted = 0

    # Delete face profile
    c = conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
    deleted += c.rowcount
    print(f"  {green('✓')} Face profile removed  ({person_name})")

    # Delete tagged episodic memories
    try:
        c = conn.execute(
            "DELETE FROM episodic WHERE content LIKE ?",
            (f"%{person_name}%",)
        )
        deleted += c.rowcount
        print(f"  {green('✓')} Episodic memories referencing '{person_name}' removed  ({c.rowcount} rows)")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()

    # Remove face image cache if present
    face_cache = DATA_DIR / "faces" / person_name.lower().replace(" ", "_")
    if face_cache.exists():
        remove_dir(face_cache, f"Face image cache ({person_name})")

    print()
    print(f"  {green('✅ Done.')} {dim(f'{deleted} records removed for {person_name}.')}")
    print(f"  {dim('All other memory is untouched.')}")


def factory_reset():
    """Full memory wipe + reset personality to compiled defaults."""
    section("Factory Reset")
    print(f"  {red(bold('⚠  WARNING:'))}  This erases ALL learned data AND resets personality.")
    print(f"  {dim('AXON will return to its initial compiled state — no experience, no drift.')}")
    print()

    if not confirm("Are you absolutely sure? This cannot be undone.", default_no=True):
        print(f"  {dim('Cancelled.')}")
        return

    # Run full memory wipe first
    full_memory_wipe()

    # Reset personality
    print()
    print(f"  {bold('Personality reset:')}")
    if PERSONALITY.parent.exists() or not PERSONALITY.exists():
        PERSONALITY.parent.mkdir(parents=True, exist_ok=True)
        with open(PERSONALITY, "w") as f:
            json.dump(PERSONA_DEFAULTS, f, indent=2)
        print(f"  {green('✓')} personality.json reset to compiled defaults")
    else:
        print(f"  {dim('personality.json not found — nothing to reset.')}")

    print()
    print(f"  {green('✅ Factory reset complete.')}")
    print(f"  {dim('AXON is back to its initial state. Start normally:  python run.py')}")


# ─────────────────────────────────────────────────────────────────────────────
# Menu
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print()
    print(f"  {CYAN}{BOLD}AXON Reset Utility{RESET}")
    divider("═")
    print(f"  {bold('1.')}  Full Memory Wipe          {dim('— clear all learned history, beliefs, faces')}")
    person_label = "remove one person's profile and memories"
    print(f"  {bold('2.')}  User Data Deletion         {dim(person_label)}")
    print(f"  {bold('3.')}  Factory Reset              {dim('— full wipe + reset personality to defaults')}")
    print(f"  {bold('0.')}  Cancel")
    print()

    try:
        choice = input("  Select [0-3]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        print(f"  {dim('Cancelled.')}")
        sys.exit(0)

    if choice == "1":
        print()
        wipe_warn = "⚠  This will erase all of AXON's learned memory."
        print(f"  {yellow(wipe_warn)}")
        if confirm("Continue with Full Memory Wipe?"):
            full_memory_wipe()
        else:
            print(f"  {dim('Cancelled.')}")

    elif choice == "2":
        user_data_deletion()

    elif choice == "3":
        factory_reset()

    elif choice == "0":
        print(f"  {dim('Cancelled.')}")
    else:
        print(f"  {red('Invalid choice.')} Run again and select 0–3.")

    print()


if __name__ == "__main__":
    main()
