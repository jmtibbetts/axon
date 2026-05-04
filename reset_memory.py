#!/usr/bin/env python3
"""
AXON — Memory Reset
Clears all learned data so AXON starts fresh on next run.
Run this from the axon directory:  python reset_memory.py
"""
import os
import shutil
import sqlite3
from pathlib import Path

DATA_DIR = Path("data")
DB_PATH  = DATA_DIR / "memory" / "axon.db"

def reset():
    print("\n  AXON Memory Reset")
    print("  " + "─" * 40)

    if not DB_PATH.exists():
        print("  No database found — already clean.")
        return

    # Connect and wipe all tables (preserve schema)
    conn = sqlite3.connect(str(DB_PATH))
    tables = ["episodic", "semantic", "hebbian", "topics", "people"]
    for t in tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            conn.execute(f"DELETE FROM {t}")
            print(f"  ✓ Cleared {t} ({count} rows)")
        except Exception as e:
            print(f"  ⚠ {t}: {e}")
    conn.commit()
    conn.close()

    # Also clear any cached neural state files
    neural_dir = DATA_DIR / "neural"
    if neural_dir.exists():
        for f in neural_dir.glob("*.pt"):
            f.unlink()
            print(f"  ✓ Removed {f.name}")
        for f in neural_dir.glob("*.json"):
            f.unlink()
            print(f"  ✓ Removed {f.name}")

    print("\n  ✅ Memory wiped — all episodic, semantic, and person profiles cleared.")
    print("     AXON will start fresh with no knowledge of anyone on next run.")
    print("     Start normally:  python run.py\n")

if __name__ == "__main__":
    confirm = input("  This will erase all of AXON's learned memory. Continue? [y/N] ").strip().lower()
    if confirm == "y":
        reset()
    else:
        print("  Cancelled.")
