"""
A1 — Download ForecastBench question sets and resolution sets.

Clones the forecastbench-datasets repo (or pulls if already cloned) and
copies the 5 biweekly question set files + matching resolution files into
data/raw/.
"""

import json
import shutil
import subprocess
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
REPO_DIR = ROOT / "forecastbench-datasets"

QUESTION_SETS = [
    # Oct–Nov 2025 (resolution dates after Aug 2025 GPT-5.4 cutoff)
    "2025-10-26-llm.json",
    "2025-11-09-llm.json",
    "2025-11-23-llm.json",
    # Dec 2025 – Apr 2026
    "2025-12-07-llm.json",
    "2025-12-21-llm.json",
    "2026-01-04-llm.json",
    "2026-01-18-llm.json",
    "2026-02-01-llm.json",
    "2026-02-15-llm.json",
    "2026-03-01-llm.json",
    "2026-03-15-llm.json",
    "2026-03-29-llm.json",
    "2026-04-12-llm.json",
]

REPO_URL = "https://github.com/forecastingresearch/forecastbench-datasets.git"


def clone_or_pull() -> None:
    if REPO_DIR.exists():
        print(f"Repo already exists at {REPO_DIR}, pulling latest…")
        result = subprocess.run(
            ["git", "-C", str(REPO_DIR), "pull", "--ff-only"],
            capture_output=True, text=True
        )
        print(result.stdout.strip() or "Already up to date.")
    else:
        print(f"Cloning {REPO_URL} …")
        subprocess.run(
            ["git", "clone", "--depth=1", REPO_URL, str(REPO_DIR)],
            check=True
        )
        print("Clone complete.")


def copy_files() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    q_src = REPO_DIR / "datasets" / "question_sets"
    r_src = REPO_DIR / "datasets" / "resolution_sets"

    copied_q, copied_r = 0, 0

    for fname in QUESTION_SETS:
        src = q_src / fname
        dst = RAW_DIR / fname
        if not src.exists():
            print(f"  [WARN] Question set not found: {src}")
            continue
        shutil.copy2(src, dst)
        print(f"  [Q] {fname}")
        copied_q += 1

        # Matching resolution file — naming: 2026-02-01_resolution_set.json
        date_part = fname.replace("-llm.json", "")  # e.g. "2026-02-01"
        res_fname = f"{date_part}_resolution_set.json"
        res_src = r_src / res_fname
        if res_src.exists():
            shutil.copy2(res_src, RAW_DIR / f"resolution_{fname}")
            print(f"  [R] resolution_{fname} (from {res_fname})")
            copied_r += 1
        else:
            available = sorted(r_src.glob("*.json")) if r_src.exists() else []
            print(f"  [WARN] No resolution file found for {fname}")
            if available:
                print(f"         Available: {[f.name for f in available[:5]]}")

    print(f"\nCopied {copied_q} question sets and {copied_r} resolution files to {RAW_DIR}")


def validate() -> None:
    """Quick sanity check: open each question set and print question counts."""
    print("\nValidating copied files…")
    for fname in QUESTION_SETS:
        path = RAW_DIR / fname
        if not path.exists():
            print(f"  [MISSING] {fname}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        n = len(data.get("questions", []))
        fdd = data.get("forecast_due_date", "?")
        print(f"  {fname}: {n} questions  (forecast_due_date={fdd})")

    # Show resolution files available
    res_files = sorted(RAW_DIR.glob("resolution_*.json"))
    print(f"\nResolution files: {len(res_files)}")
    for rf in res_files:
        with open(rf, encoding="utf-8") as f:
            content = json.load(f)
        # Resolution files may be a list or dict
        if isinstance(content, list):
            print(f"  {rf.name}: {len(content)} records (list)")
        elif isinstance(content, dict):
            print(f"  {rf.name}: keys={list(content.keys())[:5]}")


if __name__ == "__main__":
    clone_or_pull()
    copy_files()
    validate()
    print("\nA1 complete.")
