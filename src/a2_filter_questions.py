"""
A2 — Parse, filter, deduplicate, and join questions with resolution values.

Rules (from spec):
  1. Keep only standard questions: combination_of == "N/A"
  2. Keep only market questions: source in {manifold, metaculus, polymarket}
  3. Deduplicate across the 5 question sets by (id, source)
  4. Join with resolution sets; keep only resolved questions (binary 0 or 1)

Output: data/labeled/filtered_questions.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "labeled"
OUT_FILE = OUT_DIR / "filtered_questions.json"

QUESTION_SETS = [
    # Oct 2025 – Apr 2026: resolution dates safely after Aug 2025 (GPT-5.4 cutoff)
    "2025-10-26-llm.json",
    "2025-11-09-llm.json",
    "2025-11-23-llm.json",
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

# Contamination safeguard: exclude questions resolving on or before this date
RESOLUTION_CUTOFF = "2025-09-01"

MARKET_SOURCES = {"manifold", "metaculus", "polymarket"}


# ── Resolution loading ─────────────────────────────────────────────────────────

def load_resolutions() -> dict[tuple[str, str], int | float]:
    """
    Build a lookup: (id, source) -> resolved_to value.

    ForecastBench resolution files have structure:
      {
        "forecast_due_date": "...",
        "question_set": "...",
        "resolutions": [
          {"id": "...", "source": "...", "resolved": true, "resolved_to": 0.0, ...},
          ...
        ]
      }
    Only records with resolved=true are included.
    """
    resolutions: dict[tuple[str, str], int | float] = {}

    res_files = sorted(RAW_DIR.glob("resolution_*.json"))
    if not res_files:
        print("[WARN] No resolution files found. All questions will be unresolved.")
        return resolutions

    for rf in res_files:
        with open(rf, encoding="utf-8") as f:
            content = json.load(f)

        records: list = []
        if isinstance(content, list):
            records = content
        elif isinstance(content, dict):
            records = content.get("resolutions", [])

        for rec in records:
            if not isinstance(rec, dict):
                continue
            if not rec.get("resolved", False):
                continue  # skip unresolved
            qid = str(rec.get("id", ""))
            src = str(rec.get("source", ""))
            raw_val = rec.get("resolved_to")
            if qid and src and raw_val is not None:
                try:
                    resolutions[(qid, src)] = float(raw_val)
                except (ValueError, TypeError):
                    pass

    print(f"Loaded {len(resolutions)} resolved records from {len(res_files)} files.")
    return resolutions


# ── Question loading & filtering ───────────────────────────────────────────────

def load_and_filter() -> list[dict]:
    seen: set[tuple[str, str]] = set()
    all_questions: list[dict] = []

    stats = {
        "total": 0,
        "non_standard": 0,
        "non_market": 0,
        "duplicates": 0,
        "kept": 0,
    }

    for fname in QUESTION_SETS:
        path = RAW_DIR / fname
        if not path.exists():
            print(f"[WARN] Missing: {path}")
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        questions = data.get("questions", [])
        forecast_due_date = data.get("forecast_due_date", "")
        print(f"  {fname}: {len(questions)} raw questions")
        stats["total"] += len(questions)

        for q in questions:
            # Rule 1: standard only
            combo = q.get("combination_of", "N/A")
            if combo != "N/A":
                stats["non_standard"] += 1
                continue

            # Rule 2: market sources only
            source = q.get("source", "")
            if source not in MARKET_SOURCES:
                stats["non_market"] += 1
                continue

            # Rule 3: deduplicate
            qid = str(q.get("id", ""))
            key = (qid, source)
            if key in seen:
                stats["duplicates"] += 1
                continue
            seen.add(key)

            # Attach forecast_due_date from the question set
            q["forecast_due_date"] = q.get("forecast_due_date", forecast_due_date)
            all_questions.append(q)
            stats["kept"] += 1

    print(f"\nFiltering stats:")
    print(f"  Total questions seen:   {stats['total']}")
    print(f"  Dropped (non-standard): {stats['non_standard']}")
    print(f"  Dropped (non-market):   {stats['non_market']}")
    print(f"  Dropped (duplicates):   {stats['duplicates']}")
    print(f"  Kept:                   {stats['kept']}")

    return all_questions


def join_resolutions(questions: list[dict], resolutions: dict) -> list[dict]:
    resolved = []
    unresolved_count = 0
    non_binary_count = 0
    cutoff_dropped = 0

    for q in questions:
        qid = str(q.get("id", ""))
        src = q.get("source", "")
        key = (qid, src)

        res_val = resolutions.get(key)
        if res_val is None:
            unresolved_count += 1
            continue

        # Keep only binary resolutions: 0 or 1
        if res_val not in (0.0, 1.0):
            non_binary_count += 1
            continue

        # Contamination safeguard: drop questions resolving on/before cutoff
        fdd = q.get("resolution_date", "")
        if fdd and fdd <= RESOLUTION_CUTOFF:
            cutoff_dropped += 1
            continue

        q["resolution_value"] = int(res_val)
        resolved.append(q)

    print(f"\nResolution join:")
    print(f"  Unresolved (no match):  {unresolved_count}")
    print(f"  Non-binary resolution:  {non_binary_count}")
    print(f"  Dropped (cutoff guard): {cutoff_dropped}  (resolved on/before {RESOLUTION_CUTOFF})")
    print(f"  Resolved binary:        {len(resolved)}")

    return resolved


def clean_question(q: dict) -> dict:
    """Normalize fields and handle {forecast_due_date} placeholders."""
    fdd = q.get("forecast_due_date", "")

    def fill(text: str) -> str:
        if not isinstance(text, str):
            return text or ""
        return text.replace("{forecast_due_date}", fdd).replace(
            "{resolution_date}", fdd
        )

    # Handle resolution_dates for dataset questions (shouldn't be any after filter)
    res_dates = q.get("resolution_dates", "N/A")
    resolution_date = fdd
    if isinstance(res_dates, list) and res_dates:
        resolution_date = res_dates[-1]

    return {
        "question_id": str(q.get("id", "")),
        "source": q.get("source", ""),
        "question": fill(q.get("question", "")),
        "background": fill(q.get("background", "")),
        "resolution_criteria": fill(q.get("resolution_criteria", "")),
        "source_intro": q.get("source_intro", ""),
        "freeze_datetime": q.get("freeze_datetime", ""),
        "freeze_datetime_value": str(q.get("freeze_datetime_value", "")),
        "freeze_datetime_value_explanation": q.get("freeze_datetime_value_explanation", ""),
        "resolution_value": q.get("resolution_value"),
        "forecast_due_date": fdd,
        "resolution_date": resolution_date,
    }


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading resolutions ===")
    resolutions = load_resolutions()

    print("\n=== Loading and filtering questions ===")
    questions = load_and_filter()

    print("\n=== Joining with resolutions ===")
    resolved = join_resolutions(questions, resolutions)

    print("\n=== Cleaning and normalizing ===")
    cleaned = [clean_question(q) for q in resolved]

    # Source breakdown
    from collections import Counter
    src_counts = Counter(q["source"] for q in cleaned)
    print(f"\nSource breakdown: {dict(src_counts)}")

    # Freeze value distribution
    values = []
    for q in cleaned:
        try:
            values.append(float(q["freeze_datetime_value"]))
        except (ValueError, TypeError):
            pass
    if values:
        import statistics
        print(f"Freeze value stats: min={min(values):.3f}, max={max(values):.3f}, "
              f"mean={statistics.mean(values):.3f}, n={len(values)}")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(cleaned)} filtered questions to {OUT_FILE}")
    print("\nA2 complete.")
