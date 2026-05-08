"""
Task 6: Market baseline sanity check.

Samples 10 questions (3-4 per source) and produces a CSV for manual verification
that freeze_datetime_value is indeed the pre-resolution market probability.

Output: results/analysis/baseline_sanity_check.csv
"""

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_FILE = ROOT / "data" / "pilot_questions.json"
OUT_FILE = ROOT / "results" / "analysis" / "baseline_sanity_check.csv"

SAMPLE_PER_SOURCE = {"manifold": 4, "polymarket": 3, "metaculus": 3}
RANDOM_SEED = 42

COLS = [
    "question_id",
    "source",
    "question",
    "topic",
    "freeze_datetime",
    "freeze_datetime_value",
    "freeze_datetime_value_explanation",
    "resolution_date",
    "resolution_value",
    "freeze_before_resolution",  # sanity flag: True = OK
]


def main():
    with open(IN_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    by_source = {}
    for q in questions:
        by_source.setdefault(q["source"], []).append(q)

    print("Questions per source:", {s: len(v) for s, v in by_source.items()})

    rng = random.Random(RANDOM_SEED)
    sample = []
    for source, n in SAMPLE_PER_SOURCE.items():
        pool = by_source.get(source, [])
        picked = rng.sample(pool, min(n, len(pool)))
        sample.extend(picked)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        writer.writeheader()
        for q in sample:
            freeze = q.get("freeze_datetime", "")
            res_date = q.get("resolution_date", "") or q.get("forecast_due_date", "")
            # Simple lexicographic comparison works for ISO dates
            freeze_before = freeze[:10] < res_date[:10] if freeze and res_date else "N/A"
            writer.writerow({
                "question_id": q["question_id"],
                "source": q["source"],
                "question": q["question"],
                "topic": q.get("topic", ""),
                "freeze_datetime": freeze,
                "freeze_datetime_value": q.get("freeze_datetime_value", ""),
                "freeze_datetime_value_explanation": q.get("freeze_datetime_value_explanation", ""),
                "resolution_date": res_date,
                "resolution_value": q.get("resolution_value", ""),
                "freeze_before_resolution": freeze_before,
            })

    print(f"\nSaved {len(sample)} rows to {OUT_FILE}\n")

    # Print summary table
    print(f"{'source':<12} {'question_id':<30} {'freeze_date':<12} {'res_date':<12} "
          f"{'freeze_val':>10} {'res_val':>8} {'OK?'}")
    print("-" * 95)
    for q in sample:
        freeze = q.get("freeze_datetime", "")[:10]
        res_date = (q.get("resolution_date", "") or q.get("forecast_due_date", ""))[:10]
        ok = "YES" if freeze < res_date else "*** NO ***"
        fv = float(q.get("freeze_datetime_value", 0))
        rv = q.get("resolution_value", "?")
        qtext = q["question"][:45] + "..." if len(q["question"]) > 45 else q["question"]
        print(f"{q['source']:<12} {q['question_id']:<30} {freeze:<12} {res_date:<12} "
              f"{fv:>10.4f} {rv:>8} {ok}")
        print(f"  Q: {qtext}")

    # Overall check
    all_ok = all(
        (q.get("freeze_datetime", "")[:10] < (q.get("resolution_date","") or q.get("forecast_due_date",""))[:10])
        for q in sample
    )
    print(f"\nAll freeze dates pre-resolution? {'YES' if all_ok else 'NO — check flagged rows'}")


if __name__ == "__main__":
    main()
