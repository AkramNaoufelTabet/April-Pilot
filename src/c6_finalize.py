"""
Task 7: Produce the final analysis-ready dataset.

Reads pilot_results_clean_after_reruns.json, canonicalizes columns, and writes
pilot_results_final.json -- a flat list ready for statistical analysis.

Canonical logic:
  forecast_final    = forecast_v2  (v2 parser / rerun / kept v1 for truncated rows)
  parse_success     = parse_success_v2
  brier_score_final = brier_score_v2  (None for unparsed rows)

Dropped: raw v1 forecast/brier, intermediate rerun metadata.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
IN_FILE  = ROOT / "results" / "merged" / "pilot_results_clean_after_reruns.json"
OUT_FILE = ROOT / "results" / "merged" / "pilot_results_final.json"

KEEP = [
    "model_key", "model_id", "model_type",
    "question_id", "source", "topic",
    "market_prob",
    "prompt", "condition",
    "forecast_final", "parse_success", "brier_score_final",
    "resolution_value",
    "latency_seconds", "input_tokens", "output_tokens", "reasoning_tokens",
    "parser_version", "notes",
]


def _bool(v):
    if isinstance(v, bool): return v
    return str(v).lower() in ("true", "1", "yes")

def _float_or_none(v):
    if v is None or str(v).lower() in ("none", "", "null"): return None
    try: return float(v)
    except (ValueError, TypeError): return None

def _int_or_none(v):
    if v is None or str(v).lower() in ("none", "", "null"): return None
    try: return int(v)
    except (ValueError, TypeError): return None


def main():
    with open(IN_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    rows = raw["results"]
    print(f"Loaded {len(rows):,} rows from {IN_FILE.name}")

    final = []
    for r in rows:
        ps = _bool(r.get("parse_success_v2", False))
        fv = _float_or_none(r.get("forecast_v2"))
        bs = _float_or_none(r.get("brier_score_v2"))

        row = {}
        for k in KEEP:
            if k not in ("forecast_final", "parse_success", "brier_score_final"):
                row[k] = r.get(k)
        row["forecast_final"]    = fv if ps else None
        row["parse_success"]     = ps
        row["brier_score_final"] = bs if ps else None
        row["market_prob"]       = _float_or_none(r.get("market_prob"))
        row["resolution_value"]  = _float_or_none(r.get("resolution_value"))
        row["latency_seconds"]   = _float_or_none(r.get("latency_seconds"))
        row["input_tokens"]      = _int_or_none(r.get("input_tokens"))
        row["output_tokens"]     = _int_or_none(r.get("output_tokens"))
        row["reasoning_tokens"]  = _int_or_none(r.get("reasoning_tokens"))
        final.append(row)

    n_total  = len(final)
    n_parsed = sum(1 for r in final if r["parse_success"])
    n_failed = n_total - n_parsed

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    SEP = "=" * 60
    sep = "-" * 45

    print(f"\n{SEP}")
    print("FINAL DATASET SUMMARY")
    print(SEP)
    print(f"Output:          {OUT_FILE.name}")
    print(f"Total rows:      {n_total:,}")
    print(f"Parse successes: {n_parsed:,}  ({n_parsed/n_total*100:.2f}%)")
    print(f"Parse failures:  {n_failed:,}  ({n_failed/n_total*100:.2f}%)")

    print(f"\n{sep}")
    print("Parse rate by source:")
    src_counts = defaultdict(lambda: [0, 0])
    for r in final:
        src_counts[r["source"]][1] += 1
        if r["parse_success"]: src_counts[r["source"]][0] += 1
    for src, (ok, tot) in sorted(src_counts.items()):
        print(f"  {src:<12}  {ok:>4}/{tot:<4}  ({ok/tot*100:.1f}%)")

    print(f"\n{sep}")
    print("Parse rate by model:")
    mdl_counts = defaultdict(lambda: [0, 0])
    for r in final:
        mdl_counts[r["model_key"]][1] += 1
        if r["parse_success"]: mdl_counts[r["model_key"]][0] += 1
    for mdl, (ok, tot) in sorted(mdl_counts.items()):
        flag = " !" if ok < tot else ""
        print(f"  {mdl:<30}  {ok:>3}/{tot:<3}  ({ok/tot*100:.1f}%){flag}")

    print(f"\n{sep}")
    print("Parse rate by prompt:")
    pmt_counts = defaultdict(lambda: [0, 0])
    for r in final:
        pmt_counts[r["prompt"]][1] += 1
        if r["parse_success"]: pmt_counts[r["prompt"]][0] += 1
    for pmt, (ok, tot) in sorted(pmt_counts.items()):
        print(f"  {pmt:<4}  {ok:>4}/{tot:<4}  ({ok/tot*100:.1f}%)")

    print(f"\n{sep}")
    print("Notes distribution:")
    notes_cnt = Counter(r.get("notes", "") for r in final)
    for note, cnt in notes_cnt.most_common():
        print(f"  {note:<35} {cnt:>5}")

    if n_failed:
        print(f"\n{sep}")
        print("Remaining failures (model x prompt x condition):")
        fail_cells = Counter(
            (r["model_key"], r["prompt"], r["condition"])
            for r in final if not r["parse_success"]
        )
        for (mdl, pmt, cond), cnt in sorted(fail_cells.items()):
            print(f"  {mdl:<30}  {pmt}  {cond:<18}  {cnt}")

    print(f"\n{SEP}")
    print(f"Saved {n_total:,} rows -> {OUT_FILE}")


if __name__ == "__main__":
    main()
