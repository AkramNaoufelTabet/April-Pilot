"""
Task 4: Re-parse all rows with the v2 parser.

Rules:
  - parse_success=True  AND raw_response truncated (len==2000):
      keep v1 forecast unchanged. v2 on truncated text would pick up
      intermediate Bayesian posteriors, not the final answer.
  - parse_success=True  AND raw_response complete (len<2000):
      apply v2; fall back to v1 if v2 fails.
  - parse_success=False AND v2 recovers from stored text:
      genuine recovery (v1 failed even on full response, so the format
      was broken and v2's match is the best available signal).
  - parse_success=False AND v2 also fails:
      mark parse_success_v2=False; notes="requires_rerun".

Adds columns: forecast_v2, parse_success_v2, brier_score_v2,
              parser_version, notes.
"""

import json, re
from collections import defaultdict, Counter
from pathlib import Path

ROOT     = Path(__file__).parent.parent
IN_FILE  = ROOT / "results" / "merged" / "pilot_results.json"

# ── Load + dedup qwen3-max ─────────────────────────────────────────────────────
with open(IN_FILE, encoding="utf-8") as f:
    data = json.load(f)
rows = data["results"]

qwen_keys = defaultdict(list)
non_qwen  = [r for r in rows if r["model_key"] != "qwen3-max"]
for r in rows:
    if r["model_key"] == "qwen3-max":
        k = (r["question_id"], r["prompt"], r["condition"])
        qwen_keys[k].append(r)
qwen_deduped = []
for k, entries in qwen_keys.items():
    success = [e for e in entries if e.get("parse_success")]
    qwen_deduped.append(success[-1] if success else entries[-1])
all_rows = non_qwen + qwen_deduped

# ── v2 parser ──────────────────────────────────────────────────────────────────
LABELS = (
    r"forecast"
    r"|probability"
    r"|my\s+estimate"
    r"|final\s+(?:answer|estimate|probability|posterior)"
    r"|p\s*\(\s*yes\s*\)"
    r"|p\s*\(\s*event\s*\)"
    r"|posterior(?:\s+probability)?"
)
LABELED_RE = rf"(?:{LABELS})[^\S\n]*[:\=][^\S\n]*(\d+(?:\.\d+)?)[^\S\n]*(%?)"


def _to_prob(num_str, pct_flag):
    try:
        val = float(num_str)
    except ValueError:
        return None, False
    if val != val:
        return None, False
    if pct_flag or val > 1.0:
        val /= 100.0
    val = round(min(max(val, 0.0), 1.0), 4)
    return val, True


def parse_forecast_v2(raw: str):
    if not raw or not raw.strip():
        return None, False
    t = raw
    for ch in (" ", " ", " ", " ", " ", "　"):
        t = t.replace(ch, " ")
    t = t.replace("\\%", "%")
    t = re.sub(r"\\boxed\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\\text\{([^}]*)\}",  r"\1", t)
    t = re.sub(r"\*+", "", t)
    t = re.sub(r"`+",  "", t)
    ms = list(re.finditer(LABELED_RE, t, re.IGNORECASE))
    if ms:
        return _to_prob(ms[-1].group(1), ms[-1].group(2))
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    for line in reversed(lines):
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%", line)
        if m:
            return _to_prob(m.group(1), "%")
        m = re.fullmatch(r"(\d+(?:\.\d+)?)", line)
        if m:
            v = float(m.group(1))
            if 0.0 <= v <= 1.0:
                return _to_prob(m.group(1), "")
    return None, False


# ── Apply rules and add v2 columns ────────────────────────────────────────────
stats = Counter()
enriched = []

for r in all_rows:
    row = dict(r)
    raw          = r.get("raw_response") or ""
    is_truncated = len(raw) == 2000
    v1_ok        = bool(r.get("parse_success"))
    v1_forecast  = r.get("forecast")
    resolution   = r.get("resolution_value")

    if v1_ok and is_truncated:
        # Rule 1: keep v1 — v2 on truncated text is unreliable
        row["forecast_v2"]       = v1_forecast
        row["parse_success_v2"]  = True
        row["brier_score_v2"]    = r.get("brier_score")
        row["parser_version"]    = "v1"
        row["notes"]             = "kept_v1_truncated_response"
        stats["kept_v1_truncated"] += 1

    elif v1_ok and not is_truncated:
        # Rule 2: apply v2 on complete text; fall back to v1 if v2 fails
        val_v2, ok_v2 = parse_forecast_v2(raw)
        if ok_v2:
            brier_v2 = round((val_v2 - resolution) ** 2, 4) if resolution is not None else None
            row["forecast_v2"]      = val_v2
            row["parse_success_v2"] = True
            row["brier_score_v2"]   = brier_v2
            row["parser_version"]   = "v2"
            row["notes"]            = ("v2_agrees" if abs(val_v2 - v1_forecast) <= 0.01
                                       else "v2_differs_complete_response")
            stats["v2_on_complete"] += 1
        else:
            # v2 failed on complete text — keep v1
            row["forecast_v2"]      = v1_forecast
            row["parse_success_v2"] = True
            row["brier_score_v2"]   = r.get("brier_score")
            row["parser_version"]   = "v1"
            row["notes"]            = "kept_v1_v2_failed_complete"
            stats["kept_v1_v2_failed"] += 1

    else:
        # Rule 3/4: v1 failed — try v2
        val_v2, ok_v2 = parse_forecast_v2(raw)
        if ok_v2:
            brier_v2 = round((val_v2 - resolution) ** 2, 4) if resolution is not None else None
            row["forecast_v2"]      = val_v2
            row["parse_success_v2"] = True
            row["brier_score_v2"]   = brier_v2
            row["parser_version"]   = "v2"
            row["notes"]            = "recovered_by_v2_parser"
            stats["recovered_by_v2"] += 1
        else:
            row["forecast_v2"]      = None
            row["parse_success_v2"] = False
            row["brier_score_v2"]   = None
            row["parser_version"]   = None
            row["notes"]            = "requires_rerun"
            stats["requires_rerun"] += 1

    enriched.append(row)

# ── Summary ────────────────────────────────────────────────────────────────────
print("=== TASK 4: Re-parse summary ===\n")
print(f"  Total rows (post-dedup): {len(enriched)}")
print(f"\n  Row disposition:")
for k, v in sorted(stats.items()):
    print(f"    {k:<35} {v}")

v1_total   = sum(1 for r in enriched if r.get("parse_success"))
v2_total   = sum(1 for r in enriched if r.get("parse_success_v2"))
recovered  = sum(1 for r in enriched if r["notes"] == "recovered_by_v2_parser")
rerun      = sum(1 for r in enriched if r["notes"] == "requires_rerun")
print(f"\n  v1 parse rate: {v1_total}/{len(enriched)}  ({100*v1_total/len(enriched):.2f}%)")
print(f"  v2 parse rate: {v2_total}/{len(enriched)}  ({100*v2_total/len(enriched):.2f}%)")
print(f"  Recovered by v2:          {recovered}")
print(f"  Still unresolvable:       {rerun}  (empty API response OR Forecast beyond char 2000)")

# ── Per-cell for the 6 flagged cells ──────────────────────────────────────────
FLAGGED = [
    ("gpt-oss-120b",  "P2", "closed_book"),
    ("gpt-oss-120b",  "P2", "shared_evidence"),
    ("gpt-oss-120b",  "P3", "closed_book"),
    ("gpt-oss-120b",  "P3", "shared_evidence"),
    ("kimi-k2.6",     "P3", "closed_book"),
    ("mistral-large", "P2", "closed_book"),
]
print(f"\n  Per-cell parse rates for flagged cells:")
print(f"  {'Cell':<44} {'v1':>12}  {'v2':>12}  {'recovered':>9}  {'rerun_needed':>12}")
print("  " + "-" * 94)
for model, prompt, cond in FLAGGED:
    cell = [r for r in enriched
            if r["model_key"] == model and r["prompt"] == prompt and r["condition"] == cond]
    v1ok  = sum(1 for r in cell if r.get("parse_success"))
    v2ok  = sum(1 for r in cell if r.get("parse_success_v2"))
    rec   = sum(1 for r in cell if r["notes"] == "recovered_by_v2_parser")
    rerun_n = sum(1 for r in cell if r["notes"] == "requires_rerun")
    n = len(cell)
    print(f"  {model+' '+prompt+'_'+cond:<44} "
          f"{v1ok}/{n} ({100*v1ok/n:.0f}%)  "
          f"{v2ok}/{n} ({100*v2ok/n:.0f}%)  "
          f"{rec:>9}  {rerun_n:>12}")

# ── Regression check: rows where v1 ok but v2 is now False ───────────────────
regressions = [r for r in enriched if r.get("parse_success") and not r.get("parse_success_v2")]
print(f"\n  Regressions (v1 ok, v2 False): {len(regressions)}  (should be 0)")

# ── Rows where v2 differs from v1 on complete text ───────────────────────────
differs = [r for r in enriched if r["notes"] == "v2_differs_complete_response"]
print(f"  v2 differs from v1 (complete text only): {len(differs)}")
if differs[:5]:
    print("  Examples:")
    for r in differs[:5]:
        print(f"    {r['model_key']} {r['prompt']}_{r['condition']}  "
              f"v1={r.get('forecast')}  v2={r.get('forecast_v2')}  "
              f"len_raw={len(r.get('raw_response',''))}")

print(f"\n  Saving enriched dataset...")

# ── Save ───────────────────────────────────────────────────────────────────────
out = ROOT / "results" / "pilot_results_v2.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump({
        "description": (
            "Pilot results with v2 parser columns. "
            "forecast_v2/brier_score_v2 are canonical where parse_success_v2=True. "
            "Rows with notes='requires_rerun' need new API calls."
        ),
        "n_rows":     len(enriched),
        "n_v2_ok":    v2_total,
        "n_rerun":    rerun,
        "results":    enriched,
    }, f, indent=2, ensure_ascii=False)
print(f"  Saved to {out}")
