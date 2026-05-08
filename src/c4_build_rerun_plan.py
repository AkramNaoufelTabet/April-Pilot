"""
Task 5: Build rerun plan for all parse_success_v2=False rows.

Classifies each failure as:
  - truncation_likely : output_tokens >= 90% of MAX_TOKENS (32768) → rerun with 2x tokens
  - empty_response    : output_tokens == 0 or None, or raw_response == "" → API error, rerun as-is
  - unknown           : output_tokens present but well below max → inspect before deciding
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT      = Path(__file__).parent.parent
IN_FILE   = ROOT / "results" / "merged" / "pilot_results_v2.json"
OUT_FILE  = ROOT / "results" / "reruns" / "rerun_plan.json"

MAX_TOKENS      = 32768
TRUNC_THRESHOLD = 0.90   # output_tokens >= 90% of max → likely truncated

with open(IN_FILE, encoding="utf-8") as f:
    data = json.load(f)

all_rows  = data["results"]
failed    = [r for r in all_rows if not r.get("parse_success_v2")]

print(f"Total rows:            {len(all_rows)}")
print(f"parse_success_v2=True: {len(all_rows) - len(failed)}")
print(f"parse_success_v2=False (need rerun): {len(failed)}\n")

# ── Classify each failure ──────────────────────────────────────────────────────
def classify(r):
    out_tok = r.get("output_tokens")
    raw     = r.get("raw_response") or ""
    if out_tok is None or out_tok == 0 or raw.strip() == "":
        return "empty_response"
    if out_tok >= MAX_TOKENS * TRUNC_THRESHOLD:
        return "truncation_likely"
    return "unknown_low_tokens"

for r in failed:
    r["_failure_class"] = classify(r)

class_counts = Counter(r["_failure_class"] for r in failed)
print("Failure classification:")
for k, v in class_counts.most_common():
    print(f"  {k:<25} {v}")

# ── Group by cell ──────────────────────────────────────────────────────────────
print("\n\nPer-cell breakdown of failures:")
print(f"  {'Model':<28} {'Prompt_Cond':<22} {'total':>5}  {'empty':>5}  {'trunc':>5}  {'low_tok':>7}")
print("  " + "-" * 75)

cell_map = defaultdict(list)
for r in failed:
    cell_map[(r["model_key"], r["prompt"], r["condition"])].append(r)

for (model, prompt, cond), cell_rows in sorted(cell_map.items()):
    empty = sum(1 for r in cell_rows if r["_failure_class"] == "empty_response")
    trunc = sum(1 for r in cell_rows if r["_failure_class"] == "truncation_likely")
    low   = sum(1 for r in cell_rows if r["_failure_class"] == "unknown_low_tokens")
    print(f"  {model:<28} {prompt+'_'+cond:<22} {len(cell_rows):>5}  {empty:>5}  {trunc:>5}  {low:>7}")

# ── Token stats for each class ─────────────────────────────────────────────────
print("\n\nOutput token stats by failure class:")
for cls in ["truncation_likely", "empty_response", "unknown_low_tokens"]:
    toks = [r["output_tokens"] for r in failed
            if r["_failure_class"] == cls and r.get("output_tokens") is not None]
    if toks:
        print(f"  {cls}: n={len(toks)}  "
              f"min={min(toks)}  median={sorted(toks)[len(toks)//2]}  max={max(toks)}")
    else:
        print(f"  {cls}: n={len([r for r in failed if r['_failure_class']==cls])}  (all None/0)")

# ── Unknown low-token cases: inspect ──────────────────────────────────────────
low_tok_rows = [r for r in failed if r["_failure_class"] == "unknown_low_tokens"]
if low_tok_rows:
    print(f"\n\nUnknown low-token failures ({len(low_tok_rows)}) — sample responses:")
    for r in low_tok_rows[:10]:
        raw_tail = (r.get("raw_response") or "")[-200:].encode("ascii","replace").decode("ascii")
        print(f"\n  {r['model_key']} {r['prompt']}_{r['condition']}  "
              f"qid={r['question_id'][:25]}  out_tok={r.get('output_tokens')}")
        print(f"  tail: {raw_tail}")

# ── Build rerun plan ───────────────────────────────────────────────────────────
# Recommended max_tokens:
#   truncation_likely  → 65536 (2× current MAX_TOKENS)
#   empty_response     → 32768 (same; API error, not a token issue)
#   unknown_low_tokens → 32768 (same; inspect first but include in plan)

rerun_items = []
for r in failed:
    cls = r["_failure_class"]
    recommended_max_tokens = 65536 if cls == "truncation_likely" else 32768
    rerun_items.append({
        "model_key":             r["model_key"],
        "model_id":              r["model_id"],
        "question_id":           r["question_id"],
        "prompt":                r["prompt"],
        "condition":             r["condition"],
        "failure_class":         cls,
        "output_tokens_original": r.get("output_tokens"),
        "recommended_max_tokens": recommended_max_tokens,
    })

# Summary by model + recommended_max_tokens
print("\n\nRerun plan summary:")
print(f"  {'Model':<28} {'32768':>6}  {'65536':>6}  {'total':>6}")
print("  " + "-" * 55)
by_model = defaultdict(lambda: Counter())
for item in rerun_items:
    by_model[item["model_key"]][item["recommended_max_tokens"]] += 1
for model in sorted(by_model):
    c = by_model[model]
    print(f"  {model:<28} {c[32768]:>6}  {c[65536]:>6}  {c[32768]+c[65536]:>6}")
total_32 = sum(c[32768] for c in by_model.values())
total_64 = sum(c[65536] for c in by_model.values())
print(f"  {'TOTAL':<28} {total_32:>6}  {total_64:>6}  {total_32+total_64:>6}")

# Save
plan = {
    "description": (
        "Rows requiring API re-runs. "
        "failure_class='truncation_likely': output near 32768 token limit, "
        "rerun with 65536 max_tokens and patched logger (raw_response cap raised). "
        "failure_class='empty_response': API returned nothing, rerun with same settings. "
        "failure_class='unknown_low_tokens': output well below limit, inspect before running."
    ),
    "total_reruns": len(rerun_items),
    "by_failure_class": dict(class_counts),
    "items": rerun_items,
}
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(plan, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {OUT_FILE}  ({len(rerun_items)} items)")
