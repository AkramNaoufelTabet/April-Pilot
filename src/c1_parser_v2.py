import json, re
from collections import defaultdict, Counter

with open("results/pilot_results.json", encoding="utf-8") as f:
    data = json.load(f)
rows = data["results"]
qwen_keys = defaultdict(list)
non_qwen = [r for r in rows if r["model_key"] != "qwen3-max"]
for r in rows:
    if r["model_key"] == "qwen3-max":
        k = (r["question_id"], r["prompt"], r["condition"])
        qwen_keys[k].append(r)
qwen_deduped = []
for k, entries in qwen_keys.items():
    success = [e for e in entries if e.get("parse_success")]
    qwen_deduped.append(success[-1] if success else entries[-1])
all_rows = non_qwen + qwen_deduped

# ── Final parser ───────────────────────────────────────────────────────────────
LABELS = (
    r"forecast"
    r"|probability"
    r"|my\s+estimate"
    r"|final\s+(?:answer|estimate|probability|posterior)"
    r"|p\s*\(\s*yes\s*\)"
    r"|p\s*\(\s*event\s*\)"
    r"|posterior(?:\s+probability)?"
)
# Separator: horizontal whitespace only on both sides of : or =
# Blocks "Forecast:\n1. Item" (newline after colon) and "Posterior 2: 15%" (digit before colon)
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


def parse_forecast(raw: str):
    if not raw or not raw.strip():
        return None, False
    t = raw
    for ch in (" ", " ", " ", " ", " ", "　"):
        t = t.replace(ch, " ")
    t = t.replace("\\%", "%")
    t = re.sub(r"\\boxed\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\\text\{([^}]*)\}",  r"\1", t)
    t = re.sub(r"\*+", "", t)
    t = re.sub(r"`+",  "", t)
    matches = list(re.finditer(LABELED_RE, t, re.IGNORECASE))
    if matches:
        return _to_prob(matches[-1].group(1), matches[-1].group(2))
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    for line in reversed(lines):
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%", line)
        if m:
            return _to_prob(m.group(1), "%")
        m = re.fullmatch(r"(\d+(?:\.\d+)?)", line)
        if m:
            val = float(m.group(1))
            if 0.0 <= val <= 1.0:
                return _to_prob(m.group(1), "")
    return None, False


# ── Unit tests ─────────────────────────────────────────────────────────────────
TESTS = [
    ("Forecast: 35%",                         0.35),
    ("Forecast: 0.35",                        0.35),
    ("**Forecast:** 35%",                     0.35),
    ("Forecast: **35%**",                     0.35),
    ("Probability: 0.35",                     0.35),
    ("My estimate: 35%",                      0.35),
    ("Final answer: 0.35",                    0.35),
    ("P(yes) = 0.35",                         0.35),
    ("some text\n0.35",                       0.35),
    ("some text\n35%",                        0.35),
    ("Posterior: 38%\nPosterior: 23%",        0.23),
    ("Posterior 2: 15%",                      None),
    ("Forecast:\n1. Item one",                None),
    ("Forecast: 35 %",                   0.35),
    ("\\boxed{\\text{Forecast: }75\\%}",      0.75),
    ("Forecast: 0%",                          0.0),
    ("Forecast: 100%",                        1.0),
    ("The forecast is\nForecast: 42%\nDone",  0.42),
]
print("=== UNIT TESTS ===")
all_ok = True
for text, expected in TESTS:
    val, ok = parse_forecast(text)
    result = val if ok else None
    status = "OK" if result == expected else "FAIL"
    if status == "FAIL":
        all_ok = False
    print(f"  {status}  expected={str(expected):6}  got={str(result):6}  input={repr(text[:55])}")
print(f"\nAll unit tests passed: {all_ok}")

# ── Flagged cell recovery ──────────────────────────────────────────────────────
FLAGGED = [
    ("gpt-oss-120b",  "P2", "closed_book"),
    ("gpt-oss-120b",  "P2", "shared_evidence"),
    ("gpt-oss-120b",  "P3", "closed_book"),
    ("gpt-oss-120b",  "P3", "shared_evidence"),
    ("kimi-k2.6",     "P3", "closed_book"),
    ("mistral-large", "P2", "closed_book"),
]
failures = [r for r in all_rows if not r.get("parse_success")]

print("\n=== FLAGGED CELL RECOVERY ===\n")
print(f"  {'Cell':<42} {'old_fail':>8}  {'recovered':>9}  {'still_fail':>10}")
print("  " + "-" * 73)
total_rec, total_still = 0, 0
for model, prompt, cond in FLAGGED:
    cf = [r for r in failures
          if r["model_key"] == model and r["prompt"] == prompt and r["condition"] == cond]
    rec   = sum(1 for r in cf if parse_forecast(r.get("raw_response", ""))[1])
    still = len(cf) - rec
    total_rec  += rec
    total_still += still
    print(f"  {model+' '+prompt+'_'+cond:<42} {len(cf):>8}  {rec:>9}  {still:>10}")
print(f"\n  TOTAL  recovered={total_rec}  still_fail={total_still}")

# ── Regression check ───────────────────────────────────────────────────────────
print("\n=== REGRESSION CHECK ===\n")
successes     = [r for r in all_rows if r.get("parse_success") and r.get("raw_response")]
testable_re   = rf"(?:{LABELS})[^\S\n]*[:\=][^\S\n]*\d"
truly_testable = [r for r in successes
                  if re.search(testable_re, r.get("raw_response", ""), re.IGNORECASE)]

regressions, disagree = [], []
for r in truly_testable:
    val_v2, ok_v2 = parse_forecast(r.get("raw_response", ""))
    if not ok_v2:
        regressions.append(r)
    elif abs((val_v2 or 0) - (r.get("forecast") or 0)) > 0.01:
        disagree.append((r, val_v2))

print(f"Truly testable rows (label+digit on same line in raw_response): {len(truly_testable)}")
print(f"Regressions (v2 returns None):                                  {len(regressions)}")
print(f"Disagreements (>0.01 diff from v1):                             {len(disagree)}")

if disagree:
    lc = Counter()
    for r, v2 in disagree:
        t = re.sub(r"\*+", "", r.get("raw_response", ""))
        ms = list(re.finditer(LABELED_RE, t, re.IGNORECASE))
        if ms:
            lc[ms[-1].group(0)[:35].strip().lower()] += 1
    print("\n  Disagreement: label of last v2 match (top 10):")
    for label, cnt in lc.most_common(10):
        print(f"    '{label}': {cnt}")
    print("\n  First 5 examples (v2 last-match vs v1 first-match):")
    for r, v2 in disagree[:5]:
        print(f"    {r['model_key']} {r['prompt']}_{r['condition']}  v1={r.get('forecast')}  v2={v2}")

# ── Overall parse rates ────────────────────────────────────────────────────────
print("\n=== OVERALL PARSE RATES ===\n")
n_total = len(all_rows)
n_v1    = sum(1 for r in all_rows if r.get("parse_success"))
n_v2    = sum(1 for r in all_rows
              if r.get("parse_success") or parse_forecast(r.get("raw_response", ""))[1])
print(f"  Total rows:                      {n_total}")
print(f"  v1 parsed:           {n_v1}/{n_total}  ({100*n_v1/n_total:.2f}%)")
print(f"  v2 parsed (stored):  {n_v2}/{n_total}  ({100*n_v2/n_total:.2f}%)")
print(f"  Net new parses from stored text: +{n_v2 - n_v1}")
remaining = n_total - n_v2
print(f"  Remaining failures:              {remaining}  (need rerun or empty API response)")
print()
print("  Per-cell for flagged cells:")
for model, prompt, cond in FLAGGED:
    cell = [r for r in all_rows
            if r["model_key"] == model and r["prompt"] == prompt and r["condition"] == cond]
    v1ok = sum(1 for r in cell if r.get("parse_success"))
    v2ok = sum(1 for r in cell
               if r.get("parse_success") or parse_forecast(r.get("raw_response", ""))[1])
    print(f"    {model} {prompt}_{cond:<16}  "
          f"v1={v1ok}/{len(cell)} ({100*v1ok/len(cell):.0f}%)  "
          f"v2={v2ok}/{len(cell)} ({100*v2ok/len(cell):.0f}%)")
