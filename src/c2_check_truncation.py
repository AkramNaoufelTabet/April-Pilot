import json, re
from collections import defaultdict

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

def _to_prob(n, p):
    try: v = float(n)
    except: return None, False
    if p or v > 1.0: v /= 100.0
    return round(min(max(v, 0.0), 1.0), 4), True

def parse_forecast(raw):
    if not raw or not raw.strip(): return None, False
    t = raw
    for ch in (" ", " ", " ", " ", " ", "　"):
        t = t.replace(ch, " ")
    t = t.replace("\\%", "%")
    t = re.sub(r"\\boxed\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\\text\{([^}]*)\}",  r"\1", t)
    t = re.sub(r"\*+", "", t)
    t = re.sub(r"`+",  "", t)
    ms = list(re.finditer(LABELED_RE, t, re.IGNORECASE))
    if ms: return _to_prob(ms[-1].group(1), ms[-1].group(2))
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    for line in reversed(lines):
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%", line)
        if m: return _to_prob(m.group(1), "%")
        m = re.fullmatch(r"(\d+(?:\.\d+)?)", line)
        if m:
            v = float(m.group(1))
            if 0.0 <= v <= 1.0: return _to_prob(m.group(1), "")
    return None, False

def safe(s):
    return (s or "").encode("ascii", "replace").decode("ascii")

testable_re = rf"(?:{LABELS})[^\S\n]*[:\=][^\S\n]*\d"
successes = [r for r in all_rows if r.get("parse_success") and r.get("raw_response")]
truly_testable = [r for r in successes
                  if re.search(testable_re, r.get("raw_response", ""), re.IGNORECASE)]

disagree_rows = []
for r in truly_testable:
    val_v2, ok_v2 = parse_forecast(r.get("raw_response", ""))
    if ok_v2 and abs((val_v2 or 0) - (r.get("forecast") or 0)) > 0.01:
        disagree_rows.append((r, val_v2))

print(f"Total disagreement rows: {len(disagree_rows)}")

truncated     = [(r, v2) for r, v2 in disagree_rows if len(r.get("raw_response","")) == 2000]
not_truncated = [(r, v2) for r, v2 in disagree_rows if len(r.get("raw_response","")) < 2000]

print(f"  Truncated at 2000 chars: {len(truncated)}")
print(f"  Not truncated (<2000):   {len(not_truncated)}")

# ── Not-truncated examples ─────────────────────────────────────────────────────
print(f"\n--- NOT-TRUNCATED disagreement examples ({len(not_truncated)} total) ---")
for r, v2 in not_truncated[:10]:
    raw = r.get("raw_response", "")
    t   = re.sub(r"\*+", "", raw)
    ms  = list(re.finditer(LABELED_RE, t, re.IGNORECASE))
    print(f"\n  {r['model_key']} {r['prompt']}_{r['condition']}  "
          f"v1={r.get('forecast')}  v2={v2}  len={len(raw)}")
    print(f"  All labeled matches ({len(ms)}):")
    for m in ms:
        print(f"    pos={m.start():4d}  '{safe(m.group(0)[:55])}'")
    # Is v1's value correct or is v2's?
    # v1 used full response; check if "1.0" or similar looks like a list-item parse
    if abs((r.get("forecast") or 0) - 1.0) < 0.001:
        print(f"  NOTE: v1=1.0 — likely matched list item '1' in 'forecast:\\n1.'")

# ── Truncated examples: show last 300 chars ────────────────────────────────────
print(f"\n--- 10 TRUNCATED disagreement examples (last 300 chars) ---")
for r, v2 in truncated[:10]:
    raw  = r.get("raw_response", "")
    t    = re.sub(r"\*+", "", raw)
    ms   = list(re.finditer(LABELED_RE, t, re.IGNORECASE))
    tail = safe(raw[-300:])
    print(f"\n  {r['model_key']} {r['prompt']}_{r['condition']}  v1={r.get('forecast')}  v2={v2}")
    print(f"  Labeled matches in stored 2000 chars: {len(ms)}")
    if ms:
        print(f"  Last v2 match: '{safe(ms[-1].group(0)[:50])}'  pos={ms[-1].start()}")
    print(f"  Tail of stored text:")
    print(f"    ...{tail}")
    print(f"  Ends mid-sentence (truncated): {not tail.rstrip().endswith(('.', '%', ')', '!', '?'))}")
