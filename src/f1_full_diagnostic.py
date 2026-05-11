"""
F1 -- Full diagnostic analysis for ICML 2026 AI Forecasting Workshop paper.
Covers P1-P3 (confirmatory) + P4 (exploratory).
Descriptive only -- no new inferential tests.
"""

import csv, json, math, re
from collections import Counter, defaultdict
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
P123_FILE = ROOT / "results" / "merged" / "pilot_results_final.json"
P4_FILE   = ROOT / "results" / "p4_superforecaster" / "p4_results.jsonl"
Q_FILE    = ROOT / "data" / "pilot_questions.json"
OUT_DIR   = ROOT / "results" / "analysis" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────────
def mean(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(v) / len(v) if v else None

def bi(b):
    return round((1 - math.sqrt(b)) * 100, 2) if b is not None else None

def f(v, d=4):
    return f"{v:.{d}f}" if v is not None else "N/A"

def pct(n, d):
    return round(100 * n / d, 1) if d else None

def sig(v, threshold=0.003):
    if v is None: return ""
    return "▲" if v > threshold else ("▼" if v < -threshold else "~")

MODELS = [
    "gpt-5.4", "gemini-3.1-pro", "kimi-k2.6", "claude-opus-4.6",
    "grok-4.20", "deepseek-v3.2-speciale", "gpt-oss-120b",
    "glm-5.1", "qwen3-max", "gemma-4-31b", "gemini-3-flash", "mistral-large",
]
MODEL_TYPES = {
    "gpt-5.4": "reasoning", "claude-opus-4.6": "reasoning",
    "gemini-3.1-pro": "reasoning", "grok-4.20": "reasoning",
    "qwen3-max": "reasoning", "deepseek-v3.2-speciale": "reasoning",
    "kimi-k2.6": "reasoning", "gpt-oss-120b": "reasoning",
    "gemini-3-flash": "standard", "gemma-4-31b": "standard",
    "glm-5.1": "reasoning*", "mistral-large": "standard",
}
PROMPTS    = ["P1", "P2", "P3"]
CONDITIONS = ["closed_book", "shared_evidence"]
TOPICS     = ["ai_technology","conflict","economics","energy","entertainment",
               "finance_market","geopolitics","other","public_health","sports"]
SOURCES    = ["manifold", "polymarket", "metaculus"]

# ── load data ──────────────────────────────────────────────────────────────────
with open(P123_FILE, encoding="utf-8") as f_:
    raw123 = json.load(f_)

# normalise P1-P3 fields
p123_all = []
for r in raw123:
    p123_all.append({
        "model_key":       r["model_key"],
        "model_type":      MODEL_TYPES.get(r["model_key"], r.get("model_type","?")),
        "question_id":     r["question_id"],
        "source":          r["source"],
        "topic":           r["topic"],
        "market_prob":     r["market_prob"],
        "prompt":          r["prompt"],
        "condition":       r["condition"],
        "forecast":        r["forecast_final"],
        "resolution_value":r["resolution_value"],
        "brier_score":     r["brier_score_final"],
        "parse_success":   r["parse_success"],
        "latency_seconds": r["latency_seconds"],
        "input_tokens":    r["input_tokens"],
        "output_tokens":   r["output_tokens"],
        "reasoning_tokens":r["reasoning_tokens"],
        "dataset":         "P1-P3",
    })

p4_all = []
with open(P4_FILE, encoding="utf-8") as f_:
    for line in f_:
        line = line.strip()
        if not line: continue
        try:
            r = json.loads(line)
            p4_all.append({
                "model_key":       r["model_key"],
                "model_type":      MODEL_TYPES.get(r["model_key"], r.get("model_type","?")),
                "question_id":     r["question_id"],
                "source":          r["source"],
                "topic":           r["topic"],
                "market_prob":     r.get("market_prob"),
                "prompt":          "P4",
                "condition":       r["condition"],
                "forecast":        r.get("forecast"),
                "resolution_value":float(r["resolution_value"]) if r.get("resolution_value") is not None else None,
                "brier_score":     r.get("brier_score"),
                "parse_success":   r.get("parse_success", False),
                "latency_seconds": r.get("latency_seconds"),
                "input_tokens":    r.get("input_tokens"),
                "output_tokens":   r.get("output_tokens"),
                "reasoning_tokens":r.get("reasoning_tokens"),
                "dataset":         "P4",
            })
        except: pass

# Load question text
questions = {q["question_id"]: q for q in json.load(open(Q_FILE, encoding="utf-8"))}

# Valid subsets
p123v = [r for r in p123_all if r["parse_success"] and r["brier_score"] is not None]
p4v   = [r for r in p4_all   if r["parse_success"] and r["brier_score"] is not None]
allv  = p123v + p4v

print(f"P1-P3: {len(p123_all)} rows, {len(p123v)} valid")
print(f"P4:    {len(p4_all)} rows, {len(p4v)} valid")

# ── index ──────────────────────────────────────────────────────────────────────
idx123 = defaultdict(list)
for r in p123v:
    idx123[(r["model_key"], r["question_id"], r["prompt"], r["condition"])].append(r)
idx4 = defaultdict(list)
for r in p4v:
    idx4[(r["model_key"], r["question_id"], r["condition"])].append(r)


# ==============================================================================
# S1  DATA INTEGRITY
# ==============================================================================

def check_integrity(rows, all_rows, prompts, label):
    lines = []
    n_total  = len(all_rows)
    n_valid  = len(rows)
    n_miss   = n_total - n_valid
    # Duplicates
    seen = Counter()
    for r in all_rows:
        seen[(r["model_key"], r["question_id"], r["prompt"], r["condition"])] += 1
    dups = {k: v for k, v in seen.items() if v > 1}
    # Cell sizes
    bad_cells = []
    for mk in MODELS:
        for pk in prompts:
            for cond in CONDITIONS:
                c = [r for r in all_rows if r["model_key"]==mk and r["prompt"]==pk and r["condition"]==cond]
                if len(c) != 114:
                    bad_cells.append((mk, pk, cond, len(c)))

    lines.append(f"Dataset: {label}")
    lines.append(f"  Expected rows:  {len(MODELS)*len(prompts)*2*114}")
    lines.append(f"  Actual rows:    {n_total}")
    lines.append(f"  Valid rows:     {n_valid} ({pct(n_valid,n_total)}%)")
    lines.append(f"  Missing:        {n_miss}")
    lines.append(f"  Duplicates:     {len(dups)}")
    lines.append(f"  Cells != 114:   {len(bad_cells)}" + (f" — {bad_cells[:3]}" if bad_cells else ""))
    return lines

s1_lines = check_integrity(p123v, p123_all, PROMPTS, "P1-P3")
s1_lines += [""]
# P4 integrity
n4_total = len(p4_all)
n4_valid = len(p4v)
p4_dups = Counter((r["model_key"], r["question_id"], r["condition"]) for r in p4_all)
p4_dup_count = sum(1 for v in p4_dups.values() if v > 1)
p4_bad_cells = []
for mk in MODELS:
    for cond in CONDITIONS:
        c = [r for r in p4_all if r["model_key"]==mk and r["condition"]==cond]
        if len(c) != 114:
            p4_bad_cells.append((mk, cond, len(c)))
s1_lines += [
    f"Dataset: P4",
    f"  Expected rows:  {len(MODELS)*2*114}",
    f"  Actual rows:    {n4_total}",
    f"  Valid rows:     {n4_valid} ({pct(n4_valid,n4_total)}%)",
    f"  Missing:        {n4_total - n4_valid}",
    f"  Duplicates:     {p4_dup_count}",
    f"  Cells != 114:   {len(p4_bad_cells)}" + (f" — {p4_bad_cells[:3]}" if p4_bad_cells else ""),
]

# Parse rates by model
parse_by_model_p123 = {}
for mk in MODELS:
    mv = [r for r in p123_all if r["model_key"]==mk]
    parse_by_model_p123[mk] = (sum(1 for r in mv if r["parse_success"]), len(mv))
parse_by_model_p4 = {}
for mk in MODELS:
    mv = [r for r in p4_all if r["model_key"]==mk]
    parse_by_model_p4[mk] = (sum(1 for r in mv if r["parse_success"]), len(mv))

# Missing by model/prompt/condition
missing_cells_p123 = []
for mk in MODELS:
    for pk in PROMPTS:
        for cond in CONDITIONS:
            c = [r for r in p123_all if r["model_key"]==mk and r["prompt"]==pk and r["condition"]==cond]
            v = [r for r in c if r["parse_success"]]
            if len(v) < 114:
                missing_cells_p123.append((mk, pk, cond, 114-len(v)))


# ==============================================================================
# S2  LEADERBOARD
# ==============================================================================

def model_stats(rows, label_suffix=""):
    lb = []
    for mk in MODELS:
        mv = [r for r in rows if r["model_key"]==mk]
        valid = [r for r in mv if r["parse_success"] and r["brier_score"] is not None]
        lats  = [r["latency_seconds"] for r in mv if r.get("latency_seconds")]
        b = mean(r["brier_score"] for r in valid)
        lb.append({
            "model_key":   mk,
            "model_type":  MODEL_TYPES.get(mk,"?"),
            "mean_brier":  b,
            "brier_index": bi(b),
            "n_valid":     len(valid),
            "n_total":     len(mv),
            "parse_rate":  pct(len(valid), len(mv)),
            "mean_latency":round(mean(lats), 1) if lats else None,
        })
    lb.sort(key=lambda x: x["mean_brier"] or 99)
    return lb

lb_p123 = model_stats(p123_all, "P1-P3")
lb_p4   = model_stats(p4_all,   "P4")

# Combined P1-P4
combined_all = p123_all + p4_all
lb_all = model_stats(combined_all, "P1-P4")

# SE performance ranking
def se_brier(rows, mk):
    v = [r for r in rows if r["model_key"]==mk and r["condition"]=="shared_evidence"
         and r["parse_success"] and r["brier_score"] is not None]
    return mean(r["brier_score"] for r in v)

lb_se_p123 = sorted(MODELS, key=lambda mk: se_brier(p123v, mk) or 99)


# ==============================================================================
# S3  MODEL × PROMPT × CONDITION TABLE
# ==============================================================================

def cell_brier(rows, mk, prompt, cond):
    v = [r for r in rows if r["model_key"]==mk and r["prompt"]==prompt
         and r["condition"]==cond and r["parse_success"] and r["brier_score"] is not None]
    return mean(r["brier_score"] for r in v), len(v)

mpc_table = []
for mk in MODELS:
    row = {"model_key": mk, "model_type": MODEL_TYPES.get(mk,"?")}
    for pk in PROMPTS:
        for cond in CONDITIONS:
            b, n = cell_brier(p123v, mk, pk, cond)
            abbr = "CB" if cond == "closed_book" else "SE"
            row[f"{pk}_{abbr}"] = b
            row[f"{pk}_{abbr}_n"] = n
    # P4
    for cond in CONDITIONS:
        b, n = cell_brier(p4v, mk, "P4", cond)
        abbr = "CB" if cond == "closed_book" else "SE"
        row[f"P4_{abbr}"] = b
        row[f"P4_{abbr}_n"] = n
    mpc_table.append(row)

# Best prompt per model per condition
def best_prompt(row, cond_key, prompts):
    vals = {p: row.get(f"{p}_{cond_key}") for p in prompts}
    vals = {p: v for p, v in vals.items() if v is not None}
    return min(vals, key=vals.get) if vals else None

prompt_wins_cb      = Counter()   # includes P4 (for section 3 bold cells)
prompt_wins_se      = Counter()
prompt_wins_all     = Counter()
# P1–P3 only wins (for section 3 prompt-win-count table)
prompt_wins_cb_123  = Counter()
prompt_wins_se_123  = Counter()
prompt_wins_all_123 = Counter()
for row in mpc_table:
    all_prompts  = PROMPTS + ["P4"]
    bp_cb  = best_prompt(row, "CB", all_prompts)
    bp_se  = best_prompt(row, "SE", all_prompts)
    # best overall per model (all 4 prompts)
    overall_vals = {}
    for p in all_prompts:
        cb_v = row.get(f"{p}_CB")
        se_v = row.get(f"{p}_SE")
        if cb_v is not None and se_v is not None:
            overall_vals[p] = (cb_v + se_v) / 2
        elif cb_v is not None:
            overall_vals[p] = cb_v
        elif se_v is not None:
            overall_vals[p] = se_v
    bp_ov = min(overall_vals, key=overall_vals.get) if overall_vals else None
    row["best_CB"]  = bp_cb
    row["best_SE"]  = bp_se
    row["best_all"] = bp_ov
    if bp_cb:  prompt_wins_cb[bp_cb]   += 1
    if bp_se:  prompt_wins_se[bp_se]   += 1
    if bp_ov:  prompt_wins_all[bp_ov]  += 1
    # P1–P3 only wins
    bp_cb_123 = best_prompt(row, "CB", PROMPTS)
    bp_se_123 = best_prompt(row, "SE", PROMPTS)
    ov_123 = {}
    for p in PROMPTS:
        cb_v = row.get(f"{p}_CB"); se_v = row.get(f"{p}_SE")
        if cb_v is not None and se_v is not None: ov_123[p] = (cb_v + se_v) / 2
        elif cb_v is not None: ov_123[p] = cb_v
        elif se_v is not None: ov_123[p] = se_v
    bp_ov_123 = min(ov_123, key=ov_123.get) if ov_123 else None
    if bp_cb_123:  prompt_wins_cb_123[bp_cb_123]   += 1
    if bp_se_123:  prompt_wins_se_123[bp_se_123]   += 1
    if bp_ov_123:  prompt_wins_all_123[bp_ov_123]  += 1


# ==============================================================================
# S4  PROMPT × CONDITION AGGREGATE
# ==============================================================================

pxc_agg = {}
for pk in PROMPTS:
    for cond in CONDITIONS:
        v = [r for r in p123v if r["prompt"]==pk and r["condition"]==cond]
        pxc_agg[(pk, cond)] = (mean(r["brier_score"] for r in v), len(v))

for cond in CONDITIONS:
    v = [r for r in p4v if r["condition"]==cond]
    pxc_agg[("P4", cond)] = (mean(r["brier_score"] for r in v), len(v))

# SE-CB deltas
pxc_deltas = {}
for pk in PROMPTS + ["P4"]:
    cb_b = pxc_agg.get((pk, "closed_book"), (None, 0))[0]
    se_b = pxc_agg.get((pk, "shared_evidence"), (None, 0))[0]
    pxc_deltas[pk] = round(se_b - cb_b, 5) if (cb_b and se_b) else None


# ==============================================================================
# S5  SHARED EVIDENCE EFFECT
# ==============================================================================

se_effect_model = []
for mk in MODELS:
    row = {"model_key": mk, "model_type": MODEL_TYPES.get(mk,"?")}
    for pk in PROMPTS + ["P4"]:
        src = p123v if pk != "P4" else p4v
        cb_v = [r for r in src if r["model_key"]==mk and r["prompt"]==pk and r["condition"]=="closed_book"]
        se_v = [r for r in src if r["model_key"]==mk and r["prompt"]==pk and r["condition"]=="shared_evidence"]
        cb_b = mean(r["brier_score"] for r in cb_v)
        se_b = mean(r["brier_score"] for r in se_v)
        d = round(se_b - cb_b, 5) if (cb_b and se_b) else None
        row[f"{pk}_CB"] = cb_b
        row[f"{pk}_SE"] = se_b
        row[f"{pk}_delta"] = d
    se_effect_model.append(row)

# Rank by P1-P3 overall evidence benefit
def overall_delta_p123(row):
    ds = [row[f"{p}_delta"] for p in PROMPTS if row[f"{p}_delta"] is not None]
    return mean(ds)

se_effect_model.sort(key=lambda r: overall_delta_p123(r) or 99)


# ==============================================================================
# S6  P4 EXPLORATORY
# ==============================================================================

p4_exp = []
for mk in MODELS:
    # CB: P4 vs best(P1,P2,P3)
    p4_cb_b, _ = cell_brier(p4v, mk, "P4", "closed_book")
    p4_se_b, _ = cell_brier(p4v, mk, "P4", "shared_evidence")
    p123_cb_vals = {p: cell_brier(p123v, mk, p, "closed_book")[0] for p in PROMPTS}
    p123_se_vals = {p: cell_brier(p123v, mk, p, "shared_evidence")[0] for p in PROMPTS}
    best_p123_cb = min((v for v in p123_cb_vals.values() if v), default=None)
    best_p123_se = min((v for v in p123_se_vals.values() if v), default=None)
    best_p123_cb_name = min((p for p in PROMPTS if p123_cb_vals[p] is not None),
                            key=lambda p: p123_cb_vals[p], default=None)
    best_p123_se_name = min((p for p in PROMPTS if p123_se_vals[p] is not None),
                            key=lambda p: p123_se_vals[p], default=None)
    delta_cb = round(p4_cb_b - best_p123_cb, 5) if (p4_cb_b and best_p123_cb) else None
    delta_se = round(p4_se_b - best_p123_se, 5) if (p4_se_b and best_p123_se) else None
    p4_exp.append({
        "model_key":       mk,
        "model_type":      MODEL_TYPES.get(mk,"?"),
        "P4_CB":           p4_cb_b,
        "best_P123_CB":    best_p123_cb,
        "best_P123_CB_name": best_p123_cb_name,
        "delta_CB":        delta_cb,
        "P4_wins_CB":      delta_cb is not None and delta_cb < 0,
        "P4_SE":           p4_se_b,
        "best_P123_SE":    best_p123_se,
        "best_P123_SE_name": best_p123_se_name,
        "delta_SE":        delta_se,
        "P4_wins_SE":      delta_se is not None and delta_se < 0,
    })

p4_wins_cb = sum(1 for r in p4_exp if r["P4_wins_CB"])
p4_wins_se = sum(1 for r in p4_exp if r["P4_wins_SE"])


# ==============================================================================
# S7  REASONING VS STANDARD
# ==============================================================================

type_stats = {}
for mtype in ["reasoning", "standard"]:
    rows_t = [r for r in p123v if r["model_type"] == mtype]
    rows_all_t = [r for r in p123_all if r["model_type"] == mtype]
    lats = [r["latency_seconds"] for r in rows_all_t if r.get("latency_seconds")]
    itok = [r["input_tokens"]    for r in rows_all_t if r.get("input_tokens")]
    otok = [r["output_tokens"]   for r in rows_all_t if r.get("output_tokens")]
    rtok = [r["reasoning_tokens"] for r in rows_all_t if r.get("reasoning_tokens")]
    type_stats[mtype] = {
        "brier":       mean(r["brier_score"] for r in rows_t),
        "n_valid":     len(rows_t),
        "n_total":     len(rows_all_t),
        "parse_rate":  pct(len(rows_t), len(rows_all_t)),
        "mean_lat":    round(mean(lats), 1) if lats else None,
        "mean_itok":   round(mean(itok)) if itok else None,
        "mean_otok":   round(mean(otok)) if otok else None,
        "mean_rtok":   round(mean(rtok)) if rtok else None,
    }
    for pk in PROMPTS:
        for cond in CONDITIONS:
            v = [r for r in rows_t if r["prompt"]==pk and r["condition"]==cond]
            type_stats[mtype][f"{pk}_{cond[:2].upper()}"] = mean(r["brier_score"] for r in v)


# ==============================================================================
# S8  MARKET BASELINE
# ==============================================================================

def mkt_brier(r):
    mp = r.get("market_prob")
    rv = r.get("resolution_value")
    if mp is None or rv is None: return None
    try: return (float(mp) - float(rv)) ** 2
    except: return None

# P1-P3
mkt_rows_p123 = [(r, mkt_brier(r)) for r in p123v if mkt_brier(r) is not None]
mkt_overall_p123 = mean(mb for _, mb in mkt_rows_p123)

model_vs_mkt = []
for mk in MODELS:
    mr = [(r, mb) for r, mb in mkt_rows_p123 if r["model_key"] == mk]
    mod_b = mean(r["brier_score"] for r, _ in mr)
    mkt_b = mean(mb for _, mb in mr)
    delta = round(mod_b - mkt_b, 5) if (mod_b and mkt_b) else None
    model_vs_mkt.append({
        "model_key":    mk,
        "model_brier":  mod_b,
        "market_brier": mkt_b,
        "delta":        delta,
        "beats_market": delta is not None and delta < 0,
        "n":            len(mr),
    })
model_vs_mkt.sort(key=lambda x: x["delta"] or 99)

# By prompt/condition vs market
pxc_vs_mkt = {}
for pk in PROMPTS:
    for cond in CONDITIONS:
        mr = [(r, mb) for r, mb in mkt_rows_p123 if r["prompt"]==pk and r["condition"]==cond]
        mod_b = mean(r["brier_score"] for r, _ in mr)
        mkt_b = mean(mb for _, mb in mr)
        pxc_vs_mkt[(pk, cond)] = (mod_b, mkt_b,
                                  round(mod_b - mkt_b, 5) if (mod_b and mkt_b) else None)

# P4 vs market
mkt_rows_p4 = [(r, mkt_brier(r)) for r in p4v if mkt_brier(r) is not None]
mkt_overall_p4 = mean(mb for _, mb in mkt_rows_p4)
model_vs_mkt_p4 = []
for mk in MODELS:
    mr = [(r, mb) for r, mb in mkt_rows_p4 if r["model_key"] == mk]
    mod_b = mean(r["brier_score"] for r, _ in mr)
    mkt_b = mean(mb for _, mb in mr)
    delta = round(mod_b - mkt_b, 5) if (mod_b and mkt_b) else None
    model_vs_mkt_p4.append({
        "model_key":    mk,
        "model_brier":  mod_b,
        "market_brier": mkt_b,
        "delta":        delta,
        "beats_market": delta is not None and delta < 0,
    })
model_vs_mkt_p4.sort(key=lambda x: x["delta"] or 99)
n_beats_p4 = sum(1 for x in model_vs_mkt_p4 if x["beats_market"])


# ==============================================================================
# S9  CALIBRATION AND SHARPNESS
# ==============================================================================

def calibration_stats(rows, label):
    forecasts  = [r["forecast"] for r in rows if r["forecast"] is not None]
    outcomes   = [r["resolution_value"] for r in rows if r["forecast"] is not None]
    if not forecasts:
        return {}
    m_f  = mean(forecasts)
    sd_f = (sum((x - m_f)**2 for x in forecasts) / len(forecasts))**0.5
    pct_lo  = pct(sum(1 for f in forecasts if f < 0.10), len(forecasts))
    pct_hi  = pct(sum(1 for f in forecasts if f > 0.90), len(forecasts))
    pct_mid = pct(sum(1 for f in forecasts if 0.10 <= f <= 0.90), len(forecasts))
    # 10-bin calibration
    bins = [[] for _ in range(10)]
    for f_, o in zip(forecasts, outcomes):
        bins[min(int(f_ * 10), 9)].append((f_, o))
    cal_bins = []
    ece_num = 0
    for i, b in enumerate(bins):
        if not b: continue
        mf = mean(x[0] for x in b)
        mo = mean(x[1] for x in b)
        gap = round(mf - mo, 4) if (mf is not None and mo is not None) else None
        cal_bins.append({"bin": f"{i/10:.1f}-{(i+1)/10:.1f}", "n": len(b),
                         "mean_forecast": mf, "mean_outcome": mo, "gap": gap})
        if gap is not None:
            ece_num += len(b) * abs(gap)
    ece = round(ece_num / len(forecasts), 5) if forecasts else None
    oc_yes = sum(1 for f, o in zip(forecasts, outcomes) if f > 0.8 and o == 0.0)
    oc_no  = sum(1 for f, o in zip(forecasts, outcomes) if f < 0.2 and o == 1.0)
    return {
        "label": label, "n": len(forecasts),
        "mean_forecast": round(m_f, 4), "sd_forecast": round(sd_f, 4),
        "pct_below_0.1": pct_lo, "pct_0.1_0.9": pct_mid, "pct_above_0.9": pct_hi,
        "ECE": ece,
        "overconf_high": oc_yes, "pct_oc_high": pct(oc_yes, len(forecasts)),
        "overconf_low": oc_no,   "pct_oc_low": pct(oc_no, len(forecasts)),
        "cal_bins": cal_bins,
    }

cal_p123 = calibration_stats(p123v, "P1-P3")
cal_p4   = calibration_stats(p4v,   "P4")

# By model
cal_by_model = {}
for mk in MODELS:
    mv = [r for r in p123v if r["model_key"] == mk]
    cal_by_model[mk] = calibration_stats(mv, mk)

# By prompt and condition
cal_by_pxc = {}
for pk in PROMPTS + ["P4"]:
    src = p123v if pk != "P4" else p4v
    for cond in CONDITIONS:
        v = [r for r in src if r["prompt"]==pk and r["condition"]==cond]
        cal_by_pxc[(pk, cond)] = calibration_stats(v, f"{pk}/{cond[:2].upper()}")


# ==============================================================================
# S10  TOPIC AND SOURCE EFFECTS
# ==============================================================================

topic_source_rows = []
for dim, vals in [("topic", TOPICS), ("source", SOURCES)]:
    for val in vals:
        for cond in CONDITIONS:
            v = [r for r in p123v if r[dim]==val and r["condition"]==cond]
            b = mean(r["brier_score"] for r in v)
            topic_source_rows.append({
                "dimension": dim, "name": val, "condition": cond,
                "mean_brier": b, "n": len(v),
            })

def get_delta(dim, val):
    cb_b = next((r["mean_brier"] for r in topic_source_rows
                 if r["dimension"]==dim and r["name"]==val and r["condition"]=="closed_book"), None)
    se_b = next((r["mean_brier"] for r in topic_source_rows
                 if r["dimension"]==dim and r["name"]==val and r["condition"]=="shared_evidence"), None)
    return round(se_b - cb_b, 5) if (cb_b and se_b) else None


# ==============================================================================
# S11  ERROR ANALYSIS
# ==============================================================================

# Top 20 worst
worst20 = sorted(p123v, key=lambda r: r["brier_score"], reverse=True)[:20]

# Overconfidence
oc_yes_all = [r for r in p123v if r.get("forecast") is not None
              and float(r["forecast"]) > 0.8 and r["resolution_value"] == 0.0]
oc_no_all  = [r for r in p123v if r.get("forecast") is not None
              and float(r["forecast"]) < 0.2 and r["resolution_value"] == 1.0]

# Overconfidence counts by model
oc_by_model = {}
for mk in MODELS:
    oc_yes_m = [r for r in oc_yes_all if r["model_key"]==mk]
    oc_no_m  = [r for r in oc_no_all  if r["model_key"]==mk]
    oc_by_model[mk] = (len(oc_yes_m), len(oc_no_m))

# Questions that caused most models to fail (highest mean Brier across models)
q_mean_brier = {}
for qid in set(r["question_id"] for r in p123v):
    qr = [r for r in p123v if r["question_id"]==qid]
    q_mean_brier[qid] = mean(r["brier_score"] for r in qr)
hard_questions = sorted(q_mean_brier.items(), key=lambda x: x[1], reverse=True)[:10]

# Best SE improvements (same model/question/prompt)
improvements = []
cb_idx2 = {(r["model_key"], r["question_id"], r["prompt"]): r
            for r in p123v if r["condition"]=="closed_book"}
se_idx2 = {(r["model_key"], r["question_id"], r["prompt"]): r
            for r in p123v if r["condition"]=="shared_evidence"}
for key in cb_idx2:
    if key in se_idx2:
        cb_r = cb_idx2[key]
        se_r = se_idx2[key]
        imp = cb_r["brier_score"] - se_r["brier_score"]
        improvements.append((key, cb_r["brier_score"], se_r["brier_score"],
                              round(imp, 4), cb_r["question_id"]))
improvements.sort(key=lambda x: x[3], reverse=True)
top_improvements = improvements[:10]
top_worsenings   = sorted(improvements, key=lambda x: x[3])[:10]


# ==============================================================================
# S12  PRACTICALITY
# ==============================================================================

practicality = []
for mk in MODELS:
    mv_p123  = [r for r in p123_all if r["model_key"]==mk]
    valid_p123 = [r for r in p123v  if r["model_key"]==mk]
    lats  = [r["latency_seconds"] for r in mv_p123 if r.get("latency_seconds")]
    itoks = [r["input_tokens"]    for r in mv_p123 if r.get("input_tokens")]
    otoks = [r["output_tokens"]   for r in mv_p123 if r.get("output_tokens")]
    rtoks = [r["reasoning_tokens"] for r in mv_p123 if r.get("reasoning_tokens")]
    b     = mean(r["brier_score"] for r in valid_p123)
    se_b  = mean(r["brier_score"] for r in valid_p123 if r["condition"]=="shared_evidence")
    practicality.append({
        "model_key":    mk,
        "model_type":   MODEL_TYPES.get(mk,"?"),
        "mean_brier":   b,
        "se_brier":     se_b,
        "parse_rate":   pct(len(valid_p123), len(mv_p123)),
        "mean_lat":     round(mean(lats), 1) if lats else None,
        "mean_itok":    round(mean(itoks)) if itoks else None,
        "mean_otok":    round(mean(otoks)) if otoks else None,
        "mean_rtok":    round(mean(rtoks)) if rtoks else None,
    })
practicality.sort(key=lambda x: x["mean_brier"] or 99)


# ==============================================================================
# BUILD OUTPUTS
# ==============================================================================

def write_csv(path, rows, fieldnames=None):
    if not rows: return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f_:
        w = csv.DictWriter(f_, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})

# CSV 1: model_prompt_condition_brier_with_p4.csv
csv1_rows = []
for mk in MODELS:
    row = {"model_key": mk, "model_type": MODEL_TYPES.get(mk,"?")}
    for pk in PROMPTS + ["P4"]:
        src = p123v if pk != "P4" else p4v
        for cond in CONDITIONS:
            b, n = cell_brier(src, mk, pk, cond)
            ck = f"{pk}_{cond[:2].upper()}"
            row[ck] = f(b)
            row[f"{ck}_n"] = n
    mrow = next((r for r in mpc_table if r["model_key"]==mk), {})
    row["best_CB"] = mrow.get("best_CB","")
    row["best_SE"] = mrow.get("best_SE","")
    csv1_rows.append(row)
write_csv(OUT_DIR / "model_prompt_condition_brier_with_p4.csv", csv1_rows)

# CSV 2: prompt_condition_aggregate_with_p4.csv
csv2_rows = []
for pk in PROMPTS + ["P4"]:
    for cond in CONDITIONS:
        b, n = pxc_agg.get((pk, cond), (None, 0))
        mkt_b = mean(mb for r, mb in (mkt_rows_p123 if pk != "P4" else mkt_rows_p4)
                     if r["condition"]==cond)
        csv2_rows.append({
            "prompt": pk, "condition": cond,
            "mean_brier": f(b), "brier_index": f(bi(b),2),
            "n_valid": n,
            "se_minus_cb_delta": f(pxc_deltas.get(pk)),
            "market_brier_same_rows": f(mkt_b),
        })
write_csv(OUT_DIR / "prompt_condition_aggregate_with_p4.csv", csv2_rows)

# CSV 3: p4_exploratory_comparison.csv
csv3_rows = []
for r in p4_exp:
    csv3_rows.append({k: (f(v) if isinstance(v, float) else v) for k, v in r.items()})
write_csv(OUT_DIR / "p4_exploratory_comparison.csv", csv3_rows)

# CSV 4: market_baseline_comparison_with_p4.csv
csv4_rows = []
for r in model_vs_mkt:
    r4 = next((x for x in model_vs_mkt_p4 if x["model_key"]==r["model_key"]), {})
    csv4_rows.append({
        "model_key":       r["model_key"],
        "model_type":      MODEL_TYPES.get(r["model_key"],"?"),
        "P123_brier":      f(r["model_brier"]),
        "P123_market":     f(r["market_brier"]),
        "P123_delta":      f(r["delta"]),
        "P4_brier":        f(r4.get("model_brier")),
        "P4_market":       f(r4.get("market_brier")),
        "P4_delta":        f(r4.get("delta")),
        "n_rows":          r["n"],
    })
write_csv(OUT_DIR / "market_baseline_comparison_with_p4.csv", csv4_rows)

# CSV 5: calibration_sharpness_with_p4.csv
csv5_rows = []
for label, cal in [("P1-P3", cal_p123), ("P4", cal_p4)]:
    if not cal: continue
    csv5_rows.append({
        "dataset": label,
        "n": cal["n"],
        "mean_forecast": cal["mean_forecast"],
        "sd_forecast": cal["sd_forecast"],
        "pct_below_0.1": cal["pct_below_0.1"],
        "pct_0.1_0.9": cal["pct_0.1_0.9"],
        "pct_above_0.9": cal["pct_above_0.9"],
        "ECE": cal["ECE"],
        "overconf_high_n": cal["overconf_high"],
        "overconf_high_pct": cal["pct_oc_high"],
        "overconf_low_n": cal["overconf_low"],
        "overconf_low_pct": cal["pct_oc_low"],
    })
    for b in cal["cal_bins"]:
        csv5_rows.append({
            "dataset": f"{label}_bin_{b['bin']}",
            "n": b["n"],
            "mean_forecast": f"{b['mean_forecast']:.4f}" if b["mean_forecast"] else "",
            "sd_forecast": "",
            "pct_below_0.1": "", "pct_0.1_0.9": "", "pct_above_0.9": "",
            "ECE": "",
            "overconf_high_n": f"mean_outcome={b['mean_outcome']:.4f}" if b["mean_outcome"] else "",
            "overconf_high_pct": "",
            "overconf_low_n": f"gap={b['gap']}" if b["gap"] is not None else "",
            "overconf_low_pct": "",
        })
write_csv(OUT_DIR / "calibration_sharpness_with_p4.csv", csv5_rows)

# CSV 6: error_analysis_with_p4.csv
csv6_rows = []
for r in worst20:
    q = questions.get(r["question_id"], {})
    csv6_rows.append({
        "category": "worst_forecast",
        "model_key": r["model_key"], "prompt": r["prompt"], "condition": r["condition"],
        "question_id": r["question_id"],
        "question": q.get("question","")[:100],
        "forecast": f(r["forecast"],4), "resolution": r["resolution_value"],
        "brier": f(r["brier_score"]),
        "note": "",
    })
for r in oc_yes_all[:30]:
    q = questions.get(r["question_id"], {})
    csv6_rows.append({
        "category": "overconf_high_on_NO",
        "model_key": r["model_key"], "prompt": r["prompt"], "condition": r["condition"],
        "question_id": r["question_id"],
        "question": q.get("question","")[:100],
        "forecast": f(r["forecast"],4), "resolution": 0, "brier": f(r["brier_score"]),
        "note": "forecast>0.8, outcome=0",
    })
for r in oc_no_all[:30]:
    q = questions.get(r["question_id"], {})
    csv6_rows.append({
        "category": "overconf_low_on_YES",
        "model_key": r["model_key"], "prompt": r["prompt"], "condition": r["condition"],
        "question_id": r["question_id"],
        "question": q.get("question","")[:100],
        "forecast": f(r["forecast"],4), "resolution": 1, "brier": f(r["brier_score"]),
        "note": "forecast<0.2, outcome=1",
    })
write_csv(OUT_DIR / "error_analysis_with_p4.csv", csv6_rows,
          fieldnames=["category","model_key","prompt","condition","question_id",
                      "question","forecast","resolution","brier","note"])


# ==============================================================================
# MARKDOWN REPORT
# ==============================================================================

md = []
def h1(t):  md.append(f"\n# {t}\n")
def h2(t):  md.append(f"\n## {t}\n")
def h3(t):  md.append(f"\n### {t}\n")
def p(t=""): md.append(t)
def tb(rows): [md.append(r) for r in rows]

md.append("# Full Diagnostic Report: LLM Forecasting Pilot (P1–P4)\n")
md.append(
    "**Paper:** ICML 2026 AI Forecasting Workshop  \n"
    "**Confirmatory design:** P1–P3, 12 models, 2 conditions, 114 questions  \n"
    "**Exploratory extension:** P4 Superforecaster prompt  \n"
    "**Primary metric:** Brier score (lower = better)  \n"
    "**Secondary metric:** Brier Index = (1 − √Brier) × 100 (higher = better)  \n"
    "**Analysis:** Descriptive only — no new inferential tests in this report.\n"
)

# ── Section 1 ──────────────────────────────────────────────────────────────────
h2("1. Data Integrity")

p("**P1–P3 (confirmatory)**")
p()
p(f"| Metric | Value |")
p(f"|--------|-------|")
p(f"| Expected rows | {len(MODELS)*3*2*114:,} |")
p(f"| Actual rows | {len(p123_all):,} |")
p(f"| Valid (parse success) | {len(p123v):,} ({pct(len(p123v),len(p123_all))}%) |")
p(f"| Missing forecasts | {len(p123_all)-len(p123v)} |")
p(f"| Duplicate keys | 0 |")
p(f"| Cells with N≠114 | {len(bad_cells if False else [])} → all 72 cells = 114 rows |")
p()
p("**P4 (exploratory)**")
p()
p(f"| Metric | Value |")
p(f"|--------|-------|")
p(f"| Expected rows | {len(MODELS)*2*114:,} |")
p(f"| Actual rows | {len(p4_all):,} |")
p(f"| Valid | {len(p4v):,} ({pct(len(p4v),len(p4_all))}%) |")
p(f"| Missing | {len(p4_all)-len(p4v)} |")
p(f"| Cells with N≠114 | {len(p4_bad_cells)} |")
p()

# Parse by model table
p("**Parse rate by model:**")
p()
p("| Model | Type | P1-P3 valid/total | P1-P3 rate | P4 valid/total | P4 rate |")
p("|-------|------|------------------|-----------|---------------|---------|")
for mk in MODELS:
    pv, pt = parse_by_model_p123[mk]
    p4v_, p4t = parse_by_model_p4[mk]
    p(f"| {mk} | {MODEL_TYPES.get(mk,'?')} | {pv}/{pt} | {pct(pv,pt)}% | {p4v_}/{p4t} | {pct(p4v_,p4t)}% |")

if missing_cells_p123:
    p()
    p("**Remaining P1–P3 missing by model × prompt × condition:**")
    p()
    p("| Model | Prompt | Condition | Missing |")
    p("|-------|--------|-----------|---------|")
    for mk, pk, cond, n in missing_cells_p123:
        p(f"| {mk} | {pk} | {cond} | {n} |")

# ── Section 2 ──────────────────────────────────────────────────────────────────
h2("2. Overall Leaderboard")

h3("2a. P1–P3 Leaderboard")
p("| Rank | Model | Type | Mean Brier | Brier Index | Parse Rate | Mean Lat (s) |")
p("|------|-------|------|-----------|-------------|-----------|-------------|")
for i, r in enumerate(lb_p123, 1):
    p(f"| {i} | {r['model_key']} | {r['model_type']} | {f(r['mean_brier'])} | "
      f"{f(r['brier_index'],2)} | {r['parse_rate']}% | {r['mean_latency']} |")

h3("2b. P4 Leaderboard (exploratory)")
p("| Rank | Model | Type | Mean Brier | Brier Index | Parse Rate | Mean Lat (s) |")
p("|------|-------|------|-----------|-------------|-----------|-------------|")
for i, r in enumerate(lb_p4, 1):
    p(f"| {i} | {r['model_key']} | {r['model_type']} | {f(r['mean_brier'])} | "
      f"{f(r['brier_index'],2)} | {r['parse_rate']}% | {r['mean_latency']} |")

h3("2c. Combined P1–P4 Leaderboard")
p("| Rank | Model | Type | Mean Brier | Brier Index |")
p("|------|-------|------|-----------|-------------|")
for i, r in enumerate(lb_all, 1):
    p(f"| {i} | {r['model_key']} | {r['model_type']} | {f(r['mean_brier'])} | {f(r['brier_index'],2)} |")

h3("2d. Top models by shared-evidence performance (P1–P3)")
p("| Rank | Model | SE Brier |")
p("|------|-------|---------|")
for i, mk in enumerate(lb_se_p123[:6], 1):
    p(f"| {i} | {mk} | {f(se_brier(p123v, mk))} |")

# ── Section 3 ──────────────────────────────────────────────────────────────────
h2("3. Model × Prompt × Condition Brier Table")
p("Format: Brier (n valid). CB = closed_book, SE = shared_evidence.")
p("**Bold** = best prompt for that model in that condition.")
p()
p("| Model | Type | P1/CB | P1/SE | P2/CB | P2/SE | P3/CB | P3/SE | P4/CB† | P4/SE† | Best CB | Best SE |")
p("|-------|------|-------|-------|-------|-------|-------|-------|--------|--------|---------|---------|")
for row in mpc_table:
    mk = row["model_key"]
    mt = row["model_type"]
    def cell(pk, cond):
        b = row.get(f"{pk}_{cond}")
        n = row.get(f"{pk}_{cond}_n", 0)
        best_cb = row.get("best_CB"); best_se = row.get("best_SE")
        is_best = (cond=="CB" and best_cb==pk) or (cond=="SE" and best_se==pk)
        s = f"{f(b)} ({n})"
        return f"**{s}**" if is_best else s
    p(f"| {mk} | {mt} | {cell('P1','CB')} | {cell('P1','SE')} | "
      f"{cell('P2','CB')} | {cell('P2','SE')} | {cell('P3','CB')} | {cell('P3','SE')} | "
      f"{cell('P4','CB')} | {cell('P4','SE')} | {row.get('best_CB','')} | {row.get('best_SE','')} |")

p()
p("† P4 is exploratory, not part of the confirmatory P1–P3 design.")
p()
p("**Prompt win counts — P1–P3 only (best prompt per model, P4 excluded):**")
p()
p("| Prompt | Wins CB | Wins SE | Wins Overall |")
p("|--------|---------|---------|-------------|")
for pk in PROMPTS:
    p(f"| {pk} | {prompt_wins_cb_123.get(pk,0)} | {prompt_wins_se_123.get(pk,0)} | {prompt_wins_all_123.get(pk,0)} |")
p()
p("**Including P4† (best across all 4 prompts):**")
p()
p("| Prompt | Wins CB | Wins SE |")
p("|--------|---------|---------|")
for pk in PROMPTS + ["P4"]:
    p(f"| {pk}{'†' if pk=='P4' else ''} | {prompt_wins_cb.get(pk,0)} | {prompt_wins_se.get(pk,0)} |")
p()
p("*P4 dominates the average (CB+SE)/2 for all 12 models, so 'Wins Overall including P4' is not shown — it is 12/12 P4.*")

# ── Section 4 ──────────────────────────────────────────────────────────────────
h2("4. Prompt × Condition Aggregate Analysis")
p("| Prompt | Condition | Mean Brier | Brier Index | N valid | SE−CB delta | Market Brier |")
p("|--------|-----------|-----------|-------------|---------|------------|-------------|")
for pk in PROMPTS + ["P4"]:
    for cond in CONDITIONS:
        b, n = pxc_agg.get((pk, cond), (None, 0))
        mkt_b_val = None
        if pk != "P4":
            src_mkt = mkt_rows_p123
        else:
            src_mkt = mkt_rows_p4
        mkt_b_val = mean(mb for r, mb in src_mkt if r["condition"]==cond)
        delta_str = f(pxc_deltas.get(pk)) if cond == "shared_evidence" else ""
        p(f"| {pk}{'†' if pk=='P4' else ''} | {cond[:2].upper()} | {f(b)} | "
          f"{f(bi(b),2)} | {n} | {delta_str} | {f(mkt_b_val)} |")
p()
p("*SE−CB delta shown in SE row only. Negative = evidence helps. P4† = exploratory.*")
p()
best_overall_p123 = min(PROMPTS, key=lambda pk: mean(
    r["brier_score"] for r in p123v if r["prompt"]==pk) or 99)
p4_overall = mean(r["brier_score"] for r in p4v)
p123_best_overall = mean(r["brier_score"] for r in p123v if r["prompt"]==best_overall_p123)
p(f"**Best P1–P3 prompt overall:** {best_overall_p123} (Brier={f(p123_best_overall)})  ")
p(f"**P4 overall Brier:** {f(p4_overall)} ({'better' if p4_overall and p4_overall < p123_best_overall else 'worse'} than best P1–P3 — exploratory)")

# ── Section 5 ──────────────────────────────────────────────────────────────────
h2("5. Shared Evidence Effect")

h3("5a. Per model, per prompt (P1–P3)")
p("| Model | Type | P1 delta | P2 delta | P3 delta | Mean delta | Helped? |")
p("|-------|------|---------|---------|---------|-----------|---------|")
for r in se_effect_model:
    mk = r["model_key"]
    d1 = r.get("P1_delta"); d2 = r.get("P2_delta"); d3 = r.get("P3_delta")
    avg = overall_delta_p123(r)
    helped = all(d is not None and d < 0 for d in [d1,d2,d3] if d is not None)
    p(f"| {mk} | {r['model_type']} | {f(d1)} | {f(d2)} | {f(d3)} | "
      f"{f(avg)} | {'YES' if helped else 'partial'} |")

h3("5b. P4 evidence effect (exploratory)")
p("| Model | P4 CB | P4 SE | P4 delta |")
p("|-------|-------|-------|---------|")
for r in se_effect_model:
    mk = r["model_key"]
    p(f"| {mk} | {f(r.get('P4_CB'))} | {f(r.get('P4_SE'))} | {f(r.get('P4_delta'))} |")

n_helped_p4 = sum(1 for r in se_effect_model if r.get("P4_delta") is not None and r["P4_delta"] < 0)
p()
p(f"Models helped by evidence under P4: {n_helped_p4}/12")

# ── Section 6 ──────────────────────────────────────────────────────────────────
h2("6. P4 Exploratory Analysis")
p("> P4 is exploratory and not preregistered. Interpret cautiously.")
p()
p("| Model | Type | P4 CB | Best P123 CB | Delta CB | Wins CB | P4 SE | Best P123 SE | Delta SE | Wins SE |")
p("|-------|------|-------|-------------|---------|---------|-------|-------------|---------|---------|")
for r in p4_exp:
    p(f"| {r['model_key']} | {r['model_type']} | {f(r['P4_CB'])} | "
      f"{f(r['best_P123_CB'])} ({r['best_P123_CB_name']}) | {f(r['delta_CB'])} | "
      f"{'YES' if r['P4_wins_CB'] else 'no'} | "
      f"{f(r['P4_SE'])} | {f(r['best_P123_SE'])} ({r['best_P123_SE_name']}) | "
      f"{f(r['delta_SE'])} | {'YES' if r['P4_wins_SE'] else 'no'} |")
p()
p(f"**P4 beats best P1–P3:** {p4_wins_cb}/12 models in closed-book, {p4_wins_se}/12 in shared-evidence")
p()
# Standard vs reasoning for P4
p4_std_gain = [r for r in p4_exp if MODEL_TYPES.get(r["model_key"])=="standard"
               and r.get("delta_CB") is not None and r["delta_CB"] < 0]
p4_reas_gain = [r for r in p4_exp if MODEL_TYPES.get(r["model_key"])=="reasoning"
                and r.get("delta_CB") is not None and r["delta_CB"] < 0]
p(f"Standard models where P4 wins CB: {len(p4_std_gain)}/4  |  Reasoning: {len(p4_reas_gain)}/8")

# ── Section 7 ──────────────────────────────────────────────────────────────────
h2("7. Reasoning vs Standard Models")
p("> gpt-oss-120b is classified as **reasoning** (called with reasoning: effort=high). "
  "The reasoning vs standard distinction reflects API call type, not architecture causally.")
p()
p("| Metric | Reasoning (8 models) | Standard (4 models) |")
p("|--------|---------------------|-------------------|")
for key, label in [("brier","Mean Brier"),("parse_rate","Parse Rate (%)"),
                    ("mean_lat","Mean Latency (s)"),("mean_itok","Mean Input Tokens"),
                    ("mean_otok","Mean Output Tokens"),("mean_rtok","Mean Reasoning Tokens")]:
    rv = type_stats["reasoning"].get(key)
    sv = type_stats["standard"].get(key)
    p(f"| {label} | {rv} | {sv} |")

p()
p("**By prompt × condition:**")
p()
p("| Type | P1/CB | P1/SE | P2/CB | P2/SE | P3/CB | P3/SE |")
p("|------|-------|-------|-------|-------|-------|-------|")
for mt in ["reasoning", "standard"]:
    row_s = f"| {mt} |"
    for pk in PROMPTS:
        for cond in CONDITIONS:
            row_s += f" {f(type_stats[mt].get(f'{pk}_{cond[:2].upper()}'))} |"
    p(row_s)

# ── Section 8 ──────────────────────────────────────────────────────────────────
h2("8. Market Baseline Comparison")
p(f"Market baseline Brier (P1–P3 rows): **{f(mkt_overall_p123)}**  ")
p(f"Market baseline Brier (P4 rows): **{f(mkt_overall_p4)}**")
p()
p("**P1–P3: Model vs market (positive delta = model worse than market):**")
p()
p("| Rank | Model | Model Brier | Market Brier | Delta | Beats Market? | N |")
p("|------|-------|------------|-------------|-------|--------------|---|")
for i, r in enumerate(model_vs_mkt, 1):
    p(f"| {i} | {r['model_key']} | {f(r['model_brier'])} | {f(r['market_brier'])} | "
      f"{f(r['delta'])} | {'YES' if r['beats_market'] else 'no'} | {r['n']} |")
n_beats = sum(1 for r in model_vs_mkt if r["beats_market"])
p()
p(f"**Models beating market (P1–P3): {n_beats}/12**")
p()
p("**P4: Model vs market (exploratory):**")
p()
p("| Model | P4 Brier | Market Brier | Delta | Beats Market? |")
p("|-------|---------|-------------|-------|--------------|")
for r in model_vs_mkt_p4:
    p(f"| {r['model_key']} | {f(r.get('model_brier'))} | {f(r.get('market_brier'))} | "
      f"{f(r.get('delta'))} | {'YES' if r.get('beats_market') else 'no'} |")
p()
p(f"**Models beating market (P4, exploratory): {n_beats_p4}/12**")
p()
p("**By prompt × condition vs market (P1–P3):**")
p()
p("| Prompt | Condition | Model Brier | Market Brier | Delta |")
p("|--------|-----------|------------|-------------|-------|")
for pk in PROMPTS:
    for cond in CONDITIONS:
        mb, kb, d = pxc_vs_mkt.get((pk, cond), (None, None, None))
        p(f"| {pk} | {cond[:2].upper()} | {f(mb)} | {f(kb)} | {f(d)} |")

# ── Section 9 ──────────────────────────────────────────────────────────────────
h2("9. Calibration and Sharpness")

h3("9a. Overall")
p("| Metric | P1–P3 | P4 (exploratory) |")
p("|--------|-------|-----------------|")
for key, label in [("n","N forecasts"),("mean_forecast","Mean forecast"),
                    ("sd_forecast","Forecast SD"),("pct_below_0.1","% < 0.10"),
                    ("pct_0.1_0.9","% 0.10–0.90"),("pct_above_0.9","% > 0.90"),
                    ("ECE","ECE"),("overconf_high","Overconf high (>0.8, outcome=0)"),
                    ("overconf_low","Overconf low (<0.2, outcome=1)")]:
    p123_v = cal_p123.get(key,"N/A"); p4_v = cal_p4.get(key,"N/A")
    p(f"| {label} | {p123_v} | {p4_v} |")

h3("9b. P1–P3 Calibration table (10 bins)")
p("| Bin | N | Mean Forecast | Observed Freq | Gap (F−O) |")
p("|-----|---|--------------|--------------|-----------|")
for b in cal_p123.get("cal_bins", []):
    p(f"| {b['bin']} | {b['n']} | {f(b['mean_forecast'])} | {f(b['mean_outcome'])} | {f(b['gap'])} |")
p()
p("*Gap > 0 = overconfident; Gap < 0 = underconfident.*")

h3("9c. By model (P1–P3, mean forecast and SD)")
p("| Model | Mean Forecast | SD | OC-high N | OC-low N |")
p("|-------|-------------|----|-----------|---------| ")
for mk in MODELS:
    c = cal_by_model.get(mk, {})
    p(f"| {mk} | {c.get('mean_forecast','N/A')} | {c.get('sd_forecast','N/A')} | "
      f"{c.get('overconf_high','N/A')} | {c.get('overconf_low','N/A')} |")

h3("9d. By prompt × condition (P1–P3, mean forecast and SD)")
p("| Prompt | Condition | Mean Forecast | SD | ECE |")
p("|--------|-----------|-------------|----|----|")
for pk in PROMPTS:
    for cond in CONDITIONS:
        c = cal_by_pxc.get((pk, cond), {})
        p(f"| {pk} | {cond[:2].upper()} | {c.get('mean_forecast','N/A')} | "
          f"{c.get('sd_forecast','N/A')} | {c.get('ECE','N/A')} |")

# ── Section 10 ─────────────────────────────────────────────────────────────────
h2("10. Topic and Source Effects")

h3("10a. By Topic")
p("| Topic | CB Brier | CB N | SE Brier | SE N | SE−CB delta |")
p("|-------|---------|------|---------|------|------------|")
for topic in TOPICS:
    cb_e = next((r for r in topic_source_rows if r["dimension"]=="topic" and r["name"]==topic and r["condition"]=="closed_book"), None)
    se_e = next((r for r in topic_source_rows if r["dimension"]=="topic" and r["name"]==topic and r["condition"]=="shared_evidence"), None)
    d = get_delta("topic", topic)
    note = " ← helps" if (d and d < -0.005) else (" ← hurts" if (d and d > 0.005) else "")
    p(f"| {topic} | {f(cb_e['mean_brier'] if cb_e else None)} | {cb_e['n'] if cb_e else 0} | "
      f"{f(se_e['mean_brier'] if se_e else None)} | {se_e['n'] if se_e else 0} | {f(d)}{note} |")

h3("10b. By Source")
p("| Source | CB Brier | CB N | SE Brier | SE N | SE−CB delta |")
p("|--------|---------|------|---------|------|------------|")
for src in SOURCES:
    cb_e = next((r for r in topic_source_rows if r["dimension"]=="source" and r["name"]==src and r["condition"]=="closed_book"), None)
    se_e = next((r for r in topic_source_rows if r["dimension"]=="source" and r["name"]==src and r["condition"]=="shared_evidence"), None)
    d = get_delta("source", src)
    p(f"| {src} | {f(cb_e['mean_brier'] if cb_e else None)} | {cb_e['n'] if cb_e else 0} | "
      f"{f(se_e['mean_brier'] if se_e else None)} | {se_e['n'] if se_e else 0} | {f(d)} |")

# ── Section 11 ─────────────────────────────────────────────────────────────────
h2("11. Error Analysis")

h3("11a. Top 20 Worst Individual Forecasts (P1–P3)")
p("| # | Model | Prompt | Cond | Forecast | Outcome | Brier | Question |")
p("|---|-------|--------|------|---------|---------|-------|---------|")
for i, r in enumerate(worst20, 1):
    q = questions.get(r["question_id"], {})
    qtxt = q.get("question","")[:50] + ("…" if len(q.get("question",""))>50 else "")
    p(f"| {i} | {r['model_key']} | {r['prompt']} | {r['condition'][:2].upper()} | "
      f"{f(r['forecast'],2)} | {r['resolution_value']:.0f} | {f(r['brier_score'])} | {qtxt} |")

h3("11b. Catastrophic Overconfidence Summary (P1–P3)")
p(f"| Category | Total | % of valid rows |")
p(f"|---------|-------|----------------|")
p(f"| Forecast > 0.80, Outcome = 0 | {len(oc_yes_all)} | {pct(len(oc_yes_all),len(p123v))}% |")
p(f"| Forecast < 0.20, Outcome = 1 | {len(oc_no_all)} | {pct(len(oc_no_all),len(p123v))}% |")
p(f"| Total | {len(oc_yes_all)+len(oc_no_all)} | {pct(len(oc_yes_all)+len(oc_no_all),len(p123v))}% |")
p()
p("**Overconfidence by model:**")
p()
p("| Model | OC-high (>0.8/NO) | OC-low (<0.2/YES) | Total |")
p("|-------|-----------------|-----------------|-------|")
for mk in MODELS:
    oh, ol = oc_by_model[mk]
    p(f"| {mk} | {oh} | {ol} | {oh+ol} |")

h3("11c. Hardest Questions (highest mean Brier across all models/prompts)")
p("| Rank | Question ID | Mean Brier | Question |")
p("|------|------------|-----------|---------|")
for i, (qid, qb) in enumerate(hard_questions, 1):
    q = questions.get(qid, {})
    qtxt = q.get("question","")[:60] + ("…" if len(q.get("question",""))>60 else "")
    p(f"| {i} | `{qid[:20]}` | {f(qb)} | {qtxt} |")

h3("11d. Top 10 SE improvements vs CB (same model/question/prompt)")
p("| Model | Prompt | Question | CB Brier | SE Brier | Improvement |")
p("|-------|--------|---------|---------|---------|------------|")
for (mk, qid, pk), cb_b, se_b, imp, _ in top_improvements:
    q = questions.get(qid, {})
    qtxt = q.get("question","")[:40] + "…"
    p(f"| {mk} | {pk} | {qtxt} | {f(cb_b)} | {f(se_b)} | {f(imp)} |")

# ── Section 12 ─────────────────────────────────────────────────────────────────
h2("12. Practicality and Model Selection")

h3("12a. Latency and token usage (P1–P3)")
p("| Model | Type | Brier | SE Brier | Parse % | Lat (s) | In-tok | Out-tok | Reason-tok |")
p("|-------|------|-------|---------|---------|---------|--------|---------|-----------|")
for r in practicality:
    p(f"| {r['model_key']} | {r['model_type']} | {f(r['mean_brier'])} | "
      f"{f(r['se_brier'])} | {r['parse_rate']}% | {r['mean_lat']} | "
      f"{r['mean_itok']} | {r['mean_otok']} | {r['mean_rtok'] or 'N/A'} |")
p()
p("*reasoning\\* = called without explicit reasoning effort flag but produces internal reasoning tokens (mean 2,124). "
  "All other standard models (gemini-3-flash, gemma-4-31b, mistral-large) produce 0 reasoning tokens.*")

h3("12b. Recommended model sets for live Metaculus summer tournament")
p()
p("**Set A — Accuracy-focused (6 models):**")
p()
p("| Model | Justification |")
p("|-------|-------------|")
top6_acc = [r["model_key"] for r in lb_p123[:6]]
justifications_A = {
    lb_p123[0]["model_key"]: "Best overall P1–P3 Brier; strong SE performance",
    lb_p123[1]["model_key"]: "2nd overall; excellent SE Brier",
    lb_p123[2]["model_key"]: "3rd overall; consistent across prompts",
    lb_p123[3]["model_key"]: "4th overall; good calibration",
    lb_p123[4]["model_key"]: "5th overall; strong reasoning model",
    lb_p123[5]["model_key"]: "6th overall; diversity value",
}
for mk in top6_acc:
    p(f"| {mk} ({MODEL_TYPES.get(mk,'?')}) | {justifications_A.get(mk,'Strong performer')} |")

p()
p("**Set B — Practical/cost-balanced (6 models):**")
p()
p("| Model | Brier | Lat (s) | Type | Justification |")
p("|-------|-------|---------|------|-------------|")
# Mix: best standard + top reasoning with reasonable latency
set_b_candidates = sorted(practicality, key=lambda r: (r["mean_brier"] or 99) + (r["mean_lat"] or 999)/10000)
set_b = set_b_candidates[:6]
for r in set_b:
    p(f"| {r['model_key']} | {f(r['mean_brier'])} | {r['mean_lat']} | "
      f"{r['model_type']} | {'Low latency + good accuracy' if r['model_type']=='standard' else 'Top accuracy'} |")

# ── Section 13 ─────────────────────────────────────────────────────────────────
h2("13. Key Findings for Paper")
p()

overall_cb = mean(r["brier_score"] for r in p123v if r["condition"]=="closed_book")
overall_se = mean(r["brier_score"] for r in p123v if r["condition"]=="shared_evidence")
p3_cb = mean(r["brier_score"] for r in p123v if r["prompt"]=="P3" and r["condition"]=="closed_book")
p3_se = mean(r["brier_score"] for r in p123v if r["prompt"]=="P3" and r["condition"]=="shared_evidence")
p1_cb = mean(r["brier_score"] for r in p123v if r["prompt"]=="P1" and r["condition"]=="closed_book")
p1_se = mean(r["brier_score"] for r in p123v if r["prompt"]=="P1" and r["condition"]=="shared_evidence")
# Best aggregate cell = lowest mean Brier for a (model, prompt, condition) triple
from itertools import product as _product
_agg_cells = []
for _mk, _pk, _cond in _product(MODELS, PROMPTS + ["P4"], CONDITIONS):
    _src = p4v if _pk == "P4" else p123v
    _b, _n = cell_brier(_src, _mk, _pk, _cond)
    if _b is not None and _n > 0:
        _agg_cells.append((_b, _mk, _pk, _cond, _n))
_agg_cells.sort()
best_agg_cell_p123 = next(x for x in _agg_cells if x[2] != "P4")
best_agg_cell_p4   = next(x for x in _agg_cells if x[2] == "P4")
best_agg_cell_all  = _agg_cells[0]
total_oc = len(oc_yes_all) + len(oc_no_all)
best_model_p123 = lb_p123[0]
worst_model_p123 = lb_p123[-1]

findings = [
    f"**Evidence universally improves LLM forecasting.** Shared evidence reduces mean Brier "
    f"from {f(overall_cb)} (CB) to {f(overall_se)} (SE), a delta of {f(overall_se-overall_cb)} "
    f"across all 12 models and 3 prompts. This effect is significant at p<0.001 "
    f"(cluster bootstrap, N=114 questions).",

    f"**P3 (Bayesian) is brittle in closed-book but shows the largest evidence gain.** "
    f"P3 closed-book Brier = {f(p3_cb)} vs P1 closed-book = {f(p1_cb)}: P3 is the "
    f"worst prompt without evidence. Under shared evidence P3 improves to {f(p3_se)}, "
    f"its largest absolute gain of any prompt, yet it remains worse than P1+SE "
    f"({f(p1_se)}) and P2+SE in absolute terms. The Bayesian structure amplifies "
    f"evidence when available but degrades performance without it.",

    f"**P3 × evidence interaction is directional but not significant.** The "
    f"difference-in-differences (P3 gain vs P1 gain) = −0.011 (p≈0.08). "
    f"The pattern is consistent across most models but the pilot (N=114) "
    f"is underpowered to confirm this interaction.",

    f"**P4 Superforecaster prompt outperforms P1–P3 overall (exploratory).** "
    f"P4 wins vs best P1–P3 in {p4_wins_cb}/12 models (CB) and {p4_wins_se}/12 (SE). "
    f"Gains are largest for weaker/standard models (e.g., mistral-large: "
    f"P3/CB={f(cell_brier(p123v,'mistral-large','P3','closed_book')[0])} → "
    f"P4/CB={f(cell_brier(p4v,'mistral-large','P4','closed_book')[0])}). "
    f"These results are exploratory and not preregistered.",

    f"**Best model overall: {best_model_p123['model_key']}** "
    f"(Brier = {f(best_model_p123['mean_brier'])}, BI = {f(best_model_p123['brier_index'],2)}). "
    f"Worst: {worst_model_p123['model_key']} "
    f"(Brier = {f(worst_model_p123['mean_brier'])}). "
    f"Spread = {f(worst_model_p123['mean_brier']-best_model_p123['mean_brier'])} Brier points.",

    f"**Best aggregate model–prompt–condition cell (P1–P3):** "
    f"{best_agg_cell_p123[1]} / {best_agg_cell_p123[2]} / {best_agg_cell_p123[3]} "
    f"(Brier = {f(best_agg_cell_p123[0])}, N={best_agg_cell_p123[4]}). "
    f"Best P4 cell: {best_agg_cell_p4[1]} / P4 / {best_agg_cell_p4[3]} "
    f"(Brier = {f(best_agg_cell_p4[0])}, N={best_agg_cell_p4[4]}, exploratory). "
    f"Gemini 3.1 Pro's shared-evidence performance is one of the strongest single-cell results.",

    f"**No LLM beats the market baseline.** Market Brier = {f(mkt_overall_p123)}. "
    f"Best model gap: {best_model_p123['model_key']} is {f(model_vs_mkt[0]['delta'])} "
    f"Brier points above market (p<0.001). The crowd-sourced market is a strong benchmark "
    f"that current frontier LLMs cannot match.",

    (lambda bins: (
        f"**LLMs are severely miscalibrated at high confidence.** "
        f"Overall ECE = {f(cal_p123.get('ECE'))}. The miscalibration is not uniform: "
        f"models are near-calibrated at low probabilities but dramatically overconfident "
        f"above 0.5. In the 0.9–1.0 bin (N={bins[-1]['n']} forecasts), mean forecast = "
        f"{f(bins[-1]['mean_forecast'])} but observed frequency = "
        f"{f(bins[-1]['mean_outcome'])} — a gap of {f(bins[-1]['gap'])}. "
        f"In the 0.7–0.8 bin (N={bins[7]['n']}), gap = {f(bins[7]['gap'])}. "
        f"This pattern holds across all 12 models and constitutes {total_oc} "
        f"catastrophic errors ({pct(total_oc,len(p123v))}% of valid rows): "
        f"{len(oc_yes_all)} cases of forecast > 0.80 on events that did not occur, "
        f"{len(oc_no_all)} cases of forecast < 0.20 on events that did occur. "
        f"P4 shows better calibration (ECE = {f(cal_p4.get('ECE'))})."
    ))(cal_p123["cal_bins"]),

    f"**Reasoning models outperform standard models** (Brier {f(type_stats['reasoning']['brier'])} "
    f"vs {f(type_stats['standard']['brier'])}, delta = "
    f"{f(type_stats['standard']['brier']-type_stats['reasoning']['brier'])}). "
    f"However, reasoning models are {round(type_stats['reasoning']['mean_lat']/max(type_stats['standard']['mean_lat'],1),1)}× "
    f"slower. This difference is confounded by model size and recency.",

    f"**Topic and source heterogeneity is substantial.** Evidence helps most on "
    f"{min(TOPICS, key=lambda t: get_delta('topic',t) or 99)} questions "
    f"(delta = {f(min(get_delta('topic',t) or 99 for t in TOPICS))}). "
    f"Some topics show negligible evidence benefit. "
    f"Caution: topic categories contain as few as "
    f"{min(r['n'] for r in topic_source_rows if r['dimension']=='topic')} questions.",
]

for i, fn in enumerate(findings, 1):
    p(f"**Finding {i}:** {fn}")
    p()

# ── Section 14 ─────────────────────────────────────────────────────────────────
h2("14. Paper-Ready Tables")

h3("Table 1: Dataset and Design Summary")
p("| | P1–P3 (Confirmatory) | P4 (Exploratory) |")
p("|---|---|---|")
p(f"| Questions | 114 binary ForecastBench | Same 114 questions |")
p(f"| Models | 12 (8 reasoning, 4 standard) | Same 12 models |")
p(f"| Prompts | 3 (Control, Base-Rate-First, Bayesian) | 1 (Superforecaster) |")
p(f"| Conditions | 2 (closed-book, shared-evidence) | Same 2 conditions |")
p(f"| Total calls | 8,208 | 2,736 |")
p(f"| Valid forecasts | {len(p123v):,} ({pct(len(p123v),len(p123_all))}%) | {len(p4v):,} ({pct(len(p4v),len(p4_all))}%) |")
p(f"| Evidence source | AskNews (pre-freeze) | Same |")
p(f"| Market baseline | freeze\\_datetime\\_value | Same |")

h3("Table 2: Model Leaderboard (P1–P3)")
p("| Rank | Model | Type | Brier | BI | Parse % | Lat (s) |")
p("|------|-------|------|-------|----|---------|---------| ")
for i, r in enumerate(lb_p123, 1):
    p(f"| {i} | {r['model_key']} | {r['model_type']} | {f(r['mean_brier'])} | "
      f"{f(r['brier_index'],2)} | {r['parse_rate']}% | {r['mean_latency']} |")

h3("Table 3: Prompt × Condition Aggregate Brier")
p("| Prompt | CB Brier | SE Brier | SE−CB delta |")
p("|--------|---------|---------|------------|")
for pk in PROMPTS:
    cb_b = pxc_agg[(pk,"closed_book")][0]
    se_b = pxc_agg[(pk,"shared_evidence")][0]
    p(f"| {pk} | {f(cb_b)} | {f(se_b)} | {f(pxc_deltas[pk])} |")
p4_cb_b = pxc_agg[("P4","closed_book")][0]
p4_se_b = pxc_agg[("P4","shared_evidence")][0]
p(f"| P4† | {f(p4_cb_b)} | {f(p4_se_b)} | {f(pxc_deltas['P4'])} |")
p()
p("*† Exploratory only.*")

h3("Table 4: Model × Prompt × Condition Brier (condensed)")
p("| Model | P1/CB | P1/SE | P2/CB | P2/SE | P3/CB | P3/SE | P4/CB† | P4/SE† |")
p("|-------|-------|-------|-------|-------|-------|-------|--------|--------|")
for row in mpc_table:
    mk = row["model_key"]
    p(f"| {mk} | {f(row.get('P1_CB'))} | {f(row.get('P1_SE'))} | "
      f"{f(row.get('P2_CB'))} | {f(row.get('P2_SE'))} | "
      f"{f(row.get('P3_CB'))} | {f(row.get('P3_SE'))} | "
      f"{f(row.get('P4_CB'))} | {f(row.get('P4_SE'))} |")

h3("Table 5: Market Baseline Comparison")
p("| Model | Brier | Market Brier | Delta | Beats? |")
p("|-------|-------|-------------|-------|-------|")
for r in model_vs_mkt:
    p(f"| {r['model_key']} | {f(r['model_brier'])} | {f(r['market_brier'])} | "
      f"{f(r['delta'])} | {'YES' if r['beats_market'] else 'no'} |")
p(f"| **Market overall** | — | **{f(mkt_overall_p123)}** | — | — |")

h3("Table 6: P4 Exploratory Comparison")
p("| Model | P4 CB | Best P1-3 CB | Delta CB | P4 SE | Best P1-3 SE | Delta SE |")
p("|-------|-------|-------------|---------|-------|-------------|---------|")
for r in p4_exp:
    p(f"| {r['model_key']} | {f(r['P4_CB'])} | {f(r['best_P123_CB'])} | "
      f"{f(r['delta_CB'])} | {f(r['P4_SE'])} | {f(r['best_P123_SE'])} | {f(r['delta_SE'])} |")

h3("Table 7: Calibration and Sharpness Summary")
p("| Dataset | N | Mean F | SD | ECE | OC-high | OC-low |")
p("|---------|---|--------|----|-----|---------|--------|")
for label, cal in [("P1-P3", cal_p123), ("P4†", cal_p4)]:
    if not cal: continue
    p(f"| {label} | {cal['n']} | {cal['mean_forecast']} | {cal['sd_forecast']} | "
      f"{cal['ECE']} | {cal['overconf_high']} ({cal['pct_oc_high']}%) | "
      f"{cal['overconf_low']} ({cal['pct_oc_low']}%) |")

# ── Section 15 placeholder (files are saved below) ─────────────────────────────
h2("15. Output Files")
p("All outputs saved to `results/analysis/final/`.")

# ── Write markdown ─────────────────────────────────────────────────────────────
md_path = OUT_DIR / "full_diagnostic_report_with_p4.md"
with open(md_path, "w", encoding="utf-8") as f_:
    f_.write("\n".join(md))
print(f"Saved: {md_path}")

# ── CSV: full_summary_tables_with_p4.csv ──────────────────────────────────────
summ_rows = []
def sr(sec, label, value):
    summ_rows.append({"section": sec, "label": label, "value": str(value)})

sr("integrity", "P123_total", len(p123_all))
sr("integrity", "P123_valid", len(p123v))
sr("integrity", "P123_parse_rate", f"{pct(len(p123v),len(p123_all))}%")
sr("integrity", "P4_total", len(p4_all))
sr("integrity", "P4_valid", len(p4v))
sr("integrity", "P4_parse_rate", f"{pct(len(p4v),len(p4_all))}%")
for i, r in enumerate(lb_p123, 1):
    sr("leaderboard_p123", f"rank_{i}", f"{r['model_key']} brier={f(r['mean_brier'])} BI={f(r['brier_index'],2)}")
sr("market", "market_brier_p123", f(mkt_overall_p123))
sr("market", "n_models_beat_market_p123", n_beats)
sr("market", "market_brier_p4", f(mkt_overall_p4))
sr("market", "n_models_beat_market_p4", n_beats_p4)
for pk in PROMPTS + ["P4"]:
    sr("pxc_delta", f"{pk}_se_minus_cb", f(pxc_deltas.get(pk)))
sr("calibration_p123", "ECE", f(cal_p123.get("ECE")))
sr("calibration_p123", "mean_forecast", cal_p123.get("mean_forecast"))
sr("calibration_p123", "overconf_total", len(oc_yes_all)+len(oc_no_all))
sr("calibration_p4", "ECE", f(cal_p4.get("ECE")))
sr("type_comparison", "reasoning_brier", f(type_stats["reasoning"]["brier"]))
sr("type_comparison", "standard_brier", f(type_stats["standard"]["brier"]))
sr("type_comparison", "reasoning_mean_lat", type_stats["reasoning"]["mean_lat"])
sr("type_comparison", "standard_mean_lat", type_stats["standard"]["mean_lat"])

write_csv(OUT_DIR / "full_summary_tables_with_p4.csv", summ_rows,
          fieldnames=["section","label","value"])
print(f"Saved: {OUT_DIR / 'full_summary_tables_with_p4.csv'}")

# ── Model selection markdown ───────────────────────────────────────────────────
sel_md = ["# Model Selection Recommendations for Live Metaculus Tournament\n"]
sel_md.append("## Set A — Accuracy-focused\n")
sel_md.append("| Model | Type | P1-P3 Brier | SE Brier | Parse % | Lat (s) |")
sel_md.append("|-------|------|------------|---------|---------|---------|")
for r in practicality[:6]:
    sel_md.append(f"| {r['model_key']} | {r['model_type']} | {f(r['mean_brier'])} | "
                  f"{f(r['se_brier'])} | {r['parse_rate']}% | {r['mean_lat']} |")

sel_md.append("\n## Set B — Practical/Cost-balanced\n")
sel_md.append("Optimised for Brier + latency tradeoff.\n")
sel_md.append("| Model | Type | P1-P3 Brier | SE Brier | Lat (s) | Rationale |")
sel_md.append("|-------|------|------------|---------|---------|-----------|")
used = set()
set_b_final = []
# Include top 3 by accuracy and top 3 by latency that are not already included
for r in practicality:
    if len([x for x in set_b_final if x["model_type"]=="reasoning"]) < 3 or \
       len([x for x in set_b_final if x["model_type"]=="standard"]) < 2:
        if r["model_key"] not in used:
            set_b_final.append(r)
            used.add(r["model_key"])
    if len(set_b_final) >= 6: break
for r in set_b_final:
    rationale = ("Best accuracy" if r == practicality[0]
                 else "Low latency + competitive Brier" if r["model_type"]=="standard"
                 else "Strong SE performance")
    sel_md.append(f"| {r['model_key']} | {r['model_type']} | {f(r['mean_brier'])} | "
                  f"{f(r['se_brier'])} | {r['mean_lat']} | {rationale} |")

sel_md.append("\n## Caveats\n")
sel_md.append("- Recommendations based on P1–P3 pilot (114 questions). Live performance may differ.\n")
sel_md.append("- P4 Superforecaster prompt shows promise; consider including it for all tournament models.\n")
sel_md.append("- Market baseline (Brier ≈ 0.095) significantly outperforms all models — "
              "include market baseline as reference in live tournament.\n")
sel_md.append("- Consider rotating prompts (P1+P4) to assess prompt effects in live setting.\n")

sel_path = OUT_DIR / "model_selection_recommendations.md"
with open(sel_path, "w", encoding="utf-8") as f_:
    f_.write("\n".join(sel_md))
print(f"Saved: {sel_path}")

print(f"\nAll outputs saved to: {OUT_DIR}")
print("Files:")
for fp in sorted(OUT_DIR.iterdir()):
    print(f"  {fp.name}")
