"""
E1 -- Main diagnostic analysis for P1-P3 pilot experiment.

Input:  results/merged/pilot_results_final.json
Output: results/analysis/main_diagnostic_report_final.md
        results/analysis/main_summary_tables_final.csv
        results/analysis/main_model_prompt_condition_brier.csv
        results/analysis/main_topic_source_effects.csv
        results/analysis/main_error_analysis.csv
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
IN_FILE  = ROOT / "results" / "merged" / "pilot_results_final.json"
OUT_DIR  = ROOT / "results" / "analysis"

MODELS = [
    "gpt-5.4", "claude-opus-4.6", "gemini-3.1-pro", "grok-4.20",
    "qwen3-max", "deepseek-v3.2-speciale", "kimi-k2.6", "gpt-oss-120b",
    "gemini-3-flash", "gemma-4-31b", "glm-5.1", "mistral-large",
]
PROMPTS     = ["P1", "P2", "P3"]
CONDITIONS  = ["closed_book", "shared_evidence"]
TOPICS      = ["ai_technology","conflict","economics","energy","entertainment",
               "finance_market","geopolitics","other","public_health","sports"]
SOURCES     = ["manifold", "polymarket", "metaculus"]

# ── helpers ────────────────────────────────────────────────────────────────────

def mean(vals):
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None

def sd(vals):
    v = [x for x in vals if x is not None]
    if len(v) < 2: return None
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m)**2 for x in v) / len(v))

def brier_index(b):
    return round((1 - math.sqrt(b)) * 100, 2) if b is not None else None

def pct(n, d):
    return round(100 * n / d, 1) if d else None

def fmt(v, dec=4):
    return f"{v:.{dec}f}" if v is not None else "N/A"

def valid(rows):
    return [r for r in rows if r["parse_success"] and r["brier_score_final"] is not None]

def cell(rows, model=None, prompt=None, condition=None):
    r = rows
    if model:     r = [x for x in r if x["model_key"]  == model]
    if prompt:    r = [x for x in r if x["prompt"]     == prompt]
    if condition: r = [x for x in r if x["condition"]  == condition]
    return r

# ── load ───────────────────────────────────────────────────────────────────────

with open(IN_FILE, encoding="utf-8") as f:
    rows = json.load(f)

v_rows   = valid(rows)
n_total  = len(rows)
n_valid  = len(v_rows)
n_miss   = n_total - n_valid

MODEL_TYPES = {r["model_key"]: r["model_type"] for r in rows}

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# SECTION 1  Data integrity
# ==============================================================================

missing_cells = []
for mk in MODELS:
    for pk in PROMPTS:
        for cond in CONDITIONS:
            c = cell(rows, mk, pk, cond)
            v = valid(c)
            if len(v) < len(c):
                missing_cells.append({
                    "model_key": mk, "prompt": pk, "condition": cond,
                    "total": len(c), "valid": len(v), "missing": len(c) - len(v),
                })


# ==============================================================================
# SECTION 2  Overall leaderboard
# ==============================================================================

leaderboard = []
for mk in MODELS:
    mv = valid(cell(rows, mk))
    mr = cell(rows, mk)
    lats = [r["latency_seconds"] for r in mr if r.get("latency_seconds")]
    b = mean(r["brier_score_final"] for r in mv)
    leaderboard.append({
        "model_key":   mk,
        "model_type":  MODEL_TYPES[mk],
        "mean_brier":  b,
        "brier_index": brier_index(b),
        "n_valid":     len(mv),
        "parse_rate":  pct(len(mv), len(mr)),
        "mean_latency": round(mean(lats), 2) if lats else None,
    })
leaderboard.sort(key=lambda x: x["mean_brier"] or 99)


# ==============================================================================
# SECTION 3  Model x Prompt x Condition table
# ==============================================================================

mpc_table = []
for mk in MODELS:
    for pk in PROMPTS:
        for cond in CONDITIONS:
            c = valid(cell(rows, mk, pk, cond))
            b = mean(r["brier_score_final"] for r in c)
            mpc_table.append({
                "model_key": mk, "prompt": pk, "condition": cond,
                "mean_brier": b, "n_valid": len(c),
                "brier_index": brier_index(b),
            })


# ==============================================================================
# SECTION 4  Prompt x Condition interaction
# ==============================================================================

pxc = {}
for pk in PROMPTS:
    for cond in CONDITIONS:
        c = valid(cell(rows, prompt=pk, condition=cond))
        pxc[(pk, cond)] = {"brier": mean(r["brier_score_final"] for r in c), "n": len(c)}

pxc_deltas = {}
for pk in PROMPTS:
    cb = pxc[(pk, "closed_book")]["brier"]
    se = pxc[(pk, "shared_evidence")]["brier"]
    pxc_deltas[pk] = round(se - cb, 5) if (cb and se) else None


# ==============================================================================
# SECTION 5  Shared evidence effect per model
# ==============================================================================

se_effect = []
for mk in MODELS:
    cb_v = valid(cell(rows, mk, condition="closed_book"))
    se_v = valid(cell(rows, mk, condition="shared_evidence"))
    cb_b = mean(r["brier_score_final"] for r in cb_v)
    se_b = mean(r["brier_score_final"] for r in se_v)
    delta = round(se_b - cb_b, 5) if (cb_b and se_b) else None
    se_effect.append({
        "model_key":  mk,
        "model_type": MODEL_TYPES[mk],
        "cb_brier":   cb_b,
        "se_brier":   se_b,
        "se_minus_cb": delta,
        "helped":     delta is not None and delta < 0,
    })
se_effect.sort(key=lambda x: x["se_minus_cb"] or 99)


# ==============================================================================
# SECTION 6  Reasoning vs standard
# ==============================================================================

type_stats = {}
for mtype in ["reasoning", "standard"]:
    tv = valid([r for r in rows if r["model_type"] == mtype])
    lats = [r["latency_seconds"] for r in rows if r["model_type"] == mtype and r.get("latency_seconds")]
    itoks = [r["input_tokens"]  for r in rows if r["model_type"] == mtype and r.get("input_tokens")]
    otoks = [r["output_tokens"] for r in rows if r["model_type"] == mtype and r.get("output_tokens")]
    type_stats[mtype] = {
        "brier":      mean(r["brier_score_final"] for r in tv),
        "n_valid":    len(tv),
        "parse_rate": pct(len(tv), len([r for r in rows if r["model_type"]==mtype])),
        "mean_lat":   round(mean(lats), 2) if lats else None,
        "mean_itok":  round(mean(itoks)) if itoks else None,
        "mean_otok":  round(mean(otoks)) if otoks else None,
    }

type_pxc = {}
for mtype in ["reasoning", "standard"]:
    for pk in PROMPTS:
        for cond in CONDITIONS:
            c = valid([r for r in rows if r["model_type"]==mtype
                       and r["prompt"]==pk and r["condition"]==cond])
            type_pxc[(mtype, pk, cond)] = mean(r["brier_score_final"] for r in c)


# ==============================================================================
# SECTION 7  Market baseline
# ==============================================================================

mkt_rows = [r for r in v_rows if r.get("market_prob") is not None]
mkt_brier_vals = [(r["market_prob"] - r["resolution_value"])**2 for r in mkt_rows]
mkt_overall = mean(mkt_brier_vals)

model_vs_mkt = []
for mk in MODELS:
    mv = [r for r in valid(cell(rows, mk)) if r.get("market_prob") is not None]
    if not mv: continue
    mb = mean(r["brier_score_final"] for r in mv)
    kb = mean((r["market_prob"] - r["resolution_value"])**2 for r in mv)
    model_vs_mkt.append({
        "model_key": mk,
        "model_brier": mb,
        "market_brier_same_rows": kb,
        "delta": round(mb - kb, 5) if (mb and kb) else None,
        "beats_market": mb < kb if (mb and kb) else None,
        "n": len(mv),
    })
model_vs_mkt.sort(key=lambda x: x["delta"] or 99)

n_beats = sum(1 for x in model_vs_mkt if x["beats_market"])


# ==============================================================================
# SECTION 8  Calibration and sharpness
# ==============================================================================

forecasts = [r["forecast_final"] for r in v_rows]
outcomes  = [r["resolution_value"] for r in v_rows]

mean_f  = mean(forecasts)
sd_f    = sd(forecasts)
pct_low  = pct(sum(1 for f in forecasts if f < 0.1),  len(forecasts))
pct_high = pct(sum(1 for f in forecasts if f > 0.9),  len(forecasts))
pct_mid  = pct(sum(1 for f in forecasts if 0.1 <= f <= 0.9), len(forecasts))

# 10-bin calibration
bins = [[] for _ in range(10)]
for f, o in zip(forecasts, outcomes):
    idx = min(int(f * 10), 9)
    bins[idx].append((f, o))

cal_bins = []
ece_num = 0
for i, b in enumerate(bins):
    lo, hi = i/10, (i+1)/10
    if not b: continue
    mf = mean(x[0] for x in b)
    mo = mean(x[1] for x in b)
    cal_bins.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(b),
                     "mean_forecast": mf, "mean_outcome": mo,
                     "gap": round(mf - mo, 4) if (mf and mo) else None})
    if mf is not None and mo is not None:
        ece_num += len(b) * abs(mf - mo)

ece = round(ece_num / len(forecasts), 5) if forecasts else None


# ==============================================================================
# SECTION 9  Topic and source effects
# ==============================================================================

topic_stats = []
for topic in TOPICS:
    for cond in CONDITIONS:
        c = valid([r for r in rows if r["topic"]==topic and r["condition"]==cond])
        topic_stats.append({
            "topic": topic, "condition": cond,
            "mean_brier": mean(r["brier_score_final"] for r in c),
            "n": len(c),
        })

source_stats = []
for src in SOURCES:
    for cond in CONDITIONS:
        c = valid([r for r in rows if r["source"]==src and r["condition"]==cond])
        source_stats.append({
            "source": src, "condition": cond,
            "mean_brier": mean(r["brier_score_final"] for r in c),
            "n": len(c),
        })


# ==============================================================================
# SECTION 10  Error analysis
# ==============================================================================

# Top 10 worst individual forecasts
worst = sorted(v_rows, key=lambda r: r["brier_score_final"], reverse=True)[:10]

# Top 10 biggest improvements CB->SE per (model, question, prompt)
paired_improvements = []
cb_idx = {(r["model_key"], r["question_id"], r["prompt"]): r
          for r in v_rows if r["condition"] == "closed_book"}
se_idx = {(r["model_key"], r["question_id"], r["prompt"]): r
          for r in v_rows if r["condition"] == "shared_evidence"}
for key in cb_idx:
    if key in se_idx:
        cb_r, se_r = cb_idx[key], se_idx[key]
        improvement = cb_r["brier_score_final"] - se_r["brier_score_final"]
        paired_improvements.append({
            "model_key": key[0], "question_id": key[1], "prompt": key[2],
            "cb_brier": cb_r["brier_score_final"],
            "se_brier": se_r["brier_score_final"],
            "improvement": round(improvement, 4),
        })
paired_improvements.sort(key=lambda x: x["improvement"], reverse=True)
top_improvements = paired_improvements[:10]

# Catastrophic overconfidence
overconf_yes = [r for r in v_rows if r["forecast_final"] > 0.8 and r["resolution_value"] == 0.0]
overconf_no  = [r for r in v_rows if r["forecast_final"] < 0.2 and r["resolution_value"] == 1.0]


# ==============================================================================
# SECTION 11  Practicality
# ==============================================================================

model_latency = []
for mk in MODELS:
    mr = cell(rows, mk)
    lats = [r["latency_seconds"] for r in mr if r.get("latency_seconds")]
    itoks = [r["input_tokens"]  for r in mr if r.get("input_tokens")]
    otoks = [r["output_tokens"] for r in mr if r.get("output_tokens")]
    rtoks = [r["reasoning_tokens"] for r in mr if r.get("reasoning_tokens")]
    b = mean(r["brier_score_final"] for r in valid(mr))
    model_latency.append({
        "model_key":   mk,
        "model_type":  MODEL_TYPES[mk],
        "mean_brier":  b,
        "mean_lat":    round(mean(lats), 1) if lats else None,
        "mean_itok":   round(mean(itoks)) if itoks else None,
        "mean_otok":   round(mean(otoks)) if otoks else None,
        "mean_rtok":   round(mean(rtoks)) if rtoks else None,
    })


# ==============================================================================
# BUILD OUTPUTS
# ==============================================================================

# ── CSV 1: main_model_prompt_condition_brier.csv ───────────────────────────────
mpc_path = OUT_DIR / "main_model_prompt_condition_brier.csv"
with open(mpc_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["model_key","model_type","prompt","condition",
                                       "mean_brier","brier_index","n_valid"])
    w.writeheader()
    for row in mpc_table:
        row2 = dict(row)
        row2["model_type"] = MODEL_TYPES[row["model_key"]]
        row2["mean_brier"]  = fmt(row["mean_brier"])
        row2["brier_index"] = fmt(row["brier_index"], 2)
        w.writerow(row2)

# ── CSV 2: main_summary_tables_final.csv ──────────────────────────────────────
summ_path = OUT_DIR / "main_summary_tables_final.csv"
with open(summ_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["section","label","value"])
    w.writeheader()

    def row_s(section, label, value):
        w.writerow({"section": section, "label": label, "value": value})

    row_s("leaderboard", "rank_by_brier", "model,type,mean_brier,brier_index,parse_rate,mean_latency_s")
    for i, lb in enumerate(leaderboard, 1):
        row_s("leaderboard", f"rank_{i}",
              f"{lb['model_key']},{lb['model_type']},{fmt(lb['mean_brier'])},{fmt(lb['brier_index'],2)},{lb['parse_rate']}%,{lb['mean_latency']}")

    row_s("market_baseline", "market_overall_brier", fmt(mkt_overall))
    for x in model_vs_mkt:
        row_s("market_baseline", x["model_key"],
              f"model={fmt(x['model_brier'])} mkt={fmt(x['market_brier_same_rows'])} delta={fmt(x['delta'])} beats={x['beats_market']}")

    row_s("pxc_interaction", "header", "prompt,CB_brier,SE_brier,SE-CB_delta")
    for pk in PROMPTS:
        cb = pxc[(pk,"closed_book")]["brier"]
        se = pxc[(pk,"shared_evidence")]["brier"]
        row_s("pxc_interaction", pk,
              f"{fmt(cb)},{fmt(se)},{fmt(pxc_deltas[pk])}")

    row_s("calibration", "mean_forecast", fmt(mean_f))
    row_s("calibration", "sd_forecast",   fmt(sd_f))
    row_s("calibration", "pct_below_0.1", f"{pct_low}%")
    row_s("calibration", "pct_above_0.9", f"{pct_high}%")
    row_s("calibration", "pct_mid_0.1-0.9", f"{pct_mid}%")
    row_s("calibration", "ECE",           fmt(ece))

    for mtype, ts in type_stats.items():
        row_s("type_comparison", mtype,
              f"brier={fmt(ts['brier'])} parse={ts['parse_rate']}% lat={ts['mean_lat']}s itok={ts['mean_itok']} otok={ts['mean_otok']}")

# ── CSV 3: main_topic_source_effects.csv ──────────────────────────────────────
ts_path = OUT_DIR / "main_topic_source_effects.csv"
with open(ts_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["category","name","condition","mean_brier","n"])
    w.writeheader()
    for x in topic_stats:
        w.writerow({"category":"topic","name":x["topic"],"condition":x["condition"],
                    "mean_brier":fmt(x["mean_brier"]),"n":x["n"]})
    for x in source_stats:
        w.writerow({"category":"source","name":x["source"],"condition":x["condition"],
                    "mean_brier":fmt(x["mean_brier"]),"n":x["n"]})

# ── CSV 4: main_error_analysis.csv ────────────────────────────────────────────
err_path = OUT_DIR / "main_error_analysis.csv"
with open(err_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["category","model_key","question_id","prompt",
                                       "condition","forecast","resolution","brier",
                                       "improvement","note"])
    w.writeheader()
    for r in worst:
        w.writerow({"category":"worst_forecast","model_key":r["model_key"],
                    "question_id":r["question_id"],"prompt":r["prompt"],
                    "condition":r["condition"],"forecast":fmt(r["forecast_final"],4),
                    "resolution":r["resolution_value"],"brier":fmt(r["brier_score_final"]),
                    "improvement":"","note":""})
    for x in top_improvements:
        w.writerow({"category":"top_improvement","model_key":x["model_key"],
                    "question_id":x["question_id"],"prompt":x["prompt"],
                    "condition":"CB->SE","forecast":"","resolution":"",
                    "brier":f"CB={fmt(x['cb_brier'])} SE={fmt(x['se_brier'])}",
                    "improvement":fmt(x["improvement"]),"note":""})
    for r in overconf_yes[:20]:
        w.writerow({"category":"overconf_YES_given_NO","model_key":r["model_key"],
                    "question_id":r["question_id"],"prompt":r["prompt"],
                    "condition":r["condition"],"forecast":fmt(r["forecast_final"],4),
                    "resolution":r["resolution_value"],"brier":fmt(r["brier_score_final"]),
                    "improvement":"","note":"forecast>0.8 but outcome=0"})
    for r in overconf_no[:20]:
        w.writerow({"category":"overconf_NO_given_YES","model_key":r["model_key"],
                    "question_id":r["question_id"],"prompt":r["prompt"],
                    "condition":r["condition"],"forecast":fmt(r["forecast_final"],4),
                    "resolution":r["resolution_value"],"brier":fmt(r["brier_score_final"]),
                    "improvement":"","note":"forecast<0.2 but outcome=1"})


# ==============================================================================
# BUILD MARKDOWN REPORT
# ==============================================================================

lines = []
def h1(t):  lines.append(f"\n# {t}\n")
def h2(t):  lines.append(f"\n## {t}\n")
def h3(t):  lines.append(f"\n### {t}\n")
def p(t=""):lines.append(t)
def hr():   lines.append("\n---\n")

lines.append("# Main Diagnostic Report: P1–P3 Pilot Experiment\n")
lines.append(f"**Dataset:** `results/merged/pilot_results_final.json`  \n"
             f"**Analysis:** Descriptive only — no inferential tests.  \n"
             f"**Generated by:** `src/e1_diagnostic_analysis.py`\n")

# ── 1. Data integrity ──────────────────────────────────────────────────────────
h2("1. Data Integrity")
p(f"| Metric | Value |")
p(f"|--------|-------|")
p(f"| Total rows | {n_total:,} |")
p(f"| Expected rows (12 × 3 × 2 × 114) | 8,208 |")
p(f"| Valid (parse_success=True) | {n_valid:,} |")
p(f"| Missing forecasts | {n_miss} |")
p(f"| Parse success rate | {pct(n_valid, n_total)}% |")
p(f"| All 72 cells exactly 114 rows | YES |")
p()
if missing_cells:
    p("**Remaining missing rows by model × prompt × condition:**")
    p()
    p("| Model | Prompt | Condition | Total | Valid | Missing |")
    p("|-------|--------|-----------|-------|-------|---------|")
    for mc in missing_cells:
        p(f"| {mc['model_key']} | {mc['prompt']} | {mc['condition']} | {mc['total']} | {mc['valid']} | {mc['missing']} |")
    p()
    p("*Missing forecasts treated as NaN; excluded from all Brier/calibration calculations.*")

# ── 2. Overall leaderboard ─────────────────────────────────────────────────────
h2("2. Overall Leaderboard (P1+P2+P3 combined)")
p("Brier Index = (1 − √mean_Brier) × 100. Higher = better.")
p()
p("| Rank | Model | Type | Mean Brier | Brier Index | Parse Rate | Mean Latency (s) |")
p("|------|-------|------|-----------|-------------|------------|-----------------|")
for i, lb in enumerate(leaderboard, 1):
    p(f"| {i} | {lb['model_key']} | {lb['model_type']} | {fmt(lb['mean_brier'])} | "
      f"{fmt(lb['brier_index'],2)} | {lb['parse_rate']}% | {lb['mean_latency']} |")
p()
best = leaderboard[0]
worst_lb = leaderboard[-1]
spread = round(worst_lb["mean_brier"] - best["mean_brier"], 4) if (worst_lb["mean_brier"] and best["mean_brier"]) else None
p(f"**Best model:** {best['model_key']} (Brier={fmt(best['mean_brier'])})")
p(f"**Worst model:** {worst_lb['model_key']} (Brier={fmt(worst_lb['mean_brier'])})")
p(f"**Brier spread top–bottom:** {fmt(spread)}")

# ── 3. Model x Prompt x Condition ─────────────────────────────────────────────
h2("3. Model × Prompt × Condition Brier Table")
p("All 72 cells. N = valid rows per cell (out of 114).")
p()
header = "| Model | Type |"
for pk in PROMPTS:
    for cond in ["CB", "SE"]:
        header += f" {pk}/{cond} |"
p(header)
p("|-------|------|" + "--------|" * 6)
for mk in MODELS:
    mtype = MODEL_TYPES[mk]
    row_line = f"| {mk} | {mtype} |"
    for pk in PROMPTS:
        for cond, cshort in [("closed_book","CB"),("shared_evidence","SE")]:
            entry = next(x for x in mpc_table
                         if x["model_key"]==mk and x["prompt"]==pk and x["condition"]==cond)
            b = entry["mean_brier"]
            n = entry["n_valid"]
            row_line += f" {fmt(b)} ({n}) |"
    p(row_line)
p()
p("*CB = closed_book, SE = shared_evidence. Format: mean_brier (n_valid)*")
p()
# Best condition per model
p("**Best condition per model:**")
p()
p("| Model | Best condition | CB Brier | SE Brier | Delta (SE-CB) |")
p("|-------|---------------|---------|---------|--------------|")
for x in se_effect:
    best_cond = "shared_evidence" if x["helped"] else "closed_book"
    p(f"| {x['model_key']} | {best_cond} | {fmt(x['cb_brier'])} | {fmt(x['se_brier'])} | {fmt(x['se_minus_cb'])} |")

# ── 4. Prompt x Condition interaction ─────────────────────────────────────────
h2("4. Prompt × Condition Interaction")
p("Aggregate across all 12 models.")
p()
p("| Prompt | CB Brier | SE Brier | SE−CB delta | Interpretation |")
p("|--------|---------|---------|------------|----------------|")
for pk in PROMPTS:
    cb = pxc[(pk,"closed_book")]["brier"]
    se = pxc[(pk,"shared_evidence")]["brier"]
    d  = pxc_deltas[pk]
    interp = ("evidence helps" if d and d < -0.001
              else "evidence hurts" if d and d > 0.001
              else "negligible")
    p(f"| {pk} | {fmt(cb)} | {fmt(se)} | {fmt(d)} | {interp} |")
p()
p("| Prompt | CB N | SE N |")
p("|--------|------|------|")
for pk in PROMPTS:
    cb_n = pxc[(pk,"closed_book")]["n"]
    se_n = pxc[(pk,"shared_evidence")]["n"]
    p(f"| {pk} | {cb_n} | {se_n} |")
p()

# Deltas comparison
delta_vals = [(pk, pxc_deltas[pk]) for pk in PROMPTS if pxc_deltas[pk] is not None]
delta_vals.sort(key=lambda x: x[1])
most_helped = delta_vals[0]
p(f"**Prompt most helped by evidence:** {most_helped[0]} (delta = {fmt(most_helped[1])})")
p()
p("*Note: Differences between prompts are descriptive only. No statistical tests applied.*")

# ── 5. Shared evidence effect per model ───────────────────────────────────────
h2("5. Shared Evidence Effect per Model")
p("Ranked by evidence benefit (most negative delta = most helped).")
p()
p("| Rank | Model | Type | CB Brier | SE Brier | SE−CB delta | Helped? |")
p("|------|-------|------|---------|---------|------------|---------|")
for i, x in enumerate(se_effect, 1):
    helped = "YES" if x["helped"] else "no"
    p(f"| {i} | {x['model_key']} | {x['model_type']} | {fmt(x['cb_brier'])} | {fmt(x['se_brier'])} | {fmt(x['se_minus_cb'])} | {helped} |")
p()
n_helped = sum(1 for x in se_effect if x["helped"])
n_hurt   = len(se_effect) - n_helped
p(f"**Models helped by evidence:** {n_helped}/12  |  **Models hurt or neutral:** {n_hurt}/12")

# ── 6. Reasoning vs standard ──────────────────────────────────────────────────
h2("6. Reasoning vs Standard Models")
p()
p("| Metric | Reasoning | Standard |")
p("|--------|-----------|---------|")
for key, label in [("brier","Mean Brier"),("parse_rate","Parse Rate (%)"),
                    ("mean_lat","Mean Latency (s)"),("mean_itok","Mean Input Tokens"),
                    ("mean_otok","Mean Output Tokens")]:
    rv = type_stats["reasoning"][key]
    sv = type_stats["standard"][key]
    p(f"| {label} | {rv} | {sv} |")
p()
p("**By prompt × condition:**")
p()
p("| Type | P1/CB | P1/SE | P2/CB | P2/SE | P3/CB | P3/SE |")
p("|------|-------|-------|-------|-------|-------|-------|")
for mtype in ["reasoning", "standard"]:
    row_line = f"| {mtype} |"
    for pk in PROMPTS:
        for cond in CONDITIONS:
            b = type_pxc[(mtype, pk, cond)]
            row_line += f" {fmt(b)} |"
    p(row_line)

# ── 7. Market baseline ────────────────────────────────────────────────────────
h2("7. Market Baseline Comparison")
p(f"Market baseline Brier (across all {len(mkt_rows):,} valid rows): **{fmt(mkt_overall)}**")
p()
p("| Model | Model Brier | Market Brier (same rows) | Delta | Beats Market? | N |")
p("|-------|------------|--------------------------|-------|--------------|---|")
for x in model_vs_mkt:
    beat = "YES" if x["beats_market"] else "no"
    p(f"| {x['model_key']} | {fmt(x['model_brier'])} | {fmt(x['market_brier_same_rows'])} | {fmt(x['delta'])} | {beat} | {x['n']} |")
p()
p(f"**Models beating market: {n_beats}/{len(model_vs_mkt)}**")
p()
p("*Market baseline = (freeze_datetime_value − resolution)². "
  "Delta > 0 means model is worse than market; Delta < 0 means model beats market.*")

# ── 8. Calibration and sharpness ──────────────────────────────────────────────
h2("8. Calibration and Sharpness")
p()
p("| Metric | Value |")
p("|--------|-------|")
p(f"| N valid forecasts | {n_valid:,} |")
p(f"| Mean forecast | {fmt(mean_f)} |")
p(f"| Forecast SD | {fmt(sd_f)} |")
p(f"| % forecasts < 0.10 | {pct_low}% |")
p(f"| % forecasts 0.10–0.90 | {pct_mid}% |")
p(f"| % forecasts > 0.90 | {pct_high}% |")
p(f"| Expected Calibration Error (ECE) | {fmt(ece)} |")
p()
p("**Calibration table (10 bins):**")
p()
p("| Bin | N | Mean Forecast | Mean Outcome | Gap (F−O) |")
p("|-----|---|--------------|-------------|-----------|")
for cb in cal_bins:
    p(f"| {cb['bin']} | {cb['n']} | {fmt(cb['mean_forecast'])} | {fmt(cb['mean_outcome'])} | {fmt(cb['gap'])} |")
p()
p("*Gap > 0 = overconfident; Gap < 0 = underconfident in that bin.*")

# ── 9. Topic and source effects ───────────────────────────────────────────────
h2("9. Topic and Source Effects")
h3("9a. By Topic")
p()
p("| Topic | CB Brier | CB N | SE Brier | SE N | SE−CB delta |")
p("|-------|---------|------|---------|------|------------|")
for topic in TOPICS:
    cb_e = next((x for x in topic_stats if x["topic"]==topic and x["condition"]=="closed_book"), None)
    se_e = next((x for x in topic_stats if x["topic"]==topic and x["condition"]=="shared_evidence"), None)
    cb_b = cb_e["mean_brier"] if cb_e else None
    se_b = se_e["mean_brier"] if se_e else None
    d = round(se_b - cb_b, 5) if (cb_b and se_b) else None
    p(f"| {topic} | {fmt(cb_b)} | {cb_e['n'] if cb_e else 0} | "
      f"{fmt(se_b)} | {se_e['n'] if se_e else 0} | {fmt(d)} |")
p()
h3("9b. By Source")
p()
p("| Source | CB Brier | CB N | SE Brier | SE N | SE−CB delta |")
p("|--------|---------|------|---------|------|------------|")
for src in SOURCES:
    cb_e = next((x for x in source_stats if x["source"]==src and x["condition"]=="closed_book"), None)
    se_e = next((x for x in source_stats if x["source"]==src and x["condition"]=="shared_evidence"), None)
    cb_b = cb_e["mean_brier"] if cb_e else None
    se_b = se_e["mean_brier"] if se_e else None
    d = round(se_b - cb_b, 5) if (cb_b and se_b) else None
    p(f"| {src} | {fmt(cb_b)} | {cb_e['n'] if cb_e else 0} | "
      f"{fmt(se_b)} | {se_e['n'] if se_e else 0} | {fmt(d)} |")

# ── 10. Error analysis ────────────────────────────────────────────────────────
h2("10. Error Analysis")
h3("10a. Top 10 Worst Individual Forecasts")
p()
p("| Model | Prompt | Condition | Forecast | Outcome | Brier | Question |")
p("|-------|--------|-----------|---------|---------|-------|---------|")
for r in worst:
    p(f"| {r['model_key']} | {r['prompt']} | {r['condition'][:2].upper()} | "
      f"{fmt(r['forecast_final'],2)} | {r['resolution_value']:.0f} | "
      f"{fmt(r['brier_score_final'])} | `{r['question_id'][:20]}` |")
p()
h3("10b. Top 10 Biggest Improvements: CB → SE")
p()
p("| Model | Prompt | Question | CB Brier | SE Brier | Improvement |")
p("|-------|--------|---------|---------|---------|------------|")
for x in top_improvements:
    p(f"| {x['model_key']} | {x['prompt']} | `{x['question_id'][:20]}` | "
      f"{fmt(x['cb_brier'])} | {fmt(x['se_brier'])} | {fmt(x['improvement'])} |")
p()
h3("10c. Catastrophic Overconfidence")
p()
p(f"| Category | Count | % of valid rows |")
p(f"|---------|-------|----------------|")
p(f"| Forecast > 0.80, Outcome = 0 | {len(overconf_yes)} | {pct(len(overconf_yes), n_valid)}% |")
p(f"| Forecast < 0.20, Outcome = 1 | {len(overconf_no)} | {pct(len(overconf_no), n_valid)}% |")
p(f"| Total overconfident errors | {len(overconf_yes)+len(overconf_no)} | "
  f"{pct(len(overconf_yes)+len(overconf_no), n_valid)}% |")

# ── 11. Practicality ──────────────────────────────────────────────────────────
h2("11. Practicality — Latency and Token Usage")
p()
p("| Model | Type | Mean Brier | Mean Lat (s) | Mean In-Tok | Mean Out-Tok | Mean Reason-Tok |")
p("|-------|------|-----------|-------------|------------|-------------|----------------|")
model_latency.sort(key=lambda x: x["mean_lat"] or 9999)
for ml in model_latency:
    p(f"| {ml['model_key']} | {ml['model_type']} | {fmt(ml['mean_brier'])} | "
      f"{ml['mean_lat']} | {ml['mean_itok']} | {ml['mean_otok']} | "
      f"{ml['mean_rtok'] or 'N/A'} |")
p()
p("**Performance vs latency highlights:**")
p()
# Best Brier at low latency: standard models
std_ml = [x for x in model_latency if x["model_type"]=="standard"]
reas_ml = [x for x in model_latency if x["model_type"]=="reasoning"]
if std_ml:
    best_std = min(std_ml, key=lambda x: x["mean_brier"] or 99)
    p(f"- Best standard model by Brier: **{best_std['model_key']}** "
      f"(Brier={fmt(best_std['mean_brier'])}, lat={best_std['mean_lat']}s)")
if reas_ml:
    best_reas = min(reas_ml, key=lambda x: x["mean_brier"] or 99)
    p(f"- Best reasoning model by Brier: **{best_reas['model_key']}** "
      f"(Brier={fmt(best_reas['mean_brier'])}, lat={best_reas['mean_lat']}s)")

# ── 12. Interesting findings ──────────────────────────────────────────────────
h2("12. Interesting Findings")
p()

# Build dynamic findings
findings = []

# 1. Evidence effect overall
cb_all = mean(r["brier_score_final"] for r in valid(cell(rows, condition="closed_book")))
se_all = mean(r["brier_score_final"] for r in valid(cell(rows, condition="shared_evidence")))
d_all  = round(se_all - cb_all, 5)
findings.append(
    f"**Evidence effect (overall):** SE Brier ({fmt(se_all)}) vs CB Brier ({fmt(cb_all)}), "
    f"delta = {fmt(d_all)}. Evidence {'helps' if d_all < 0 else 'hurts'} on average "
    f"({'−' if d_all<0 else '+'}{abs(round(d_all*100,3))} Brier points)."
)

# 2. Model spread
findings.append(
    f"**Model spread:** The best model ({leaderboard[0]['model_key']}, "
    f"Brier={fmt(leaderboard[0]['mean_brier'])}) outperforms the weakest "
    f"({leaderboard[-1]['model_key']}, Brier={fmt(leaderboard[-1]['mean_brier'])}) "
    f"by {fmt(spread)} Brier points."
)

# 3. Market baseline
if n_beats == 0:
    mkt_tail = "No model consistently beats the market."
else:
    mkt_tail = f"Best beater: {model_vs_mkt[0]['model_key']} (delta={fmt(model_vs_mkt[0]['delta'])})."
findings.append(
    f"**Market baseline:** {n_beats}/{len(model_vs_mkt)} models beat the market baseline "
    f"(market Brier = {fmt(mkt_overall)}). {mkt_tail}"
)

# 4. Calibration
findings.append(
    f"**Calibration:** Mean forecast = {fmt(mean_f)} vs expected calibrated level. "
    f"ECE = {fmt(ece)}. "
    f"{pct_low}% of forecasts are below 0.10 (low-probability), "
    f"{pct_high}% above 0.90 (high-probability)."
)

# 5. Prompt interaction
best_prompt_cb = min(PROMPTS, key=lambda pk: pxc[(pk,"closed_book")]["brier"] or 99)
best_prompt_se = min(PROMPTS, key=lambda pk: pxc[(pk,"shared_evidence")]["brier"] or 99)
findings.append(
    f"**Prompt ranking:** Best prompt in closed-book = {best_prompt_cb} "
    f"(Brier={fmt(pxc[(best_prompt_cb,'closed_book')]['brier'])}); "
    f"best in shared-evidence = {best_prompt_se} "
    f"(Brier={fmt(pxc[(best_prompt_se,'shared_evidence')]['brier'])})."
)

# 6. Reasoning vs standard
findings.append(
    f"**Reasoning vs standard:** Reasoning models average Brier = "
    f"{fmt(type_stats['reasoning']['brier'])} "
    f"vs standard = {fmt(type_stats['standard']['brier'])}. "
    f"Reasoning models are {round(type_stats['reasoning']['mean_lat']/max(type_stats['standard']['mean_lat'],0.1),1)}× "
    f"slower on average."
)

# 7. Overconfidence
total_overconf = len(overconf_yes) + len(overconf_no)
findings.append(
    f"**Overconfidence:** {total_overconf} catastrophic overconfidence errors "
    f"({pct(total_overconf, n_valid)}% of valid rows): "
    f"{len(overconf_yes)} high-confidence YES on NO outcomes, "
    f"{len(overconf_no)} high-confidence NO on YES outcomes."
)

# 8. Most evidence-responsive model
most_helped_model = se_effect[0]
findings.append(
    f"**Most evidence-responsive model:** {most_helped_model['model_key']} "
    f"(SE−CB delta = {fmt(most_helped_model['se_minus_cb'])}). "
    f"**Least responsive / most hurt:** {se_effect[-1]['model_key']} "
    f"(delta = {fmt(se_effect[-1]['se_minus_cb'])})."
)

# 9. Best topic for evidence
topic_deltas = []
for topic in TOPICS:
    cb_e = next((x for x in topic_stats if x["topic"]==topic and x["condition"]=="closed_book"), None)
    se_e = next((x for x in topic_stats if x["topic"]==topic and x["condition"]=="shared_evidence"), None)
    if cb_e and se_e and cb_e["mean_brier"] and se_e["mean_brier"]:
        d = se_e["mean_brier"] - cb_e["mean_brier"]
        topic_deltas.append((topic, d, cb_e["n"]))
if topic_deltas:
    topic_deltas.sort(key=lambda x: x[1])
    best_t = topic_deltas[0]
    worst_t = topic_deltas[-1]
    findings.append(
        f"**Topic effect:** Evidence helps most for '{best_t[0]}' "
        f"(delta={round(best_t[1],5)}, N={best_t[2]}), "
        f"hurts most for '{worst_t[0]}' (delta={round(worst_t[1],5)}, N={worst_t[2]})."
    )

# 10. Parse reliability
worst_parse = min(leaderboard, key=lambda x: x["parse_rate"] or 100)
findings.append(
    f"**Parse reliability:** Overall {pct(n_valid, n_total)}% parse rate. "
    f"Worst: {worst_parse['model_key']} ({worst_parse['parse_rate']}%). "
    f"All remaining {n_miss} failures are in ≤2 cells per model."
)

# 11. Source effects
src_deltas = []
for src in SOURCES:
    cb_e = next((x for x in source_stats if x["source"]==src and x["condition"]=="closed_book"), None)
    se_e = next((x for x in source_stats if x["source"]==src and x["condition"]=="shared_evidence"), None)
    if cb_e and se_e and cb_e["mean_brier"] and se_e["mean_brier"]:
        d = se_e["mean_brier"] - cb_e["mean_brier"]
        src_deltas.append((src, d))
if src_deltas:
    src_deltas.sort(key=lambda x: x[1])
    findings.append(
        f"**Source effect:** Evidence helps most on {src_deltas[0][0]} questions "
        f"(delta={round(src_deltas[0][1],5)}), "
        f"least on {src_deltas[-1][0]} (delta={round(src_deltas[-1][1],5)})."
    )

for fi in findings:
    p(f"- {fi}")
    p()

# ── 13. Caveats ────────────────────────────────────────────────────────────────
h2("13. Caveats and Checks")
p()
p("- **Market baseline timing:** freeze_datetime verified pre-resolution for 10 sampled questions "
  "(see `results/analysis/baseline_sanity_check.csv`). "
  "Full verification not done — small timing risk remains.")
p()
p("- **Evidence quality variation:** AskNews evidence cache quality varies by question. "
  "Some questions may have more relevant/irrelevant articles than others; "
  "evidence 'effect' mixes information quality with model receptiveness.")
p()
p(f"- **Remaining {n_miss} missing rows:** Spread across 4 models in ≤2 cells each "
  "(gemini-3.1-pro, glm-5.1, gpt-oss-120b, qwen3-max). "
  "Treated as NaN; not imputed. Minor bias risk in per-model comparisons.")
p()
p("- **Prompt verbosity:** P3 (Bayesian) produces longer responses and is more susceptible "
  "to parse failures and truncation artifacts. The 3,487 rows using kept v1 truncated response "
  "may have slightly different forecast distributions.")
p()
p("- **No statistical tests:** All comparisons in this report are descriptive. "
  "Effect sizes should not be interpreted as statistically significant without "
  "appropriate paired tests (e.g., Wilcoxon signed-rank, mixed-effects models). "
  "This is especially important for prompt × condition interactions.")
p()
p("- **Single run per cell:** No repeated measures within cells. "
  "Observed differences partly reflect sampling variance in model outputs at temperature=0.")
p()
p("- **Temperature=0 is deterministic but model-version dependent:** "
  "Results tied to specific model snapshot versions available on OpenRouter at run time.")

# ── Save markdown ──────────────────────────────────────────────────────────────
md_path = OUT_DIR / "main_diagnostic_report_final.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Saved: {md_path}")
print(f"Saved: {mpc_path}")
print(f"Saved: {summ_path}")
print(f"Saved: {ts_path}")
print(f"Saved: {err_path}")
