#!/usr/bin/env python3
"""
Comprehensive analysis of April Pilot LLM forecasting experiment.
"""

import json
import csv
import os
import math
from collections import defaultdict

BASE = "c:/Users/Akram/Desktop/April Pilot"
RESULTS_DIR = os.path.join(BASE, "results")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
FIGURES_DIR = os.path.join(ANALYSIS_DIR, "key_figures")

os.makedirs(ANALYSIS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────

print("Loading merged results...")
with open(os.path.join(RESULTS_DIR, "pilot_results.json"), encoding="utf-8") as f:
    raw = json.load(f)

rows = raw["results"]
print(f"  Total rows: {len(rows)}")

# Load question metadata (for freeze_datetime_value)
with open(os.path.join(BASE, "data", "pilot_questions.json"), encoding="utf-8") as f:
    questions_list = json.load(f)

q_meta = {}
for q in questions_list:
    qid = q["question_id"]
    fdv = q.get("freeze_datetime_value")
    try:
        fdv_float = float(fdv) if fdv is not None and fdv != "" else None
    except (ValueError, TypeError):
        fdv_float = None
    q_meta[qid] = {
        "freeze_datetime_value": fdv_float,
        "resolution_value": q.get("resolution_value"),
        "topic": q.get("topic", "unknown"),
        "source": q.get("source", "unknown"),
    }

# ── Constants ─────────────────────────────────────────────────────────────────

REASONING_MODELS = {
    "gpt-5.4", "claude-opus-4.6", "gemini-3.1-pro", "grok-4.20",
    "qwen3-max", "deepseek-v3.2-speciale", "kimi-k2.6"
}
STANDARD_MODELS = {"gemini-3-flash", "gemma-4-31b", "glm-5.1", "gpt-oss-120b", "mistral-large"}
ALL_MODELS = sorted(REASONING_MODELS | STANDARD_MODELS)
PROMPTS = ["P1", "P2", "P3"]
CONDITIONS = ["closed_book", "shared_evidence"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None

def stdev(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

def fmt4(v):
    return f"{v:.4f}" if v is not None else "N/A"

def fmt2(v):
    return f"{v:.2f}" if v is not None else "N/A"

# ── Completeness ──────────────────────────────────────────────────────────────

print("\nComputing completeness...")
completeness = defaultdict(int)
for r in rows:
    key = (r["model_key"], r["prompt"], r["condition"])
    completeness[key] += 1

print("Rows per (model, prompt, condition):")
for model in ALL_MODELS:
    for prompt in PROMPTS:
        for cond in CONDITIONS:
            n = completeness.get((model, prompt, cond), 0)
            flag = " *** MISSING" if n == 0 else (" LOW" if n < 100 else "")
            print(f"  {model:35s} {prompt} {cond:15s}: {n:4d}{flag}")

# ── Aggregate rows by key ──────────────────────────────────────────────────────

# Group rows
def group_rows(rows, key_fn):
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    return groups

# ── Analysis 1: Overall leaderboard ───────────────────────────────────────────

print("\n=== Analysis 1: Overall Leaderboard ===")

model_data = defaultdict(list)
model_parse = defaultdict(lambda: {"success": 0, "total": 0})
for r in rows:
    m = r["model_key"]
    bs = r.get("brier_score")
    if bs is not None:
        model_data[m].append(bs)
    model_parse[m]["total"] += 1
    if r.get("parse_success"):
        model_parse[m]["success"] += 1

leaderboard = []
for m in ALL_MODELS:
    bs_list = model_data[m]
    mb = mean(bs_list)
    n = model_parse[m]["total"]
    ps = model_parse[m]["success"]
    parse_rate = ps / n if n > 0 else None
    mtype = "reasoning" if m in REASONING_MODELS else "standard"
    leaderboard.append({
        "model": m,
        "type": mtype,
        "mean_brier": mb,
        "n_rows": n,
        "parse_rate": parse_rate,
    })

leaderboard.sort(key=lambda x: (x["mean_brier"] if x["mean_brier"] is not None else 999))

print("Rank  Model                                Type       Brier   ParseRate  N")
for i, row in enumerate(leaderboard, 1):
    print(f"  {i:2d}. {row['model']:35s} {row['type']:10s} {fmt4(row['mean_brier'])}  {fmt4(row['parse_rate'])}  {row['n_rows']}")

# ── Analysis 2: Brier by model × prompt × condition ───────────────────────────

print("\n=== Analysis 2: Brier by model × prompt × condition ===")

mpc_brier = defaultdict(list)
mpc_total = defaultdict(int)
mpc_success = defaultdict(int)
for r in rows:
    key = (r["model_key"], r["prompt"], r["condition"])
    bs = r.get("brier_score")
    if bs is not None:
        mpc_brier[key].append(bs)
    mpc_total[key] += 1
    if r.get("parse_success"):
        mpc_success[key] += 1

# Build 72-cell table
heatmap_rows = []
for m in ALL_MODELS:
    mtype = "reasoning" if m in REASONING_MODELS else "standard"
    for p in PROMPTS:
        for c in CONDITIONS:
            key = (m, p, c)
            vals = mpc_brier[key]
            mb = mean(vals)
            n = mpc_total[key]
            ns = mpc_success[key]
            pr = ns / n if n > 0 else None
            heatmap_rows.append({
                "model": m,
                "model_type": mtype,
                "prompt": p,
                "condition": c,
                "mean_brier": mb,
                "n_rows": n,
                "parse_rate": pr,
                "flagged": (pr is not None and pr < 0.5),
            })

# ── Analysis 3: Prompt × condition interaction ────────────────────────────────

print("\n=== Analysis 3: Prompt × Condition Interaction ===")

pc_brier = defaultdict(list)
for r in rows:
    key = (r["prompt"], r["condition"])
    bs = r.get("brier_score")
    if bs is not None:
        pc_brier[key].append(bs)

pc_means = {}
for p in PROMPTS:
    for c in CONDITIONS:
        key = (p, c)
        pc_means[key] = mean(pc_brier[key])

print("Prompt  CB_Brier  SE_Brier  Delta(SE-CB)  N_CB    N_SE")
prompt_condition_rows = []
for p in PROMPTS:
    cb = pc_means[(p, "closed_book")]
    se = pc_means[(p, "shared_evidence")]
    delta = (se - cb) if (se is not None and cb is not None) else None
    n_cb = len(pc_brier[(p, "closed_book")])
    n_se = len(pc_brier[(p, "shared_evidence")])
    print(f"  {p}  {fmt4(cb)}  {fmt4(se)}  {fmt4(delta)}  {n_cb}  {n_se}")
    prompt_condition_rows.append({
        "prompt": p,
        "cb_brier": cb,
        "se_brier": se,
        "delta_se_minus_cb": delta,
        "n_cb": n_cb,
        "n_se": n_se,
    })

# ── Analysis 4: Evidence effect per model ─────────────────────────────────────

print("\n=== Analysis 4: Evidence Effect per Model ===")

model_cond_brier = defaultdict(list)
for r in rows:
    key = (r["model_key"], r["condition"])
    bs = r.get("brier_score")
    if bs is not None:
        model_cond_brier[key].append(bs)

evidence_effect = []
for m in ALL_MODELS:
    cb = mean(model_cond_brier[(m, "closed_book")])
    se = mean(model_cond_brier[(m, "shared_evidence")])
    delta = (se - cb) if (se is not None and cb is not None) else None
    mtype = "reasoning" if m in REASONING_MODELS else "standard"
    evidence_effect.append({
        "model": m,
        "model_type": mtype,
        "cb_brier": cb,
        "se_brier": se,
        "delta_se_minus_cb": delta,
    })

evidence_effect.sort(key=lambda x: (x["delta_se_minus_cb"] if x["delta_se_minus_cb"] is not None else 0))

print("Model                                Type       CB_Brier  SE_Brier  Delta(SE-CB)")
for e in evidence_effect:
    print(f"  {e['model']:35s} {e['model_type']:10s} {fmt4(e['cb_brier'])}  {fmt4(e['se_brier'])}  {fmt4(e['delta_se_minus_cb'])}")

# ── Analysis 5: Reasoning vs Standard ────────────────────────────────────────

print("\n=== Analysis 5: Reasoning vs Standard ===")

type_pc_brier = defaultdict(list)
for r in rows:
    key = (r["model_type"], r["prompt"], r["condition"])
    bs = r.get("brier_score")
    if bs is not None:
        type_pc_brier[key].append(bs)

reasoning_vs_standard = []
for mtype in ["reasoning", "standard"]:
    for p in PROMPTS:
        for c in CONDITIONS:
            key = (mtype, p, c)
            vals = type_pc_brier[key]
            mb = mean(vals)
            reasoning_vs_standard.append({
                "model_type": mtype,
                "prompt": p,
                "condition": c,
                "mean_brier": mb,
                "n_rows": len(vals),
            })
            print(f"  {mtype:10s} {p} {c:15s}: {fmt4(mb)} (n={len(vals)})")

# ── Analysis 6: Calibration ──────────────────────────────────────────────────

print("\n=== Analysis 6: Calibration ===")

model_forecasts = defaultdict(list)
for r in rows:
    if r.get("parse_success") and r.get("forecast") is not None:
        model_forecasts[r["model_key"]].append(r["forecast"])

calibration = []
for m in ALL_MODELS:
    fc = model_forecasts[m]
    if not fc:
        calibration.append({
            "model": m,
            "mean_forecast": None,
            "std_forecast": None,
            "pct_below_01": None,
            "pct_above_09": None,
            "pct_01_to_09": None,
            "n": 0,
        })
        continue
    mf = mean(fc)
    sf = stdev(fc)
    n = len(fc)
    below_01 = sum(1 for v in fc if v < 0.1) / n * 100
    above_09 = sum(1 for v in fc if v > 0.9) / n * 100
    mid = sum(1 for v in fc if 0.1 <= v <= 0.9) / n * 100
    calibration.append({
        "model": m,
        "mean_forecast": mf,
        "std_forecast": sf,
        "pct_below_01": below_01,
        "pct_above_09": above_09,
        "pct_01_to_09": mid,
        "n": n,
    })
    print(f"  {m:35s}: mean={fmt4(mf)} std={fmt4(sf)} <0.1={below_01:.1f}% >0.9={above_09:.1f}% mid={mid:.1f}%")

# ── Analysis 7: Parse reliability ────────────────────────────────────────────

print("\n=== Analysis 7: Parse Reliability ===")

parse_data = defaultdict(lambda: {"success": 0, "total": 0})
for r in rows:
    key = (r["model_key"], r["prompt"], r["condition"])
    parse_data[key]["total"] += 1
    if r.get("parse_success"):
        parse_data[key]["success"] += 1

parse_rows = []
flagged_cells = []
for m in ALL_MODELS:
    for p in PROMPTS:
        for c in CONDITIONS:
            key = (m, p, c)
            d = parse_data[key]
            n = d["total"]
            s = d["success"]
            fail_n = n - s
            fail_pct = fail_n / n * 100 if n > 0 else None
            flagged = fail_pct is not None and fail_pct > 10
            parse_rows.append({
                "model": m,
                "prompt": p,
                "condition": c,
                "n_total": n,
                "n_success": s,
                "n_fail": fail_n,
                "fail_pct": fail_pct,
                "flagged": flagged,
            })
            if flagged:
                flagged_cells.append((m, p, c, fail_pct))

print(f"  Total cells flagged (>10% failure): {len(flagged_cells)}")
for m, p, c, fp in flagged_cells:
    print(f"    *** {m} {p} {c}: {fp:.1f}% failures")

# ── Analysis 8: Latency/cost ──────────────────────────────────────────────────

print("\n=== Analysis 8: Latency / Cost ===")

model_latency = defaultdict(list)
model_reasoning_tokens = defaultdict(list)
for r in rows:
    m = r["model_key"]
    lat = r.get("latency_seconds")
    rt = r.get("reasoning_tokens")
    if lat is not None:
        model_latency[m].append(lat)
    if rt is not None:
        model_reasoning_tokens[m].append(rt)

latency_rows = []
for m in ALL_MODELS:
    ml = mean(model_latency[m])
    mrt = mean(model_reasoning_tokens[m])
    mb = mean(model_data[m])
    # Efficiency: lower brier per second is better (lower = better)
    # We'll define efficiency as brier / latency (lower is better, but we want brier-efficient models)
    # Actually let's just report brier and latency
    efficiency = mb / ml if (mb is not None and ml is not None and ml > 0) else None
    latency_rows.append({
        "model": m,
        "mean_latency_s": ml,
        "mean_reasoning_tokens": mrt,
        "mean_brier": mb,
        "brier_per_second": efficiency,
    })
    print(f"  {m:35s}: latency={fmt2(ml)}s  reasoning_tok={fmt2(mrt)}  brier={fmt4(mb)}  brier/s={fmt4(efficiency)}")

latency_rows.sort(key=lambda x: (x["brier_per_second"] if x["brier_per_second"] is not None else 999))

# ── Analysis 9: Market baseline ───────────────────────────────────────────────

print("\n=== Analysis 9: Market Baseline ===")

# Market brier per question
q_market_brier = {}
for qid, meta in q_meta.items():
    fdv = meta["freeze_datetime_value"]
    rv = meta["resolution_value"]
    if fdv is not None and rv is not None:
        try:
            rv_f = float(rv)
            q_market_brier[qid] = (fdv - rv_f) ** 2
        except (TypeError, ValueError):
            pass

print(f"  Questions with valid market Brier: {len(q_market_brier)}")

# For each (model, prompt, condition), compute model brier and market brier on same rows
mpc_market_brier = defaultdict(list)
mpc_model_brier_mkt = defaultdict(list)

for r in rows:
    qid = r["question_id"]
    key = (r["model_key"], r["prompt"], r["condition"])
    mb = r.get("brier_score")
    mkt_b = q_market_brier.get(qid)
    if mb is not None and mkt_b is not None:
        mpc_model_brier_mkt[key].append(mb)
        mpc_market_brier[key].append(mkt_b)

market_rows = []
print("Model                                Prompt Condition       ModelBrier  MarketBrier  Delta(model-mkt)  BeatMkt?")
for m in ALL_MODELS:
    for p in PROMPTS:
        for c in CONDITIONS:
            key = (m, p, c)
            mod_b = mean(mpc_model_brier_mkt[key])
            mkt_b = mean(mpc_market_brier[key])
            delta = (mod_b - mkt_b) if (mod_b is not None and mkt_b is not None) else None
            beat = "YES" if (delta is not None and delta < 0) else "no"
            market_rows.append({
                "model": m,
                "prompt": p,
                "condition": c,
                "model_brier": mod_b,
                "market_brier": mkt_b,
                "delta_model_minus_market": delta,
                "beats_market": beat,
            })

# Summary: how many cells beat market?
beat_count = sum(1 for r in market_rows if r["beats_market"] == "YES")
print(f"  Cells beating market: {beat_count} / {len(market_rows)}")

# Overall model vs market
model_vs_market = []
for m in ALL_MODELS:
    all_mod = []
    all_mkt = []
    for r in rows:
        qid = r["question_id"]
        if r["model_key"] != m:
            continue
        mb = r.get("brier_score")
        mkt_b = q_market_brier.get(qid)
        if mb is not None and mkt_b is not None:
            all_mod.append(mb)
            all_mkt.append(mkt_b)
    mod_mean = mean(all_mod)
    mkt_mean = mean(all_mkt)
    delta = (mod_mean - mkt_mean) if (mod_mean is not None and mkt_mean is not None) else None
    model_vs_market.append({
        "model": m,
        "model_brier": mod_mean,
        "market_brier": mkt_mean,
        "delta": delta,
        "beats_market": "YES" if (delta is not None and delta < 0) else "no",
    })
    print(f"  {m:35s}: model={fmt4(mod_mean)}  market={fmt4(mkt_mean)}  delta={fmt4(delta)}  {('BEATS' if delta is not None and delta < 0 else '')}")

# ── Analysis 10: Topic effects ────────────────────────────────────────────────

print("\n=== Analysis 10: Topic Effects ===")

topic_cond_brier = defaultdict(list)
for r in rows:
    topic = r.get("topic", "unknown")
    cond = r["condition"]
    bs = r.get("brier_score")
    if bs is not None:
        topic_cond_brier[(topic, cond)].append(bs)

# Get all topics
topics = sorted(set(r.get("topic", "unknown") for r in rows))

topic_rows = []
print("Topic                   CB_Brier  SE_Brier  Delta(SE-CB)")
for t in topics:
    cb = mean(topic_cond_brier[(t, "closed_book")])
    se = mean(topic_cond_brier[(t, "shared_evidence")])
    delta = (se - cb) if (cb is not None and se is not None) else None
    n_cb = len(topic_cond_brier[(t, "closed_book")])
    n_se = len(topic_cond_brier[(t, "shared_evidence")])
    topic_rows.append({
        "topic": t,
        "cb_brier": cb,
        "se_brier": se,
        "delta_se_minus_cb": delta,
        "n_cb": n_cb,
        "n_se": n_se,
    })
    print(f"  {t:25s}: CB={fmt4(cb)}  SE={fmt4(se)}  delta={fmt4(delta)}  n_cb={n_cb} n_se={n_se}")

# ── Analysis 11: Source effects ───────────────────────────────────────────────

print("\n=== Analysis 11: Source Effects ===")

source_cond_brier = defaultdict(list)
for r in rows:
    source = r.get("source", "unknown")
    cond = r["condition"]
    bs = r.get("brier_score")
    if bs is not None:
        source_cond_brier[(source, cond)].append(bs)

sources = sorted(set(r.get("source", "unknown") for r in rows))

source_rows = []
print("Source       CB_Brier  SE_Brier  Delta(SE-CB)")
for s in sources:
    cb = mean(source_cond_brier[(s, "closed_book")])
    se = mean(source_cond_brier[(s, "shared_evidence")])
    delta = (se - cb) if (cb is not None and se is not None) else None
    n_cb = len(source_cond_brier[(s, "closed_book")])
    n_se = len(source_cond_brier[(s, "shared_evidence")])
    source_rows.append({
        "source": s,
        "cb_brier": cb,
        "se_brier": se,
        "delta_se_minus_cb": delta,
        "n_cb": n_cb,
        "n_se": n_se,
    })
    print(f"  {s:15s}: CB={fmt4(cb)}  SE={fmt4(se)}  delta={fmt4(delta)}  n_cb={n_cb} n_se={n_se}")

# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT FILES
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== Writing output files ===")

# ── A. summary_tables.csv ─────────────────────────────────────────────────────

csv_path = os.path.join(ANALYSIS_DIR, "summary_tables.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)

    # Section 1: Leaderboard
    w.writerow(["SECTION: LEADERBOARD"])
    w.writerow(["rank", "model", "model_type", "mean_brier", "n_rows", "parse_rate"])
    for i, row in enumerate(leaderboard, 1):
        w.writerow([i, row["model"], row["type"], fmt4(row["mean_brier"]),
                    row["n_rows"], fmt4(row["parse_rate"])])
    w.writerow([])

    # Section 2: Brier by model×prompt×condition (72 cells)
    w.writerow(["SECTION: BRIER BY MODEL x PROMPT x CONDITION (72 cells)"])
    w.writerow(["model", "model_type", "prompt", "condition", "mean_brier", "n_rows", "parse_rate", "flagged_low_parse"])
    for hr in heatmap_rows:
        w.writerow([hr["model"], hr["model_type"], hr["prompt"], hr["condition"],
                    fmt4(hr["mean_brier"]), hr["n_rows"], fmt4(hr["parse_rate"]),
                    "YES" if hr["flagged"] else ""])
    w.writerow([])

    # Section 3: Prompt × Condition interaction
    w.writerow(["SECTION: PROMPT x CONDITION INTERACTION"])
    w.writerow(["prompt", "cb_brier", "se_brier", "delta_se_minus_cb", "n_cb", "n_se"])
    for row in prompt_condition_rows:
        w.writerow([row["prompt"], fmt4(row["cb_brier"]), fmt4(row["se_brier"]),
                    fmt4(row["delta_se_minus_cb"]), row["n_cb"], row["n_se"]])
    w.writerow([])

    # Section 4: Evidence effect per model
    w.writerow(["SECTION: EVIDENCE EFFECT PER MODEL (ranked most helped to most hurt)"])
    w.writerow(["model", "model_type", "cb_brier", "se_brier", "delta_se_minus_cb"])
    for e in evidence_effect:
        w.writerow([e["model"], e["model_type"], fmt4(e["cb_brier"]),
                    fmt4(e["se_brier"]), fmt4(e["delta_se_minus_cb"])])
    w.writerow([])

    # Section 5: Reasoning vs Standard
    w.writerow(["SECTION: REASONING VS STANDARD BY PROMPT x CONDITION"])
    w.writerow(["model_type", "prompt", "condition", "mean_brier", "n_rows"])
    for row in reasoning_vs_standard:
        w.writerow([row["model_type"], row["prompt"], row["condition"],
                    fmt4(row["mean_brier"]), row["n_rows"]])
    w.writerow([])

    # Section 6: Calibration
    w.writerow(["SECTION: CALIBRATION"])
    w.writerow(["model", "mean_forecast", "std_forecast", "pct_below_01",
                "pct_above_09", "pct_01_to_09", "n_forecasts"])
    for c in calibration:
        w.writerow([c["model"], fmt4(c["mean_forecast"]), fmt4(c["std_forecast"]),
                    fmt4(c["pct_below_01"]), fmt4(c["pct_above_09"]),
                    fmt4(c["pct_01_to_09"]), c["n"]])
    w.writerow([])

    # Section 7: Parse reliability
    w.writerow(["SECTION: PARSE RELIABILITY"])
    w.writerow(["model", "prompt", "condition", "n_total", "n_success", "n_fail", "fail_pct", "flagged"])
    for row in parse_rows:
        w.writerow([row["model"], row["prompt"], row["condition"], row["n_total"],
                    row["n_success"], row["n_fail"], fmt4(row["fail_pct"]),
                    "YES" if row["flagged"] else ""])
    w.writerow([])

    # Section 8: Latency/cost
    w.writerow(["SECTION: LATENCY AND COST"])
    w.writerow(["model", "mean_latency_s", "mean_reasoning_tokens", "mean_brier", "brier_per_second"])
    for row in latency_rows:
        w.writerow([row["model"], fmt2(row["mean_latency_s"]),
                    fmt2(row["mean_reasoning_tokens"]), fmt4(row["mean_brier"]),
                    fmt4(row["brier_per_second"])])
    w.writerow([])

    # Section 9: Market baseline comparison
    w.writerow(["SECTION: MARKET BASELINE COMPARISON (overall per model)"])
    w.writerow(["model", "model_brier", "market_brier", "delta_model_minus_market", "beats_market"])
    for row in model_vs_market:
        w.writerow([row["model"], fmt4(row["model_brier"]), fmt4(row["market_brier"]),
                    fmt4(row["delta"]), row["beats_market"]])
    w.writerow([])

    w.writerow(["SECTION: MARKET BASELINE BY MODEL x PROMPT x CONDITION"])
    w.writerow(["model", "prompt", "condition", "model_brier", "market_brier",
                "delta_model_minus_market", "beats_market"])
    for row in market_rows:
        w.writerow([row["model"], row["prompt"], row["condition"],
                    fmt4(row["model_brier"]), fmt4(row["market_brier"]),
                    fmt4(row["delta_model_minus_market"]), row["beats_market"]])
    w.writerow([])

    # Section 10: Topic effects
    w.writerow(["SECTION: TOPIC EFFECTS"])
    w.writerow(["topic", "cb_brier", "se_brier", "delta_se_minus_cb", "n_cb", "n_se"])
    for row in topic_rows:
        w.writerow([row["topic"], fmt4(row["cb_brier"]), fmt4(row["se_brier"]),
                    fmt4(row["delta_se_minus_cb"]), row["n_cb"], row["n_se"]])
    w.writerow([])

    # Section 11: Source effects
    w.writerow(["SECTION: SOURCE EFFECTS"])
    w.writerow(["source", "cb_brier", "se_brier", "delta_se_minus_cb", "n_cb", "n_se"])
    for row in source_rows:
        w.writerow([row["source"], fmt4(row["cb_brier"]), fmt4(row["se_brier"]),
                    fmt4(row["delta_se_minus_cb"]), row["n_cb"], row["n_se"]])

print(f"  Saved: {csv_path}")

# ── C. Key figures CSVs ───────────────────────────────────────────────────────

# brier_heatmap.csv
path = os.path.join(FIGURES_DIR, "brier_heatmap.csv")
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "model_type", "prompt", "condition", "mean_brier", "n_rows", "parse_rate"])
    for hr in heatmap_rows:
        w.writerow([hr["model"], hr["model_type"], hr["prompt"], hr["condition"],
                    fmt4(hr["mean_brier"]), hr["n_rows"], fmt4(hr["parse_rate"])])
print(f"  Saved: {path}")

# evidence_effect.csv
path = os.path.join(FIGURES_DIR, "evidence_effect.csv")
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "model_type", "cb_brier", "se_brier", "delta_se_minus_cb"])
    for e in evidence_effect:
        w.writerow([e["model"], e["model_type"], fmt4(e["cb_brier"]),
                    fmt4(e["se_brier"]), fmt4(e["delta_se_minus_cb"])])
print(f"  Saved: {path}")

# calibration.csv
path = os.path.join(FIGURES_DIR, "calibration.csv")
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "mean_forecast", "std_forecast", "pct_below_01", "pct_above_09", "pct_01_to_09", "n"])
    for c in calibration:
        w.writerow([c["model"], fmt4(c["mean_forecast"]), fmt4(c["std_forecast"]),
                    fmt4(c["pct_below_01"]), fmt4(c["pct_above_09"]),
                    fmt4(c["pct_01_to_09"]), c["n"]])
print(f"  Saved: {path}")

# topic_brier.csv
path = os.path.join(FIGURES_DIR, "topic_brier.csv")
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["topic", "cb_brier", "se_brier", "delta_se_minus_cb", "n_cb", "n_se"])
    for row in topic_rows:
        w.writerow([row["topic"], fmt4(row["cb_brier"]), fmt4(row["se_brier"]),
                    fmt4(row["delta_se_minus_cb"]), row["n_cb"], row["n_se"]])
print(f"  Saved: {path}")

# prompt_condition_means.csv
path = os.path.join(FIGURES_DIR, "prompt_condition_means.csv")
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["prompt", "condition", "mean_brier", "n_rows"])
    for row in prompt_condition_rows:
        for cond_key, brier_key, n_key in [("closed_book", "cb_brier", "n_cb"), ("shared_evidence", "se_brier", "n_se")]:
            w.writerow([row["prompt"], cond_key, fmt4(row[brier_key]), row[n_key]])
print(f"  Saved: {path}")

# ──────────────────────────────────────────────────────────────────────────────
# Collect all numbers for the markdown report
# ──────────────────────────────────────────────────────────────────────────────

# Best and worst models
best_model = leaderboard[0]
worst_model = leaderboard[-1]

# Best prompt×condition combo
pc_list = [(p, c, pc_means[(p, c)]) for p in PROMPTS for c in CONDITIONS if pc_means[(p, c)] is not None]
pc_list.sort(key=lambda x: x[2])
best_pc = pc_list[0]
worst_pc = pc_list[-1]

# Evidence effect direction
most_helped = evidence_effect[0]  # most negative delta = most helped
most_hurt = evidence_effect[-1]   # most positive delta = most hurt

# Prompt that benefits most from evidence (most negative delta)
prompt_benefit = min(prompt_condition_rows, key=lambda x: x["delta_se_minus_cb"] if x["delta_se_minus_cb"] is not None else 999)

# Models beating market
market_beaters = [r for r in model_vs_market if r["beats_market"] == "YES"]

# Overall market brier (same set of rows)
all_mkt_brieries = []
for qid, mb in q_market_brier.items():
    all_mkt_brieries.append(mb)
overall_mkt = mean(all_mkt_brieries)
print(f"\nOverall market Brier (all questions): {fmt4(overall_mkt)}")

# Reasoning vs standard overall
reasoning_brier = mean([r.get("brier_score") for r in rows if r.get("model_type") == "reasoning" and r.get("brier_score") is not None])
standard_brier = mean([r.get("brier_score") for r in rows if r.get("model_type") == "standard" and r.get("brier_score") is not None])
print(f"Reasoning mean Brier: {fmt4(reasoning_brier)}")
print(f"Standard mean Brier:  {fmt4(standard_brier)}")

# Overall parse rate
total_rows = len(rows)
total_success = sum(1 for r in rows if r.get("parse_success"))
overall_parse_rate = total_success / total_rows * 100

# CB vs SE overall
all_cb = [r.get("brier_score") for r in rows if r["condition"] == "closed_book" and r.get("brier_score") is not None]
all_se = [r.get("brier_score") for r in rows if r["condition"] == "shared_evidence" and r.get("brier_score") is not None]
overall_cb = mean(all_cb)
overall_se = mean(all_se)
overall_evidence_delta = overall_se - overall_cb if (overall_cb and overall_se) else None

print(f"Overall CB Brier: {fmt4(overall_cb)}")
print(f"Overall SE Brier: {fmt4(overall_se)}")
print(f"Overall SE-CB delta: {fmt4(overall_evidence_delta)}")
print(f"Overall parse rate: {overall_parse_rate:.2f}%")

# ── B. short_diagnostic_report.md ────────────────────────────────────────────

# Completeness table data
completeness_table = []
for model in ALL_MODELS:
    row_data = {"model": model}
    total = 0
    for p in PROMPTS:
        for c in CONDITIONS:
            n = completeness.get((model, p, c), 0)
            row_data[f"{p}_{c[:2].upper()}"] = n
            total += n
    row_data["total"] = total
    completeness_table.append(row_data)

# Topic that benefits most from evidence
topic_rows_sorted = sorted([t for t in topic_rows if t["delta_se_minus_cb"] is not None],
                            key=lambda x: x["delta_se_minus_cb"])
most_helped_topic = topic_rows_sorted[0] if topic_rows_sorted else None
most_hurt_topic = topic_rows_sorted[-1] if topic_rows_sorted else None

# Source that benefits most
source_rows_sorted = sorted([s for s in source_rows if s["delta_se_minus_cb"] is not None],
                              key=lambda x: x["delta_se_minus_cb"])

# Flagged cells
flagged_parse_cells = [(r["model"], r["prompt"], r["condition"], r["fail_pct"])
                        for r in parse_rows if r["flagged"]]

md_lines = []
md_lines.append("# April Pilot: Short Diagnostic Report")
md_lines.append("")
md_lines.append(f"**Generated:** 2026-04-28  ")
md_lines.append(f"**Total rows analyzed:** {len(rows):,}  ")
md_lines.append(f"**Models:** {len(ALL_MODELS)} (7 reasoning, 5 standard)  ")
md_lines.append(f"**Questions:** 114 resolved binary  ")
md_lines.append(f"**Prompts:** P1 (Control), P2 (Base-rate-first), P3 (Bayesian)  ")
md_lines.append(f"**Conditions:** closed_book (CB), shared_evidence (SE)  ")
md_lines.append("")

# Executive summary
md_lines.append("## Executive Summary")
md_lines.append("")
md_lines.append(
    f"The pilot ran {len(rows):,} LLM forecasting calls across 12 models, 3 prompts, and 2 conditions on 114 binary questions, "
    f"achieving an overall parse rate of {overall_parse_rate:.1f}%. "
    f"The best-performing model was **{best_model['model']}** (mean Brier = {fmt4(best_model['mean_brier'])}), "
    f"and the weakest was **{worst_model['model']}** (mean Brier = {fmt4(worst_model['mean_brier'])}). "
    f"Overall, shared evidence {'improved' if overall_evidence_delta is not None and overall_evidence_delta < 0 else 'worsened'} "
    f"forecasting accuracy relative to closed-book (SE−CB delta = {fmt4(overall_evidence_delta)}), "
    f"and {len(market_beaters)} of 12 models beat the market baseline on average. "
    f"The Prompt×Condition interaction — the key hypothesis — shows that "
    f"**{prompt_benefit['prompt']}** benefits most from evidence (delta = {fmt4(prompt_benefit['delta_se_minus_cb'])}), "
    f"though effect sizes are modest at this pilot scale."
)
md_lines.append("")

# Data completeness
md_lines.append("## Data Completeness")
md_lines.append("")
md_lines.append(f"Expected rows per cell: 114 (one per question). Expected total: 8,208. Actual: {len(rows):,}.")
md_lines.append("")
md_lines.append("| Model | Type | P1_CB | P1_SE | P2_CB | P2_SE | P3_CB | P3_SE | Total |")
md_lines.append("|-------|------|-------|-------|-------|-------|-------|-------|-------|")
for ct in completeness_table:
    mtype = "R" if ct["model"] in REASONING_MODELS else "S"
    # recompute per prompt/condition
    row_vals = []
    for p in PROMPTS:
        for c in ["closed_book", "shared_evidence"]:
            n = completeness.get((ct["model"], p, c), 0)
            flag = " !" if n == 0 else (" ~" if n < 100 else "")
            row_vals.append(f"{n}{flag}")
    md_lines.append(f"| {ct['model']} | {mtype} | " + " | ".join(row_vals) + f" | {ct['total']} |")
md_lines.append("")

# Analysis 1: Leaderboard
md_lines.append("## Analysis 1: Overall Leaderboard")
md_lines.append("")
md_lines.append("| Rank | Model | Type | Mean Brier | Parse Rate | N |")
md_lines.append("|------|-------|------|------------|------------|---|")
for i, row in enumerate(leaderboard, 1):
    md_lines.append(f"| {i} | {row['model']} | {row['type']} | {fmt4(row['mean_brier'])} | {fmt4(row['parse_rate'])} | {row['n_rows']} |")
md_lines.append("")
md_lines.append(f"Market baseline (freeze-datetime probability): mean Brier = **{fmt4(overall_mkt)}**")
md_lines.append("")

# Analysis 2: 72-cell table
md_lines.append("## Analysis 2: Brier by Model × Prompt × Condition (72 cells)")
md_lines.append("")
md_lines.append("*CB = closed_book, SE = shared_evidence. Flagged cells (parse rate < 50%) marked with \\*.*")
md_lines.append("")
header_cols = ["Model", "Type"]
for p in PROMPTS:
    header_cols += [f"{p}_CB", f"{p}_SE"]
md_lines.append("| " + " | ".join(header_cols) + " |")
md_lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")
for m in ALL_MODELS:
    mtype = "R" if m in REASONING_MODELS else "S"
    vals = [m, mtype]
    for p in PROMPTS:
        for c in ["closed_book", "shared_evidence"]:
            hr_match = next((hr for hr in heatmap_rows if hr["model"] == m and hr["prompt"] == p and hr["condition"] == c), None)
            if hr_match:
                cell = fmt4(hr_match["mean_brier"])
                if hr_match["flagged"]:
                    cell += " *"
            else:
                cell = "N/A"
            vals.append(cell)
    md_lines.append("| " + " | ".join(vals) + " |")
md_lines.append("")

# Analysis 3: Prompt × Condition interaction
md_lines.append("## Analysis 3: Prompt × Condition Interaction")
md_lines.append("")
md_lines.append("| Prompt | CB Brier | SE Brier | Delta (SE−CB) | N_CB | N_SE |")
md_lines.append("|--------|----------|----------|---------------|------|------|")
for row in prompt_condition_rows:
    md_lines.append(f"| {row['prompt']} | {fmt4(row['cb_brier'])} | {fmt4(row['se_brier'])} | {fmt4(row['delta_se_minus_cb'])} | {row['n_cb']} | {row['n_se']} |")
md_lines.append("")
md_lines.append(f"**Key finding:** {prompt_benefit['prompt']} benefits most from evidence (delta = {fmt4(prompt_benefit['delta_se_minus_cb'])}).")
md_lines.append(f"Overall SE−CB delta across all prompts: {fmt4(overall_evidence_delta)}.")
md_lines.append("")

# Analysis 4: Evidence effect per model
md_lines.append("## Analysis 4: Evidence Effect per Model")
md_lines.append("")
md_lines.append("*Ranked from most helped (most negative delta) to most hurt by evidence.*")
md_lines.append("")
md_lines.append("| Model | Type | CB Brier | SE Brier | Delta (SE−CB) |")
md_lines.append("|-------|------|----------|----------|---------------|")
for e in evidence_effect:
    md_lines.append(f"| {e['model']} | {e['model_type']} | {fmt4(e['cb_brier'])} | {fmt4(e['se_brier'])} | {fmt4(e['delta_se_minus_cb'])} |")
md_lines.append("")

# Analysis 5: Reasoning vs Standard
md_lines.append("## Analysis 5: Reasoning vs Standard Models")
md_lines.append("")
md_lines.append(f"- Overall reasoning mean Brier: **{fmt4(reasoning_brier)}**")
md_lines.append(f"- Overall standard mean Brier: **{fmt4(standard_brier)}**")
md_lines.append("")
md_lines.append("| Type | Prompt | Condition | Mean Brier | N |")
md_lines.append("|------|--------|-----------|------------|---|")
for row in reasoning_vs_standard:
    md_lines.append(f"| {row['model_type']} | {row['prompt']} | {row['condition']} | {fmt4(row['mean_brier'])} | {row['n_rows']} |")
md_lines.append("")

# Analysis 6: Calibration
md_lines.append("## Analysis 6: Calibration")
md_lines.append("")
md_lines.append("| Model | Mean Forecast | Std Dev | % < 0.1 | % > 0.9 | % 0.1–0.9 | N |")
md_lines.append("|-------|--------------|---------|---------|---------|-----------|---|")
for c in calibration:
    md_lines.append(f"| {c['model']} | {fmt4(c['mean_forecast'])} | {fmt4(c['std_forecast'])} | "
                    f"{fmt4(c['pct_below_01'])} | {fmt4(c['pct_above_09'])} | {fmt4(c['pct_01_to_09'])} | {c['n']} |")
md_lines.append("")

# Analysis 7: Parse reliability
md_lines.append("## Analysis 7: Parse Reliability")
md_lines.append("")
md_lines.append(f"Overall parse rate: **{overall_parse_rate:.2f}%** ({total_success:,}/{total_rows:,} rows)")
md_lines.append("")
if flagged_parse_cells:
    md_lines.append(f"**{len(flagged_parse_cells)} cell(s) flagged with >10% failure rate:**")
    md_lines.append("")
    for m, p, c, fp in flagged_parse_cells:
        md_lines.append(f"- {m} / {p} / {c}: {fp:.1f}% failures")
    md_lines.append("")
else:
    md_lines.append("No cells exceed the 10% failure threshold.")
    md_lines.append("")

# Full parse table (summary by model only to keep it brief)
md_lines.append("Parse rate by model (across all prompts/conditions):")
md_lines.append("")
md_lines.append("| Model | Parse Rate | N Fail / N Total |")
md_lines.append("|-------|------------|-----------------|")
for m in ALL_MODELS:
    d = model_parse[m]
    n = d["total"]
    s = d["success"]
    pr = s / n if n > 0 else None
    md_lines.append(f"| {m} | {fmt4(pr)} | {n-s}/{n} |")
md_lines.append("")

# Analysis 8: Latency
md_lines.append("## Analysis 8: Latency and Cost")
md_lines.append("")
md_lines.append("*Ranked by Brier/second (lower = more efficient).*")
md_lines.append("")
md_lines.append("| Model | Mean Latency (s) | Mean Reasoning Tokens | Mean Brier | Brier/s |")
md_lines.append("|-------|------------------|-----------------------|------------|---------|")
for row in latency_rows:
    md_lines.append(f"| {row['model']} | {fmt2(row['mean_latency_s'])} | {fmt2(row['mean_reasoning_tokens'])} | "
                    f"{fmt4(row['mean_brier'])} | {fmt4(row['brier_per_second'])} |")
md_lines.append("")

# Analysis 9: Market baseline
md_lines.append("## Analysis 9: Market Baseline Comparison")
md_lines.append("")
md_lines.append(f"Market baseline mean Brier (freeze-datetime probability): **{fmt4(overall_mkt)}**")
md_lines.append("")
md_lines.append(f"Models beating the market: **{len(market_beaters)}/{len(ALL_MODELS)}**")
md_lines.append("")
md_lines.append("| Model | Model Brier | Market Brier | Delta | Beats Market? |")
md_lines.append("|-------|-------------|--------------|-------|---------------|")
for row in sorted(model_vs_market, key=lambda x: x["delta"] if x["delta"] is not None else 999):
    md_lines.append(f"| {row['model']} | {fmt4(row['model_brier'])} | {fmt4(row['market_brier'])} | "
                    f"{fmt4(row['delta'])} | {row['beats_market']} |")
md_lines.append("")

# Analysis 10: Topic effects
md_lines.append("## Analysis 10: Topic Effects")
md_lines.append("")
md_lines.append("| Topic | CB Brier | SE Brier | Delta (SE−CB) | N_CB | N_SE |")
md_lines.append("|-------|----------|----------|---------------|------|------|")
for row in sorted(topic_rows, key=lambda x: x["delta_se_minus_cb"] if x["delta_se_minus_cb"] is not None else 0):
    md_lines.append(f"| {row['topic']} | {fmt4(row['cb_brier'])} | {fmt4(row['se_brier'])} | "
                    f"{fmt4(row['delta_se_minus_cb'])} | {row['n_cb']} | {row['n_se']} |")
md_lines.append("")

# Analysis 11: Source effects
md_lines.append("## Analysis 11: Source Effects")
md_lines.append("")
md_lines.append("| Source | CB Brier | SE Brier | Delta (SE−CB) | N_CB | N_SE |")
md_lines.append("|--------|----------|----------|---------------|------|------|")
for row in source_rows_sorted:
    md_lines.append(f"| {row['source']} | {fmt4(row['cb_brier'])} | {fmt4(row['se_brier'])} | "
                    f"{fmt4(row['delta_se_minus_cb'])} | {row['n_cb']} | {row['n_se']} |")
md_lines.append("")

# Interesting findings
md_lines.append("## Interesting Findings")
md_lines.append("")

# Generate dynamic bullets
bullets = []

# 1. Best/worst model
bullets.append(
    f"**Model spread:** Best model ({best_model['model']}) achieves Brier = {fmt4(best_model['mean_brier'])}, "
    f"while worst ({worst_model['model']}) scores {fmt4(worst_model['mean_brier'])} — "
    f"a gap of {abs(best_model['mean_brier'] - worst_model['mean_brier']):.4f} Brier points."
)

# 2. Evidence effect direction
direction = "improves" if overall_evidence_delta is not None and overall_evidence_delta < 0 else "worsens"
bullets.append(
    f"**Evidence effect overall:** Shared evidence {direction} accuracy relative to closed-book "
    f"(mean SE−CB = {fmt4(overall_evidence_delta)}). "
    f"{most_helped['model']} benefits most (delta = {fmt4(most_helped['delta_se_minus_cb'])}) and "
    f"{most_hurt['model']} is most hurt (delta = {fmt4(most_hurt['delta_se_minus_cb'])})."
)

# 3. Prompt interaction
bullets.append(
    f"**Prompt × Condition interaction:** {prompt_benefit['prompt']} benefits most from evidence "
    f"(SE−CB delta = {fmt4(prompt_benefit['delta_se_minus_cb'])}). "
    f"This is the core hypothesis; the direction {'is consistent with' if prompt_benefit['prompt'] in ['P2', 'P3'] else 'does not support'} "
    f"structured prompts gaining more from evidence."
)

# 4. Market baseline
pct_beating = len(market_beaters) / len(ALL_MODELS) * 100
bullets.append(
    f"**Market baseline:** Market Brier = {fmt4(overall_mkt)}. "
    f"{len(market_beaters)}/{len(ALL_MODELS)} models ({pct_beating:.0f}%) beat the market overall. "
    f"{'Most models outperform the crowd.' if pct_beating > 50 else 'Most models fail to beat the crowd.'}"
)

# 5. Reasoning vs Standard
diff_rs = reasoning_brier - standard_brier if (reasoning_brier and standard_brier) else None
bullets.append(
    f"**Reasoning vs Standard:** Reasoning models average Brier = {fmt4(reasoning_brier)}, "
    f"standard = {fmt4(standard_brier)} (difference = {fmt4(diff_rs)}). "
    f"{'Reasoning models are' if diff_rs is not None and diff_rs < 0 else 'Standard models are'} better on average."
)

# 6. Topic finding
if most_helped_topic:
    bullets.append(
        f"**Topic effects:** '{most_helped_topic['topic']}' benefits most from evidence "
        f"(SE−CB = {fmt4(most_helped_topic['delta_se_minus_cb'])}); "
        f"'{most_hurt_topic['topic']}' is most hurt (SE−CB = {fmt4(most_hurt_topic['delta_se_minus_cb'])})."
    )

# 7. Calibration finding — most extreme
cal_sorted_mean = sorted([c for c in calibration if c["mean_forecast"] is not None],
                          key=lambda x: x["mean_forecast"])
if cal_sorted_mean:
    lowest_mean = cal_sorted_mean[0]
    highest_mean = cal_sorted_mean[-1]
    bullets.append(
        f"**Calibration spread:** {lowest_mean['model']} has the lowest mean forecast ({fmt4(lowest_mean['mean_forecast'])}), "
        f"suggesting conservative/underconfident tendencies; "
        f"{highest_mean['model']} has the highest ({fmt4(highest_mean['mean_forecast'])})."
    )

# 8. Parse reliability
if flagged_parse_cells:
    bullets.append(
        f"**Parse failures:** {len(flagged_parse_cells)} model/prompt/condition cell(s) exceed the 10% failure threshold, "
        f"flagging potential reliability concerns in those subsets."
    )
else:
    bullets.append(
        f"**Parse reliability:** No cells exceed the 10% failure threshold — all models produced parseable outputs in ≥90% of calls."
    )

# 9. Latency
latency_rows_sorted_by_latency = sorted(latency_rows, key=lambda x: x["mean_latency_s"] if x["mean_latency_s"] is not None else 0)
fastest = latency_rows_sorted_by_latency[0]
slowest = latency_rows_sorted_by_latency[-1]
bullets.append(
    f"**Latency range:** {fastest['model']} is fastest ({fmt2(fastest['mean_latency_s'])}s/call), "
    f"{slowest['model']} is slowest ({fmt2(slowest['mean_latency_s'])}s/call). "
    f"Slowest model is {slowest['mean_latency_s']/fastest['mean_latency_s']:.1f}x slower."
)

# 10. Source finding
if source_rows_sorted:
    best_src = source_rows_sorted[0]
    worst_src = source_rows_sorted[-1]
    bullets.append(
        f"**Source effects:** '{best_src['source']}' questions benefit most from evidence "
        f"(delta = {fmt4(best_src['delta_se_minus_cb'])}); "
        f"'{worst_src['source']}' benefits least or is hurt (delta = {fmt4(worst_src['delta_se_minus_cb'])})."
    )

for b in bullets:
    md_lines.append(f"- {b}")
    md_lines.append("")

# Red flags
md_lines.append("## Red Flags / Checks Before Writing the Paper")
md_lines.append("")
red_flags = [
    f"**Sample size per cell is small.** Each (model, prompt, condition) cell has ~114 rows. "
    f"Effect sizes should be treated with caution — small differences in mean Brier may not be statistically robust. "
    f"Run permutation tests or bootstrap CIs before making strong claims.",

    f"**Evidence effect is ambiguous in direction.** With a delta of {fmt4(overall_evidence_delta)}, the evidence effect is "
    f"{'small and possibly noise-level' if overall_evidence_delta is not None and abs(overall_evidence_delta) < 0.01 else 'potentially meaningful but requires statistical testing'}. "
    f"Do not claim a clear benefit or harm without significance testing.",

    f"**freeze_datetime_value used as market Brier.** The market baseline uses the freeze-datetime crowd probability. "
    f"Verify this is the correct pre-freeze snapshot (not a post-resolution price) before comparing to model performance.",

    f"**Model versions and API routing.** Model keys like 'gpt-5.4', 'gemini-3.1-pro', etc., may correspond to non-public or internal model versions. "
    f"Confirm exact model IDs and dates before any publication — routing changes could affect reproducibility.",

    f"**Parse success ≠ calibrated forecast.** Even parsed forecasts may be anchored, hedged, or miscalibrated. "
    f"Inspect raw_response samples for each model to verify forecasts are genuine probability estimates, "
    f"not percentage points (e.g., 50 vs 0.50 errors).",
]
for rf in red_flags:
    md_lines.append(f"- {rf}")
    md_lines.append("")

# Interpretation notes
md_lines.append("## Notes on Cautious Interpretation")
md_lines.append("")
md_lines.append(
    "This is a pilot diagnostic, not a definitive study. Effect sizes are described but not statistically tested. "
    "The 114-question sample covers multiple topics and sources with unequal representation, so aggregate numbers "
    "mask within-topic and within-source heterogeneity. The Prompt×Condition interaction is the central hypothesis "
    "and should be the focus of follow-up analysis with confidence intervals. "
    "Market Brier comparisons assume the freeze-datetime value is a valid baseline; verify this for each source. "
    "Latency numbers include API wait time and may vary significantly across runs."
)
md_lines.append("")
md_lines.append("---")
md_lines.append("*Report generated by analyze.py from pilot_results.json and pilot_questions.json.*")

md_path = os.path.join(ANALYSIS_DIR, "short_diagnostic_report.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))
print(f"  Saved: {md_path}")

print("\n=== ALL DONE ===")
print(f"Output directory: {ANALYSIS_DIR}")
print(f"  summary_tables.csv")
print(f"  short_diagnostic_report.md")
print(f"  key_figures/brier_heatmap.csv")
print(f"  key_figures/evidence_effect.csv")
print(f"  key_figures/calibration.csv")
print(f"  key_figures/topic_brier.csv")
print(f"  key_figures/prompt_condition_means.csv")
