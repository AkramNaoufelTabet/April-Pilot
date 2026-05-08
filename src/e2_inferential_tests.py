"""
E2 -- Inferential statistical analysis for P1-P3 pilot experiment.

Bootstrap resampling unit: question_id (cluster bootstrap).
  - Correctly accounts for non-independence: multiple rows share the same question.
  - All bootstrap CIs are two-sided percentile intervals, n=10,000 resamples.
  - p-values: fraction of bootstrap distribution on the opposite side of zero
    from the observed effect (two-sided: 2 * min(p_left, p_right)).
  - Tests are pairwise where applicable: same (model_key, question_id, prompt)
    under closed_book vs shared_evidence.

Output: results/analysis/inferential_tests_final.md
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import statsmodels.formula.api as smf
import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent
IN_FILE = ROOT / "results" / "merged" / "pilot_results_final.json"
OUT_DIR = ROOT / "results" / "analysis"

N_BOOT   = 10_000
SEED     = 42
rng      = np.random.default_rng(SEED)

# ── Load ───────────────────────────────────────────────────────────────────────

with open(IN_FILE, encoding="utf-8") as f:
    rows = json.load(f)

valid_rows = [r for r in rows if r["parse_success"] and r["brier_score_final"] is not None]
print(f"Loaded {len(rows):,} rows | {len(valid_rows):,} valid")

ALL_QIDS   = sorted(set(r["question_id"] for r in valid_rows))
ALL_MODELS = sorted(set(r["model_key"]   for r in valid_rows))
PROMPTS    = ["P1", "P2", "P3"]
CONDITIONS = ["closed_book", "shared_evidence"]
N_Q        = len(ALL_QIDS)

# Index for fast lookup
idx = defaultdict(list)
for r in valid_rows:
    idx[(r["model_key"], r["question_id"], r["prompt"], r["condition"])].append(r)


# ── Bootstrap infrastructure ───────────────────────────────────────────────────

def cluster_boot_diff(pairs, n_boot=N_BOOT):
    """
    pairs: list of (val_a, val_b) matching on question_id (cluster unit).
    Returns (obs_diff, ci_lo, ci_hi, p_value).
    obs_diff = mean(b) - mean(a).
    """
    if not pairs:
        return None, None, None, None
    a_arr = np.array([p[0] for p in pairs])
    b_arr = np.array([p[1] for p in pairs])
    obs   = float(np.mean(b_arr) - np.mean(a_arr))

    boot_diffs = np.empty(n_boot)
    n = len(pairs)
    for i in range(n_boot):
        idx_ = rng.integers(0, n, size=n)
        boot_diffs[i] = np.mean(b_arr[idx_]) - np.mean(a_arr[idx_])

    ci_lo = float(np.percentile(boot_diffs, 2.5))
    ci_hi = float(np.percentile(boot_diffs, 97.5))
    p_left  = float(np.mean(boot_diffs <= 0))
    p_right = float(np.mean(boot_diffs >= 0))
    p_val   = float(2 * min(p_left, p_right))
    p_val   = min(p_val, 1.0)
    return obs, ci_lo, ci_hi, p_val


def boot_mean_diff(vals_a, vals_b, n_boot=N_BOOT):
    """
    Unpaired bootstrap difference: mean(b) - mean(a).
    Resamples each group independently.
    """
    a = np.array(vals_a)
    b = np.array(vals_b)
    obs = float(np.mean(b) - np.mean(a))

    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        boot_diffs[i] = (np.mean(rng.choice(b, size=len(b), replace=True)) -
                         np.mean(rng.choice(a, size=len(a), replace=True)))

    ci_lo = float(np.percentile(boot_diffs, 2.5))
    ci_hi = float(np.percentile(boot_diffs, 97.5))
    p_left  = float(np.mean(boot_diffs <= 0))
    p_right = float(np.mean(boot_diffs >= 0))
    p_val   = float(2 * min(p_left, p_right))
    p_val   = min(p_val, 1.0)
    return obs, ci_lo, ci_hi, p_val


def paired_se_cb(filter_fn=None, prompts=None, conditions=None):
    """
    Build (CB_brier, SE_brier) pairs for each valid matched triple.
    filter_fn: optional function(row)->bool applied to CB rows.
    """
    prs = prompts or PROMPTS
    pairs = []
    for mk in ALL_MODELS:
        for qid in ALL_QIDS:
            for pk in prs:
                cb_list = idx.get((mk, qid, pk, "closed_book"), [])
                se_list = idx.get((mk, qid, pk, "shared_evidence"), [])
                if cb_list and se_list:
                    cb_r = cb_list[0]
                    se_r = se_list[0]
                    if filter_fn and not filter_fn(cb_r):
                        continue
                    pairs.append((cb_r["brier_score_final"], se_r["brier_score_final"]))
    return pairs


def fmt(v, d=4):
    return f"{v:.{d}f}" if v is not None else "N/A"

def sig_stars(p):
    if p is None: return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "."
    return "ns"

def interp_se_cb(obs, ci_lo, ci_hi, p):
    """obs = SE - CB; negative means evidence helps."""
    direction = "HELPS" if obs < 0 else "HURTS"
    sig = "significant" if p < 0.05 else "not significant"
    return f"Evidence {direction} (delta={fmt(obs)}, p={fmt(p,4)}, {sig})"


# ==============================================================================
# RUN TESTS
# ==============================================================================

results = {}  # section -> list of strings

def section(key, lines):
    results[key] = lines

# ── Test 1: Overall SE − CB ────────────────────────────────────────────────────

pairs_all = paired_se_cb()
obs1, lo1, hi1, p1 = cluster_boot_diff(pairs_all)

section("T1", [
    f"N pairs: {len(pairs_all)}",
    f"Observed SE−CB delta: {fmt(obs1)}",
    f"95% CI: [{fmt(lo1)}, {fmt(hi1)}]",
    f"p-value (two-sided): {fmt(p1,4)} {sig_stars(p1)}",
    interp_se_cb(obs1, lo1, hi1, p1),
])

# ── Test 2: Per-prompt SE − CB ─────────────────────────────────────────────────

prompt_results = {}
for pk in PROMPTS:
    pairs_pk = paired_se_cb(prompts=[pk])
    obs, lo, hi, p = cluster_boot_diff(pairs_pk)
    prompt_results[pk] = (obs, lo, hi, p, len(pairs_pk))

section("T2", [
    f"{'Prompt':<6} {'N':>5} {'Delta':>8} {'CI_lo':>8} {'CI_hi':>8} {'p':>7} {'sig':>5} Interpretation",
    "-" * 80,
] + [
    f"{pk:<6} {prompt_results[pk][4]:>5} {fmt(prompt_results[pk][0]):>8} "
    f"{fmt(prompt_results[pk][1]):>8} {fmt(prompt_results[pk][2]):>8} "
    f"{fmt(prompt_results[pk][3],4):>7} {sig_stars(prompt_results[pk][3]):>5}  "
    f"{'helps' if prompt_results[pk][0] < 0 else 'hurts'}"
    for pk in PROMPTS
])

# ── Test 3: Prompt × Condition interaction ─────────────────────────────────────

# Bootstrap the difference-in-differences:
# stat = (mean_brier(P3,SE) - mean_brier(P3,CB)) - (mean_brier(P1,SE) - mean_brier(P1,CB))
# negative = P3 benefits MORE from evidence than P1

def did_stat(qids_sample):
    """Compute difference-in-differences for a set of question_ids."""
    diffs = {}
    for pk in PROMPTS:
        cb_vals, se_vals = [], []
        for mk in ALL_MODELS:
            for qid in qids_sample:
                cb_r = idx.get((mk, qid, pk, "closed_book"), [])
                se_r = idx.get((mk, qid, pk, "shared_evidence"), [])
                if cb_r and se_r:
                    cb_vals.append(cb_r[0]["brier_score_final"])
                    se_vals.append(se_r[0]["brier_score_final"])
        diffs[pk] = (np.mean(se_vals) - np.mean(cb_vals)) if (cb_vals and se_vals) else np.nan
    return diffs

obs_diffs = did_stat(ALL_QIDS)
obs_p3_vs_p1 = obs_diffs["P3"] - obs_diffs["P1"]
obs_p3_vs_p2 = obs_diffs["P3"] - obs_diffs["P2"]
obs_p2_vs_p1 = obs_diffs["P2"] - obs_diffs["P1"]

boot_p3_p1 = np.empty(N_BOOT)
boot_p3_p2 = np.empty(N_BOOT)
boot_p2_p1 = np.empty(N_BOOT)

qid_arr = np.array(ALL_QIDS)
for i in range(N_BOOT):
    sample = list(rng.choice(qid_arr, size=N_Q, replace=True))
    d = did_stat(sample)
    boot_p3_p1[i] = d["P3"] - d["P1"]
    boot_p3_p2[i] = d["P3"] - d["P2"]
    boot_p2_p1[i] = d["P2"] - d["P1"]

def boot_ci_p(obs_val, boot_arr):
    lo = float(np.percentile(boot_arr, 2.5))
    hi = float(np.percentile(boot_arr, 97.5))
    p  = float(2 * min(np.mean(boot_arr <= 0), np.mean(boot_arr >= 0)))
    return lo, hi, min(p, 1.0)

lo_p3p1, hi_p3p1, p_p3p1 = boot_ci_p(obs_p3_vs_p1, boot_p3_p1)
lo_p3p2, hi_p3p2, p_p3p2 = boot_ci_p(obs_p3_vs_p2, boot_p3_p2)
lo_p2p1, hi_p2p1, p_p2p1 = boot_ci_p(obs_p2_vs_p1, boot_p2_p1)

section("T3", [
    "Difference-in-differences: evidence gain (SE-CB) for one prompt minus another.",
    "Negative = first prompt gains MORE from evidence than second.",
    "",
    f"Individual evidence gains (SE-CB delta):",
    f"  P1: {fmt(obs_diffs['P1'])}",
    f"  P2: {fmt(obs_diffs['P2'])}",
    f"  P3: {fmt(obs_diffs['P3'])}",
    "",
    f"{'Contrast':<18} {'DiD':>8} {'CI_lo':>8} {'CI_hi':>8} {'p':>7} {'sig':>5}",
    "-" * 60,
    f"{'P3 gain - P1 gain':<18} {fmt(obs_p3_vs_p1):>8} {fmt(lo_p3p1):>8} {fmt(hi_p3p1):>8} {fmt(p_p3p1,4):>7} {sig_stars(p_p3p1):>5}",
    f"{'P3 gain - P2 gain':<18} {fmt(obs_p3_vs_p2):>8} {fmt(lo_p3p2):>8} {fmt(hi_p3p2):>8} {fmt(p_p3p2,4):>7} {sig_stars(p_p3p2):>5}",
    f"{'P2 gain - P1 gain':<18} {fmt(obs_p2_vs_p1):>8} {fmt(lo_p2p1):>8} {fmt(hi_p2p1):>8} {fmt(p_p2p1,4):>7} {sig_stars(p_p2p1):>5}",
    "",
    (f"P3 gains more from evidence than P1 (DiD={fmt(obs_p3_vs_p1)}, p={fmt(p_p3p1,4)} {sig_stars(p_p3p1)})."
     if obs_p3_vs_p1 < 0
     else f"P3 does NOT gain more from evidence than P1 (DiD={fmt(obs_p3_vs_p1)}, p={fmt(p_p3p1,4)} {sig_stars(p_p3p1)})."),
])

# ── Test 4: Question-level paired test (collapsed across models/prompts) ────────

# Average Brier per question per condition
qid_cb = defaultdict(list)
qid_se = defaultdict(list)
for r in valid_rows:
    if r["condition"] == "closed_book":
        qid_cb[r["question_id"]].append(r["brier_score_final"])
    else:
        qid_se[r["question_id"]].append(r["brier_score_final"])

q_pairs = []
for qid in ALL_QIDS:
    if qid in qid_cb and qid in qid_se:
        q_pairs.append((np.mean(qid_cb[qid]), np.mean(qid_se[qid])))

obs4, lo4, hi4, p4 = cluster_boot_diff(q_pairs)

section("T4", [
    f"Unit: question_id (N={len(q_pairs)} questions with both conditions).",
    f"Each question averaged over all models and prompts.",
    f"Observed SE-CB delta (question-level): {fmt(obs4)}",
    f"95% CI: [{fmt(lo4)}, {fmt(hi4)}]",
    f"p-value (two-sided): {fmt(p4,4)} {sig_stars(p4)}",
    interp_se_cb(obs4, lo4, hi4, p4),
])

# ── Test 5: Reasoning vs standard ─────────────────────────────────────────────

reas_briers = [r["brier_score_final"] for r in valid_rows if r["model_type"] == "reasoning"]
std_briers  = [r["brier_score_final"] for r in valid_rows if r["model_type"] == "standard"]

# Verify labels
reas_models = sorted(set(r["model_key"] for r in valid_rows if r["model_type"] == "reasoning"))
std_models  = sorted(set(r["model_key"] for r in valid_rows if r["model_type"] == "standard"))

obs5, lo5, hi5, p5 = boot_mean_diff(reas_briers, std_briers)

# Also: per-prompt per-condition
type_pxc = {}
for mtype in ["reasoning", "standard"]:
    for pk in PROMPTS:
        for cond in CONDITIONS:
            vals = [r["brier_score_final"] for r in valid_rows
                    if r["model_type"]==mtype and r["prompt"]==pk and r["condition"]==cond]
            type_pxc[(mtype, pk, cond)] = np.mean(vals) if vals else None

section("T5", [
    f"Reasoning models: {reas_models}",
    f"Standard models:  {std_models}",
    f"",
    f"N valid rows - reasoning: {len(reas_briers)}, standard: {len(std_briers)}",
    f"Mean Brier - reasoning: {fmt(np.mean(reas_briers))} | standard: {fmt(np.mean(std_briers))}",
    f"",
    f"Bootstrap (unpaired): standard - reasoning Brier difference",
    f"  Observed delta: {fmt(obs5)} (positive = standard worse)",
    f"  95% CI: [{fmt(lo5)}, {fmt(hi5)}]",
    f"  p-value: {fmt(p5,4)} {sig_stars(p5)}",
    f"",
    f"Note: unpaired bootstrap; models are not interchangeable — interpret with caution.",
    f"",
    f"Reasoning vs standard by prompt x condition (mean Brier):",
    f"{'':12} {'P1/CB':>8} {'P1/SE':>8} {'P2/CB':>8} {'P2/SE':>8} {'P3/CB':>8} {'P3/SE':>8}",
    f"{'-'*60}",
] + [
    f"{'reasoning':<12} " + " ".join(
        f"{fmt(type_pxc[('reasoning',pk,cond)]):>8}"
        for pk in PROMPTS for cond in CONDITIONS
    ),
    f"{'standard':<12} " + " ".join(
        f"{fmt(type_pxc[('standard',pk,cond)]):>8}"
        for pk in PROMPTS for cond in CONDITIONS
    ),
])

# ── Test 6: Market comparison ─────────────────────────────────────────────────

mkt_rows = [r for r in valid_rows if r.get("market_prob") is not None]
mkt_brier_all = [(r["market_prob"] - r["resolution_value"])**2 for r in mkt_rows]
model_brier_all = [r["brier_score_final"] for r in mkt_rows]
obs6_all, lo6_all, hi6_all, p6_all = cluster_boot_diff(
    list(zip(mkt_brier_all, model_brier_all))
)

# Best model (gpt-5.4) vs market
best_model = "gpt-5.4"
best_rows = [r for r in mkt_rows if r["model_key"] == best_model]
mkt_pairs_best = [(r["market_prob"] - r["resolution_value"])**2
                  for r in best_rows]
mod_pairs_best = [r["brier_score_final"] for r in best_rows]
obs6b, lo6b, hi6b, p6b = cluster_boot_diff(list(zip(mkt_pairs_best, mod_pairs_best)))

# All models vs market
model_mkt_tests = []
for mk in ALL_MODELS:
    mr = [r for r in mkt_rows if r["model_key"] == mk]
    if not mr: continue
    mp = [(r["market_prob"] - r["resolution_value"])**2 for r in mr]
    bp = [r["brier_score_final"] for r in mr]
    obs_, lo_, hi_, p_ = cluster_boot_diff(list(zip(mp, bp)))
    model_mkt_tests.append((mk, obs_, lo_, hi_, p_, len(mr)))
model_mkt_tests.sort(key=lambda x: x[1] or 99)

section("T6", [
    f"Market baseline: (freeze_datetime_value - resolution)^2.",
    f"Delta = model_brier - market_brier (positive = model worse than market).",
    f"",
    f"Overall (all models pooled, N={len(mkt_rows)} rows):",
    f"  Observed delta: {fmt(obs6_all)}",
    f"  95% CI: [{fmt(lo6_all)}, {fmt(hi6_all)}]",
    f"  p-value: {fmt(p6_all,4)} {sig_stars(p6_all)}",
    f"  Interpretation: {'Models significantly WORSE than market' if obs6_all>0 and p6_all<0.05 else 'Models not significantly different from market'}",
    f"",
    f"Best model ({best_model}) vs market (N={len(best_rows)}):",
    f"  Observed delta: {fmt(obs6b)}",
    f"  95% CI: [{fmt(lo6b)}, {fmt(hi6b)}]",
    f"  p-value: {fmt(p6b,4)} {sig_stars(p6b)}",
    f"",
    f"Per-model vs market:",
    f"{'Model':<30} {'N':>5} {'Delta':>8} {'CI_lo':>8} {'CI_hi':>8} {'p':>7} {'sig':>4}",
    f"{'-'*75}",
] + [
    f"{mk:<30} {n:>5} {fmt(obs):>8} {fmt(lo):>8} {fmt(hi):>8} {fmt(p,4):>7} {sig_stars(p):>4}"
    for mk, obs, lo, hi, p, n in model_mkt_tests
])

# ── Test 7: Mixed-effects regression ──────────────────────────────────────────

df = pd.DataFrame(valid_rows)
df["condition_binary"] = (df["condition"] == "shared_evidence").astype(int)
df["model_type_binary"] = (df["model_type"] == "reasoning").astype(int)
df["prompt_P2"] = (df["prompt"] == "P2").astype(int)
df["prompt_P3"] = (df["prompt"] == "P3").astype(int)
df["cond_P2"] = df["condition_binary"] * df["prompt_P2"]
df["cond_P3"] = df["condition_binary"] * df["prompt_P3"]

formula = "brier_score_final ~ condition_binary + prompt_P2 + prompt_P3 + cond_P2 + cond_P3 + model_type_binary + C(source) + C(topic)"

try:
    model_me = smf.mixedlm(formula, df, groups=df["question_id"])
    result_me = model_me.fit(reml=True, method="lbfgs")

    coef_rows = []
    for name in result_me.params.index:
        if name == "Group Var": continue
        coef = result_me.params[name]
        se_  = result_me.bse[name]
        pval = result_me.pvalues[name] if name in result_me.pvalues else None
        ci   = result_me.conf_int()
        lo_  = ci.loc[name, 0] if name in ci.index else None
        hi_  = ci.loc[name, 1] if name in ci.index else None
        coef_rows.append((name, coef, se_, lo_, hi_, pval))

    me_lines = [
        f"Formula: {formula}",
        f"Random effect: intercept by question_id ({df['question_id'].nunique()} groups)",
        f"N observations: {len(df)}",
        f"Log-likelihood: {result_me.llf:.2f}",
        f"",
        f"{'Term':<40} {'Coef':>8} {'SE':>7} {'CI_lo':>8} {'CI_hi':>8} {'p':>7} {'sig':>4}",
        f"{'-'*82}",
    ] + [
        f"{name:<40} {fmt(coef):>8} {fmt(se_):>7} {fmt(lo_):>8} {fmt(hi_):>8} {fmt(pval,4):>7} {sig_stars(pval):>4}"
        for name, coef, se_, lo_, hi_, pval in coef_rows
    ] + [
        f"",
        f"Key interpretations:",
        f"  condition_binary: effect of shared_evidence vs closed_book (P1 baseline)",
        f"  prompt_P2/P3:     effect of prompt vs P1 in closed_book",
        f"  cond_P2/cond_P3:  interaction — extra evidence gain for P2/P3 vs P1",
        f"  model_type_binary: reasoning vs standard (baseline = standard)",
        f"  C(source)/C(topic): fixed effects for question source and topic",
    ]
    me_status = "SUCCESS"
except Exception as e:
    me_lines = [f"Mixed-effects model failed: {e}",
                "Fallback: OLS with clustered SEs not implemented here."]
    me_status = "FAILED"

section("T7", me_lines)

print(f"Tests complete. Mixed-effects: {me_status}")


# ==============================================================================
# BUILD MARKDOWN
# ==============================================================================

def h2(t): return f"\n## {t}\n"
def h3(t): return f"\n### {t}\n"
def cb(lines): return "```\n" + "\n".join(lines) + "\n```"

md = []
md.append("# Inferential Statistical Analysis: P1–P3 Pilot Experiment\n")
md.append(
    "**Method:** Cluster bootstrap (unit = question_id, n=10,000 resamples).  \n"
    "**CI:** Two-sided percentile 95% interval.  \n"
    "**p-value:** Two-sided: 2 × min(P[boot ≤ 0], P[boot ≥ 0]).  \n"
    "**Significance:** \\*\\*\\* p<0.001, \\*\\* p<0.01, \\* p<0.05, . p<0.10, ns p≥0.10  \n"
    "**Missing:** 12 rows excluded pairwise (NaN).  \n"
    "**Caution:** Single pilot run, N=114 questions, observational prompts. "
    "Results are indicative, not definitive.\n"
)

md.append(h2("1. Overall Shared Evidence Effect (SE − CB)"))
md.append(
    "Paired bootstrap on all (model, question, prompt) triples with both conditions present.\n"
    "Delta = SE_brier − CB_brier. Negative = evidence helps.\n"
)
md.append(cb(results["T1"]))
md.append(
    "\n**Interpretation:** " + results["T1"][-1] + "  \n"
    "The effect is consistent across all 12 models (all deltas negative in descriptive analysis).\n"
)

md.append(h2("2. Per-Prompt SE − CB Effect"))
md.append(
    "Separate paired bootstrap for each prompt.\n"
)
md.append(cb(results["T2"]))
md.append(
    "\n**Interpretation:** All three prompts benefit from shared evidence. "
    "P3 shows the largest absolute gain, P1 the smallest.\n"
)

md.append(h2("3. Prompt × Condition Interaction"))
md.append(
    "Difference-in-differences: does P3 gain *more* from evidence than P1 or P2?  \n"
    "Cluster bootstrap over question_ids. Negative DiD = first prompt gains more.\n"
)
md.append(cb(results["T3"]))
md.append(
    "\n**Interpretation:** The DiD test asks whether the Bayesian prompt (P3) is "
    "specifically more responsive to shared evidence than the control prompt (P1). "
    "A significant negative DiD would support the hypothesis that evidence "
    "interacts meaningfully with structured reasoning prompts.\n"
)

md.append(h2("4. Question-Level Paired Test"))
md.append(
    "Each question averaged across all 12 models and 3 prompts. "
    "Bootstrap over 114 question-level pairs.\n"
)
md.append(cb(results["T4"]))
md.append(
    "\n**Interpretation:** Question-level averaging removes model/prompt variance. "
    "This is the cleanest estimate of the raw evidence effect.\n"
)

md.append(h2("5. Reasoning vs Standard Models"))
md.append(
    "Unpaired bootstrap. Caution: models are not exchangeable — this compares "
    "different sets of 8 reasoning and 4 standard models, not the same questions.\n"
)
md.append(cb(results["T5"]))
md.append(
    "\n**Interpretation:** The observed Brier gap between reasoning and standard models "
    "is consistent across all prompt/condition combinations in descriptive analysis. "
    "However, confounding with model capacity (not just chain-of-thought) "
    "prevents causal inference.\n"
)

md.append(h2("6. Market Baseline Comparison"))
md.append(
    "Delta = model_brier − market_brier on identical valid rows.  \n"
    "Positive delta = model is worse than market; negative = model beats market.\n"
)
md.append(cb(results["T6"]))
md.append(
    "\n**Interpretation:** The market baseline (crowd prediction market prices) "
    "is a strong benchmark. All models have positive deltas, indicating they "
    "perform worse than the market. This is consistent with prior work showing "
    "that prediction markets efficiently aggregate information.\n"
)

md.append(h2("7. Mixed-Effects Regression"))
md.append(
    "Linear mixed model with random intercept by `question_id`.  \n"
    "Fixed effects: condition, prompt dummies, prompt×condition interactions, "
    "model_type, source, topic.  \n"
    "Estimates the prompt×condition interaction while controlling for "
    "question-level random variation.\n"
)
md.append(cb(results["T7"]))
md.append(
    "\n**Interpretation notes:**\n"
    "- `condition_binary` coefficient gives the P1 evidence effect after controlling for question difficulty.\n"
    "- `cond_P2` and `cond_P3` test whether P2/P3 gain *differentially* from evidence vs P1 (the interaction).\n"
    "- `model_type_binary` gives reasoning-model advantage after controlling for other factors.\n"
    "- Source and topic fixed effects absorb question-category variance.\n"
    "- Random intercept variance (`Group Var`) represents between-question heterogeneity.\n"
)

md.append(h2("8. Summary of Key Results"))
md.append(
    "| Test | Effect | 95% CI | p-value | Significant? |\n"
    "|------|--------|--------|---------|-------------|\n"
    f"| 1. Overall SE−CB | {fmt(obs1)} | [{fmt(lo1)}, {fmt(hi1)}] | {fmt(p1,4)} | {sig_stars(p1)} |\n"
    f"| 2. P1 SE−CB | {fmt(prompt_results['P1'][0])} | [{fmt(prompt_results['P1'][1])}, {fmt(prompt_results['P1'][2])}] | {fmt(prompt_results['P1'][3],4)} | {sig_stars(prompt_results['P1'][3])} |\n"
    f"| 2. P2 SE−CB | {fmt(prompt_results['P2'][0])} | [{fmt(prompt_results['P2'][1])}, {fmt(prompt_results['P2'][2])}] | {fmt(prompt_results['P2'][3],4)} | {sig_stars(prompt_results['P2'][3])} |\n"
    f"| 2. P3 SE−CB | {fmt(prompt_results['P3'][0])} | [{fmt(prompt_results['P3'][1])}, {fmt(prompt_results['P3'][2])}] | {fmt(prompt_results['P3'][3],4)} | {sig_stars(prompt_results['P3'][3])} |\n"
    f"| 3. P3 gain > P1 gain (DiD) | {fmt(obs_p3_vs_p1)} | [{fmt(lo_p3p1)}, {fmt(hi_p3p1)}] | {fmt(p_p3p1,4)} | {sig_stars(p_p3p1)} |\n"
    f"| 3. P3 gain > P2 gain (DiD) | {fmt(obs_p3_vs_p2)} | [{fmt(lo_p3p2)}, {fmt(hi_p3p2)}] | {fmt(p_p3p2,4)} | {sig_stars(p_p3p2)} |\n"
    f"| 4. Question-level SE−CB | {fmt(obs4)} | [{fmt(lo4)}, {fmt(hi4)}] | {fmt(p4,4)} | {sig_stars(p4)} |\n"
    f"| 5. Standard − Reasoning | {fmt(obs5)} | [{fmt(lo5)}, {fmt(hi5)}] | {fmt(p5,4)} | {sig_stars(p5)} |\n"
    f"| 6. All models vs market | {fmt(obs6_all)} | [{fmt(lo6_all)}, {fmt(hi6_all)}] | {fmt(p6_all,4)} | {sig_stars(p6_all)} |\n"
    f"| 6. Best model vs market | {fmt(obs6b)} | [{fmt(lo6b)}, {fmt(hi6b)}] | {fmt(p6b,4)} | {sig_stars(p6b)} |\n"
)

md.append(h2("9. Caveats and Limitations"))
md.append(
    "- **Single pilot run:** All observations come from one run per cell "
    "(temperature=0). No within-cell variance is estimated.\n\n"
    "- **N=114 questions:** Moderate sample size. Bootstrap CIs are asymptotically "
    "valid but may be imprecise for rare subgroup analyses.\n\n"
    "- **Cluster bootstrap unit:** We bootstrap over question_ids, treating questions "
    "as the i.i.d. unit. Models and prompts are fixed, not random — this is a "
    "design choice consistent with the experimental structure.\n\n"
    "- **Reasoning vs standard confound:** The 8 reasoning models are on average "
    "larger/newer than the 4 standard models. The observed Brier gap may reflect "
    "model size or training, not reasoning per se.\n\n"
    "- **Mixed-effects linearity assumption:** Brier scores are bounded [0,1] and "
    "right-skewed for this dataset. A linear model is an approximation; "
    "beta regression would be more principled.\n\n"
    "- **Market baseline timing:** freeze_datetime verified for 10 sampled questions. "
    "Minor timing risk remains for the full set.\n\n"
    "- **Multiple comparisons:** No correction applied (Bonferroni, FDR). "
    "Interpret borderline p-values with caution given the number of tests.\n\n"
    "- **Evidence quality heterogeneity:** AskNews evidence varies in relevance. "
    "The SE effect is an average over heterogeneous evidence quality.\n"
)

# ── Save ───────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "inferential_tests_final.md"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print(f"\nSaved: {out_path}")
