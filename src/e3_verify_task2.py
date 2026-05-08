"""
Task 2: Three alternative mixed-effects / regression specifications.
A) MixedLM: brier ~ condition*prompt + model_type | random intercept by question_id
B) Same fixed effects, crossed random intercepts question_id + model_key (via VC)
C) OLS with C(model_key) + C(question_id) as fixed effects, cluster-robust SEs by question_id
"""

import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
IN_FILE = ROOT / "results" / "merged" / "pilot_results_final.json"

with open(IN_FILE, encoding="utf-8") as f:
    rows = json.load(f)
valid = [r for r in rows if r["parse_success"] and r["brier_score_final"] is not None]

df = pd.DataFrame(valid)
df["condition_binary"] = (df["condition"] == "shared_evidence").astype(float)
df["model_type_binary"] = (df["model_type"] == "reasoning").astype(float)

# Prompt dummies (P1 baseline)
df["P2"] = (df["prompt"] == "P2").astype(float)
df["P3"] = (df["prompt"] == "P3").astype(float)

# Interaction terms
df["cond_P2"] = df["condition_binary"] * df["P2"]
df["cond_P3"] = df["condition_binary"] * df["P3"]

print(f"N observations: {len(df)}")
print(f"Questions: {df['question_id'].nunique()}, Models: {df['model_key'].nunique()}")
print()

def fmt(v, d=4):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "nan"
    return f"{v:.{d}f}"

def sig(p):
    if p is None or np.isnan(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "."
    return "ns"

def print_interaction_table(params, bse, pvals, conf, label):
    print(f"\n{'='*70}")
    print(f"MODEL {label} — Interaction terms (condition*prompt):")
    print(f"{'='*70}")
    terms = ["condition_binary", "P2", "P3", "cond_P2", "cond_P3", "model_type_binary"]
    print(f"{'Term':<22} {'Coef':>8} {'SE':>7} {'CI_lo':>8} {'CI_hi':>8} {'p':>8} {'sig':>4}")
    print("-" * 70)
    for t in terms:
        if t not in params.index: continue
        c   = params[t]
        se_ = bse[t] if t in bse.index else np.nan
        p_  = pvals[t] if t in pvals.index else np.nan
        lo_ = conf.loc[t, 0] if t in conf.index else np.nan
        hi_ = conf.loc[t, 1] if t in conf.index else np.nan
        print(f"{t:<22} {fmt(c):>8} {fmt(se_):>7} {fmt(lo_):>8} {fmt(hi_):>8} {fmt(p_,4):>8} {sig(p_):>4}")

# ── Model A ────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("MODEL A: MixedLM — RE by question_id only")
print("formula: brier ~ condition_binary + P2 + P3 + cond_P2 + cond_P3 + model_type_binary")
print("="*70)

formula_A = "brier_score_final ~ condition_binary + P2 + P3 + cond_P2 + cond_P3 + model_type_binary"
status_A = "FAILED"
try:
    with warnings.catch_warnings(record=True) as caught_A:
        warnings.simplefilter("always")
        mod_A = smf.mixedlm(formula_A, df, groups=df["question_id"])
        res_A = mod_A.fit(reml=True, method="lbfgs")
        status_A = "OK"
        warn_msgs_A = [str(w.message) for w in caught_A if issubclass(w.category, (UserWarning, RuntimeWarning))]

    print(f"Status: {status_A}")
    print(f"Log-likelihood: {res_A.llf:.4f}")
    print(f"Random effect (Group Var): {res_A.cov_re.values[0,0]:.6f}")
    if warn_msgs_A:
        print(f"Warnings: {'; '.join(set(warn_msgs_A))}")
    print_interaction_table(res_A.params, res_A.bse, res_A.pvalues, res_A.conf_int(), "A")
    res_A_summary = res_A
except Exception as e:
    print(f"FAILED: {e}")
    res_A_summary = None

# ── Model B ────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("MODEL B: MixedLM — Crossed RE via variance components (question_id + model_key)")
print("="*70)

status_B = "FAILED"
try:
    with warnings.catch_warnings(record=True) as caught_B:
        warnings.simplefilter("always")
        # Use question_id as primary grouping, model_key as additional VC
        vcf = {"model_key": "0 + C(model_key)"}
        mod_B = smf.mixedlm(
            formula_A, df,
            groups=df["question_id"],
            vc_formula=vcf
        )
        res_B = mod_B.fit(reml=True, method="lbfgs")
        status_B = "OK"
        warn_msgs_B = [str(w.message) for w in caught_B if issubclass(w.category, (UserWarning, RuntimeWarning))]

    print(f"Status: {status_B}")
    print(f"Log-likelihood: {res_B.llf:.4f}")
    if warn_msgs_B:
        print(f"Warnings: {'; '.join(set(warn_msgs_B))}")
    print_interaction_table(res_B.params, res_B.bse, res_B.pvalues, res_B.conf_int(), "B")
    res_B_summary = res_B
except Exception as e:
    print(f"FAILED: {e}")
    res_B_summary = None
    status_B = f"FAILED: {str(e)[:120]}"

# ── Model C ────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("MODEL C: OLS + C(model_key) + C(question_id) fixed effects, cluster-robust SEs")
print("="*70)

formula_C = ("brier_score_final ~ condition_binary + P2 + P3 + cond_P2 + cond_P3 "
             "+ C(model_key) + C(question_id)")
status_C = "FAILED"
try:
    mod_C = smf.ols(formula_C, df)
    res_C_naive = mod_C.fit()
    # Cluster-robust SEs by question_id
    res_C = res_C_naive.get_robustcov_results(
        cov_type="cluster",
        groups=df["question_id"].values,
        use_correction=True
    )
    status_C = "OK"
    print(f"Status: {status_C}")
    print(f"R-squared: {res_C_naive.rsquared:.4f}")
    print(f"N clusters (question_id): {df['question_id'].nunique()}")

    # Extract only the interaction terms (not the 100+ model/question dummies)
    params_C = res_C.params
    bse_C    = res_C.bse
    pval_C   = res_C.pvalues
    conf_C   = res_C.conf_int()
    print_interaction_table(params_C, bse_C, pval_C, conf_C, "C")
    res_C_summary = res_C
except Exception as e:
    print(f"FAILED: {e}")
    res_C_summary = None
    status_C = f"FAILED: {str(e)[:120]}"

# ── Comparison across models ───────────────────────────────────────────────────
print("\n" + "="*70)
print("COMPARISON: condition_binary and interaction terms across A / B / C")
print("="*70)
print(f"{'Term':<22} {'A_coef':>8} {'A_p':>8} {'B_coef':>8} {'B_p':>8} {'C_coef':>8} {'C_p':>8}")
print("-"*70)
terms = ["condition_binary", "P2", "P3", "cond_P2", "cond_P3", "model_type_binary"]
for t in terms:
    def get(res, t):
        if res is None: return ("nan","nan")
        try: return (fmt(res.params[t]), fmt(res.pvalues[t],4))
        except: return ("nan","nan")
    ac, ap = get(res_A_summary, t)
    bc, bp = get(res_B_summary if status_B=="OK" else None, t)
    cc, cp = get(res_C_summary if status_C=="OK" else None, t)
    print(f"{t:<22} {ac:>8} {ap:>8} {bc:>8} {bp:>8} {cc:>8} {cp:>8}")

print()
print("RECOMMENDATION:")
print(" A) MixedLM question_id RE: appropriate if question-level clustering matters")
print("    most; simplest; interpret with caution if Group Var ~0.")
print(" B) Crossed RE: preferred if model-to-model variation is substantial;")
print("    convergence issues common with vc_formula on small group counts.")
print(" C) OLS two-way FE + cluster-robust SEs: most conservative; absorbs all")
print("    question and model variance; cluster SEs correct for within-question")
print("    correlation. RECOMMENDED for this design: no distributional assumptions,")
print("    robust to heteroskedasticity, interaction terms directly interpretable.")
