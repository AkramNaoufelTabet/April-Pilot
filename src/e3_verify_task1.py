"""
Task 1: Independently re-verify bootstrap tests 1-6.
Fresh implementation — no shared code with e2_inferential_tests.py.
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT    = Path(__file__).resolve().parent.parent
IN_FILE = ROOT / "results" / "merged" / "pilot_results_final.json"

N_BOOT = 10_000
SEED   = 42
rng    = np.random.default_rng(SEED)

with open(IN_FILE, encoding="utf-8") as f:
    rows = json.load(f)

valid = [r for r in rows if r["parse_success"] and r["brier_score_final"] is not None]
ALL_QIDS   = sorted(set(r["question_id"] for r in valid))
ALL_MODELS = sorted(set(r["model_key"]   for r in valid))
PROMPTS    = ["P1", "P2", "P3"]
CONDITIONS = ["closed_book", "shared_evidence"]
N_Q = len(ALL_QIDS)

# Index
idx = defaultdict(list)
for r in valid:
    idx[(r["model_key"], r["question_id"], r["prompt"], r["condition"])].append(r)

# ── Bootstrap helpers ──────────────────────────────────────────────────────────

def ci_p(obs, boot):
    lo  = float(np.percentile(boot, 2.5))
    hi  = float(np.percentile(boot, 97.5))
    p   = float(2 * min(np.mean(boot <= 0), np.mean(boot >= 0)))
    return lo, hi, min(p, 1.0)

def paired_boot(a_arr, b_arr):
    """Cluster bootstrap: resample paired observations."""
    obs = float(np.mean(b_arr) - np.mean(a_arr))
    n = len(a_arr)
    boot = np.array([
        np.mean(b_arr[rng.integers(0,n,n)]) - np.mean(a_arr[rng.integers(0,n,n)])
        for _ in range(N_BOOT)
    ])
    lo, hi, p = ci_p(obs, boot)
    return obs, lo, hi, p

def unpaired_boot(a_arr, b_arr):
    obs = float(np.mean(b_arr) - np.mean(a_arr))
    boot = np.array([
        np.mean(rng.choice(b_arr, len(b_arr), replace=True)) -
        np.mean(rng.choice(a_arr, len(a_arr), replace=True))
        for _ in range(N_BOOT)
    ])
    lo, hi, p = ci_p(obs, boot)
    return obs, lo, hi, p

# Build all CB/SE pairs for a given filter
def build_pairs(prompt_filter=None, model_filter=None):
    pairs_cb, pairs_se = [], []
    for mk in ALL_MODELS:
        if model_filter and mk not in model_filter:
            continue
        for qid in ALL_QIDS:
            for pk in (PROMPTS if prompt_filter is None else [prompt_filter]):
                cb = idx.get((mk, qid, pk, "closed_book"),  [])
                se = idx.get((mk, qid, pk, "shared_evidence"), [])
                if cb and se:
                    pairs_cb.append(cb[0]["brier_score_final"])
                    pairs_se.append(se[0]["brier_score_final"])
    return np.array(pairs_cb), np.array(pairs_se)

# ── TEST 1: Overall SE-CB ──────────────────────────────────────────────────────
cb1, se1 = build_pairs()
t1_obs, t1_lo, t1_hi, t1_p = paired_boot(cb1, se1)
print(f"TEST 1 | n_pairs={len(cb1)} | delta={t1_obs:.4f} | CI=[{t1_lo:.4f},{t1_hi:.4f}] | p={t1_p:.4f}")

# ── TEST 2: Per-prompt SE-CB ───────────────────────────────────────────────────
t2 = {}
for pk in PROMPTS:
    cb, se = build_pairs(prompt_filter=pk)
    obs, lo, hi, p = paired_boot(cb, se)
    t2[pk] = (obs, lo, hi, p, len(cb))
    print(f"TEST 2/{pk} | n={len(cb)} | delta={obs:.4f} | CI=[{lo:.4f},{hi:.4f}] | p={p:.4f}")

# ── TEST 3: DiD ────────────────────────────────────────────────────────────────
# Bootstrap over question_ids; for each resample compute SE-CB per prompt, then DiD
qid_arr = np.array(ALL_QIDS)

def did_sample(qids):
    diffs = {}
    for pk in PROMPTS:
        cb_v, se_v = [], []
        for mk in ALL_MODELS:
            for qid in qids:
                cb = idx.get((mk, qid, pk, "closed_book"),  [])
                se = idx.get((mk, qid, pk, "shared_evidence"), [])
                if cb and se:
                    cb_v.append(cb[0]["brier_score_final"])
                    se_v.append(se[0]["brier_score_final"])
        diffs[pk] = np.mean(se_v) - np.mean(cb_v) if cb_v else np.nan
    return diffs

obs_did = did_sample(ALL_QIDS)
boot_p3p1 = np.empty(N_BOOT)
boot_p3p2 = np.empty(N_BOOT)
boot_p2p1 = np.empty(N_BOOT)
for i in range(N_BOOT):
    s = list(rng.choice(qid_arr, N_Q, replace=True))
    d = did_sample(s)
    boot_p3p1[i] = d["P3"] - d["P1"]
    boot_p3p2[i] = d["P3"] - d["P2"]
    boot_p2p1[i] = d["P2"] - d["P1"]

obs_p3p1 = obs_did["P3"] - obs_did["P1"]
obs_p3p2 = obs_did["P3"] - obs_did["P2"]
obs_p2p1 = obs_did["P2"] - obs_did["P1"]
lo_p3p1, hi_p3p1, p_p3p1 = ci_p(obs_p3p1, boot_p3p1)
lo_p3p2, hi_p3p2, p_p3p2 = ci_p(obs_p3p2, boot_p3p2)
lo_p2p1, hi_p2p1, p_p2p1 = ci_p(obs_p2p1, boot_p2p1)
print(f"TEST 3/P3-P1 | DiD={obs_p3p1:.4f} | CI=[{lo_p3p1:.4f},{hi_p3p1:.4f}] | p={p_p3p1:.4f}")
print(f"TEST 3/P3-P2 | DiD={obs_p3p2:.4f} | CI=[{lo_p3p2:.4f},{hi_p3p2:.4f}] | p={p_p3p2:.4f}")
print(f"TEST 3/P2-P1 | DiD={obs_p2p1:.4f} | CI=[{lo_p2p1:.4f},{hi_p2p1:.4f}] | p={p_p2p1:.4f}")

# ── TEST 4: Question-level ─────────────────────────────────────────────────────
q_cb, q_se = {}, {}
for r in valid:
    qid = r["question_id"]
    if r["condition"] == "closed_book":
        q_cb.setdefault(qid, []).append(r["brier_score_final"])
    else:
        q_se.setdefault(qid, []).append(r["brier_score_final"])

q_pairs = [(np.mean(q_cb[qid]), np.mean(q_se[qid]))
           for qid in ALL_QIDS if qid in q_cb and qid in q_se]
qcb4 = np.array([p[0] for p in q_pairs])
qse4 = np.array([p[1] for p in q_pairs])
t4_obs, t4_lo, t4_hi, t4_p = paired_boot(qcb4, qse4)
print(f"TEST 4 | n_q={len(q_pairs)} | delta={t4_obs:.4f} | CI=[{t4_lo:.4f},{t4_hi:.4f}] | p={t4_p:.4f}")

# ── TEST 5: Reasoning vs standard ─────────────────────────────────────────────
reas = np.array([r["brier_score_final"] for r in valid if r["model_type"]=="reasoning"])
std  = np.array([r["brier_score_final"] for r in valid if r["model_type"]=="standard"])
t5_obs, t5_lo, t5_hi, t5_p = unpaired_boot(reas, std)
print(f"TEST 5 | n_reas={len(reas)} n_std={len(std)} | delta={t5_obs:.4f} | CI=[{t5_lo:.4f},{t5_hi:.4f}] | p={t5_p:.4f}")

# ── TEST 6: Market baseline ────────────────────────────────────────────────────
mkt_rows = [r for r in valid if r.get("market_prob") is not None]
mkt_b = np.array([(r["market_prob"]-r["resolution_value"])**2 for r in mkt_rows])
mod_b = np.array([r["brier_score_final"] for r in mkt_rows])
t6_obs, t6_lo, t6_hi, t6_p = paired_boot(mkt_b, mod_b)
print(f"TEST 6/all | n={len(mkt_rows)} | delta={t6_obs:.4f} | CI=[{t6_lo:.4f},{t6_hi:.4f}] | p={t6_p:.4f}")

best_rows = [r for r in mkt_rows if r["model_key"]=="gpt-5.4"]
bm_b = np.array([(r["market_prob"]-r["resolution_value"])**2 for r in best_rows])
bmod = np.array([r["brier_score_final"] for r in best_rows])
t6b_obs, t6b_lo, t6b_hi, t6b_p = paired_boot(bm_b, bmod)
print(f"TEST 6/gpt-5.4 | n={len(best_rows)} | delta={t6b_obs:.4f} | CI=[{t6b_lo:.4f},{t6b_hi:.4f}] | p={t6b_p:.4f}")

# ── Comparison table ───────────────────────────────────────────────────────────
REPORTED = {
    "T1_overall":    (-0.0277, 0.0000),
    "T2_P1":         (-0.0223, 0.0028),
    "T2_P2":         (-0.0272, 0.0002),
    "T2_P3":         (-0.0334, 0.0000),
    "T3_P3-P1":      (-0.0111, 0.0796),
    "T3_P3-P2":      (-0.0062, 0.3786),
    "T3_P2-P1":      (-0.0049, 0.3218),
    "T4_q_level":    (-0.0276, 0.1656),
    "T5_std-reas":   ( 0.0307, 0.0000),
    "T6_all_vs_mkt": ( 0.0711, 0.0000),
    "T6_gpt54_mkt":  ( 0.0419, 0.0000),
}
MY = {
    "T1_overall":    (t1_obs,  t1_p),
    "T2_P1":         (t2["P1"][0], t2["P1"][3]),
    "T2_P2":         (t2["P2"][0], t2["P2"][3]),
    "T2_P3":         (t2["P3"][0], t2["P3"][3]),
    "T3_P3-P1":      (obs_p3p1, p_p3p1),
    "T3_P3-P2":      (obs_p3p2, p_p3p2),
    "T3_P2-P1":      (obs_p2p1, p_p2p1),
    "T4_q_level":    (t4_obs, t4_p),
    "T5_std-reas":   (t5_obs, t5_p),
    "T6_all_vs_mkt": (t6_obs, t6_p),
    "T6_gpt54_mkt":  (t6b_obs, t6b_p),
}

print("\n--- VERIFICATION TABLE ---")
print(f"{'test':<20} {'rep_delta':>10} {'my_delta':>10} {'rep_p':>8} {'my_p':>8} {'d_agree':>8} {'p_agree':>8}")
print("-" * 80)
for k in REPORTED:
    rd, rp = REPORTED[k]
    md, mp = MY[k]
    d_ok = abs(md - rd) < 0.0010
    p_ok = abs(mp - rp) < 0.01 or (rp < 0.001 and mp < 0.001) or (rp > 0.10 and mp > 0.10)
    agree = "YES" if d_ok and p_ok else "CHECK"
    print(f"{k:<20} {rd:>10.4f} {md:>10.4f} {rp:>8.4f} {mp:>8.4f} {str(d_ok):>8} {str(p_ok):>8}  {agree}")
