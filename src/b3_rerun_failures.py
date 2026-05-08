"""
B3 — Rerun the 172 rows from results/rerun_plan.json that failed parsing.

Changes vs b2_run_experiment.py:
  - raw_response stored without truncation (full response, up to 200000 chars)
  - max_tokens = 65536 for truncation_likely rows, 32768 for all others
  - outputs to results/reruns/rerun_outputs.jsonl (append — crash-safe)
  - v2 parser applied at collection time
  - never touches pilot_results.json or pilot_results_v2.json

After rerun:
  python src/b3_rerun_failures.py --merge
  → produces results/pilot_results_clean_after_reruns.json
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT           = Path(__file__).parent.parent
QUESTIONS_FILE = ROOT / "data" / "pilot_questions.json"
CACHE_DIR      = ROOT / "data" / "evidence_cache"
PLAN_FILE      = ROOT / "results" / "reruns" / "rerun_plan.json"
V2_FILE        = ROOT / "results" / "merged" / "pilot_results_v2.json"
RERUNS_DIR     = ROOT / "results" / "reruns"
JSONL_FILE     = RERUNS_DIR / "rerun_outputs.jsonl"
CLEAN_FILE     = ROOT / "results" / "merged" / "pilot_results_clean_after_reruns.json"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

RAW_RESPONSE_CAP = 200_000   # ← no effective truncation; full response stored
TIMEOUT          = 900       # 15 min; some models take >10 min on hard P3 questions

MODELS = {
    "gpt-5.4":                "openai/gpt-5.4",
    "claude-opus-4.6":        "anthropic/claude-opus-4.6",
    "gemini-3.1-pro":         "google/gemini-3.1-pro-preview",
    "grok-4.20":              "x-ai/grok-4.20",
    "qwen3-max":              "qwen/qwen3-max-thinking",
    "deepseek-v3.2-speciale": "deepseek/deepseek-v3.2-speciale",
    "kimi-k2.6":              "moonshotai/kimi-k2.6",
    "gemini-3-flash":         "google/gemini-3-flash-preview",
    "gemma-4-31b":            "google/gemma-4-31b-it",
    "glm-5.1":                "z-ai/glm-5.1",
    "gpt-oss-120b":           "openai/gpt-oss-120b",
    "mistral-large":          "mistralai/mistral-large-2512",
}

MODEL_TYPES = {
    "gpt-5.4": "reasoning", "claude-opus-4.6": "reasoning",
    "gemini-3.1-pro": "reasoning", "grok-4.20": "reasoning",
    "qwen3-max": "reasoning", "deepseek-v3.2-speciale": "reasoning",
    "kimi-k2.6": "reasoning", "gpt-oss-120b": "reasoning",
    "gemini-3-flash": "standard", "gemma-4-31b": "standard",
    "glm-5.1": "standard", "mistral-large": "standard",
}
REASONING_MODEL_KEYS = {k for k, v in MODEL_TYPES.items() if v == "reasoning"}

EVIDENCE_BLOCK_HEADER = (
    "\n\nThe following recent news articles may be relevant to this question. "
    "Use them if helpful; disregard if not.\n\n"
)


# ── Prompt builders (identical to b2) ─────────────────────────────────────────

def _base(q, evidence=""):
    ev = (EVIDENCE_BLOCK_HEADER + evidence) if evidence else ""
    return (
        f"Question: {q['question']}\n\n"
        f"Background: {q['background']}\n\n"
        f"Resolution Criteria: {q['resolution_criteria']}\n\n"
        f"Resolution Date: {q['forecast_due_date']}{ev}"
    )

def build_p1(q, evidence=""): return (
    "Please answer the following question with a probabilistic estimate expressed "
    "between 0% and 100%, and format your response as: 'Forecast: X%'.\n\n" + _base(q, evidence))

def build_p2(q, evidence=""): return (
    "Please answer the following question with a probabilistic estimate expressed "
    "between 0% and 100%, and format your response as: 'Forecast: X%'.\n\n"
    "Before considering the specific details of this question, what is the "
    "historical frequency of similar events? Using this base rate as your starting "
    "point, adjust your probability estimate based on the particular circumstances "
    "of this case.\n\n" + _base(q, evidence))

def build_p3(q, evidence=""): return (
    "Consider the following question in terms of Bayesian reasoning. Start with a "
    "prior probability based on historical data or general knowledge. Then, update "
    "this prior using more specific information about the case under discussion. "
    "For each new piece of information, produce an updated posterior estimate of "
    "the outcome using the principle behind Bayes rule. Conclude with the final "
    "posterior probability, formatted as: 'Forecast: X%'\n\n" + _base(q, evidence))

PROMPT_BUILDERS = {"P1": build_p1, "P2": build_p2, "P3": build_p3}


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
    return round(min(max(val, 0.0), 1.0), 4), True


def parse_forecast_v2(raw):
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


# ── API call ───────────────────────────────────────────────────────────────────

def load_evidence(question_id):
    path = CACHE_DIR / f"{question_id}.json"
    if not path.exists():
        return ""
    try:
        d = json.load(open(path, encoding="utf-8"))
        return d.get("prompt_optimized_string", "") or ""
    except Exception:
        return ""


def call_model(model_id, prompt, max_tokens, is_reasoning=False):
    t0 = time.time()
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if is_reasoning:
        payload["reasoning"] = {"effort": "high"}

    for attempt in range(4):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                         "Content-Type": "application/json"},
                json=payload,
                timeout=TIMEOUT,
            )
            latency = round(time.time() - t0, 2)
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"[429 rate-limit, waiting {wait}s...]", end=" ", flush=True)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                return {"status": "http_error", "code": resp.status_code,
                        "error": resp.text[:200], "latency": latency}
            data  = resp.json()
            usage = data.get("usage", {})
            msg   = data["choices"][0]["message"]
            content = (msg.get("content") or msg.get("reasoning") or "").strip()
            return {
                "status":           "ok",
                "response":         content,
                "latency":          latency,
                "input_tokens":     usage.get("prompt_tokens"),
                "output_tokens":    usage.get("completion_tokens"),
                "reasoning_tokens": usage.get("completion_tokens_details", {})
                                        .get("reasoning_tokens"),
            }
        except Exception as e:
            return {"status": "exception", "error": str(e),
                    "latency": round(time.time() - t0, 2)}
    return {"status": "http_error", "code": 429,
            "error": "Rate limited after 3 retries",
            "latency": round(time.time() - t0, 2)}


# ── Crash-recovery helpers ─────────────────────────────────────────────────────

def load_done():
    """Return set of (model_key, question_id, prompt, condition) already in JSONL."""
    done = set()
    if JSONL_FILE.exists():
        with open(JSONL_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done.add((r["model_key"], r["question_id"], r["prompt"], r["condition"]))
                except Exception:
                    pass
    return done


def append_jsonl(row):
    with open(JSONL_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── Merge ──────────────────────────────────────────────────────────────────────

def merge():
    print(f"Loading v2 base dataset from {V2_FILE} ...")
    with open(V2_FILE, encoding="utf-8") as f:
        v2_data = json.load(f)
    base_rows = v2_data["results"]

    print(f"Loading reruns from {JSONL_FILE} ...")
    rerun_rows = []
    if JSONL_FILE.exists():
        with open(JSONL_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rerun_rows.append(json.loads(line))
                    except Exception:
                        pass
    print(f"  {len(rerun_rows)} rerun records loaded")

    # Build lookup: (model_key, question_id, prompt, condition) → rerun row
    rerun_index = {}
    for r in rerun_rows:
        k = (r["model_key"], r["question_id"], r["prompt"], r["condition"])
        # If duplicate reruns exist, prefer parse_success=True, then latest
        if k not in rerun_index or (r.get("parse_success_v2") and
                                     not rerun_index[k].get("parse_success_v2")):
            rerun_index[k] = r

    merged = []
    replaced = 0
    kept_failed = 0

    for row in base_rows:
        if row.get("notes") != "requires_rerun":
            merged.append(row)
            continue
        k = (row["model_key"], row["question_id"], row["prompt"], row["condition"])
        if k in rerun_index:
            merged.append(rerun_index[k])
            if rerun_index[k].get("parse_success_v2"):
                replaced += 1
            else:
                kept_failed += 1
        else:
            # Rerun not attempted yet — keep original
            row["notes"] = "rerun_not_attempted"
            merged.append(row)

    # Summary
    v2_ok = sum(1 for r in merged if r.get("parse_success_v2"))
    still_fail = sum(1 for r in merged if not r.get("parse_success_v2"))
    print(f"\n  Merge summary:")
    print(f"    Total rows:            {len(merged)}")
    print(f"    parse_success_v2=True: {v2_ok}  ({100*v2_ok/len(merged):.2f}%)")
    print(f"    Still failed:          {still_fail}")
    print(f"    Replaced by rerun:     {replaced}")
    print(f"    Rerun also failed:     {kept_failed}")

    with open(CLEAN_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "description": (
                "Clean dataset: v2 parser + successful reruns merged. "
                "Rows with notes='rerun_failed' need further investigation."
            ),
            "n_rows":     len(merged),
            "n_v2_ok":    v2_ok,
            "n_still_fail": still_fail,
            "results":    merged,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {CLEAN_FILE}")

    # Per-cell summary for originally-failed cells
    from collections import Counter
    print(f"\n  Per-cell status after merge:")
    print(f"  {'Model':<28} {'Cell':<22} {'v2_ok':>6}  {'failed':>6}")
    print("  " + "-" * 65)
    flagged_models = {"gpt-oss-120b", "kimi-k2.6", "mistral-large",
                      "gemini-3.1-pro", "glm-5.1"}
    for r in merged:
        pass  # build index below
    cell_stats = {}
    for r in merged:
        if r["model_key"] not in flagged_models:
            continue
        k = (r["model_key"], r["prompt"], r["condition"])
        if k not in cell_stats:
            cell_stats[k] = {"ok": 0, "fail": 0}
        if r.get("parse_success_v2"):
            cell_stats[k]["ok"] += 1
        else:
            cell_stats[k]["fail"] += 1
    for (model, prompt, cond), s in sorted(cell_stats.items()):
        if s["fail"] > 0:
            print(f"  {model:<28} {prompt+'_'+cond:<22} {s['ok']:>6}  {s['fail']:>6}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="store_true",
                        help="Merge rerun outputs into pilot_results_clean_after_reruns.json")
    parser.add_argument("--models", nargs="+", metavar="KEY",
                        help="Run only these model keys")
    args = parser.parse_args()

    if args.merge:
        merge()
        sys.exit(0)

    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set.")
        sys.exit(1)

    # Confirm the cap
    print(f"raw_response storage cap: {RAW_RESPONSE_CAP:,} chars  (effectively unlimited)")
    print()

    # Load plan and questions
    with open(PLAN_FILE, encoding="utf-8") as f:
        plan = json.load(f)
    items = plan["items"]
    if args.models:
        items = [i for i in items if i["model_key"] in args.models]

    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        all_questions = json.load(f)
    q_index = {q["question_id"]: q for q in all_questions}

    RERUNS_DIR.mkdir(parents=True, exist_ok=True)
    done = load_done()

    print(f"Rerun plan: {len(items)} items total  |  already done: {len(done)}")
    remaining = [i for i in items
                 if (i["model_key"], i["question_id"], i["prompt"], i["condition"]) not in done]
    print(f"Remaining:  {len(remaining)}\n")

    succeeded = 0
    failed_again = 0

    for idx, item in enumerate(remaining):
        model_key   = item["model_key"]
        model_id    = item["model_id"]
        question_id = item["question_id"]
        prompt_key  = item["prompt"]
        condition   = item["condition"]
        fail_class  = item["failure_class"]
        max_tokens  = item["recommended_max_tokens"]
        is_reas     = model_key in REASONING_MODEL_KEYS

        q = q_index.get(question_id)
        if q is None:
            print(f"[{idx+1}/{len(remaining)}] SKIP — question_id {question_id} not found")
            continue

        evidence = load_evidence(question_id) if condition == "shared_evidence" else ""
        prompt   = PROMPT_BUILDERS[prompt_key](q, evidence)

        status_tag = f"[{idx+1}/{len(remaining)}] {model_key} {prompt_key}_{condition[:2]} " \
                     f"q={question_id[:20]} max_tok={max_tokens}"
        print(f"{status_tag} ...", end=" ", flush=True)

        result  = call_model(model_id, prompt, max_tokens, is_reasoning=is_reas)
        val_v2, ok_v2 = None, False
        brier_v2 = None
        resolution = q.get("resolution_value")

        if result["status"] == "ok":
            val_v2, ok_v2 = parse_forecast_v2(result["response"])
            if ok_v2 and resolution is not None:
                brier_v2 = round((val_v2 - resolution) ** 2, 4)

        if ok_v2:
            succeeded += 1
        else:
            failed_again += 1

        status_str = (f"Forecast={val_v2:.0%}  Brier={brier_v2}"
                      if ok_v2 else f"FAIL ({result['status']})")
        print(f"{result.get('latency',0):.1f}s  {status_str}")

        row = {
            # identity
            "model_key":           model_key,
            "model_id":            model_id,
            "model_type":          MODEL_TYPES.get(model_key, "?"),
            "question_id":         question_id,
            "source":              q.get("source"),
            "topic":               q.get("topic"),
            "market_prob":         q.get("freeze_datetime_value"),
            "prompt":              prompt_key,
            "condition":           condition,
            # v2 parse results
            "forecast_v2":         val_v2,
            "parse_success_v2":    ok_v2,
            "brier_score_v2":      brier_v2,
            "parser_version":      "v2" if ok_v2 else None,
            # rerun metadata
            "initial_parse_success":   False,
            "initial_failure_type":    fail_class,
            "rerun_attempted":         True,
            "rerun_parse_success":     ok_v2,
            "rerun_max_tokens":        max_tokens,
            "notes":                   "recovered_by_rerun" if ok_v2 else "rerun_failed",
            # v1 columns (None — original failed)
            "forecast":            None,
            "brier_score":         None,
            "parse_success":       False,
            # runtime
            "resolution_value":    resolution,
            "latency_seconds":     result.get("latency"),
            "input_tokens":        result.get("input_tokens"),
            "output_tokens":       result.get("output_tokens"),
            "reasoning_tokens":    result.get("reasoning_tokens"),
            "raw_response":        (result.get("response") or "")[:RAW_RESPONSE_CAP],
            "error":               result.get("error") if result["status"] != "ok" else None,
        }
        append_jsonl(row)
        time.sleep(0.3)

    print(f"\n{'='*60}")
    print(f"Rerun complete:  succeeded={succeeded}  failed_again={failed_again}")
    print(f"JSONL output:    {JSONL_FILE}")
    print(f"\nRun with --merge to produce {CLEAN_FILE.name}")
