"""
B2 — Full experiment runner.

114 questions x 12 models x 3 prompts x 2 conditions = 8,208 calls.

Each model writes to its own results/pilot_{model_key}.json so multiple
processes can run in parallel without file conflicts.

Crash recovery: before each call checks if (model_key, prompt, condition,
question_id) already exists in the output file — skips if so.

Usage:
    # Run one model
    python src/b2_run_experiment.py --models gpt-5.4

    # Run several models in one process (sequential)
    python src/b2_run_experiment.py --models gemini-3-flash gemma-4-31b glm-5.1

    # Run all models (not recommended — use parallel launches instead)
    python src/b2_run_experiment.py

    # Subset conditions or prompts
    python src/b2_run_experiment.py --models gpt-5.4 --conditions closed_book
    python src/b2_run_experiment.py --models gpt-5.4 --prompts P1 P2

Parallel launch pattern (separate terminals):
    Terminal 1: python src/b2_run_experiment.py --models gpt-5.4 claude-opus-4.6 ...
    Terminal 2: python src/b2_run_experiment.py --models deepseek-v3.2-speciale
    Terminal 3: python src/b2_run_experiment.py --models kimi-k2.6
    Terminal 4: python src/b2_run_experiment.py --models qwen3-max

After all finish, merge with:
    python src/b2_run_experiment.py --merge
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
RESULTS_DIR    = ROOT / "results" / "raw"
MERGED_FILE    = ROOT / "results" / "merged" / "pilot_results.json"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

MAX_TOKENS = 32768
TIMEOUT    = 720

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
    "kimi-k2.6": "reasoning",
    "gpt-oss-120b": "reasoning",
    "gemini-3-flash": "standard", "gemma-4-31b": "standard",
    "glm-5.1": "standard", "mistral-large": "standard",
}

REASONING_MODEL_KEYS  = {k for k, v in MODEL_TYPES.items() if v == "reasoning"}
VERBOSE_RESPONSE_MODELS = {"deepseek-v3.2-speciale", "kimi-k2.6"}

CONDITIONS = ["closed_book", "shared_evidence"]

EVIDENCE_BLOCK_HEADER = (
    "\n\nThe following recent news articles may be relevant to this question. "
    "Use them if helpful; disregard if not.\n\n"
)


# ── Prompt builders ────────────────────────────────────────────────────────────

def _base(q: dict, evidence: str = "") -> str:
    ev = (EVIDENCE_BLOCK_HEADER + evidence) if evidence else ""
    return (
        f"Question: {q['question']}\n\n"
        f"Background: {q['background']}\n\n"
        f"Resolution Criteria: {q['resolution_criteria']}\n\n"
        f"Resolution Date: {q['forecast_due_date']}{ev}"
    )


def build_p1(q: dict, evidence: str = "") -> str:
    return (
        "Please answer the following question with a probabilistic estimate expressed "
        "between 0% and 100%, and format your response as: 'Forecast: X%'.\n\n"
        + _base(q, evidence)
    )


def build_p2(q: dict, evidence: str = "") -> str:
    return (
        "Please answer the following question with a probabilistic estimate expressed "
        "between 0% and 100%, and format your response as: 'Forecast: X%'.\n\n"
        "Before considering the specific details of this question, what is the "
        "historical frequency of similar events? Using this base rate as your starting "
        "point, adjust your probability estimate based on the particular circumstances "
        "of this case.\n\n"
        + _base(q, evidence)
    )


def build_p3(q: dict, evidence: str = "") -> str:
    return (
        "Consider the following question in terms of Bayesian reasoning. Start with a "
        "prior probability based on historical data or general knowledge. Then, update "
        "this prior using more specific information about the case under discussion. "
        "For each new piece of information, produce an updated posterior estimate of "
        "the outcome using the principle behind Bayes rule. Conclude with the final "
        "posterior probability, formatted as: 'Forecast: X%'\n\n"
        + _base(q, evidence)
    )


PROMPT_BUILDERS = {"P1": build_p1, "P2": build_p2, "P3": build_p3}
PROMPT_LABELS   = {"P1": "Control", "P2": "Base-Rate-First", "P3": "Bayesian"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_forecast(text: str) -> float | None:
    match = re.search(r"Forecast:\s*([\d.]+)(%?)", text, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        if not match.group(2) and val <= 1.0:
            return round(min(max(val, 0.0), 1.0), 4)
        return round(min(max(val / 100, 0.0), 1.0), 4)
    return None


def load_evidence(question_id: str) -> str:
    path = CACHE_DIR / f"{question_id}.json"
    if not path.exists():
        return ""
    try:
        d = json.load(open(path, encoding="utf-8"))
        return d.get("prompt_optimized_string", "") or ""
    except Exception:
        return ""


def out_file(model_key: str) -> Path:
    return RESULTS_DIR / f"pilot_{model_key}.json"


def load_done(model_key: str) -> set[tuple]:
    """Return set of (prompt, condition, question_id) already completed."""
    path = out_file(model_key)
    if not path.exists():
        return set()
    try:
        d = json.load(open(path, encoding="utf-8"))
        return {(r["prompt"], r["condition"], r["question_id"])
                for r in d.get("results", []) if r.get("parse_success")}
    except Exception:
        return set()


def append_result(model_key: str, row: dict) -> None:
    """Append one result row to the model's output file (create if needed)."""
    path = out_file(model_key)
    if path.exists():
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            d = {"results": []}
    else:
        d = {"results": []}
    d["results"].append(row)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def call_model(model_id: str, prompt: str, is_reasoning: bool = False) -> dict:
    t0 = time.time()
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
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
                print(f"[429 rate limit, waiting {wait}s...]", end=" ", flush=True)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                return {"status": "http_error", "code": resp.status_code,
                        "error": resp.text[:200], "latency": latency}
            data   = resp.json()
            usage  = data.get("usage", {})
            msg    = data["choices"][0]["message"]
            content = (msg.get("content") or msg.get("reasoning") or "").strip()
            return {
                "status": "ok", "response": content, "latency": latency,
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


# ── Run one model ──────────────────────────────────────────────────────────────

def run_model(model_key: str, questions: list[dict],
              active_conditions: list[str], active_prompts: list[str]) -> None:
    model_id   = MODELS[model_key]
    mtype      = MODEL_TYPES[model_key]
    is_reas    = model_key in REASONING_MODEL_KEYS
    verbose    = model_key in VERBOSE_RESPONSE_MODELS
    done       = load_done(model_key)

    total_cells = len(questions) * len(active_conditions) * len(active_prompts)
    skipped     = sum(
        1 for q in questions for cond in active_conditions for pk in active_prompts
        if (pk, cond, q["question_id"]) in done
    )

    print(f"\n{'='*72}")
    print(f"Model: {model_key} ({mtype})  ->  {model_id}")
    print(f"  {total_cells} cells  |  {skipped} already done  |  {total_cells-skipped} to run")
    print(f"{'='*72}")

    parse_failures = []

    for cond in active_conditions:
        print(f"\n  -- Condition: {cond} --")
        for pk in active_prompts:
            builder = PROMPT_BUILDERS[pk]
            print(f"\n    -- {pk} ({PROMPT_LABELS[pk]}) --")

            for i, q in enumerate(questions):
                qid = q["question_id"]
                key = (pk, cond, qid)

                if key in done:
                    print(f"      Q{i+1:>3} SKIP")
                    continue

                evidence = load_evidence(qid) if cond == "shared_evidence" else ""
                prompt   = builder(q, evidence)

                mkt = q.get("freeze_datetime_value", "?")
                try:
                    mkt_pct = f"{float(mkt):.0%}"
                except (ValueError, TypeError):
                    mkt_pct = str(mkt)

                print(f"      Q{i+1:>3} [mkt={mkt_pct}] ...", end=" ", flush=True)

                result   = call_model(model_id, prompt, is_reasoning=is_reas)
                forecast = None
                parse_ok = False

                if result["status"] == "ok":
                    forecast = extract_forecast(result["response"])
                    parse_ok = forecast is not None
                    brier    = round((forecast - q["resolution_value"]) ** 2, 4) if parse_ok else None
                    rtoks    = result.get("reasoning_tokens")
                    rtok_str = f"  r_tok={rtoks}" if rtoks else ""
                    status_str = (f"Forecast={forecast:.0%}  Brier={brier}"
                                  if parse_ok else "PARSE FAIL")
                    print(f"{result['latency']}s  {status_str}{rtok_str}")

                    if verbose and parse_ok:
                        snippet = result["response"][-300:].strip().encode("ascii","replace").decode("ascii")
                        print(f"        [tail] {snippet}\n")
                    if not parse_ok:
                        parse_failures.append((pk, cond, qid))
                        snippet = result["response"][:200].encode("ascii","replace").decode("ascii")
                        print(f"        Response: {snippet}")
                else:
                    brier = None
                    err   = (result.get("error") or "")[:80].encode("ascii","replace").decode("ascii")
                    print(f"ERROR ({result['status']}: {err})")

                row = {
                    "model_key":        model_key,
                    "model_id":         model_id,
                    "model_type":       mtype,
                    "question_id":      qid,
                    "source":           q["source"],
                    "topic":            q.get("topic"),
                    "market_prob":      q.get("freeze_datetime_value"),
                    "prompt":           pk,
                    "condition":        cond,
                    "forecast":         forecast,
                    "resolution_value": q["resolution_value"],
                    "brier_score":      brier,
                    "parse_success":    parse_ok,
                    "latency_seconds":  result.get("latency"),
                    "input_tokens":     result.get("input_tokens"),
                    "output_tokens":    result.get("output_tokens"),
                    "reasoning_tokens": result.get("reasoning_tokens"),
                    "raw_response":     result.get("response", "")[:2000],
                    "error":            result.get("error") if result["status"] != "ok" else None,
                }
                append_result(model_key, row)
                done.add(key)

    # Per-model summary
    all_rows = json.load(open(out_file(model_key), encoding="utf-8")).get("results", [])
    parsed   = [r for r in all_rows if r["parse_success"] and
                r["condition"] in active_conditions and r["prompt"] in active_prompts]
    total    = len([r for r in all_rows if r["condition"] in active_conditions
                    and r["prompt"] in active_prompts])

    print(f"\n  {model_key} summary: parsed={len(parsed)}/{total}")
    for cond in active_conditions:
        for pk in active_prompts:
            pr = [r for r in parsed if r["condition"]==cond and r["prompt"]==pk]
            if pr:
                avg_b = sum(r["brier_score"] for r in pr) / len(pr)
                avg_l = sum(r["latency_seconds"] for r in pr if r["latency_seconds"]) / len(pr)
                print(f"    {cond:<16} {pk}  Brier={avg_b:.4f}  lat={avg_l:.1f}s  n={len(pr)}")

    if parse_failures:
        print(f"  Parse failures ({len(parse_failures)}):")
        for pk, cond, qid in parse_failures:
            print(f"    {pk}  {cond}  q={qid[:25]}")

    print(f"\n  Saved to {out_file(model_key)}")


# ── Merge all model files ──────────────────────────────────────────────────────

def merge_results() -> None:
    all_rows = []
    found = []
    for model_key in MODELS:
        path = out_file(model_key)
        if path.exists():
            d = json.load(open(path, encoding="utf-8"))
            rows = d.get("results", [])
            all_rows.extend(rows)
            found.append(f"  {model_key}: {len(rows)} rows")
        else:
            found.append(f"  {model_key}: MISSING")

    print("Merging:")
    for line in found:
        print(line)

    with open(MERGED_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "test_type":    "full_experiment",
            "n_questions":  114,
            "n_models":     12,
            "n_prompts":    3,
            "n_conditions": 2,
            "total_rows":   len(all_rows),
            "results":      all_rows,
        }, f, indent=2, ensure_ascii=False)

    parsed = sum(1 for r in all_rows if r.get("parse_success"))
    print(f"\nMerged {len(all_rows)} rows ({parsed} parsed) -> {MERGED_FILE}")

    # Quick Brier table
    print(f"\n{'Model':<28} {'CB-P1':>7} {'CB-P2':>7} {'CB-P3':>7} {'SE-P1':>7} {'SE-P2':>7} {'SE-P3':>7}")
    print("-" * 70)
    for mk in MODELS:
        mr = [r for r in all_rows if r["model_key"] == mk]
        if not mr:
            continue
        row = f"  {mk:<26}"
        for cond, cs in [("closed_book","CB"), ("shared_evidence","SE")]:
            for pk in ["P1","P2","P3"]:
                pr = [r for r in mr if r["condition"]==cond and r["prompt"]==pk and r.get("parse_success")]
                if pr:
                    row += f"  {sum(r['brier_score'] for r in pr)/len(pr):>5.4f}"
                else:
                    row += f"  {'N/A':>5}"
        print(row)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", metavar="KEY",
                        help="Model keys to run (default: all)")
    parser.add_argument("--conditions", nargs="+",
                        choices=["closed_book", "shared_evidence"],
                        default=["closed_book", "shared_evidence"])
    parser.add_argument("--prompts", nargs="+",
                        choices=["P1", "P2", "P3"],
                        default=["P1", "P2", "P3"])
    parser.add_argument("--merge", action="store_true",
                        help="Merge all per-model files into pilot_results.json")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_results()
        sys.exit(0)

    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set.")
        sys.exit(1)

    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    active_models = {k: v for k, v in MODELS.items()
                     if args.models is None or k in args.models}

    print(f"Full experiment runner")
    print(f"Models: {list(active_models.keys())}")
    print(f"Conditions: {args.conditions}")
    print(f"Prompts: {args.prompts}")
    print(f"Questions: {len(questions)}")
    print(f"Total calls this process: "
          f"{len(questions) * len(active_models) * len(args.conditions) * len(args.prompts)}")

    for model_key in active_models:
        run_model(model_key, questions, args.conditions, args.prompts)
        time.sleep(1)

    print("\nAll done for this process.")
    print("When all parallel processes finish, run:")
    print("  python src/b2_run_experiment.py --merge")
