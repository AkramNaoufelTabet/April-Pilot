"""
Smoke test: 5 questions x 12 models x 3 prompts, closed-book.
Prints a compact table and saves to results/smoke_test.json.

Usage:
    python src/smoke_test.py                          # all 12 models, all 3 prompts
    python src/smoke_test.py --models deepseek-v3.2-speciale kimi-k2.6
    python src/smoke_test.py --prompts P1 P2          # subset of prompts
    python src/smoke_test.py --merge                  # merge new runs into existing JSON
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
QUESTIONS_FILE = ROOT / "data" / "pilot_questions.json"
OUT_FILE = ROOT / "results" / "smoke_test.json"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

MODELS = {
    # reasoning (7)
    "gpt-5.4":                "openai/gpt-5.4",
    "claude-opus-4.6":        "anthropic/claude-opus-4.6",
    "gemini-3.1-pro":         "google/gemini-3.1-pro-preview",
    "grok-4.20":              "x-ai/grok-4.20",
    "qwen3-max":              "qwen/qwen3-max-thinking",
    "deepseek-v3.2-speciale": "deepseek/deepseek-v3.2-speciale",
    "kimi-k2.6":              "moonshotai/kimi-k2.6",
    # standard (5)
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
    "gemini-3-flash": "standard", "gemma-4-31b": "standard",
    "glm-5.1": "standard", "gpt-oss-120b": "standard",
    "mistral-large": "standard",
}

REASONING_MODEL_KEYS = {k for k, v in MODEL_TYPES.items() if v == "reasoning"}
VERBOSE_RESPONSE_MODELS = {"deepseek-v3.2-speciale", "kimi-k2.6"}

N_QUESTIONS = 5
MAX_TOKENS = 32768   # same for all models; reasoning models spend ~80% on chain-of-thought
TIMEOUT = 300        # deepseek/kimi can take 3+ minutes


# ── Prompt templates (closed-book) ─────────────────────────────────────────────

def build_prompt_p1(q: dict) -> str:
    """P1 — Control (Schoenegger et al. Prompt #1)"""
    return (
        "Please answer the following question with a probabilistic estimate expressed "
        "between 0% and 100%, and format your response as: 'Forecast: X%'.\n\n"
        f"Question: {q['question']}\n\n"
        f"Background: {q['background']}\n\n"
        f"Resolution Criteria: {q['resolution_criteria']}\n\n"
        f"Resolution Date: {q['forecast_due_date']}"
    )


def build_prompt_p2(q: dict) -> str:
    """P2 — Base-Rate-First (Schoenegger et al. Prompt #19)"""
    return (
        "Please answer the following question with a probabilistic estimate expressed "
        "between 0% and 100%, and format your response as: 'Forecast: X%'.\n\n"
        "Before considering the specific details of this question, what is the "
        "historical frequency of similar events? Using this base rate as your starting "
        "point, adjust your probability estimate based on the particular circumstances "
        "of this case.\n\n"
        f"Question: {q['question']}\n\n"
        f"Background: {q['background']}\n\n"
        f"Resolution Criteria: {q['resolution_criteria']}\n\n"
        f"Resolution Date: {q['forecast_due_date']}"
    )


def build_prompt_p3(q: dict) -> str:
    """P3 — Bayesian Reasoning (Schoenegger et al. Prompt #27)"""
    return (
        "Consider the following question in terms of Bayesian reasoning. Start with a "
        "prior probability based on historical data or general knowledge. Then, update "
        "this prior using more specific information about the case under discussion. "
        "For each new piece of information, produce an updated posterior estimate of "
        "the outcome using the principle behind Bayes rule. Conclude with the final "
        "posterior probability, formatted as: 'Forecast: X%'\n\n"
        f"Question: {q['question']}\n\n"
        f"Background: {q['background']}\n\n"
        f"Resolution Criteria: {q['resolution_criteria']}\n\n"
        f"Resolution Date: {q['forecast_due_date']}"
    )


PROMPT_BUILDERS = {
    "P1": build_prompt_p1,
    "P2": build_prompt_p2,
    "P3": build_prompt_p3,
}

PROMPT_LABELS = {
    "P1": "Control",
    "P2": "Base-Rate-First",
    "P3": "Bayesian",
}


# ── API call ───────────────────────────────────────────────────────────────────

def extract_forecast(text: str) -> float | None:
    match = re.search(r"Forecast:\s*([\d.]+)%", text, re.IGNORECASE)
    if match:
        return round(min(max(float(match.group(1)) / 100, 0.0), 1.0), 4)
    return None


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

    for attempt in range(4):  # up to 3 retries on 429
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
            data = resp.json()
            usage = data.get("usage", {})
            msg = data["choices"][0]["message"]
            content = (msg.get("content") or msg.get("reasoning") or "").strip()
            return {
                "status": "ok",
                "response": content,
                "latency": latency,
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
            }
        except Exception as e:
            return {"status": "exception", "error": str(e),
                    "latency": round(time.time() - t0, 2)}
    return {"status": "http_error", "code": 429,
            "error": "Rate limited after 3 retries", "latency": round(time.time() - t0, 2)}


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", metavar="KEY",
                        help="Run only these model keys")
    parser.add_argument("--prompts", nargs="+", choices=["P1", "P2", "P3"],
                        default=["P1", "P2", "P3"],
                        help="Which prompts to run (default: all three)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge results into existing smoke_test.json")
    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set.")
        sys.exit(1)

    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        all_questions = json.load(f)
    questions = all_questions[:N_QUESTIONS]

    active_models = {k: v for k, v in MODELS.items()
                     if args.models is None or k in args.models}
    active_prompts = args.prompts

    print(f"Smoke test: {N_QUESTIONS} questions x {len(active_models)} models "
          f"x {len(active_prompts)} prompts ({', '.join(active_prompts)}) — closed-book\n")

    print("Questions:")
    for i, q in enumerate(questions):
        mkt = q.get("freeze_datetime_value", "?")
        try:
            mkt_pct = f"{float(mkt):.0%}"
        except (ValueError, TypeError):
            mkt_pct = str(mkt)
        print(f"  Q{i+1} [{q['source']}] {q['question'][:72]}")
        print(f"       market_prob={mkt_pct}  resolution={q['resolution_value']}  topic={q['topic']}")
    print()

    new_results = []
    parse_failures = []

    for model_key, model_id in active_models.items():
        mtype = MODEL_TYPES[model_key]
        is_reasoning = model_key in REASONING_MODEL_KEYS
        verbose = model_key in VERBOSE_RESPONSE_MODELS

        print(f"\n{'='*70}")
        print(f"Model: {model_key} ({mtype})  ->  {model_id}")
        print(f"{'='*70}")

        for prompt_key in active_prompts:
            build_fn = PROMPT_BUILDERS[prompt_key]
            print(f"\n  -- {prompt_key} ({PROMPT_LABELS[prompt_key]}) --")

            prompt_results = []

            for i, q in enumerate(questions):
                mkt = q.get("freeze_datetime_value", "?")
                try:
                    mkt_pct = f"{float(mkt):.0%}"
                except (ValueError, TypeError):
                    mkt_pct = str(mkt)

                prompt = build_fn(q)
                print(f"    Q{i+1} [mkt={mkt_pct}] ...", end=" ", flush=True)

                result = call_model(model_id, prompt, is_reasoning=is_reasoning)
                forecast = None
                parse_ok = False

                if result["status"] == "ok":
                    forecast = extract_forecast(result["response"])
                    parse_ok = forecast is not None
                    brier = round((forecast - q["resolution_value"]) ** 2, 4) if parse_ok else None
                    rtoks = result.get("reasoning_tokens")
                    rtok_str = f"  r_tok={rtoks}" if rtoks else ""
                    status_str = (f"Forecast={forecast:.0%}  Brier={brier}"
                                  if parse_ok else "PARSE FAIL")
                    print(f"{result['latency']}s  {status_str}{rtok_str}")

                    if verbose and parse_ok:
                        snippet = result["response"][-400:].strip().encode("ascii", "replace").decode("ascii")
                        print(f"      [tail] {snippet}\n")
                    if not parse_ok:
                        parse_failures.append((model_key, prompt_key, q["question_id"]))
                        print(f"      Response: {result['response'][:200]}")
                else:
                    brier = None
                    print(f"ERROR ({result['status']}: {result.get('error','')[:80]})")

                row = {
                    "model_key": model_key,
                    "model_id": model_id,
                    "model_type": mtype,
                    "question_id": q["question_id"],
                    "source": q["source"],
                    "topic": q["topic"],
                    "market_prob": q.get("freeze_datetime_value"),
                    "prompt": prompt_key,
                    "condition": "closed_book",
                    "forecast": forecast,
                    "resolution_value": q["resolution_value"],
                    "brier_score": brier,
                    "parse_success": parse_ok,
                    "latency_seconds": result.get("latency"),
                    "input_tokens": result.get("input_tokens"),
                    "output_tokens": result.get("output_tokens"),
                    "reasoning_tokens": result.get("reasoning_tokens"),
                    "raw_response": result.get("response", "")[:2000],
                    "error": result.get("error") if result["status"] != "ok" else None,
                }
                new_results.append(row)
                prompt_results.append(row)

            parsed = [r for r in prompt_results if r["parse_success"]]
            if parsed:
                avg_b = sum(r["brier_score"] for r in parsed) / len(parsed)
                avg_l = sum(r["latency_seconds"] for r in prompt_results) / len(prompt_results)
                print(f"    --> {prompt_key} avg Brier={avg_b:.4f}  avg lat={avg_l:.1f}s  "
                      f"parse={len(parsed)}/{N_QUESTIONS}")

        time.sleep(0.5)

    # Merge with existing results if --merge or --models
    all_results = new_results
    if (args.merge or args.models) and OUT_FILE.exists():
        with open(OUT_FILE, encoding="utf-8") as f:
            existing = json.load(f)
        old = existing.get("results", [])
        rerun_keys = set(active_models.keys())
        rerun_prompts = set(active_prompts)
        kept = [r for r in old
                if not (r["model_key"] in rerun_keys and r.get("prompt") in rerun_prompts)]
        all_results = kept + new_results

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*85}")
    print("SMOKE TEST SUMMARY  (closed-book, 5 questions)")
    print(f"{'='*85}")

    # Market baseline Brier
    mkt_briers = []
    for r in all_results:
        try:
            mkt = float(r.get("market_prob") or "")
            mkt_briers.append((mkt - r["resolution_value"]) ** 2)
        except (TypeError, ValueError):
            pass
    mkt_baseline = sum(mkt_briers) / len(mkt_briers) if mkt_briers else None

    header = f"{'Model':<26} {'Type':<10}"
    for pk in ["P1", "P2", "P3"]:
        header += f"  {pk+' Brier':<12}"
    header += f"  {'Avg Lat':>8}  vs Market"
    print(header)
    print("-" * 85)

    for model_key in MODELS:
        mr = [r for r in all_results if r["model_key"] == model_key]
        if not mr:
            continue
        mtype = MODEL_TYPES.get(model_key, "?")
        row = f"  {model_key:<24} {mtype:<10}"
        for pk in ["P1", "P2", "P3"]:
            pr = [r for r in mr if r.get("prompt") == pk and r["parse_success"]]
            if pr:
                avg_b = sum(r["brier_score"] for r in pr) / len(pr)
                row += f"  {avg_b:.4f}      "
            else:
                row += f"  {'N/A':<12}"
        avg_lat = sum(r["latency_seconds"] for r in mr if r["latency_seconds"]) / max(len(mr), 1)
        mkt_str = f"mkt={mkt_baseline:.4f}" if mkt_baseline else ""
        row += f"  {avg_lat:>7.1f}s  {mkt_str}"
        print(row)

    if parse_failures:
        print(f"\nParse failures ({len(parse_failures)}):")
        for mk, pk, qid in parse_failures:
            print(f"  {mk}  {pk}  q={qid[:20]}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "test_type": "smoke_test_3prompts",
            "n_questions": N_QUESTIONS,
            "n_models": len(set(r["model_key"] for r in all_results)),
            "prompts": list(set(r.get("prompt") for r in all_results)),
            "condition": "closed_book",
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUT_FILE}")
