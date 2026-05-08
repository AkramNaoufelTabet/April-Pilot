"""
Preflight checks + mini smoke test for the full experiment.

Checks:
  1. All 114 questions exist
  2. Each question has resolution_value, freeze_datetime, topic, source
  3. Evidence cache exists for every question
  4. All evidence articles are before freeze_datetime
  5. All 12 models are listed
  6. All 6 prompt templates (3 prompts x 2 conditions) render without error
  7. Total expected calls = 8,208
  8. No duplicate run keys in existing results
  9. Output directories exist
 10. API keys are set

Then runs a mini smoke test:
  12 models x 2 questions x 3 prompts x 2 conditions = 144 calls

Usage:
    python src/preflight.py             # checks only
    python src/preflight.py --smoke     # checks + mini smoke test
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
QUESTIONS_FILE  = ROOT / "data" / "pilot_questions.json"
CACHE_DIR       = ROOT / "data" / "evidence_cache"
RESULTS_DIR     = ROOT / "results" / "raw"
SMOKE_OUT       = RESULTS_DIR / "preflight_smoke.json"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
ASKNEWS_API_KEY    = os.environ.get("ASKNEWS_API_KEY", "")

N_QUESTIONS_FULL  = 114
N_MODELS          = 12
N_PROMPTS         = 3
N_CONDITIONS      = 2
EXPECTED_CALLS    = N_QUESTIONS_FULL * N_MODELS * N_PROMPTS * N_CONDITIONS  # 8208

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
    # gpt-oss-120b always emits reasoning tokens regardless of the reasoning param;
    # treating it as reasoning routes its CoT to the hidden field so content = clean answer
    "gpt-oss-120b": "reasoning",
    "gemini-3-flash": "standard", "gemma-4-31b": "standard",
    "glm-5.1": "standard", "mistral-large": "standard",
}

REASONING_MODEL_KEYS  = {k for k, v in MODEL_TYPES.items() if v == "reasoning"}
VERBOSE_RESPONSE_MODELS = {"deepseek-v3.2-speciale", "kimi-k2.6"}

REQUIRED_Q_FIELDS = ["resolution_value", "freeze_datetime", "topic", "source"]

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
CONDITIONS      = ["closed_book", "shared_evidence"]


# ── API helpers ────────────────────────────────────────────────────────────────

def extract_forecast(text: str) -> float | None:
    # Accept "Forecast: X%" or "Forecast: X" (model omits % sign)
    match = re.search(r"Forecast:\s*([\d.]+)(%?)", text, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        if not match.group(2) and val <= 1.0:
            return round(min(max(val, 0.0), 1.0), 4)
        return round(min(max(val / 100, 0.0), 1.0), 4)
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
            data = resp.json()
            usage = data.get("usage", {})
            msg = data["choices"][0]["message"]
            content = (msg.get("content") or msg.get("reasoning") or "").strip()
            return {
                "status": "ok", "response": content, "latency": latency,
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
            }
        except Exception as e:
            return {"status": "exception", "error": str(e),
                    "latency": round(time.time() - t0, 2)}
    return {"status": "http_error", "code": 429,
            "error": "Rate limited after 3 retries", "latency": round(time.time() - t0, 2)}


def load_evidence(question_id: str) -> str:
    path = CACHE_DIR / f"{question_id}.json"
    if not path.exists():
        return ""
    try:
        d = json.load(open(path, encoding="utf-8"))
        return d.get("prompt_optimized_string", "") or ""
    except Exception:
        return ""


# ── Preflight checks ───────────────────────────────────────────────────────────

def run_checks(questions: list[dict]) -> list[str]:
    failures = []

    def fail(msg: str):
        failures.append(msg)
        print(f"  FAIL: {msg}")

    def ok(msg: str):
        print(f"  ok   {msg}")

    # ── Check 1: question count ────────────────────────────────────────────────
    print("\n[1] Question count")
    if len(questions) == N_QUESTIONS_FULL:
        ok(f"{len(questions)} questions loaded")
    else:
        fail(f"Expected {N_QUESTIONS_FULL}, got {len(questions)}")

    # ── Check 2: required fields ───────────────────────────────────────────────
    print("\n[2] Required fields per question")
    missing_fields = []
    for q in questions:
        for field in REQUIRED_Q_FIELDS:
            val = q.get(field)
            if val is None or val == "":
                missing_fields.append(f"{q['question_id']} missing {field}")
    if not missing_fields:
        ok("All questions have resolution_value, freeze_datetime, topic, source")
    else:
        for m in missing_fields[:5]:
            fail(m)
        if len(missing_fields) > 5:
            fail(f"...and {len(missing_fields)-5} more")

    # ── Check 3: evidence cache ────────────────────────────────────────────────
    print("\n[3] Evidence cache completeness")
    missing_cache = []
    empty_cache   = []
    for q in questions:
        path = CACHE_DIR / f"{q['question_id']}.json"
        if not path.exists():
            missing_cache.append(q["question_id"])
        else:
            try:
                d = json.load(open(path, encoding="utf-8"))
                if not d.get("prompt_optimized_string"):
                    empty_cache.append(q["question_id"])
            except Exception:
                missing_cache.append(q["question_id"])
    if not missing_cache and not empty_cache:
        ok("Evidence cached for all 114 questions")
    for qid in missing_cache[:3]:
        fail(f"Cache file missing: {qid}")
    for qid in empty_cache[:3]:
        fail(f"Cache empty (no articles): {qid}")

    # ── Check 4: article dates before freeze_datetime ──────────────────────────
    print("\n[4] Evidence article dates before freeze_datetime")
    date_violations = []
    for q in questions:
        path = CACHE_DIR / f"{q['question_id']}.json"
        if not path.exists():
            continue
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        freeze = (q.get("freeze_datetime") or "")[:10]
        if not freeze:
            continue
        for art_date in d.get("article_dates", []):
            art_day = str(art_date)[:10]
            if art_day >= freeze:
                date_violations.append(
                    f"{q['question_id']}: article {art_day} >= freeze {freeze}")
    if not date_violations:
        ok("All evidence articles predate freeze_datetime")
    else:
        for v in date_violations[:5]:
            fail(v)
        if len(date_violations) > 5:
            fail(f"...and {len(date_violations)-5} more violations")

    # ── Check 5: model list ────────────────────────────────────────────────────
    print("\n[5] Model list")
    if len(MODELS) == N_MODELS:
        ok(f"{N_MODELS} models defined")
    else:
        fail(f"Expected {N_MODELS} models, got {len(MODELS)}")
    for key in MODELS:
        if key not in MODEL_TYPES:
            fail(f"Model '{key}' missing from MODEL_TYPES")

    # ── Check 6: prompt templates render ──────────────────────────────────────
    print("\n[6] Prompt template rendering (3 prompts x 2 conditions = 6 templates)")
    sample_q = questions[0]
    sample_ev = load_evidence(sample_q["question_id"])
    for pk, builder in PROMPT_BUILDERS.items():
        for cond in CONDITIONS:
            ev = sample_ev if cond == "shared_evidence" else ""
            try:
                rendered = builder(sample_q, ev)
                has_forecast = "Forecast: X%" in rendered
                has_question = sample_q["question"][:30] in rendered
                if not has_forecast:
                    fail(f"{pk}/{cond}: missing 'Forecast: X%' in rendered prompt")
                elif not has_question:
                    fail(f"{pk}/{cond}: question text not in rendered prompt")
                else:
                    ok(f"{pk}/{cond}: {len(rendered)} chars")
            except Exception as e:
                fail(f"{pk}/{cond}: render error: {e}")

    # ── Check 7: expected call count ──────────────────────────────────────────
    print("\n[7] Expected call count")
    actual = len(questions) * len(MODELS) * len(PROMPT_BUILDERS) * len(CONDITIONS)
    if actual == EXPECTED_CALLS:
        ok(f"{actual} total calls (114 x {N_MODELS} x {N_PROMPTS} x {N_CONDITIONS})")
    else:
        fail(f"Expected {EXPECTED_CALLS}, computed {actual}")

    # ── Check 8: duplicate run keys in existing results ────────────────────────
    print("\n[8] Duplicate run keys in existing results")
    existing_results_files = list(RESULTS_DIR.glob("*.json"))
    all_keys = []
    for rf in existing_results_files:
        try:
            d = json.load(open(rf, encoding="utf-8"))
            for row in d.get("results", []):
                key = (row.get("model_key"), row.get("prompt"),
                       row.get("condition"), row.get("question_id"))
                all_keys.append(key)
        except Exception:
            pass
    dupes = [k for k in set(all_keys) if all_keys.count(k) > 1]
    if not dupes:
        ok(f"No duplicates in {len(existing_results_files)} result file(s)")
    else:
        fail(f"{len(dupes)} duplicate run keys found")
        for d in dupes[:3]:
            fail(f"  dup: {d}")

    # ── Check 9: output directories ────────────────────────────────────────────
    print("\n[9] Output directories")
    for d in [RESULTS_DIR, CACHE_DIR]:
        if d.exists():
            ok(str(d))
        else:
            fail(f"Missing directory: {d}")

    # ── Check 10: API keys ─────────────────────────────────────────────────────
    print("\n[10] API keys")
    if OPENROUTER_API_KEY:
        ok(f"OPENROUTER_API_KEY set ({len(OPENROUTER_API_KEY)} chars)")
    else:
        fail("OPENROUTER_API_KEY not set")
    if ASKNEWS_API_KEY:
        ok(f"ASKNEWS_API_KEY set ({len(ASKNEWS_API_KEY)} chars)")
    else:
        fail("ASKNEWS_API_KEY not set")

    return failures


# ── Mini smoke test ────────────────────────────────────────────────────────────

def run_mini_smoke(questions: list[dict]) -> None:
    # Pick 2 questions from different topics
    seen_topics: set[str] = set()
    test_qs: list[dict] = []
    for q in questions:
        t = q.get("topic", "other")
        if t not in seen_topics:
            test_qs.append(q)
            seen_topics.add(t)
        if len(test_qs) == 2:
            break

    print(f"\n{'='*72}")
    print(f"MINI SMOKE TEST  (12 models x 2 questions x 3 prompts x 2 conditions = 144 calls)")
    print(f"{'='*72}")
    print("Questions:")
    for i, q in enumerate(test_qs):
        mkt = q.get("freeze_datetime_value", "?")
        try:
            mkt_pct = f"{float(mkt):.0%}"
        except (ValueError, TypeError):
            mkt_pct = str(mkt)
        print(f"  Q{i+1} [{q['source']}] [{q['topic']}] {q['question'][:65]}")
        print(f"       market_prob={mkt_pct}  resolution={q['resolution_value']}")
    print()

    new_results = []
    parse_failures = []

    for model_key, model_id in MODELS.items():
        mtype   = MODEL_TYPES[model_key]
        is_reas = model_key in REASONING_MODEL_KEYS
        verbose = model_key in VERBOSE_RESPONSE_MODELS

        print(f"\n{'='*72}")
        print(f"Model: {model_key} ({mtype})  ->  {model_id}")
        print(f"{'='*72}")

        for cond in CONDITIONS:
            print(f"\n  -- Condition: {cond} --")

            for pk, builder in PROMPT_BUILDERS.items():
                print(f"    -- {pk} ({PROMPT_LABELS[pk]}) --")

                for i, q in enumerate(test_qs):
                    evidence = load_evidence(q["question_id"]) if cond == "shared_evidence" else ""
                    prompt   = builder(q, evidence)
                    mkt = q.get("freeze_datetime_value", "?")
                    try:
                        mkt_pct = f"{float(mkt):.0%}"
                    except (ValueError, TypeError):
                        mkt_pct = str(mkt)

                    print(f"      Q{i+1} [mkt={mkt_pct}] ...", end=" ", flush=True)

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
                            parse_failures.append((model_key, pk, cond, q["question_id"]))
                            print(f"        Response: {result['response'][:150].encode('ascii','replace').decode('ascii')}")
                    else:
                        brier = None
                        err   = (result.get("error") or "")[:80].encode("ascii","replace").decode("ascii")
                        print(f"ERROR ({result['status']}: {err})")

                    new_results.append({
                        "model_key":        model_key,
                        "model_id":         model_id,
                        "model_type":       mtype,
                        "question_id":      q["question_id"],
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
                    })

        time.sleep(0.5)

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*85}")
    print("MINI SMOKE SUMMARY  (2 questions per cell)")
    print(f"{'='*85}")

    header = f"  {'Model':<26} {'Type':<10}"
    for cond_short in ["CB", "SE"]:
        for pk in ["P1", "P2", "P3"]:
            header += f"  {cond_short+'-'+pk:<8}"
    print(header)
    print("-" * 85)

    for model_key in MODELS:
        mr    = [r for r in new_results if r["model_key"] == model_key]
        mtype = MODEL_TYPES.get(model_key, "?")
        row   = f"  {model_key:<26} {mtype:<10}"
        for cond, cond_short in [("closed_book","CB"), ("shared_evidence","SE")]:
            for pk in ["P1", "P2", "P3"]:
                pr = [r for r in mr if r["condition"]==cond and r["prompt"]==pk and r["parse_success"]]
                if pr:
                    avg_b = sum(r["brier_score"] for r in pr) / len(pr)
                    row += f"  {avg_b:.3f}   "
                else:
                    pr_all = [r for r in mr if r["condition"]==cond and r["prompt"]==pk]
                    row += f"  {'ERR' if pr_all else 'N/A':<8}"
        print(row)

    if parse_failures:
        print(f"\nParse failures ({len(parse_failures)}):")
        for mk, pk, cond, qid in parse_failures:
            print(f"  {mk}  {pk}  {cond}  q={qid[:20]}")

    total   = len(new_results)
    parsed  = sum(1 for r in new_results if r["parse_success"])
    errors  = sum(1 for r in new_results if r.get("error"))
    print(f"\nTotal calls: {total}  Parsed: {parsed}/{total}  Errors: {errors}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SMOKE_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "test_type": "preflight_mini_smoke",
            "n_models":     len(MODELS),
            "n_questions":  len(test_qs),
            "n_prompts":    len(PROMPT_BUILDERS),
            "n_conditions": len(CONDITIONS),
            "total_calls":  total,
            "results": new_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {SMOKE_OUT}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Also run 144-call mini smoke test after checks")
    args = parser.parse_args()

    print("=" * 72)
    print("PREFLIGHT CHECKS")
    print("=" * 72)

    if not QUESTIONS_FILE.exists():
        print(f"FATAL: {QUESTIONS_FILE} not found. Run Phase A first.")
        sys.exit(1)

    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    failures = run_checks(questions)

    print(f"\n{'='*72}")
    if failures:
        print(f"PREFLIGHT FAILED: {len(failures)} issue(s)")
        for f in failures:
            print(f"  - {f}")
        if args.smoke:
            print("\nAborting mini smoke test due to preflight failures.")
        sys.exit(1)
    else:
        print("PREFLIGHT PASSED: all 10 checks OK")

    if args.smoke:
        if not OPENROUTER_API_KEY:
            print("\nERROR: OPENROUTER_API_KEY required for smoke test.")
            sys.exit(1)
        run_mini_smoke(questions)
