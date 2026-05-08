"""
D1 -- P4 Superforecaster prompt experiment.

P4 x 2 conditions x 12 models x 114 questions = 2,736 calls.

Per-model crash-safe JSONL files written to results/p4_superforecaster/raw/.
Merge step produces results/p4_superforecaster/p4_results.jsonl.

Usage:
    # Terminal 1 (fast batch A)
    python src/d1_run_p4.py --models gemini-3-flash gemma-4-31b glm-5.1

    # Terminal 2 (fast batch B)
    python src/d1_run_p4.py --models mistral-large gpt-5.4 grok-4.20

    # Terminal 3 (fast batch C)
    python src/d1_run_p4.py --models claude-opus-4.6 deepseek-v3.2-speciale qwen3-max

    # Terminal 4 (slow models, sequential)
    python src/d1_run_p4.py --models gemini-3.1-pro kimi-k2.6 gpt-oss-120b

    # After all finish:
    python src/d1_run_p4.py --merge
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT           = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = ROOT / "data" / "pilot_questions.json"
CACHE_DIR      = ROOT / "data" / "evidence_cache"
RAW_DIR        = ROOT / "results" / "p4_superforecaster" / "raw"
MERGED_FILE    = ROOT / "results" / "p4_superforecaster" / "p4_results.jsonl"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

MAX_TOKENS      = 32768
TIMEOUT         = 900
RAW_RESPONSE_CAP = 200_000

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

CONDITIONS = ["closed_book", "shared_evidence"]

EVIDENCE_BLOCK_HEADER = (
    "\n\nThe following recent news articles may be relevant to this question. "
    "Use them if helpful; disregard if not.\n\n"
)

STRICT_SUFFIX = "\n\nYou must end with exactly one final line: Forecast: X%"

P4_PERSONA = (
    "In this chat, you are a superforecaster who has a strong track record of "
    "accurate forecasting. You evaluate past data and trends carefully for potential "
    "clues to future events, while recognising that the past is an imperfect guide "
    "to the future so you will need to put probabilities on possible future outcomes "
    "(ranging from 0 to 100%). Your specific goal is to maximize the accuracy of "
    "these probability judgments by minimising the Brier scores that your probability "
    "judgments receive once future outcomes are known. Brier scores have two key "
    "components: calibration (across all questions you answer, the probability "
    "estimates you assign to possible future outcomes should correspond as closely as "
    "possible to the objective frequency with which outcomes occur) and resolution "
    "(across all questions, aim to assign higher probabilities to events that occur "
    "than to events that do not occur). You outline your reasons for each forecast: "
    "list the strongest evidence and arguments for making lower or higher estimates "
    "and explain how you balance the evidence to make your own forecast. You begin "
    "this analytic process by looking for reference or comparison classes of similar "
    "events and grounding your initial estimates in base rates of occurrence (how "
    "often do events of this sort occur in situations that look like the present "
    "one?). You then adjust that initial estimate in response to the latest news and "
    "distinctive features of the present situation, recognising the need for flexible "
    "adjustments but also the risks of over-adjusting and excessive volatility. "
    "Superforecasting requires weighing the risks of opposing errors: e.g., of "
    "failing to learn from useful historical patterns vs. over-relying on misleading "
    "patterns. In this process of error balancing, you draw on the 10 commandments "
    "of superforecasting (Tetlock & Gardner, 2015) as well as on other peer-reviewed "
    "research on superforecasting: 1. Triage 2. Break seemingly intractable problems "
    "into tractable sub-problems 3. Strike the right balance between inside and "
    "outside views 4. Strike the right balance between under- and overreacting to "
    "evidence 5. Look for the clashing causal forces at work in each problem 6. "
    "Strive to distinguish as many degrees of doubt as the problem permits but no "
    "more 7. Strike the right balance between under- and overconfidence, between "
    "prudence and decisiveness 8. Look for the errors behind your mistakes but "
    "beware of rearview-mirror hindsight biases 9. Bring out the best in others and "
    "let others bring out the best in you 10. Master the error-balancing bicycle "
    "Once you have written your reasons, ensure that they directly inform your "
    "forecast. Then, you will provide me with a number between 0 and 100 (up to 2 "
    "decimal places) that is your best prediction of the event in the following "
    "format: \"Forecast: X%\". Take a deep breath and work on this problem "
    "step-by-step. The question that you are forecasting as well as some background "
    "information and resolution criteria are below. Read them carefully before making "
    "your prediction."
)


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_p4(q: dict, evidence: str = "") -> str:
    ev = (EVIDENCE_BLOCK_HEADER + evidence) if evidence else ""
    return (
        P4_PERSONA + "\n\n"
        f"Question: {q['question']}\n\n"
        f"Background: {q['background']}\n\n"
        f"Resolution Criteria: {q['resolution_criteria']}\n\n"
        f"Resolution Date: {q['forecast_due_date']}"
        f"{ev}"
        f"{STRICT_SUFFIX}"
    )


# ── v2 parser (inlined) ────────────────────────────────────────────────────────

_LABELS = (
    r"forecast"
    r"|probability"
    r"|my\s+estimate"
    r"|final\s+(?:answer|estimate|probability|posterior)"
    r"|p\s*\(\s*yes\s*\)"
    r"|p\s*\(\s*event\s*\)"
    r"|posterior(?:\s+probability)?"
)
_LABELED_RE = rf"(?:{_LABELS})[^\S\n]*[:\=][^\S\n]*(\d+(?:\.\d+)?)[^\S\n]*(%?)"


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


def parse_forecast_v2(raw: str):
    if not raw or not raw.strip():
        return None, False
    t = raw
    for ch in (" ", " ", " ", " ", " ", "　"):
        t = t.replace(ch, " ")
    t = t.replace("\\%", "%")
    t = re.sub(r"\\boxed\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\\text\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\*+", "", t)
    t = re.sub(r"`+", "", t)
    ms = list(re.finditer(_LABELED_RE, t, re.IGNORECASE))
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


# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_evidence(question_id: str) -> str:
    path = CACHE_DIR / f"{question_id}.json"
    if not path.exists():
        return ""
    try:
        d = json.load(open(path, encoding="utf-8"))
        return d.get("prompt_optimized_string", "") or ""
    except Exception:
        return ""


def raw_file(model_key: str) -> Path:
    return RAW_DIR / f"p4_{model_key}.jsonl"


def load_done(model_key: str) -> set:
    path = raw_file(model_key)
    done = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["condition"], r["question_id"]))
            except Exception:
                pass
    return done


def append_row(model_key: str, row: dict) -> None:
    path = raw_file(model_key)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── API call ───────────────────────────────────────────────────────────────────

def call_model(model_id: str, prompt: str, is_reasoning: bool) -> dict:
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
            data    = resp.json()
            usage   = data.get("usage", {})
            msg     = data["choices"][0]["message"]
            content = (msg.get("content") or msg.get("reasoning") or "").strip()
            return {
                "status": "ok", "response": content, "latency": latency,
                "input_tokens":     usage.get("prompt_tokens"),
                "output_tokens":    usage.get("completion_tokens"),
                "reasoning_tokens": usage.get("completion_tokens_details", {})
                                        .get("reasoning_tokens"),
            }
        except Exception as e:
            if attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            return {"status": "exception", "error": str(e),
                    "latency": round(time.time() - t0, 2)}
    return {"status": "http_error", "code": 429,
            "error": "Rate limited after 3 retries",
            "latency": round(time.time() - t0, 2)}


# ── Run one model ──────────────────────────────────────────────────────────────

def run_model(model_key: str, questions: list, active_conditions: list) -> None:
    model_id  = MODELS[model_key]
    mtype     = MODEL_TYPES[model_key]
    is_reas   = model_key in REASONING_MODEL_KEYS
    done      = load_done(model_key)

    total = len(questions) * len(active_conditions)
    skip  = sum(1 for q in questions for c in active_conditions
                if (c, q["question_id"]) in done)

    print(f"\n{'='*72}")
    print(f"Model: {model_key} ({mtype})  ->  {model_id}")
    print(f"  {total} cells  |  {skip} already done  |  {total - skip} to run")
    print(f"{'='*72}")

    parse_failures = []

    for cond in active_conditions:
        print(f"\n  -- Condition: {cond} --")
        for i, q in enumerate(questions):
            qid = q["question_id"]
            key = (cond, qid)
            if key in done:
                print(f"    Q{i+1:>3} SKIP")
                continue

            evidence = load_evidence(qid) if cond == "shared_evidence" else ""
            prompt   = build_p4(q, evidence)

            mkt = q.get("freeze_datetime_value", "?")
            try:
                mkt_pct = f"{float(mkt):.0%}"
            except (ValueError, TypeError):
                mkt_pct = str(mkt)

            print(f"    Q{i+1:>3} [mkt={mkt_pct}] ...", end=" ", flush=True)

            result = call_model(model_id, prompt, is_reas)

            forecast  = None
            parse_ok  = False
            brier     = None

            if result["status"] == "ok":
                forecast, parse_ok = parse_forecast_v2(result["response"])
                if parse_ok and forecast is not None:
                    res_val = float(q["resolution_value"])
                    brier   = round((forecast - res_val) ** 2, 4)

                rtoks = result.get("reasoning_tokens")
                rtok_str = f"  r_tok={rtoks}" if rtoks else ""
                if parse_ok:
                    print(f"{result['latency']}s  Forecast={forecast:.0%}  Brier={brier}{rtok_str}")
                else:
                    parse_failures.append((cond, qid))
                    snippet = result["response"][:200].encode("ascii", "replace").decode("ascii")
                    print(f"{result['latency']}s  PARSE FAIL")
                    print(f"      Response: {snippet}")
            else:
                err = (result.get("error") or "")[:80].encode("ascii", "replace").decode("ascii")
                print(f"ERROR ({result['status']}: {err})")

            row = {
                "model_key":        model_key,
                "model_id":         model_id,
                "model_type":       mtype,
                "question_id":      qid,
                "source":           q["source"],
                "topic":            q.get("topic"),
                "market_prob":      q.get("freeze_datetime_value"),
                "prompt":           "P4",
                "condition":        cond,
                "forecast":         forecast,
                "resolution_value": q["resolution_value"],
                "brier_score":      brier,
                "parse_success":    parse_ok,
                "latency_seconds":  result.get("latency"),
                "input_tokens":     result.get("input_tokens"),
                "output_tokens":    result.get("output_tokens"),
                "reasoning_tokens": result.get("reasoning_tokens"),
                "raw_response":     result.get("response", "")[:RAW_RESPONSE_CAP],
                "error":            result.get("error") if result["status"] != "ok" else None,
            }
            append_row(model_key, row)
            done.add(key)

    # Per-model summary
    rows = []
    if raw_file(model_key).exists():
        with open(raw_file(model_key), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass

    parsed = [r for r in rows if r.get("parse_success") and r["condition"] in active_conditions]
    total_done = len([r for r in rows if r["condition"] in active_conditions])
    print(f"\n  {model_key} summary: parsed={len(parsed)}/{total_done}")

    for cond in active_conditions:
        pr = [r for r in parsed if r["condition"] == cond]
        if pr:
            avg_b = sum(r["brier_score"] for r in pr if r["brier_score"] is not None) / len(pr)
            lats  = [r["latency_seconds"] for r in pr if r.get("latency_seconds")]
            avg_l = sum(lats) / len(lats) if lats else 0
            print(f"    {cond:<18} P4  Brier={avg_b:.4f}  lat={avg_l:.1f}s  n={len(pr)}")

    if parse_failures:
        print(f"  Parse failures ({len(parse_failures)}):")
        for cond, qid in parse_failures:
            print(f"    {cond}  q={qid[:30]}")

    print(f"  Saved to {raw_file(model_key)}")


# ── Merge ──────────────────────────────────────────────────────────────────────

def merge_results() -> None:
    all_rows = []
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for model_key in MODELS:
        path = raw_file(model_key)
        if not path.exists():
            print(f"  {model_key}: MISSING")
            continue
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_rows.append(json.loads(line))
                        count += 1
                    except Exception:
                        pass
        print(f"  {model_key}: {count} rows")

    MERGED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MERGED_FILE, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    parsed = sum(1 for r in all_rows if r.get("parse_success"))
    print(f"\nMerged {len(all_rows)} rows  ({parsed} parsed, {parsed/len(all_rows)*100:.2f}%)")
    print(f"-> {MERGED_FILE}")

    # Quick Brier table
    print(f"\n{'Model':<30} {'CB-P4':>7} {'SE-P4':>7} {'Overall':>9}")
    print("-" * 55)
    for mk in MODELS:
        mr = [r for r in all_rows if r["model_key"] == mk and r.get("parse_success")]
        if not mr:
            continue
        cb = [r for r in mr if r["condition"] == "closed_book"]
        se = [r for r in mr if r["condition"] == "shared_evidence"]
        cb_b = f"{sum(r['brier_score'] for r in cb)/len(cb):.4f}" if cb else "N/A"
        se_b = f"{sum(r['brier_score'] for r in se)/len(se):.4f}" if se else "N/A"
        ov_b = f"{sum(r['brier_score'] for r in mr)/len(mr):.4f}"
        print(f"  {mk:<28}  {cb_b:>7}  {se_b:>7}  {ov_b:>9}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        help="Which models to run (default: all)")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                        default=CONDITIONS)
    parser.add_argument("--merge", action="store_true",
                        help="Merge per-model JSONL files into p4_results.jsonl")
    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    if args.merge:
        print("Merging P4 results...")
        merge_results()
        sys.exit(0)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    active_models = args.models if args.models else list(MODELS.keys())
    active_conditions = args.conditions

    print(f"P4 Superforecaster Experiment")
    print(f"Models:     {active_models}")
    print(f"Conditions: {active_conditions}")
    print(f"Questions:  {len(questions)}")
    print(f"Output:     {RAW_DIR}")

    for model_key in active_models:
        run_model(model_key, questions, active_conditions)

    print("\nDone.")
