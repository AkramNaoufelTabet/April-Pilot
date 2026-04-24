"""
B0 -- Model verification.

Pings each of the 12 experiment models via OpenRouter with a trivial prompt,
records latency / tokens / cost / response text, and flags any that fail.

MODELS is the single source of truth: each entry holds the OpenRouter
candidate strings to try (first hit wins) and the model_type classification.

Usage:
    python src/b0_verify_models.py           # verify all 12
    python src/b0_verify_models.py --only gpt-5.4 mistral-large

Requires: OPENROUTER_API_KEY environment variable.
Output:   results/model_verification.json
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
OUT_FILE = ROOT / "results" / "model_verification.json"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
TRIVIAL_PROMPT = "Reply with just the word 'ok' and nothing else."
TIMEOUT = 60  # seconds per request

# ---------------------------------------------------------------------------
# Single source of truth for all 12 experiment models.
#
# Each entry:
#   "key": {
#       "model_type": "reasoning" | "standard",
#       "candidates":  [<first_choice>, <fallback>, ...]  -- tried in order
#   }
#
# Key naming: plain model name, no capability suffixes.
# model_type carries the reasoning/standard classification explicitly.
# ---------------------------------------------------------------------------
MODELS = {
    # ── Reasoning models (7) ────────────────────────────────────────────────
    "gpt-5.4": {
        "model_type": "reasoning",
        "candidates": [
            "openai/gpt-5.4",
        ],
    },
    "claude-opus-4.6": {
        "model_type": "reasoning",
        "candidates": [
            "anthropic/claude-opus-4.6",
        ],
    },
    "gemini-3.1-pro": {
        "model_type": "reasoning",
        "candidates": [
            "google/gemini-3.1-pro-preview",
        ],
    },
    "grok-4.20": {
        "model_type": "reasoning",
        "candidates": [
            "x-ai/grok-4.20",
        ],
    },
    "qwen3-max": {
        "model_type": "reasoning",
        "candidates": [
            "qwen/qwen3-max-thinking",
        ],
    },
    "deepseek-v3.2-speciale": {
        "model_type": "reasoning",
        "candidates": [
            "deepseek/deepseek-v3.2-speciale",
        ],
    },
    "kimi-k2.6": {
        "model_type": "reasoning",
        "candidates": [
            "moonshotai/kimi-k2.6",
        ],
    },
    # ── Standard models (5) ─────────────────────────────────────────────────
    "gemini-3-flash": {
        "model_type": "standard",
        "candidates": [
            "google/gemini-3-flash-preview",
        ],
    },
    "gemma-4-31b": {
        "model_type": "standard",
        "candidates": [
            "google/gemma-4-31b-it",
        ],
    },
    "glm-5.1": {
        "model_type": "standard",
        "candidates": [
            "z-ai/glm-5.1",
        ],
    },
    "gpt-oss-120b": {
        "model_type": "standard",
        "candidates": [
            "openai/gpt-oss-120b",
        ],
    },
    "mistral-large": {
        "model_type": "standard",
        "candidates": [
            "mistralai/mistral-large-2512",
        ],
    },
}


def ping_model(model_id: str) -> dict:
    """Send the trivial prompt to one model string; return a result dict."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": TRIVIAL_PROMPT}],
        "temperature": 0,
        "max_tokens": 512,
    }

    t0 = time.time()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
        latency = round(time.time() - t0, 3)

        if resp.status_code != 200:
            return {
                "status": "error",
                "http_code": resp.status_code,
                "error_message": resp.text[:300],
                "latency_seconds": latency,
            }

        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]
        # Reasoning models may put output in "reasoning" when content is null
        raw_content = msg.get("content") or msg.get("reasoning") or ""
        response_text = raw_content.strip()[:120]
        usage = data.get("usage", {})

        cost_usd = data.get("cost")

        return {
            "status": "ok",
            "http_code": 200,
            "response_text": response_text,
            "latency_seconds": latency,
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cost_usd": cost_usd,
            "finish_reason": choice.get("finish_reason"),
        }

    except requests.Timeout:
        latency = round(time.time() - t0, 3)
        return {
            "status": "timeout",
            "latency_seconds": latency,
            "error_message": f"Timed out after {TIMEOUT}s",
        }
    except Exception as e:
        latency = round(time.time() - t0, 3)
        return {
            "status": "exception",
            "latency_seconds": latency,
            "error_message": str(e),
        }


def verify_all(only: list[str] | None = None) -> tuple[dict, dict]:
    """Verify models. Pass `only` to restrict to a subset of keys."""
    results = {}
    resolved_models = {}

    keys = [k for k in MODELS if (only is None or k in only)]
    print(f"Verifying {len(keys)} model(s)...\n")

    for key in keys:
        entry = MODELS[key]
        model_type = entry["model_type"]
        candidates = entry["candidates"]
        print(f"  [{key}]  (type={model_type})")

        resolved_id = None
        result = None

        for candidate in candidates:
            print(f"    Trying {candidate} ...", end=" ", flush=True)
            result = ping_model(candidate)

            if result["status"] == "ok":
                print(f"OK  ({result['latency_seconds']}s, "
                      f"in={result['input_tokens']} out={result['output_tokens']}, "
                      f"response={repr(result['response_text'][:60])})")
                resolved_id = candidate
                break
            else:
                print(f"FAIL ({result['status']}: {result.get('error_message','')[:80]})")

        if resolved_id is None:
            print(f"    !! All candidates failed for [{key}]")

        results[key] = {
            "model_key": key,
            "model_type": model_type,
            "resolved_model_id": resolved_id,
            "candidates_tried": candidates,
            **result,
        }
        resolved_models[key] = resolved_id

        time.sleep(1)

    return results, resolved_models


def print_summary(results: dict, resolved_models: dict) -> None:
    passed = [k for k, v in results.items() if v.get("resolved_model_id")]
    failed = [k for k, v in results.items() if not v.get("resolved_model_id")]
    n = len(passed) + len(failed)

    print("\n" + "=" * 70)
    print(f"VERIFICATION SUMMARY: {len(passed)}/{n} models OK")
    print("=" * 70)

    reasoning = [k for k in passed if results[k]["model_type"] == "reasoning"]
    standard  = [k for k in passed if results[k]["model_type"] == "standard"]

    print(f"\nREASONING ({len(reasoning)}):")
    for k in reasoning:
        v = results[k]
        print(f"  {k:<24} -> {v['resolved_model_id']}")
        print(f"    latency={v['latency_seconds']}s  "
              f"tokens={v['input_tokens']}/{v['output_tokens']}")

    print(f"\nSTANDARD ({len(standard)}):")
    for k in standard:
        v = results[k]
        print(f"  {k:<24} -> {v['resolved_model_id']}")
        print(f"    latency={v['latency_seconds']}s  "
              f"tokens={v['input_tokens']}/{v['output_tokens']}")

    if failed:
        print("\nFAILED:")
        for k in failed:
            v = results[k]
            print(f"  {k:<24}  tried: {v['candidates_tried']}")
            print(f"    error: {v.get('error_message','?')[:120]}")

    print("\nResolved MODELS dict (for experiment scripts):")
    print("MODELS = {")
    for k, model_id in resolved_models.items():
        mtype = results[k]["model_type"]
        if model_id:
            print(f'    "{k}": "{model_id}",  # {mtype}')
        else:
            print(f'    "{k}": None,  # {mtype} -- FAILED')
    print("}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", metavar="KEY",
                        help="Verify only these model keys (space-separated)")
    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY is not set.")
        sys.exit(1)

    results, resolved_models = verify_all(only=args.only)
    print_summary(results, resolved_models)

    # When running --only, merge new results into the existing file
    if args.only and OUT_FILE.exists():
        with open(OUT_FILE, encoding="utf-8") as f:
            existing = json.load(f)
        existing["details"].update(results)
        existing["resolved_models"].update(resolved_models)
        # Drop any keys no longer in MODELS
        for stale in [k for k in list(existing["resolved_models"]) if k not in MODELS]:
            existing["details"].pop(stale, None)
            existing["resolved_models"].pop(stale, None)
        results = existing["details"]
        resolved_models = existing["resolved_models"]

    output = {
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "passed": sum(1 for v in results.values() if v.get("resolved_model_id")),
        "failed": sum(1 for v in results.values() if not v.get("resolved_model_id")),
        "resolved_models": resolved_models,
        "details": results,
    }
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nFull results saved to {OUT_FILE}")

    if output["failed"] > 0:
        print(f"\nWARNING: {output['failed']} model(s) failed.")
        sys.exit(1)
    else:
        print("\nAll models verified. Safe to proceed to Phase B.")
