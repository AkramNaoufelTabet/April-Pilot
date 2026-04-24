"""
A3 ? Topic labeling via OpenRouter (Claude Sonnet).

Sends questions in batches to the model and assigns one of the 12 topic labels.
Saves results to data/labeled/labeled_questions.json.

Usage:
  python src/a3_label_topics.py

Requires: OPENROUTER_API_KEY environment variable.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
IN_FILE = ROOT / "data" / "labeled" / "filtered_questions.json"
OUT_FILE = ROOT / "data" / "labeled" / "labeled_questions.json"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "anthropic/claude-sonnet-4.6"
BATCH_SIZE = 15  # questions per API call
TEMPERATURE = 0

TOPIC_LABELS = [
    "geopolitics",      # international relations, elections, diplomacy, governance
    "conflict",         # wars, military operations, armed groups, peace negotiations
    "ai_technology",    # AI, ML, tech companies, software, hardware
    "economics",        # macro indicators, company performance, trade, monetary policy
    "public_health",    # disease, healthcare, medical research, epidemics
    "energy",           # oil, gas, renewables, climate policy, commodities
    "finance_market",   # stocks, crypto, markets, investment, financial instruments
    "entertainment",    # movies, music, TV, awards, celebrity, pop culture
    "sports",           # all competitive sports and sporting events
    "other",            # anything that doesn't fit above categories
]


def build_batch_prompt(questions: list[dict]) -> str:
    labels_str = ", ".join(TOPIC_LABELS)
    parts = [
        f"Classify each of the following {len(questions)} forecasting questions into "
        f"exactly one topic label.\n\n"
        f"Available labels: {labels_str}\n\n"
        f"Respond with a JSON array (one object per question) in this exact format:\n"
        f'[{{"idx": 0, "topic": "<label>", "confidence": <0.0-1.0>}}, ...]\n\n'
        f"Questions:"
    ]

    for i, q in enumerate(questions):
        bg = (q.get("background") or "")[:400]  # trim to keep context short
        parts.append(
            f"\n[{i}] Question: {q['question']}\n"
            f"Background: {bg}"
        )

    return "\n".join(parts)


def call_openrouter(prompt: str, retry: int = 3) -> str:
    """Call OpenRouter chat completions and return the assistant message text."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": 2048,
    }

    for attempt in range(retry):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.HTTPError as e:
            print(f"  HTTP error {e.response.status_code}: {e.response.text[:200]}")
            if attempt < retry - 1:
                wait = 2 ** attempt
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
        except Exception as e:
            print(f"  Error: {e}")
            if attempt < retry - 1:
                time.sleep(2)

    raise RuntimeError("OpenRouter call failed after retries.")


def parse_batch_response(text: str, batch_size: int) -> list[dict]:
    """Extract JSON array from model response, with fallback parsing."""
    # Try to find JSON array in response
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            results = json.loads(match.group())
            if isinstance(results, list):
                return results
        except json.JSONDecodeError:
            pass

    # Fallback: try parsing the whole response as JSON
    try:
        results = json.loads(text.strip())
        if isinstance(results, list):
            return results
    except json.JSONDecodeError:
        pass

    print(f"  [WARN] Could not parse response, defaulting to 'other' for batch.")
    return [{"idx": i, "topic": "other", "confidence": 0.0} for i in range(batch_size)]


def label_questions(questions: list[dict]) -> list[dict]:
    labeled = list(questions)  # copy
    # Initialize topic fields
    for q in labeled:
        q["topic"] = None
        q["topic_confidence"] = None

    total_batches = (len(questions) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Labeling {len(questions)} questions in {total_batches} batches of <={BATCH_SIZE}...\n")

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(questions))
        batch = questions[start:end]

        print(f"  Batch {batch_idx + 1}/{total_batches} (questions {start}-{end - 1})...", end=" ", flush=True)

        prompt = build_batch_prompt(batch)
        response_text = call_openrouter(prompt)
        results = parse_batch_response(response_text, len(batch))

        # Map results back by idx
        idx_map = {r.get("idx", i): r for i, r in enumerate(results)}

        for i, q in enumerate(batch):
            res = idx_map.get(i, {})
            topic = res.get("topic", "other")
            confidence = res.get("confidence", 0.0)

            # Validate topic
            if topic not in TOPIC_LABELS:
                print(f"\n  [WARN] Invalid topic '{topic}', defaulting to 'other'")
                topic = "other"

            labeled[start + i]["topic"] = topic
            labeled[start + i]["topic_confidence"] = confidence

        topics_in_batch = [labeled[start + i]["topic"] for i in range(len(batch))]
        topic_summary = {}
        for t in topics_in_batch:
            topic_summary[t] = topic_summary.get(t, 0) + 1
        print(f"done. Topics: {topic_summary}")

        # Save intermediate progress after each batch
        with open(OUT_FILE, "w", encoding="utf-8") as f:
            import json as _json
            _json.dump(labeled, f, indent=2, ensure_ascii=False)

        # Small delay between batches to avoid rate limiting
        if batch_idx < total_batches - 1:
            time.sleep(1)

    return labeled


def print_topic_distribution(questions: list[dict]) -> None:
    from collections import Counter
    counts = Counter(q.get("topic", "unknown") for q in questions)
    print("\nTopic distribution:")
    for label in TOPIC_LABELS + ["unknown"]:
        n = counts.get(label, 0)
        if n > 0:
            bar = "?" * (n // 2)
            print(f"  {label:<20} {n:>4}  {bar}")


if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY environment variable is not set.")
        sys.exit(1)

    if not IN_FILE.exists():
        print(f"ERROR: Input file not found: {IN_FILE}")
        print("Run a2_filter_questions.py first.")
        sys.exit(1)

    with open(IN_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} filtered questions.")

    # Check for partially labeled output (resume support)
    if OUT_FILE.exists():
        with open(OUT_FILE, encoding="utf-8") as f:
            existing = json.load(f)
        already_labeled = {q["question_id"] for q in existing if q.get("topic")}
        if len(already_labeled) == len(questions):
            print(f"All {len(questions)} questions already labeled. Loading existing results.")
            questions = existing
        else:
            print(f"{len(already_labeled)} already labeled, {len(questions) - len(already_labeled)} remaining.")
            # Filter to unlabeled
            unlabeled = [q for q in questions if q["question_id"] not in already_labeled]
            labeled_new = label_questions(unlabeled)
            # Merge
            labeled_map = {q["question_id"]: q for q in existing}
            for q in labeled_new:
                labeled_map[q["question_id"]] = q
            questions = [labeled_map.get(q["question_id"], q) for q in questions]
    else:
        questions = label_questions(questions)

    print_topic_distribution(questions)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(questions)} labeled questions to {OUT_FILE}")
    print("\nA3 complete.")
