"""
B1 — Retrieve and cache AskNews evidence for each pilot question.

For each question in data/pilot_questions.json:
  - Search AskNews using the question text as query
  - Time-bound to articles published BEFORE freeze_datetime (methodological requirement)
  - Store prompt_optimized_string + article metadata to data/evidence_cache/{question_id}.json
  - Skip if already cached and valid

Usage:
    python src/b1_retrieve_evidence.py               # all 114 questions
    python src/b1_retrieve_evidence.py --test        # 5 questions across topics (sanity check)
    python src/b1_retrieve_evidence.py --ids q1 q2   # specific question IDs

Requires: ASKNEWS_API_KEY environment variable
          pip install asknews
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from asknews_sdk import AskNewsSDK

ROOT = Path(__file__).parent.parent
QUESTIONS_FILE = ROOT / "data" / "pilot_questions.json"
CACHE_DIR = ROOT / "data" / "evidence_cache"

ASKNEWS_API_KEY = os.environ.get("ASKNEWS_API_KEY", "")

N_ARTICLES = 10         # plan cap; API returns in relevance order so this is top-10
SLEEP_BETWEEN = 1.2     # seconds between calls (Spelunker plan: 1 req/s burst 5)


SEARCH_WINDOW_DAYS = 90  # look back this many days before freeze_datetime


def parse_freeze_datetime(freeze_str: str) -> int | None:
    """Convert freeze_datetime string to Unix timestamp."""
    if not freeze_str:
        return None
    try:
        clean = freeze_str[:19]  # "2026-01-04T00:00:00"
        dt = datetime.strptime(clean, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except ValueError:
        return None


def is_cached(question_id: str) -> bool:
    path = CACHE_DIR / f"{question_id}.json"
    if not path.exists():
        return False
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return bool(data.get("prompt_optimized_string"))
    except Exception:
        return False


def retrieve(sdk: AskNewsSDK, q: dict) -> dict:
    question_id = q["question_id"]
    query = q["question"]
    freeze_str = q.get("freeze_datetime", "") or q.get("forecast_due_date", "")
    end_ts = parse_freeze_datetime(freeze_str)
    start_ts = (end_ts - SEARCH_WINDOW_DAYS * 86400) if end_ts else None

    result = {
        "question_id": question_id,
        "query": query,
        "freeze_datetime": freeze_str,
        "end_timestamp": end_ts,
        "start_timestamp": start_ts,
        "n_articles_returned": 0,
        "prompt_optimized_string": "",
        "article_dates": [],
        "articles": [],
        "error": None,
    }

    try:
        resp = sdk.news.search_news(
            query=query,
            n_articles=N_ARTICLES,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            # time_filter defaults to "crawl_date" — required for historical archive;
            # "pub_date" silently returns 0 results when historical=True.
            # diversify_sources and sort_by override n_articles in historical mode — omit both.
            return_type="both",
            historical=True,
            method="kw",
        )

        result["prompt_optimized_string"] = resp.as_string or ""

        articles = []
        pub_dates = []
        for art in (resp.as_dicts or []):
            pub = None
            if art.pub_date:
                try:
                    pub = art.pub_date.isoformat() if hasattr(art.pub_date, "isoformat") else str(art.pub_date)
                    pub_dates.append(pub)
                except Exception:
                    pass
            articles.append({
                "title": art.eng_title or art.title or "",
                "pub_date": pub,
                "domain": str(art.domain_url) if art.domain_url else "",
                "summary": art.summary or "",
                "url": str(art.article_url) if art.article_url else "",
            })

        result["n_articles_returned"] = len(articles)
        result["articles"] = articles
        result["article_dates"] = pub_dates

    except Exception as e:
        result["error"] = str(e)

    return result


def print_sanity(result: dict, q: dict) -> None:
    print(f"\n  Question : {q['question'][:90]}")
    print(f"  Topic    : {q.get('topic', '?')}")
    print(f"  Freeze   : {result['freeze_datetime']} (end_ts={result['end_timestamp']})")
    print(f"  Articles : {result['n_articles_returned']}")
    if result["error"]:
        print(f"  ERROR    : {result['error']}")
        return
    dates = sorted(d for d in result["article_dates"] if d)
    if dates:
        print(f"  Date range: {dates[0][:10]}  -->  {dates[-1][:10]}")
        # Warn if any article is after freeze
        if result["end_timestamp"]:
            over = [d for d in dates if d[:10] >= result["freeze_datetime"][:10]]
            if over:
                print(f"  !! WARN: {len(over)} articles may be at/after freeze date")
    snippet = (result["prompt_optimized_string"] or "")[:200].replace("\n", " ")
    snippet = snippet.encode("ascii", "replace").decode("ascii")
    print(f"  Preview  : {snippet}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Run on 5 questions spanning different topics (sanity check)")
    parser.add_argument("--ids", nargs="+", metavar="ID",
                        help="Run only these question IDs")
    parser.add_argument("--force", action="store_true",
                        help="Re-retrieve even if already cached")
    args = parser.parse_args()

    if not ASKNEWS_API_KEY:
        print("ERROR: ASKNEWS_API_KEY not set.")
        sys.exit(1)

    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        all_questions = json.load(f)

    if args.ids:
        questions = [q for q in all_questions if q["question_id"] in args.ids]
    elif args.test:
        # Pick one question per topic from 5 target topics
        target_topics = ["geopolitics", "ai_technology", "conflict", "economics", "finance_market"]
        questions = []
        seen_topics = set()
        for q in all_questions:
            t = q.get("topic", "other")
            if t in target_topics and t not in seen_topics:
                questions.append(q)
                seen_topics.add(t)
            if len(questions) == 5:
                break
    else:
        questions = all_questions

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    sdk = AskNewsSDK(api_key=ASKNEWS_API_KEY)

    print(f"Evidence retrieval: {len(questions)} questions")
    if args.test:
        print("Mode: SANITY CHECK (5 questions)")
    print()

    skipped = 0
    success = 0
    failed = 0

    for i, q in enumerate(questions):
        qid = q["question_id"]
        cache_path = CACHE_DIR / f"{qid}.json"

        if not args.force and is_cached(qid):
            print(f"[{i+1}/{len(questions)}] SKIP (cached): {qid[:30]}")
            skipped += 1
            continue

        print(f"[{i+1}/{len(questions)}] {qid[:30]}  topic={q.get('topic','?')} ...", end=" ", flush=True)

        result = retrieve(sdk, q)

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        if result["error"]:
            print(f"ERROR: {result['error'][:80]}")
            failed += 1
        else:
            print(f"ok  ({result['n_articles_returned']} articles)")
            success += 1

        if args.test:
            print_sanity(result, q)

        if i < len(questions) - 1:
            time.sleep(SLEEP_BETWEEN)

    print(f"\nDone. success={success}  skipped={skipped}  failed={failed}")
    print(f"Cache: {CACHE_DIR}")
