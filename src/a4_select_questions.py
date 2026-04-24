"""
A4 — Stratified selection of 100 pilot questions.

Selection strategy (PhD: "Forecasting for Social Innovation using Language Technologies"):
  - Hard floors: priority domains filled first (geopolitics, conflict, ai_technology,
    economics, public_health, energy, finance_market)
  - Hard caps: sports ≤ 8, entertainment ≤ 5, other ≤ 5
  - Source cap: no single platform exceeds 45 questions
  - Difficulty: natural distribution (no forced binning)

Output: data/pilot_questions.json
"""

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
IN_FILE = ROOT / "data" / "labeled" / "labeled_questions.json"
OUT_FILE = ROOT / "data" / "pilot_questions.json"

TARGET_N = 114  # 96 priority + 8 sports + 5 entertainment + 5 other
RANDOM_SEED = 42

# Hard caps — never exceed these regardless of availability
TOPIC_CAPS = {
    "sports": 8,
    "entertainment": 5,
    "other": 5,
}
TOPIC_DEFAULT_CAP = 30  # generous ceiling for priority topics

# Floors set to match previous priority allocations + caps for non-priority topics
# so Phase 1 locks in all allocations directly
TOPIC_FLOORS = {
    "geopolitics": 20,
    "conflict": 18,
    "ai_technology": 20,
    "economics": 14,
    "public_health": 6,
    "energy": 4,
    "finance_market": 14,
    "sports": 8,       # floor == cap
    "entertainment": 5, # floor == cap
    "other": 5,         # floor == cap
}

# Priority domains must collectively make up >= 70 of 114 questions
PRIORITY_TOPICS = {"geopolitics", "conflict", "ai_technology", "economics",
                   "public_health", "energy"}

MAX_PER_SOURCE = 45


# ── Quality filter ─────────────────────────────────────────────────────────────

def is_low_quality(q: dict) -> tuple[bool, str]:
    bg = (q.get("background") or "").strip()
    question = (q.get("question") or "").strip()
    res_criteria = (q.get("resolution_criteria") or "").strip()

    if len(bg) < 50:
        return True, f"background too short ({len(bg)} chars)"
    if len(question) < 10:
        return True, "question text too short"
    if len(res_criteria) < 30:
        return True, f"resolution criteria too short ({len(res_criteria)} chars)"

    url_pattern = r"https?://\S+"
    urls_in_q = re.findall(url_pattern, question)
    if urls_in_q and len(question.replace(urls_in_q[0], "").strip()) < 20:
        return True, "question is mostly a URL"

    return False, ""


# ── Diverse selection within a topic pool ──────────────────────────────────────

def diverse_select(pool: list[dict], n: int) -> list[dict]:
    """Select n from pool, round-robining across sources."""
    if len(pool) <= n:
        return list(pool)

    by_source: dict[str, list[dict]] = defaultdict(list)
    for q in pool:
        by_source[q["source"]].append(q)
    for src in by_source:
        random.shuffle(by_source[src])

    selected = []
    sources = list(by_source.keys())
    iters = {s: iter(by_source[s]) for s in sources}
    active = list(sources)

    while len(selected) < n and active:
        for src in list(active):
            if len(selected) >= n:
                break
            try:
                selected.append(next(iters[src]))
            except StopIteration:
                active.remove(src)

    # Fill remainder if round-robin ran out early
    selected_ids = {id(q) for q in selected}
    remainder = [q for q in pool if id(q) not in selected_ids]
    random.shuffle(remainder)
    selected.extend(remainder[:n - len(selected)])

    return selected[:n]


# ── Main selection ─────────────────────────────────────────────────────────────

def select_questions(questions: list[dict]) -> list[dict]:
    random.seed(RANDOM_SEED)

    # Quality filter
    quality_qs, dropped = [], []
    for q in questions:
        low_q, reason = is_low_quality(q)
        if low_q:
            dropped.append((q["question_id"], reason))
        else:
            quality_qs.append(q)

    print(f"Quality filter: {len(questions)} -> {len(quality_qs)} ({len(dropped)} dropped)")
    if dropped[:3]:
        for qid, reason in dropped[:3]:
            print(f"  {qid}: {reason}")

    # Group by topic
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for q in quality_qs:
        topic = q.get("topic") or "other"
        by_topic[topic].append(q)

    print(f"\nAvailable pool by topic:")
    for t, qs in sorted(by_topic.items(), key=lambda x: -len(x[1])):
        cap = TOPIC_CAPS.get(t, TOPIC_DEFAULT_CAP)
        floor = TOPIC_FLOORS.get(t, 0)
        print(f"  {t:<20} {len(qs):>4} available  (floor={floor}, cap={cap})")

    # ── Phase 1: guarantee floors ──────────────────────────────────────────────
    allocations: dict[str, int] = {}
    for topic in by_topic:
        cap = TOPIC_CAPS.get(topic, TOPIC_DEFAULT_CAP)
        floor = TOPIC_FLOORS.get(topic, 0)
        available = len(by_topic[topic])
        allocations[topic] = min(floor, available, cap)

    phase1_total = sum(allocations.values())
    remaining_slots = TARGET_N - phase1_total

    print(f"\nPhase 1 (floors guaranteed): {phase1_total} questions allocated")
    print(f"Remaining slots for Phase 2: {remaining_slots}")

    # ── Phase 2: distribute remaining slots proportionally ─────────────────────
    if remaining_slots > 0:
        # Headroom = how many more each topic can accept within its cap
        headroom = {}
        for topic in by_topic:
            cap = TOPIC_CAPS.get(topic, TOPIC_DEFAULT_CAP)
            available = len(by_topic[topic])
            headroom[topic] = min(available, cap) - allocations[topic]

        total_headroom = sum(h for h in headroom.values() if h > 0)
        topics_with_room = [t for t, h in headroom.items() if h > 0]

        if total_headroom > 0:
            # Distribute proportionally to headroom
            extras: dict[str, int] = {t: 0 for t in topics_with_room}
            for t in topics_with_room:
                extras[t] = round(headroom[t] / total_headroom * remaining_slots)

            # Correct rounding drift
            diff = remaining_slots - sum(extras.values())
            for t in sorted(topics_with_room, key=lambda x: -headroom[x]):
                if diff == 0:
                    break
                cap = TOPIC_CAPS.get(t, TOPIC_DEFAULT_CAP)
                can_add = min(len(by_topic[t]), cap) - allocations[t] - extras[t]
                if diff > 0 and can_add > 0:
                    extras[t] += 1
                    diff -= 1
                elif diff < 0 and extras[t] > 0:
                    extras[t] -= 1
                    diff += 1

            for t, extra in extras.items():
                allocations[t] = allocations.get(t, 0) + extra

    # ── Validate constraints ───────────────────────────────────────────────────
    total_alloc = sum(allocations.values())
    priority_alloc = sum(allocations.get(t, 0) for t in PRIORITY_TOPICS)

    print(f"\nFinal allocations (total={total_alloc}, priority={priority_alloc}):")
    for t in sorted(allocations, key=lambda x: -allocations[x]):
        cap = TOPIC_CAPS.get(t, TOPIC_DEFAULT_CAP)
        floor = TOPIC_FLOORS.get(t, 0)
        flag = ""
        if allocations[t] < floor:
            flag = f"  !! BELOW FLOOR ({floor})"
        print(f"  {t:<20} {allocations[t]:>3}  (floor={floor}, cap={cap}){flag}")

    if priority_alloc < 70:
        print(f"  !! WARNING: priority topics = {priority_alloc} (target ≥ 70)")

    # ── Select questions per topic ─────────────────────────────────────────────
    selected: list[dict] = []
    for topic, alloc in allocations.items():
        if alloc <= 0:
            continue
        pool = list(by_topic[topic])
        random.shuffle(pool)
        chosen = diverse_select(pool, alloc)
        selected.extend(chosen)

    random.shuffle(selected)

    # ── Enforce source cap ─────────────────────────────────────────────────────
    source_counts = Counter(q["source"] for q in selected)
    for src, cnt in source_counts.items():
        if cnt > MAX_PER_SOURCE:
            excess = cnt - MAX_PER_SOURCE
            removable = [q for q in selected if q["source"] == src]
            random.shuffle(removable)
            remove_ids = {id(q) for q in removable[:excess]}
            selected = [q for q in selected if id(q) not in remove_ids]
            print(f"  [source cap] Removed {excess} '{src}' questions (cap={MAX_PER_SOURCE})")

    return selected


def format_output(q: dict) -> dict:
    return {k: v for k, v in q.items()
            if not k.startswith("_") and k != "topic_confidence"}


def print_summary(selected: list[dict]) -> None:
    topic_counts = Counter(q.get("topic", "?") for q in selected)
    source_counts = Counter(q["source"] for q in selected)

    print(f"\n{'='*60}")
    print(f"FINAL SELECTION: {len(selected)} questions")
    print(f"{'='*60}")

    print("\nTopic distribution:")
    priority = ["geopolitics", "conflict", "ai_technology", "economics",
                "public_health", "energy", "finance_market"]
    other_topics = [t for t in topic_counts if t not in priority]
    for t in priority + sorted(other_topics):
        n = topic_counts.get(t, 0)
        floor = TOPIC_FLOORS.get(t, 0)
        cap = TOPIC_CAPS.get(t, TOPIC_DEFAULT_CAP)
        bar = "#" * n
        print(f"  {t:<20} {n:>3}  {bar}  (floor={floor}, cap={cap})")

    priority_n = sum(topic_counts.get(t, 0) for t in PRIORITY_TOPICS)
    print(f"\n  Priority domains total: {priority_n}/{sum(topic_counts.values())}  (target >= 70)")

    print(f"\nSource distribution:")
    for src, n in source_counts.most_common():
        print(f"  {src:<12} {n:>3}  (cap={MAX_PER_SOURCE})")

    # Difficulty histogram
    bins = [0] * 10
    for q in selected:
        try:
            val = max(0.0, min(1.0, float(q.get("freeze_datetime_value", 0.5))))
            bins[min(int(val * 10), 9)] += 1
        except (ValueError, TypeError):
            bins[5] += 1

    print(f"\nDifficulty distribution (freeze_datetime_value deciles):")
    for i, cnt in enumerate(bins):
        lo, hi = i * 0.1, (i + 1) * 0.1
        bar = "#" * cnt
        print(f"  [{lo:.1f}-{hi:.1f}]  {cnt:>3}  {bar}")


if __name__ == "__main__":
    if not IN_FILE.exists():
        print(f"ERROR: Input file not found: {IN_FILE}")
        print("Run a3_label_topics.py first.")
        sys.exit(1)

    with open(IN_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} labeled questions.")

    unlabeled = [q for q in questions if not q.get("topic")]
    if unlabeled:
        print(f"[WARN] {len(unlabeled)} questions have no topic label → assigning 'other'")
        for q in unlabeled:
            q["topic"] = "other"

    selected = select_questions(questions)
    final = [format_output(q) for q in selected]

    print_summary(final)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(final)} questions to {OUT_FILE}")
    print("\nA4 complete.")
