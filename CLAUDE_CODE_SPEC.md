# CLAUDE_CODE_SPEC.md — Paper 1 Pilot Experiment

## Overview

PhD project: "Forecasting for Social Innovation using Language Technologies."

This pipeline tests whether LLM forecasting accuracy varies across prompting strategies
and information conditions.

**Factorial design:**
- 12 models × 3 prompts × 2 conditions × 114 questions = **8,208 total LLM calls**

**Phase A — Data Preparation:** COMPLETE  
**Phase B — Experiment Execution:** Smoke test running; full run pending

---

## Phase A — Data Preparation (COMPLETE)

### A1. Download Question Sets and Resolution Sets

`src/a1_download_data.py`

Clones the ForecastBench datasets repo then copies relevant files to `data/raw/`.

**Knowledge cutoff constraint:** GPT-5.4 has an August 2025 training cutoff — the most
restrictive of all 12 models. To avoid contamination, we use only question sets whose
`forecast_due_date` is after Aug 2025.

**13 question sets used (Oct 2025 – Apr 2026):**
```python
QUESTION_SETS = [
    "2025-10-26-llm.json",  "2025-11-09-llm.json",  "2025-11-23-llm.json",
    "2025-12-07-llm.json",  "2025-12-21-llm.json",  "2026-01-04-llm.json",
    "2026-01-18-llm.json",  "2026-02-01-llm.json",  "2026-02-15-llm.json",
    "2026-03-01-llm.json",  "2026-03-15-llm.json",  "2026-03-29-llm.json",
    "2026-04-12-llm.json",
]
```

Also runs `git pull` inside `forecastbench-datasets/` to ensure all 32 resolution
files are up to date before copying.

### A2. Parse, Filter, Deduplicate, Join Resolutions

`src/a2_filter_questions.py`

**Filtering rules:**
1. Keep only **standard** questions: `combination_of == "N/A"`
2. Keep only **market** questions: `source in {manifold, metaculus, polymarket}`
3. **Deduplicate** across the 13 sets by `(id, source)` pair
4. **Join with resolution sets**: keep only binary-resolved questions (0 or 1)
5. **Contamination guard**: drop questions whose `resolution_date` is on or before
   `2025-09-01` (i.e. within GPT-5.4's training window)

```python
RESOLUTION_CUTOFF = "2025-09-01"
```

**Result:** 579 resolved binary questions from 3 sources.

ForecastBench question structure:
```json
{
  "id": "...",
  "source": "manifold",
  "question": "Will X happen by {forecast_due_date}?",
  "background": "...",
  "resolution_criteria": "...",
  "freeze_datetime_value": "0.65",
  "combination_of": "N/A"
}
```

Output: `data/labeled/filtered_questions.json`

### A3. Topic Labeling

`src/a3_label_topics.py`

Labels each question using Claude Sonnet 4.6 via OpenRouter, in batches of 15.

**10-topic taxonomy:**
```python
TOPIC_LABELS = [
    "geopolitics",     # international relations, elections, diplomacy, governance
    "conflict",        # wars, military operations, armed groups, peace negotiations
    "ai_technology",   # AI, ML, tech companies, software, hardware
    "economics",       # macro indicators, company performance, trade, monetary policy
    "public_health",   # disease, healthcare, medical research, epidemics
    "energy",          # oil, gas, renewables, climate policy, commodities
    "finance_market",  # stocks, crypto, markets, investment, financial instruments
    "entertainment",   # movies, music, TV, awards, celebrity, pop culture
    "sports",          # all competitive sports and sporting events
    "other",           # anything that doesn't fit above categories
]
```

Sends batches of 15 questions; requests JSON array output:
```json
[{"idx": 0, "topic": "geopolitics", "confidence": 0.95}, ...]
```

Saves intermediate progress after each batch (resume support).

Output: `data/labeled/labeled_questions.json`

### A4. Select 114 Pilot Questions

`src/a4_select_questions.py`

Two-phase stratified selection with hard floors and caps.

**Configuration:**
```python
TARGET_N = 114  # 96 priority + 8 sports + 5 entertainment + 5 other
RANDOM_SEED = 42

TOPIC_CAPS    = {"sports": 8, "entertainment": 5, "other": 5}
TOPIC_DEFAULT_CAP = 30   # generous ceiling for priority topics

TOPIC_FLOORS  = {
    "geopolitics": 20,   "conflict": 18,  "ai_technology": 20,
    "economics": 14,     "public_health": 6, "energy": 4,
    "finance_market": 14,
    "sports": 8,         "entertainment": 5,  "other": 5,  # floor == cap
}

PRIORITY_TOPICS = {"geopolitics", "conflict", "ai_technology",
                   "economics", "public_health", "energy"}  # must total >= 70

MAX_PER_SOURCE = 45  # no single platform exceeds this
```

**Phase 1:** Guarantee floors — each topic gets `min(floor, available, cap)` questions.  
**Phase 2:** Distribute remaining slots proportionally to topic headroom.  

**Quality filter:** Drops questions with background < 50 chars, question < 10 chars,
resolution criteria < 30 chars, or where the question text is mostly a URL.

**Selection within a topic:** `diverse_select()` round-robins across sources.

**Result:** 114 questions  
- Topic: geopolitics 20, ai_technology 20, conflict 18, economics 14, finance_market 14,
  sports 8, public_health 6, entertainment 5, other 5, energy 4  
- Priority domains: 82/114  
- Sources: polymarket 41, metaculus 37, manifold 36  

Output: `data/pilot_questions.json`

---

## Phase B — Experiment Execution

### B1. Smoke Test

`src/smoke_test.py`

5 questions × 12 models × 3 prompts (closed-book). Validates parse rates, latency,
and Brier scores before committing to the full 8,208-call run.

**CLI:**
```bash
python src/smoke_test.py                          # all 12 models, all 3 prompts
python src/smoke_test.py --models deepseek-v3.2-speciale kimi-k2.6
python src/smoke_test.py --prompts P1 P2
python src/smoke_test.py --merge                  # merge new runs into existing JSON
```

Output: `results/smoke_test.json`

### B2. Models (via OpenRouter)

All calls use `temperature=0`, `max_tokens=32768`, `timeout=720`.

**Reasoning models** additionally receive `"reasoning": {"effort": "high"}` — OpenRouter
maps this to provider-specific controls (Anthropic `budget_tokens`, Google
`thinkingBudget`, DeepSeek/Qwen `thinking_budget`, xAI native reasoning).

```python
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
```

**Notes on specific models:**
- `deepseek-v3.2-speciale` and `kimi-k2.6`: verbose chain-of-thought; last 400 chars
  of response are printed for inspection. Can take 600+ seconds per question.
- `deepseek-v3.2-speciale`: provider-side 429s are common; retry logic essential.
- All reasoning models spend ~80% of output tokens on chain-of-thought;
  `max_tokens=32768` ensures they're not truncated mid-reasoning.

**Rate limit handling** (429 responses):
```python
for attempt in range(4):   # up to 3 retries
    if resp.status_code == 429:
        wait = 15 * (attempt + 1)   # 15s, 30s, 45s
        time.sleep(wait)
```

**OpenRouter API call:**
```python
payload = {
    "model": model_id,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "max_tokens": 32768,
}
if is_reasoning:
    payload["reasoning"] = {"effort": "high"}
```

### B3. Prompts

Three prompts from Schoenegger et al. (2025), used verbatim:

**P1 — Control (Schoenegger Prompt #1):**
```
Please answer the following question with a probabilistic estimate expressed
between 0% and 100%, and format your response as: 'Forecast: X%'.

Question: {question}
Background: {background}
Resolution Criteria: {resolution_criteria}
Resolution Date: {forecast_due_date}
```

**P2 — Base-Rate-First (Schoenegger Prompt #19):**
```
Please answer the following question with a probabilistic estimate expressed
between 0% and 100%, and format your response as: 'Forecast: X%'.

Before considering the specific details of this question, what is the
historical frequency of similar events? Using this base rate as your starting
point, adjust your probability estimate based on the particular circumstances
of this case.

Question: {question}
Background: {background}
Resolution Criteria: {resolution_criteria}
Resolution Date: {forecast_due_date}
```

**P3 — Bayesian Reasoning (Schoenegger Prompt #27):**
```
Consider the following question in terms of Bayesian reasoning. Start with a
prior probability based on historical data or general knowledge. Then, update
this prior using more specific information about the case under discussion.
For each new piece of information, produce an updated posterior estimate of
the outcome using the principle behind Bayes rule. Conclude with the final
posterior probability, formatted as: 'Forecast: X%'

Question: {question}
Background: {background}
Resolution Criteria: {resolution_criteria}
Resolution Date: {forecast_due_date}
```

### B4. Response Parsing

```python
def extract_forecast(text: str) -> float | None:
    match = re.search(r"Forecast:\s*([\d.]+)%", text, re.IGNORECASE)
    if match:
        return round(min(max(float(match.group(1)) / 100, 0.0), 1.0), 4)
    return None
```

Parse failures are logged with `(model_key, prompt_key, question_id)` for review.
No automatic retry on parse failure — flag for manual inspection.

### B5. Information Conditions

**Condition 1 — Closed-Book:** Model receives only the prompt + question metadata.  
**Condition 2 — Shared Evidence (PENDING):** Same prompt + evidence block from AskNews.

Evidence is retrieved ONCE per question and shared across all 12 models × 3 prompts
(i.e. 12 models see identical articles for a given question).

```
Environment variable: ASKNEWS_API_KEY
```

### B6. Evaluation Metrics

Per `(model, prompt, condition)` cell over 114 questions:

1. **Brier Score** — mean of `(forecast - resolution)²` (lower = better)
2. **Calibration** — forecast probability vs. actual resolution rate by decile
3. **MAE** — mean `|forecast - resolution|`
4. **Parse failure rate** — fraction without extractable `Forecast: X%`
5. **Mean latency** — average seconds per call
6. **Reasoning vs. standard** — model_type as moderator

**Market baseline Brier** (computed from `freeze_datetime_value`) is printed alongside
model Brier scores in the smoke test summary table.

### B7. Output Schema

Each result row saved in `results/smoke_test.json` (smoke test) and eventually
`results/pilot_results.json` (full run):

```json
{
  "model_key":        "claude-opus-4.6",
  "model_id":         "anthropic/claude-opus-4.6",
  "model_type":       "reasoning",
  "question_id":      "abc123",
  "source":           "metaculus",
  "topic":            "ai_technology",
  "market_prob":      "0.72",
  "prompt":           "P1",
  "condition":        "closed_book",
  "forecast":         0.7,
  "resolution_value": 1,
  "brier_score":      0.09,
  "parse_success":    true,
  "latency_seconds":  12.4,
  "input_tokens":     480,
  "output_tokens":    320,
  "reasoning_tokens": 8192,
  "raw_response":     "... (truncated to 2000 chars)",
  "error":            null
}
```

---

## Project Structure

```
April Pilot/
├── CLAUDE_CODE_SPEC.md          # This file
├── .gitignore                   # Excludes forecastbench-datasets/, data/raw/, .env
├── data/
│   ├── raw/                     # Raw ForecastBench JSON files (gitignored)
│   ├── labeled/
│   │   ├── filtered_questions.json   # After A2 (579 questions)
│   │   └── labeled_questions.json    # After A3 (579 questions with topics)
│   └── pilot_questions.json          # Final 114 selected questions (A4 output)
├── src/
│   ├── a1_download_data.py      # Clone/update ForecastBench, copy raw files
│   ├── a2_filter_questions.py   # Parse, filter, deduplicate, join resolutions
│   ├── a3_label_topics.py       # Topic labeling via Claude Sonnet 4.6
│   ├── a4_select_questions.py   # Stratified selection of 114 pilot questions
│   ├── b0_verify_models.py      # Ping all 12 models with a trivial prompt
│   ├── smoke_test.py            # 5 q x 12 models x 3 prompts validation run
│   └── (b2_run_experiment.py)   # Full 114-question run (to be built)
├── results/
│   ├── smoke_test.json          # Smoke test output
│   └── (pilot_results.json)     # Full run output (pending)
└── requirements.txt
```

---

## Environment Variables

```bash
export OPENROUTER_API_KEY="sk-or-..."     # Required for all LLM calls and A3 labeling
export ASKNEWS_API_KEY="..."               # Required for shared-evidence condition (pending)
```

On Windows (PowerShell):
```powershell
$env:OPENROUTER_API_KEY="sk-or-..."
```

---

## Execution Order

```bash
# Phase A (all complete)
python src/a1_download_data.py        # Download raw files (run git pull first)
python src/a2_filter_questions.py     # 579 resolved binary questions
python src/a3_label_topics.py         # Topic labels (~8 min, 39 batches)
python src/a4_select_questions.py     # 114 pilot questions

# Phase B
python src/b0_verify_models.py        # Confirm all 12 models accessible
python src/smoke_test.py              # 5-question validation (all 3 prompts)
# python src/b2_run_experiment.py     # Full run (to be built after smoke test validated)
```

---

## Key Design Decisions

- **Temperature = 0** for all calls (deterministic).
- **No source_intro** in prompts — ForecastBench-specific framing conflicts with our prompts.
- **Single run per cell** in the pilot; variance estimation deferred to live tournament.
- **max_tokens = 32768** for all models — reasoning models spend ~80% on CoT and need room.
- **Crash recovery** — `--merge` flag merges new model runs into existing smoke_test.json
  without re-running already-completed models.
- **Windows cp1252 terminal** — all non-ASCII chars in print statements replaced with
  ASCII equivalents; verbose response snippets sanitized with `.encode("ascii","replace")`.

---

## Important Notes on Models

- **claude-opus-4.6** (not 4.7) is the pilot model. Claude 4.7 is reserved for the
  live tournament that follows this pilot.
- **deepseek-v3.2-speciale** regularly hits provider-side 429s. The retry loop
  (15/30/45s waits) is essential. Expect 600+ second latency on hard questions.
- **kimi-k2.6** also produces verbose responses; tail snippets are printed for inspection.
- **TIMEOUT = 720** (12 minutes) per call — necessary for deepseek on hard questions.
