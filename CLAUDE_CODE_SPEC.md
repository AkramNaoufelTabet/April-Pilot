# CLAUDE.md — Paper 1 Pilot Experiment

## Overview

This project runs a pilot experiment for a PhD paper studying how LLM forecasting
performance changes across different prompting strategies and information conditions.
The pipeline has two phases:

**Phase A — Data Preparation:** Collect ForecastBench questions, match with resolution
values, label topics, and select 100 binary questions for the pilot.

**Phase B — Experiment Execution:** Run 10 LLMs × 3 prompts × 2 information conditions
on the 100 questions, collect forecasts, and compute evaluation metrics.

---

## Phase A — Data Preparation

### A1. Download Question Sets and Resolution Sets

Clone the ForecastBench datasets repo:
```
git clone https://github.com/forecastingresearch/forecastbench-datasets.git
```

We need these 5 question set files from `datasets/question_sets/`:
- `2026-02-01-llm.json`
- `2026-02-15-llm.json`
- `2026-03-01-llm.json`
- `2026-03-15-llm.json`
- `2026-03-29-llm.json`

And the corresponding resolution files from `datasets/resolution_sets/`.

### A2. Parse and Filter Questions

Each question set JSON has this structure:
```json
{
  "forecast_due_date": "2026-02-01",
  "question_set": "2026-02-01-llm.json",
  "questions": [ ... ]
}
```

Each question object has these key fields:
- `id` — unique identifier within source
- `source` — e.g. "manifold", "metaculus", "polymarket", "acled", "fred", etc.
- `question` — the question text (may contain {forecast_due_date} and {resolution_date} placeholders for dataset questions)
- `background` — background context
- `resolution_criteria` — how the question resolves
- `source_intro` — intro text about the source (useful context)
- `freeze_datetime` — when the market/data value was captured
- `freeze_datetime_value` — the market probability or data value at freeze time
- `freeze_datetime_value_explanation` — what the freeze value represents
- `combination_of` — "N/A" for standard questions, array for combination questions
- `resolution_dates` — array of dates for dataset questions, "N/A" for market questions

**Filtering rules:**
1. Keep only **standard** questions: `combination_of == "N/A"`
2. Keep only **market** questions (binary): source in ["manifold", "metaculus", "polymarket"]
   - These are binary Yes/No probability questions, which is what our prompts are designed for
   - Dataset questions (ACLED, FRED, etc.) are numeric and need a different evaluation pipeline
3. **Deduplicate** across the 5 question sets — the same question ID + source pair may appear
   in multiple biweekly sets. Keep only one instance per unique (id, source) pair.
4. **Join with resolution sets** to find which questions have resolved.
   - Resolution set files contain resolution values for questions.
   - A question is "resolved" if it has a ground-truth binary outcome (0 or 1).
   - Keep only resolved questions — we need ground truth to compute Brier scores.

After filtering, we should have a pool of several hundred resolved binary questions.

### A3. Topic Labeling

Label each question using Claude Sonnet 4.6 via OpenRouter API.

**Topic labels** (assign exactly one per question):
```python
TOPIC_LABELS = [
    "geopolitics",
    "conflict",
    "economics_macro",
    "economics_company",
    "ai_technology",
    "public_health",
    "energy",
    "finance_market",
    "sports",
    "celebrity",
    "entertainment",
    "other",
]
```

**Labeling approach:**
- Send each question's `question` + `background` text to the model
- Use a structured prompt that asks the model to classify into exactly one of the above labels
- Request JSON output: `{"topic": "<label>", "confidence": <0-1>}`
- Use OpenRouter API with model `anthropic/claude-sonnet-4-20250514`
- Batch questions to minimize API calls (e.g. 10–20 questions per request)
- Save the labeled questions to a CSV/JSON file

**OpenRouter API for labeling:**
```
POST https://openrouter.ai/api/v1/chat/completions
Headers:
  Authorization: Bearer $OPENROUTER_API_KEY
  Content-Type: application/json
Body:
  {
    "model": "anthropic/claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "..."}],
    "temperature": 0
  }
```

Environment variable: `OPENROUTER_API_KEY`

### A4. Select 100 Questions

From the labeled pool, select 100 questions using stratified sampling:

**Selection criteria:**
1. **Topic diversity** — sample proportionally across topic labels, ensuring at least
   2–3 questions per represented topic. No single topic should exceed 25 questions.
2. **Source diversity** — mix across Manifold, Metaculus, Polymarket (don't pull all 100
   from one platform).
3. **Difficulty spread** — use `freeze_datetime_value` (the market probability at freeze
   time) to ensure a range of "easy" (market near 0 or 1) and "hard" (market near 0.5)
   questions. Aim for roughly uniform distribution across probability bins.
4. **Exclude low-quality questions** — skip questions with very short or empty `background`
   fields, or questions whose resolution criteria just say "Resolves to the outcome of..."
   without additional context (these are fine to keep if `background` is substantive).
5. **Question clarity** — prefer questions where the `question` text is self-contained
   enough that an LLM can understand what's being asked without visiting external URLs.

**Output:** Save final 100 questions as `data/pilot_questions.json` with fields:
```json
{
  "question_id": "unique_id",
  "source": "manifold",
  "question": "Will X happen by Y?",
  "background": "...",
  "resolution_criteria": "...",
  "source_intro": "...",
  "freeze_datetime": "2026-02-01T00:00:00+00:00",
  "freeze_datetime_value": "0.65",
  "resolution_value": 1,
  "topic": "ai_technology",
  "forecast_due_date": "2026-02-15"
}
```

---

## Phase B — Experiment Execution

### B1. Experimental Design

**Factorial:** 3 prompts × 2 information conditions × 10 models = 60 cells
**Questions per cell:** 100
**Total LLM calls:** 6,000

### B2. Models (via OpenRouter)

```python
MODELS = {
    # Reasoning models
    "gpt-5.4-thinking":       "openai/gpt-5.4-thinking",        # verify exact string
    "claude-opus-4.6":        "anthropic/claude-opus-4-6",       # verify exact string
    "gemini-3.1-pro":         "google/gemini-3.1-pro",           # verify exact string
    "grok-4.20-reasoning":    "x-ai/grok-4.20-reasoning",       # verify exact string
    "qwen3-max-thinking":     "qwen/qwen3-max-thinking",        # verify exact string
    "deepseek-v3.2-speciale": "deepseek/deepseek-v3.2-speciale", # verify exact string
    # Standard (non-reasoning) models
    "gemma-4-31b":            "google/gemma-4-31b",              # verify exact string
    "glm-5.1-reasoning":      "zhipu/glm-5.1-reasoning",        # verify exact string
    "kimi-k2.5-thinking":     "moonshot/kimi-k2.5-thinking",     # verify exact string
    "gpt-oss-120b":           "openai/gpt-oss-120b",            # verify exact string
}
```

⚠️ IMPORTANT: Before running the experiment, verify each model's exact OpenRouter
model string by checking https://openrouter.ai/models. The strings above are
placeholders — some model names may differ on OpenRouter. Build a verification step
that pings each model with a trivial prompt to confirm it's accessible and responding.

**OpenRouter API call pattern:**
```
POST https://openrouter.ai/api/v1/chat/completions
Headers:
  Authorization: Bearer $OPENROUTER_API_KEY
  Content-Type: application/json
Body:
  {
    "model": "<model_string>",
    "messages": [{"role": "user", "content": "<prompt_with_question>"}],
    "temperature": 0,
    "max_tokens": 4096
  }
```

### B3. Prompts

Three prompts from Schoenegger et al. (2025), used verbatim:

**P1 — Control (Schoenegger Prompt #1):**
```
Please answer the following question with a probabilistic estimate expressed
between 0% and 100%, and format your response as: 'Forecast: X%'.
```

**P2 — Base-Rate-First (Schoenegger Prompt #19):**
```
Please answer the following question with a probabilistic estimate expressed
between 0% and 100%, and format your response as: 'Forecast: X%'.

Before considering the specific details of this question, what is the
historical frequency of similar events? Using this base rate as your starting
point, adjust your probability estimate based on the particular circumstances
of this case.
```

**P3 — Bayesian Reasoning (Schoenegger Prompt #27):**
```
Consider the following question in terms of Bayesian reasoning. Start with a
prior probability based on historical data or general knowledge. Then, update
this prior using more specific information about the case under discussion.
For each new piece of information, produce an updated posterior estimate of
the outcome using the principle behind Bayes rule. Conclude with the final
posterior probability, formatted as: 'Forecast: X%'
```

### B4. Information Conditions

**Condition 1 — Closed-Book:**
The model receives only the prompt + question metadata. No external evidence.

**Condition 2 — Shared Evidence:**
Same as closed-book, but with an appended evidence block containing retrieved
news articles. The SAME articles are given to all models for a given question.

**Evidence retrieval** uses AskNews API:
```
GET https://api.asknews.app/v1/news/search
Headers:
  Authorization: Bearer $ASKNEWS_API_KEY
Params:
  q=<question_text_as_search_query>
  n_articles=5
  method=kw
```

Environment variable: `ASKNEWS_API_KEY` (and `ASKNEWS_CLIENT_ID` if required)

For each of the 100 questions, retrieve articles ONCE and cache them. The same
article bundle is then used across all 10 models × 3 prompts in the evidence condition.

**Evidence block format** (appended after question metadata):
```
The following recent news articles may be relevant to this question.
Use them if helpful; disregard if not.

{retrieved_articles}
```

### B5. Full Prompt Assembly

For each (model, prompt, condition, question) combination, assemble the prompt:

```
{prompt_text}

Question: {question}

Background: {background}

Resolution Criteria: {resolution_criteria}

Resolution Date: {forecast_due_date}

[IF condition == "shared_evidence":]
The following recent news articles may be relevant to this question.
Use them if helpful; disregard if not.

{retrieved_articles}
```

Note: `background` and `resolution_criteria` are part of the question itself and
appear in BOTH conditions. Only the evidence block differs.

### B6. Response Parsing

Extract the forecast from each model response using:
```python
import re

def extract_forecast(response_text: str) -> float | None:
    """Extract forecast probability from model response."""
    match = re.search(r"Forecast:\s*([\d.]+)%", response_text)
    if match:
        value = float(match.group(1))
        return value / 100  # Convert to [0, 1] for Brier score
    return None  # Flag for manual review or retry
```

**Handling failures:**
- If no `Forecast: X%` pattern is found, retry the call once with a nudge:
  "Please provide your final forecast as 'Forecast: X%'."
- If still no match, flag the response for manual review.
- Track parse failure rate per model — this feeds into the "formatting reliability"
  down-selection criterion.

### B7. Output Storage

Save all results to `results/pilot_results.json` and `results/pilot_results.csv`.

**Per-response record:**
```json
{
  "question_id": "abc123",
  "source": "manifold",
  "topic": "ai_technology",
  "model": "anthropic/claude-opus-4-6",
  "model_type": "reasoning",
  "prompt": "P1_control",
  "condition": "closed_book",
  "forecast": 0.72,
  "resolution_value": 1,
  "brier_score": 0.0784,
  "raw_response": "...",
  "parse_success": true,
  "response_time_seconds": 3.2,
  "input_tokens": 450,
  "output_tokens": 280,
  "cost_usd": 0.003,
  "timestamp": "2026-04-14T10:30:00Z",
  "retry_count": 0
}
```

Track token usage and cost from OpenRouter response headers/body:
- `usage.prompt_tokens`
- `usage.completion_tokens`
- Use model pricing from OpenRouter to compute `cost_usd`

### B8. Evaluation Metrics

Compute the following for each (model, prompt, condition) cell:

1. **Brier Score** — mean of (forecast - resolution)² across 100 questions
2. **Calibration** — bin forecasts into deciles [0-0.1, 0.1-0.2, ...], plot
   predicted probability vs. actual frequency
3. **Resolution** — variance of forecasts (how well the model discriminates)
4. **MAE** — mean absolute error |forecast - resolution|
5. **Parse failure rate** — fraction of responses where forecast couldn't be extracted
6. **Mean response time** — average seconds per call
7. **Mean cost per question** — average USD per question

**Aggregate views for down-selection:**
- Performance by model (averaged across prompts and conditions)
- Performance by prompt (averaged across models)
- Prompt × condition interaction (does evidence help or hurt each prompt?)
- Model × prompt interaction (does P2/P3 create degenerate behavior for specific models?)
- Performance by topic (robustness across categories)
- Reasoning vs. standard model comparison (treat model_type as a moderator)

---

## Project Structure

```
paper1-pilot/
├── CLAUDE.md                    # This file
├── data/
│   ├── raw/                     # Raw ForecastBench question + resolution JSONs
│   ├── labeled/                 # Questions after topic labeling
│   ├── pilot_questions.json     # Final 100 selected questions
│   └── evidence_cache/          # Cached AskNews articles per question
├── prompts/
│   └── prompt_templates.md      # The 6 prompt templates (already created)
├── src/
│   ├── a1_download_data.py      # Clone repo, extract relevant files
│   ├── a2_filter_questions.py   # Parse, filter, deduplicate, join resolutions
│   ├── a3_label_topics.py       # Topic labeling via OpenRouter
│   ├── a4_select_questions.py   # Stratified sampling of 100 questions
│   ├── b1_retrieve_evidence.py  # AskNews retrieval + caching
│   ├── b2_run_experiment.py     # Main experiment loop (all 6,000 calls)
│   ├── b3_parse_results.py      # Response parsing + failure handling
│   └── b4_evaluate.py           # Metrics computation + tables + plots
├── results/
│   ├── pilot_results.json
│   ├── pilot_results.csv
│   └── figures/                 # Calibration plots, bar charts, etc.
└── requirements.txt             # openai, requests, pandas, matplotlib, etc.
```

---

## Environment Variables Required

```bash
export OPENROUTER_API_KEY="sk-or-..."     # OpenRouter API key
export ASKNEWS_API_KEY="..."               # AskNews API key
export ASKNEWS_CLIENT_ID="..."             # AskNews client ID (if needed)
```

---

## Execution Order

1. Run Phase A scripts (a1 → a2 → a3 → a4) to produce `pilot_questions.json`
2. Run `b1_retrieve_evidence.py` to cache evidence for all 100 questions
3. Run `b2_run_experiment.py` to execute the 6,000 LLM calls
   - Implement rate limiting (respect OpenRouter rate limits per model)
   - Save intermediate results after each model completes (crash recovery)
   - Log progress: which (model, prompt, condition, question) combinations are done
4. Run `b3_parse_results.py` if any responses need re-parsing
5. Run `b4_evaluate.py` to produce metrics, tables, and plots

---

## Important Notes

- **Temperature = 0** for all calls (deterministic forecasts for this pilot).
- **Single run per cell** — no repeated sampling in the pilot (save that for the
  live tournament where you might want 3 runs for variance estimation).
- **Crash recovery** — the experiment loop must be resumable. Check what's already
  in `pilot_results.json` before making a call. Skip completed combinations.
- **Rate limiting** — OpenRouter has per-model rate limits. Implement exponential
  backoff with jitter. Parallelize across models but serialize within a model.
- **Cost monitoring** — print running cost total after each model completes.
  Set a hard budget cap (e.g. $50) that halts execution if exceeded.
- **The `source_intro` field** from ForecastBench (e.g. "We would like you to predict
  the outcome of a prediction market...") should NOT be included in the prompt.
  It's ForecastBench-specific framing that would conflict with our own prompt design.
  Use only `question`, `background`, and `resolution_criteria`.
