# Paper 1 — Prompt Templates
## 3 Prompts × 2 Information Conditions (Closed-Book / Shared Evidence)

**Design principle:** Closed-book versions reproduce Schoenegger et al. (2025) verbatim.
Shared-evidence versions are identical except for one inserted evidence block.
All performance differences between conditions are therefore attributable to the
information, not to prompt rewording.

---

## Structural Placeholders

| Placeholder             | Source                        | Present in all conditions |
|-------------------------|-------------------------------|--------------------------|
| `{question_title}`      | Metaculus / Pilot ForecastBench     | Yes                      |
| `{background}`          | Metaculus / Pilot ForecastBench     | Yes                      |
| `{resolution_criteria}` | Metaculus / Pilot ForecastBench     | Yes                      |
| `{resolution_date}`     | Metaculus / Pilot ForecastBench     | Yes                      |
| `{retrieved_articles}`  | AskNews / shared news bundle  | Evidence condition only   |

> **Note:** `{background}` and `{resolution_criteria}` are part of the *question
> itself*, not external evidence. They appear in both conditions.

---

## Evidence Block (inserted only in shared-evidence condition)

```
The following recent news articles may be relevant to this question.
Use them if helpful; disregard if not.

{retrieved_articles}
```

---

## P1 — Control Prompt (Schoenegger et al. Prompt #1)

### P1 × Closed-Book

```
Please answer the following question with a probabilistic estimate expressed
between 0% and 100%, and format your response as: 'Forecast: X%'.

Question: {question_title}

Background: {background}

Resolution Criteria: {resolution_criteria}

Resolution Date: {resolution_date}
```

### P1 × Shared Evidence

```
Please answer the following question with a probabilistic estimate expressed
between 0% and 100%, and format your response as: 'Forecast: X%'.

Question: {question_title}

Background: {background}

Resolution Criteria: {resolution_criteria}

Resolution Date: {resolution_date}

The following recent news articles may be relevant to this question.
Use them if helpful; disregard if not.

{retrieved_articles}
```

---

## P2 — Base-Rate-First Prompt (Schoenegger et al. Prompt #19)

### P2 × Closed-Book

```
Please answer the following question with a probabilistic estimate expressed
between 0% and 100%, and format your response as: 'Forecast: X%'.

Before considering the specific details of this question, what is the
historical frequency of similar events? Using this base rate as your starting
point, adjust your probability estimate based on the particular circumstances
of this case.

Question: {question_title}

Background: {background}

Resolution Criteria: {resolution_criteria}

Resolution Date: {resolution_date}
```

### P2 × Shared Evidence

```
Please answer the following question with a probabilistic estimate expressed
between 0% and 100%, and format your response as: 'Forecast: X%'.

Before considering the specific details of this question, what is the
historical frequency of similar events? Using this base rate as your starting
point, adjust your probability estimate based on the particular circumstances
of this case.

Question: {question_title}

Background: {background}

Resolution Criteria: {resolution_criteria}

Resolution Date: {resolution_date}

The following recent news articles may be relevant to this question.
Use them if helpful; disregard if not.

{retrieved_articles}
```

---

## P3 — Bayesian Reasoning Prompt (Schoenegger et al. Prompt #27)

### P3 × Closed-Book

```
Consider the following question in terms of Bayesian reasoning. Start with a
prior probability based on historical data or general knowledge. Then, update
this prior using more specific information about the case under discussion.
For each new piece of information, produce an updated posterior estimate of
the outcome using the principle behind Bayes rule. Conclude with the final
posterior probability, formatted as: 'Forecast: X%'

Question: {question_title}

Background: {background}

Resolution Criteria: {resolution_criteria}

Resolution Date: {resolution_date}
```

### P3 × Shared Evidence

```
Consider the following question in terms of Bayesian reasoning. Start with a
prior probability based on historical data or general knowledge. Then, update
this prior using more specific information about the case under discussion.
For each new piece of information, produce an updated posterior estimate of
the outcome using the principle behind Bayes rule. Conclude with the final
posterior probability, formatted as: 'Forecast: X%'

Question: {question_title}

Background: {background}

Resolution Criteria: {resolution_criteria}

Resolution Date: {resolution_date}

The following recent news articles may be relevant to this question.
Use them if helpful; disregard if not.

{retrieved_articles}
```

---
