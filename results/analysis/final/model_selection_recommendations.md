# Model Selection Recommendations for Live Metaculus Tournament

## Set A — Accuracy-focused

| Model | Type | P1-P3 Brier | SE Brier | Parse % | Lat (s) |
|-------|------|------------|---------|---------|---------|
| gpt-5.4 | reasoning | 0.1367 | 0.1274 | 100.0% | 61.6 |
| gemini-3.1-pro | reasoning | 0.1432 | 0.1134 | 99.7% | 35.4 |
| kimi-k2.6 | reasoning | 0.1440 | 0.1289 | 100.0% | 249.2 |
| claude-opus-4.6 | reasoning | 0.1464 | 0.1290 | 100.0% | 39.2 |
| grok-4.20 | reasoning | 0.1608 | 0.1439 | 100.0% | 28.5 |
| deepseek-v3.2-speciale | reasoning | 0.1644 | 0.1530 | 100.0% | 196.9 |

## Set B — Practical/Cost-balanced

Optimised for Brier + latency tradeoff.

| Model | Type | P1-P3 Brier | SE Brier | Lat (s) | Rationale |
|-------|------|------------|---------|---------|-----------|
| gpt-5.4 | reasoning | 0.1367 | 0.1274 | 61.6 | Best accuracy |
| gemini-3.1-pro | reasoning | 0.1432 | 0.1134 | 35.4 | Strong SE performance |
| kimi-k2.6 | reasoning | 0.1440 | 0.1289 | 249.2 | Strong SE performance |
| claude-opus-4.6 | reasoning | 0.1464 | 0.1290 | 39.2 | Strong SE performance |
| grok-4.20 | reasoning | 0.1608 | 0.1439 | 28.5 | Strong SE performance |
| deepseek-v3.2-speciale | reasoning | 0.1644 | 0.1530 | 196.9 | Strong SE performance |

## Caveats

- Recommendations based on P1–P3 pilot (114 questions). Live performance may differ.

- P4 Superforecaster prompt shows promise; consider including it for all tournament models.

- Market baseline (Brier ≈ 0.095) significantly outperforms all models — include market baseline as reference in live tournament.

- Consider rotating prompts (P1+P4) to assess prompt effects in live setting.
