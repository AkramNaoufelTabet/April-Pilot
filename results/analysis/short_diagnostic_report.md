# April Pilot: Short Diagnostic Report

**Generated:** 2026-04-28  
**Total rows analyzed:** 8,239  
**Models:** 12 (7 reasoning, 5 standard)  
**Questions:** 114 resolved binary  
**Prompts:** P1 (Control), P2 (Base-rate-first), P3 (Bayesian)  
**Conditions:** closed_book (CB), shared_evidence (SE)  

## Executive Summary

The pilot ran 8,239 LLM forecasting calls across 12 models, 3 prompts, and 2 conditions on 114 binary questions, achieving an overall parse rate of 96.6%. The best-performing model was **gpt-5.4** (mean Brier = 0.1367), and the weakest was **mistral-large** (mean Brier = 0.2151). Overall, shared evidence improved forecasting accuracy relative to closed-book (SE−CB delta = -0.0261), and 0 of 12 models beat the market baseline on average. The Prompt×Condition interaction — the key hypothesis — shows that **P3** benefits most from evidence (delta = -0.0318), though effect sizes are modest at this pilot scale.

## Data Completeness

Expected rows per cell: 114 (one per question). Expected total: 8,208. Actual: 8,239.

| Model | Type | P1_CB | P1_SE | P2_CB | P2_SE | P3_CB | P3_SE | Total |
|-------|------|-------|-------|-------|-------|-------|-------|-------|
| claude-opus-4.6 | R | 114 | 114 | 114 | 114 | 114 | 114 | 684 |
| deepseek-v3.2-speciale | R | 114 | 114 | 114 | 114 | 114 | 114 | 684 |
| gemini-3-flash | S | 114 | 114 | 114 | 114 | 114 | 114 | 684 |
| gemini-3.1-pro | R | 114 | 114 | 114 | 114 | 114 | 114 | 684 |
| gemma-4-31b | S | 114 | 114 | 114 | 114 | 114 | 114 | 684 |
| glm-5.1 | S | 114 | 114 | 114 | 114 | 114 | 114 | 684 |
| gpt-5.4 | R | 114 | 114 | 114 | 114 | 114 | 114 | 684 |
| gpt-oss-120b | S | 114 | 114 | 114 | 114 | 114 | 114 | 684 |
| grok-4.20 | R | 114 | 114 | 114 | 114 | 114 | 114 | 684 |
| kimi-k2.6 | R | 114 | 114 | 114 | 114 | 114 | 114 | 684 |
| mistral-large | S | 114 | 114 | 114 | 114 | 114 | 114 | 684 |
| qwen3-max | R | 114 | 117 | 121 | 120 | 124 | 119 | 715 |

## Analysis 1: Overall Leaderboard

| Rank | Model | Type | Mean Brier | Parse Rate | N |
|------|-------|------|------------|------------|---|
| 1 | gpt-5.4 | reasoning | 0.1367 | 1.0000 | 684 |
| 2 | kimi-k2.6 | reasoning | 0.1401 | 0.9459 | 684 |
| 3 | gemini-3.1-pro | reasoning | 0.1427 | 0.9620 | 684 |
| 4 | claude-opus-4.6 | reasoning | 0.1444 | 0.9868 | 684 |
| 5 | grok-4.20 | reasoning | 0.1611 | 0.9971 | 684 |
| 6 | deepseek-v3.2-speciale | reasoning | 0.1624 | 0.9883 | 684 |
| 7 | glm-5.1 | standard | 0.1729 | 0.9635 | 684 |
| 8 | qwen3-max | reasoning | 0.1758 | 0.9497 | 715 |
| 9 | gemma-4-31b | standard | 0.1786 | 0.9956 | 684 |
| 10 | gemini-3-flash | standard | 0.1796 | 1.0000 | 684 |
| 11 | gpt-oss-120b | standard | 0.1852 | 0.8567 | 684 |
| 12 | mistral-large | standard | 0.2151 | 0.9430 | 684 |

Market baseline (freeze-datetime probability): mean Brier = **0.0948**

## Analysis 2: Brier by Model × Prompt × Condition (72 cells)

*CB = closed_book, SE = shared_evidence. Flagged cells (parse rate < 50%) marked with \*.*

| Model | Type | P1_CB | P1_SE | P2_CB | P2_SE | P3_CB | P3_SE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| claude-opus-4.6 | R | 0.1611 | 0.1360 | 0.1631 | 0.1187 | 0.1593 | 0.1274 |
| deepseek-v3.2-speciale | R | 0.1763 | 0.1457 | 0.1672 | 0.1476 | 0.1700 | 0.1682 |
| gemini-3-flash | S | 0.1980 | 0.1666 | 0.1855 | 0.1671 | 0.1899 | 0.1702 |
| gemini-3.1-pro | R | 0.1657 | 0.1088 | 0.1784 | 0.1060 | 0.1783 | 0.1182 |
| gemma-4-31b | S | 0.1765 | 0.1731 | 0.1879 | 0.1620 | 0.1961 | 0.1757 |
| glm-5.1 | S | 0.1750 | 0.1632 | 0.2063 | 0.1535 | 0.1912 | 0.1488 |
| gpt-5.4 | R | 0.1421 | 0.1245 | 0.1476 | 0.1412 | 0.1484 | 0.1166 |
| gpt-oss-120b | S | 0.1671 | 0.1455 | 0.1634 | 0.1825 | 0.2474 | 0.2131 |
| grok-4.20 | R | 0.1779 | 0.1548 | 0.1764 | 0.1409 | 0.1790 | 0.1375 |
| kimi-k2.6 | R | 0.1556 | 0.1343 | 0.1622 | 0.1149 | 0.1471 | 0.1277 |
| mistral-large | S | 0.1932 | 0.2157 | 0.1600 | 0.2066 | 0.2857 | 0.2128 |
| qwen3-max | R | 0.1886 | 0.1481 | 0.1860 | 0.1550 | 0.1925 | 0.1843 |

## Analysis 3: Prompt × Condition Interaction

| Prompt | CB Brier | SE Brier | Delta (SE−CB) | N_CB | N_SE |
|--------|----------|----------|---------------|------|------|
| P1 | 0.1732 | 0.1515 | -0.0217 | 1343 | 1345 |
| P2 | 0.1741 | 0.1493 | -0.0248 | 1292 | 1336 |
| P3 | 0.1899 | 0.1581 | -0.0318 | 1316 | 1324 |

**Key finding:** P3 benefits most from evidence (delta = -0.0318).
Overall SE−CB delta across all prompts: -0.0261.

## Analysis 4: Evidence Effect per Model

*Ranked from most helped (most negative delta) to most hurt by evidence.*

| Model | Type | CB Brier | SE Brier | Delta (SE−CB) |
|-------|------|----------|----------|---------------|
| gemini-3.1-pro | reasoning | 0.1741 | 0.1109 | -0.0632 |
| glm-5.1 | standard | 0.1906 | 0.1551 | -0.0355 |
| claude-opus-4.6 | reasoning | 0.1612 | 0.1274 | -0.0338 |
| grok-4.20 | reasoning | 0.1778 | 0.1444 | -0.0334 |
| kimi-k2.6 | reasoning | 0.1552 | 0.1255 | -0.0297 |
| qwen3-max | reasoning | 0.1890 | 0.1625 | -0.0265 |
| gemini-3-flash | standard | 0.1912 | 0.1680 | -0.0232 |
| gpt-5.4 | reasoning | 0.1460 | 0.1274 | -0.0186 |
| deepseek-v3.2-speciale | reasoning | 0.1712 | 0.1538 | -0.0174 |
| gemma-4-31b | standard | 0.1869 | 0.1703 | -0.0167 |
| gpt-oss-120b | standard | 0.1918 | 0.1788 | -0.0130 |
| mistral-large | standard | 0.2188 | 0.2117 | -0.0071 |

## Analysis 5: Reasoning vs Standard Models

- Overall reasoning mean Brier: **0.1557**
- Overall standard mean Brier: **0.1862**

| Type | Prompt | Condition | Mean Brier | N |
|------|--------|-----------|------------|---|
| reasoning | P1 | closed_book | 0.1669 | 892 |
| reasoning | P1 | shared_evidence | 0.1373 | 894 |
| reasoning | P2 | closed_book | 0.1681 | 877 |
| reasoning | P2 | shared_evidence | 0.1375 | 883 |
| reasoning | P3 | closed_book | 0.1764 | 866 |
| reasoning | P3 | shared_evidence | 0.1483 | 875 |
| standard | P1 | closed_book | 0.1858 | 451 |
| standard | P1 | shared_evidence | 0.1798 | 451 |
| standard | P2 | closed_book | 0.1867 | 415 |
| standard | P2 | shared_evidence | 0.1723 | 453 |
| standard | P3 | closed_book | 0.2158 | 450 |
| standard | P3 | shared_evidence | 0.1770 | 449 |

## Analysis 6: Calibration

| Model | Mean Forecast | Std Dev | % < 0.1 | % > 0.9 | % 0.1–0.9 | N |
|-------|--------------|---------|---------|---------|-----------|---|
| claude-opus-4.6 | 0.2198 | 0.2960 | 56.5926 | 6.2222 | 37.1852 | 675 |
| deepseek-v3.2-speciale | 0.3147 | 0.3062 | 30.7692 | 5.4734 | 63.7574 | 676 |
| gemini-3-flash | 0.3247 | 0.3023 | 30.7018 | 6.1404 | 63.1579 | 684 |
| gemini-3.1-pro | 0.2283 | 0.3033 | 53.3435 | 4.8632 | 41.7933 | 658 |
| gemma-4-31b | 0.3400 | 0.2934 | 23.6417 | 5.5800 | 70.7783 | 681 |
| glm-5.1 | 0.2556 | 0.3081 | 47.3445 | 7.4355 | 45.2200 | 659 |
| gpt-5.4 | 0.2561 | 0.2664 | 37.8655 | 4.8246 | 57.3099 | 684 |
| gpt-oss-120b | 0.2926 | 0.2999 | 36.0068 | 6.8259 | 57.1672 | 586 |
| grok-4.20 | 0.3108 | 0.2443 | 18.9150 | 2.0528 | 79.0323 | 682 |
| kimi-k2.6 | 0.2323 | 0.2572 | 42.5039 | 2.3184 | 55.1777 | 647 |
| mistral-large | 0.3982 | 0.2691 | 13.4884 | 3.5659 | 82.9457 | 645 |
| qwen3-max | 0.2579 | 0.3240 | 46.5390 | 7.2165 | 46.2445 | 679 |

## Analysis 7: Parse Reliability

Overall parse rate: **96.57%** (7,956/8,239 rows)

**6 cell(s) flagged with >10% failure rate:**

- gpt-oss-120b / P2 / closed_book: 21.9% failures
- gpt-oss-120b / P2 / shared_evidence: 17.5% failures
- gpt-oss-120b / P3 / closed_book: 19.3% failures
- gpt-oss-120b / P3 / shared_evidence: 14.9% failures
- kimi-k2.6 / P3 / closed_book: 10.5% failures
- mistral-large / P2 / closed_book: 30.7% failures

Parse rate by model (across all prompts/conditions):

| Model | Parse Rate | N Fail / N Total |
|-------|------------|-----------------|
| claude-opus-4.6 | 0.9868 | 9/684 |
| deepseek-v3.2-speciale | 0.9883 | 8/684 |
| gemini-3-flash | 1.0000 | 0/684 |
| gemini-3.1-pro | 0.9620 | 26/684 |
| gemma-4-31b | 0.9956 | 3/684 |
| glm-5.1 | 0.9635 | 25/684 |
| gpt-5.4 | 1.0000 | 0/684 |
| gpt-oss-120b | 0.8567 | 98/684 |
| grok-4.20 | 0.9971 | 2/684 |
| kimi-k2.6 | 0.9459 | 37/684 |
| mistral-large | 0.9430 | 39/684 |
| qwen3-max | 0.9497 | 36/715 |

## Analysis 8: Latency and Cost

*Ranked by Brier/second (lower = more efficient).*

| Model | Mean Latency (s) | Mean Reasoning Tokens | Mean Brier | Brier/s |
|-------|------------------|-----------------------|------------|---------|
| kimi-k2.6 | 246.57 | 7633.24 | 0.1401 | 0.0006 |
| qwen3-max | 246.17 | 8767.92 | 0.1758 | 0.0007 |
| deepseek-v3.2-speciale | 197.02 | 7071.16 | 0.1624 | 0.0008 |
| glm-5.1 | 85.05 | 2325.39 | 0.1729 | 0.0020 |
| gpt-oss-120b | 83.84 | 4149.82 | 0.1852 | 0.0022 |
| gpt-5.4 | 61.62 | 2972.85 | 0.1367 | 0.0022 |
| gemini-3.1-pro | 38.63 | 2332.40 | 0.1427 | 0.0037 |
| claude-opus-4.6 | 39.06 | 859.02 | 0.1444 | 0.0037 |
| gemma-4-31b | 39.32 | 0.00 | 0.1786 | 0.0045 |
| grok-4.20 | 28.48 | 1782.47 | 0.1611 | 0.0057 |
| mistral-large | 17.32 | 0.00 | 0.2151 | 0.0124 |
| gemini-3-flash | 5.53 | 0.00 | 0.1796 | 0.0324 |

## Analysis 9: Market Baseline Comparison

Market baseline mean Brier (freeze-datetime probability): **0.0948**

Models beating the market: **0/12**

| Model | Model Brier | Market Brier | Delta | Beats Market? |
|-------|-------------|--------------|-------|---------------|
| gpt-5.4 | 0.1367 | 0.0948 | 0.0419 | no |
| gemini-3.1-pro | 0.1427 | 0.0962 | 0.0465 | no |
| kimi-k2.6 | 0.1401 | 0.0911 | 0.0490 | no |
| claude-opus-4.6 | 0.1444 | 0.0926 | 0.0518 | no |
| grok-4.20 | 0.1611 | 0.0951 | 0.0660 | no |
| deepseek-v3.2-speciale | 0.1624 | 0.0941 | 0.0683 | no |
| glm-5.1 | 0.1729 | 0.0950 | 0.0778 | no |
| qwen3-max | 0.1758 | 0.0955 | 0.0802 | no |
| gemma-4-31b | 0.1786 | 0.0952 | 0.0833 | no |
| gemini-3-flash | 0.1796 | 0.0948 | 0.0847 | no |
| gpt-oss-120b | 0.1852 | 0.0885 | 0.0966 | no |
| mistral-large | 0.2151 | 0.0922 | 0.1229 | no |

## Analysis 10: Topic Effects

| Topic | CB Brier | SE Brier | Delta (SE−CB) | N_CB | N_SE |
|-------|----------|----------|---------------|------|------|
| sports | 0.1862 | 0.0243 | -0.1619 | 278 | 284 |
| public_health | 0.2628 | 0.1700 | -0.0928 | 206 | 212 |
| other | 0.2049 | 0.1421 | -0.0628 | 177 | 173 |
| finance_market | 0.2758 | 0.2187 | -0.0571 | 489 | 500 |
| economics | 0.1235 | 0.0878 | -0.0357 | 482 | 492 |
| ai_technology | 0.1275 | 0.0981 | -0.0294 | 700 | 707 |
| geopolitics | 0.1606 | 0.1645 | 0.0039 | 689 | 695 |
| conflict | 0.1675 | 0.1877 | 0.0202 | 619 | 624 |
| entertainment | 0.1825 | 0.2132 | 0.0307 | 176 | 177 |
| energy | 0.2611 | 0.3828 | 0.1218 | 135 | 141 |

## Analysis 11: Source Effects

| Source | CB Brier | SE Brier | Delta (SE−CB) | N_CB | N_SE |
|--------|----------|----------|---------------|------|------|
| manifold | 0.1579 | 0.0855 | -0.0724 | 1252 | 1275 |
| metaculus | 0.2271 | 0.2109 | -0.0161 | 1284 | 1294 |
| polymarket | 0.1543 | 0.1606 | 0.0063 | 1415 | 1436 |

## Interesting Findings

- **Model spread:** Best model (gpt-5.4) achieves Brier = 0.1367, while worst (mistral-large) scores 0.2151 — a gap of 0.0783 Brier points.

- **Evidence effect overall:** Shared evidence improves accuracy relative to closed-book (mean SE−CB = -0.0261). gemini-3.1-pro benefits most (delta = -0.0632) and mistral-large is most hurt (delta = -0.0071).

- **Prompt × Condition interaction:** P3 benefits most from evidence (SE−CB delta = -0.0318). This is the core hypothesis; the direction is consistent with structured prompts gaining more from evidence.

- **Market baseline:** Market Brier = 0.0948. 0/12 models (0%) beat the market overall. Most models fail to beat the crowd.

- **Reasoning vs Standard:** Reasoning models average Brier = 0.1557, standard = 0.1862 (difference = -0.0306). Reasoning models are better on average.

- **Topic effects:** 'sports' benefits most from evidence (SE−CB = -0.1619); 'energy' is most hurt (SE−CB = 0.1218).

- **Calibration spread:** claude-opus-4.6 has the lowest mean forecast (0.2198), suggesting conservative/underconfident tendencies; mistral-large has the highest (0.3982).

- **Parse failures:** 6 model/prompt/condition cell(s) exceed the 10% failure threshold, flagging potential reliability concerns in those subsets.

- **Latency range:** gemini-3-flash is fastest (5.53s/call), kimi-k2.6 is slowest (246.57s/call). Slowest model is 44.5x slower.

- **Source effects:** 'manifold' questions benefit most from evidence (delta = -0.0724); 'polymarket' benefits least or is hurt (delta = 0.0063).

## Red Flags / Checks Before Writing the Paper

- **Sample size per cell is small.** Each (model, prompt, condition) cell has ~114 rows. Effect sizes should be treated with caution — small differences in mean Brier may not be statistically robust. Run permutation tests or bootstrap CIs before making strong claims.

- **Evidence effect is ambiguous in direction.** With a delta of -0.0261, the evidence effect is potentially meaningful but requires statistical testing. Do not claim a clear benefit or harm without significance testing.

- **freeze_datetime_value used as market Brier.** The market baseline uses the freeze-datetime crowd probability. Verify this is the correct pre-freeze snapshot (not a post-resolution price) before comparing to model performance.

- **Model versions and API routing.** Model keys like 'gpt-5.4', 'gemini-3.1-pro', etc., may correspond to non-public or internal model versions. Confirm exact model IDs and dates before any publication — routing changes could affect reproducibility.

- **Parse success ≠ calibrated forecast.** Even parsed forecasts may be anchored, hedged, or miscalibrated. Inspect raw_response samples for each model to verify forecasts are genuine probability estimates, not percentage points (e.g., 50 vs 0.50 errors).

## Notes on Cautious Interpretation

This is a pilot diagnostic, not a definitive study. Effect sizes are described but not statistically tested. The 114-question sample covers multiple topics and sources with unequal representation, so aggregate numbers mask within-topic and within-source heterogeneity. The Prompt×Condition interaction is the central hypothesis and should be the focus of follow-up analysis with confidence intervals. Market Brier comparisons assume the freeze-datetime value is a valid baseline; verify this for each source. Latency numbers include API wait time and may vary significantly across runs.

---
*Report generated by analyze.py from pilot_results.json and pilot_questions.json.*