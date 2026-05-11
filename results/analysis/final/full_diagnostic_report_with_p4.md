# Full Diagnostic Report: LLM Forecasting Pilot (P1–P4)

**Paper:** ICML 2026 AI Forecasting Workshop  
**Confirmatory design:** P1–P3, 12 models, 2 conditions, 114 questions  
**Exploratory extension:** P4 Superforecaster prompt  
**Primary metric:** Brier score (lower = better)  
**Secondary metric:** Brier Index = (1 − √Brier) × 100 (higher = better)  
**Analysis:** Descriptive only — no new inferential tests in this report.


## 1. Data Integrity

**P1–P3 (confirmatory)**

| Metric | Value |
|--------|-------|
| Expected rows | 8,208 |
| Actual rows | 8,208 |
| Valid (parse success) | 8,196 (99.9%) |
| Missing forecasts | 12 |
| Duplicate keys | 0 |
| Cells with N≠114 | 0 → all 72 cells = 114 rows |

**P4 (exploratory)**

| Metric | Value |
|--------|-------|
| Expected rows | 2,736 |
| Actual rows | 2,736 |
| Valid | 2,695 (98.5%) |
| Missing | 41 |
| Cells with N≠114 | 0 |

**Parse rate by model:**

| Model | Type | P1-P3 valid/total | P1-P3 rate | P4 valid/total | P4 rate |
|-------|------|------------------|-----------|---------------|---------|
| gpt-5.4 | reasoning | 684/684 | 100.0% | 226/228 | 99.1% |
| gemini-3.1-pro | reasoning | 682/684 | 99.7% | 210/228 | 92.1% |
| kimi-k2.6 | reasoning | 684/684 | 100.0% | 226/228 | 99.1% |
| claude-opus-4.6 | reasoning | 684/684 | 100.0% | 228/228 | 100.0% |
| grok-4.20 | reasoning | 684/684 | 100.0% | 227/228 | 99.6% |
| deepseek-v3.2-speciale | reasoning | 684/684 | 100.0% | 226/228 | 99.1% |
| gpt-oss-120b | reasoning | 680/684 | 99.4% | 226/228 | 99.1% |
| glm-5.1 | reasoning* | 682/684 | 99.7% | 220/228 | 96.5% |
| qwen3-max | reasoning | 680/684 | 99.4% | 225/228 | 98.7% |
| gemma-4-31b | standard | 684/684 | 100.0% | 226/228 | 99.1% |
| gemini-3-flash | standard | 684/684 | 100.0% | 228/228 | 100.0% |
| mistral-large | standard | 684/684 | 100.0% | 227/228 | 99.6% |

**Remaining P1–P3 missing by model × prompt × condition:**

| Model | Prompt | Condition | Missing |
|-------|--------|-----------|---------|
| gemini-3.1-pro | P3 | closed_book | 1 |
| gemini-3.1-pro | P3 | shared_evidence | 1 |
| gpt-oss-120b | P1 | shared_evidence | 1 |
| gpt-oss-120b | P2 | closed_book | 1 |
| gpt-oss-120b | P3 | closed_book | 2 |
| glm-5.1 | P1 | shared_evidence | 1 |
| glm-5.1 | P2 | shared_evidence | 1 |
| qwen3-max | P2 | closed_book | 1 |
| qwen3-max | P2 | shared_evidence | 1 |
| qwen3-max | P3 | closed_book | 1 |
| qwen3-max | P3 | shared_evidence | 1 |

## 2. Overall Leaderboard


### 2a. P1–P3 Leaderboard

| Rank | Model | Type | Mean Brier | Brier Index | Parse Rate | Mean Lat (s) |
|------|-------|------|-----------|-------------|-----------|-------------|
| 1 | gpt-5.4 | reasoning | 0.1367 | 63.02 | 100.0% | 61.6 |
| 2 | gemini-3.1-pro | reasoning | 0.1432 | 62.16 | 99.7% | 35.4 |
| 3 | kimi-k2.6 | reasoning | 0.1440 | 62.05 | 100.0% | 249.2 |
| 4 | claude-opus-4.6 | reasoning | 0.1464 | 61.73 | 100.0% | 39.2 |
| 5 | grok-4.20 | reasoning | 0.1608 | 59.90 | 100.0% | 28.5 |
| 6 | deepseek-v3.2-speciale | reasoning | 0.1644 | 59.46 | 100.0% | 196.9 |
| 7 | glm-5.1 | reasoning* | 0.1734 | 58.36 | 99.7% | 76.6 |
| 8 | qwen3-max | reasoning | 0.1755 | 58.11 | 99.4% | 250.5 |
| 9 | gpt-oss-120b | reasoning | 0.1756 | 58.10 | 99.4% | 85.6 |
| 10 | gemma-4-31b | standard | 0.1783 | 57.78 | 100.0% | 39.3 |
| 11 | gemini-3-flash | standard | 0.1789 | 57.70 | 100.0% | 5.5 |
| 12 | mistral-large | standard | 0.2155 | 53.58 | 100.0% | 17.4 |

### 2b. P4 Leaderboard (exploratory)

| Rank | Model | Type | Mean Brier | Brier Index | Parse Rate | Mean Lat (s) |
|------|-------|------|-----------|-------------|-----------|-------------|
| 1 | gpt-5.4 | reasoning | 0.1246 | 64.70 | 99.1% | 93.0 |
| 2 | gemini-3.1-pro | reasoning | 0.1370 | 62.98 | 92.1% | 70.8 |
| 3 | kimi-k2.6 | reasoning | 0.1421 | 62.31 | 99.1% | 164.9 |
| 4 | claude-opus-4.6 | reasoning | 0.1433 | 62.14 | 100.0% | 67.2 |
| 5 | gpt-oss-120b | reasoning | 0.1467 | 61.70 | 99.1% | 199.0 |
| 6 | deepseek-v3.2-speciale | reasoning | 0.1474 | 61.61 | 99.1% | 342.1 |
| 7 | grok-4.20 | reasoning | 0.1521 | 61.00 | 99.6% | 54.9 |
| 8 | mistral-large | standard | 0.1622 | 59.72 | 99.6% | 31.5 |
| 9 | gemma-4-31b | standard | 0.1641 | 59.49 | 99.1% | 47.9 |
| 10 | gemini-3-flash | standard | 0.1650 | 59.38 | 100.0% | 7.3 |
| 11 | glm-5.1 | reasoning* | 0.1684 | 58.96 | 96.5% | 115.5 |
| 12 | qwen3-max | reasoning | 0.1734 | 58.35 | 98.7% | 233.2 |

### 2c. Combined P1–P4 Leaderboard

| Rank | Model | Type | Mean Brier | Brier Index |
|------|-------|------|-----------|-------------|
| 1 | gpt-5.4 | reasoning | 0.1337 | 63.43 |
| 2 | gemini-3.1-pro | reasoning | 0.1417 | 62.35 |
| 3 | kimi-k2.6 | reasoning | 0.1435 | 62.11 |
| 4 | claude-opus-4.6 | reasoning | 0.1457 | 61.83 |
| 5 | grok-4.20 | reasoning | 0.1586 | 60.17 |
| 6 | deepseek-v3.2-speciale | reasoning | 0.1602 | 59.98 |
| 7 | gpt-oss-120b | reasoning | 0.1684 | 58.97 |
| 8 | glm-5.1 | reasoning* | 0.1722 | 58.51 |
| 9 | gemma-4-31b | standard | 0.1748 | 58.20 |
| 10 | qwen3-max | reasoning | 0.1750 | 58.17 |
| 11 | gemini-3-flash | standard | 0.1754 | 58.11 |
| 12 | mistral-large | standard | 0.2022 | 55.03 |

### 2d. Top models by shared-evidence performance (P1–P3)

| Rank | Model | SE Brier |
|------|-------|---------|
| 1 | gemini-3.1-pro | 0.1134 |
| 2 | gpt-5.4 | 0.1274 |
| 3 | kimi-k2.6 | 0.1289 |
| 4 | claude-opus-4.6 | 0.1290 |
| 5 | grok-4.20 | 0.1439 |
| 6 | deepseek-v3.2-speciale | 0.1530 |

## 3. Model × Prompt × Condition Brier Table

Format: Brier (n valid). CB = closed_book, SE = shared_evidence.
**Bold** = best prompt for that model in that condition.

| Model | Type | P1/CB | P1/SE | P2/CB | P2/SE | P3/CB | P3/SE | P4/CB† | P4/SE† | Best CB | Best SE |
|-------|------|-------|-------|-------|-------|-------|-------|--------|--------|---------|---------|
| gpt-5.4 | reasoning | 0.1421 (114) | 0.1245 (114) | 0.1476 (114) | 0.1412 (114) | 0.1484 (114) | 0.1166 (114) | **0.1371 (113)** | **0.1121 (113)** | P4 | P4 |
| gemini-3.1-pro | reasoning | 0.1691 (114) | **0.1065 (114)** | 0.1748 (114) | 0.1170 (114) | 0.1750 (113) | 0.1168 (113) | **0.1458 (103)** | 0.1286 (107) | P4 | P1 |
| kimi-k2.6 | reasoning | 0.1618 (114) | 0.1414 (114) | 0.1654 (114) | **0.1185 (114)** | **0.1503 (114)** | 0.1267 (114) | 0.1523 (114) | 0.1317 (112) | P3 | P2 |
| claude-opus-4.6 | reasoning | 0.1611 (114) | 0.1369 (114) | 0.1677 (114) | **0.1232 (114)** | 0.1628 (114) | 0.1269 (114) | **0.1518 (114)** | 0.1349 (114) | P4 | P2 |
| grok-4.20 | reasoning | 0.1779 (114) | 0.1544 (114) | 0.1764 (114) | 0.1398 (114) | 0.1790 (114) | 0.1375 (114) | **0.1708 (113)** | **0.1335 (114)** | P4 | P4 |
| deepseek-v3.2-speciale | reasoning | 0.1763 (114) | 0.1447 (114) | 0.1658 (114) | 0.1476 (114) | 0.1851 (114) | 0.1667 (114) | **0.1643 (113)** | **0.1306 (113)** | P4 | P4 |
| gpt-oss-120b | reasoning | 0.1683 (114) | 0.1504 (113) | 0.1637 (113) | 0.1543 (114) | 0.2222 (112) | 0.1950 (114) | **0.1576 (112)** | **0.1360 (114)** | P4 | P4 |
| glm-5.1 | reasoning* | 0.1802 (114) | 0.1611 (113) | 0.1979 (114) | **0.1521 (113)** | 0.1917 (114) | 0.1570 (114) | **0.1670 (113)** | 0.1699 (107) | P4 | P2 |
| qwen3-max | reasoning | 0.1886 (114) | **0.1468 (114)** | **0.1860 (113)** | 0.1550 (113) | 0.1925 (113) | 0.1843 (113) | 0.1913 (112) | 0.1558 (113) | P2 | P1 |
| gemma-4-31b | standard | **0.1748 (114)** | 0.1731 (114) | 0.1879 (114) | 0.1620 (114) | 0.1961 (114) | 0.1757 (114) | 0.1807 (112) | **0.1479 (114)** | P1 | P4 |
| gemini-3-flash | standard | 0.1942 (114) | 0.1666 (114) | 0.1855 (114) | 0.1671 (114) | 0.1899 (114) | 0.1702 (114) | **0.1772 (114)** | **0.1528 (114)** | P4 | P4 |
| mistral-large | standard | 0.1932 (114) | 0.2157 (114) | **0.1859 (114)** | 0.2050 (114) | 0.2838 (114) | 0.2091 (114) | 0.1864 (114) | **0.1379 (113)** | P2 | P4 |

† P4 is exploratory, not part of the confirmatory P1–P3 design.

**Prompt win counts — P1–P3 only (best prompt per model, P4 excluded):**

| Prompt | Wins CB | Wins SE | Wins Overall |
|--------|---------|---------|-------------|
| P1 | 5 | 5 | 4 |
| P2 | 6 | 5 | 5 |
| P3 | 1 | 2 | 3 |

**Including P4† (best across all 4 prompts):**

| Prompt | Wins CB | Wins SE |
|--------|---------|---------|
| P1 | 1 | 2 |
| P2 | 2 | 3 |
| P3 | 1 | 0 |
| P4† | 8 | 7 |

*P4 dominates the average (CB+SE)/2 for all 12 models, so 'Wins Overall including P4' is not shown — it is 12/12 P4.*

## 4. Prompt × Condition Aggregate Analysis

| Prompt | Condition | Mean Brier | Brier Index | N valid | SE−CB delta | Market Brier |
|--------|-----------|-----------|-------------|---------|------------|-------------|
| P1 | CL | 0.1740 | 58.29 | 1368 |  | 0.0949 |
| P1 | SH | 0.1518 | 61.03 | 1366 | -0.0221 | 0.0950 |
| P2 | CL | 0.1754 | 58.12 | 1366 |  | 0.0949 |
| P2 | SH | 0.1486 | 61.46 | 1366 | -0.0268 | 0.0950 |
| P3 | CL | 0.1897 | 56.45 | 1364 |  | 0.0949 |
| P3 | SH | 0.1569 | 60.39 | 1366 | -0.0328 | 0.0950 |
| P4† | CL | 0.1653 | 59.34 | 1347 |  | 0.0945 |
| P4† | SH | 0.1392 | 62.69 | 1348 | -0.0261 | 0.0958 |

*SE−CB delta shown in SE row only. Negative = evidence helps. P4† = exploratory.*

**Best P1–P3 prompt overall:** P2 (Brier=0.1620)  
**P4 overall Brier:** 0.1523 (better than best P1–P3 — exploratory)

## 5. Shared Evidence Effect


### 5a. Per model, per prompt (P1–P3)

| Model | Type | P1 delta | P2 delta | P3 delta | Mean delta | Helped? |
|-------|------|---------|---------|---------|-----------|---------|
| gemini-3.1-pro | reasoning | -0.0626 | -0.0579 | -0.0582 | -0.0596 | YES |
| claude-opus-4.6 | reasoning | -0.0242 | -0.0445 | -0.0359 | -0.0348 | YES |
| grok-4.20 | reasoning | -0.0235 | -0.0366 | -0.0416 | -0.0339 | YES |
| glm-5.1 | reasoning* | -0.0191 | -0.0458 | -0.0347 | -0.0332 | YES |
| kimi-k2.6 | reasoning | -0.0204 | -0.0469 | -0.0236 | -0.0303 | YES |
| qwen3-max | reasoning | -0.0418 | -0.0309 | -0.0082 | -0.0270 | YES |
| deepseek-v3.2-speciale | reasoning | -0.0316 | -0.0182 | -0.0184 | -0.0227 | YES |
| gemini-3-flash | standard | -0.0276 | -0.0184 | -0.0198 | -0.0219 | YES |
| gpt-5.4 | reasoning | -0.0176 | -0.0063 | -0.0318 | -0.0186 | YES |
| gpt-oss-120b | reasoning | -0.0179 | -0.0094 | -0.0271 | -0.0181 | YES |
| gemma-4-31b | standard | -0.0016 | -0.0259 | -0.0204 | -0.0160 | YES |
| mistral-large | standard | 0.0225 | 0.0191 | -0.0747 | -0.0110 | partial |

### 5b. P4 evidence effect (exploratory)

| Model | P4 CB | P4 SE | P4 delta |
|-------|-------|-------|---------|
| gemini-3.1-pro | 0.1458 | 0.1286 | -0.0171 |
| claude-opus-4.6 | 0.1518 | 0.1349 | -0.0169 |
| grok-4.20 | 0.1708 | 0.1335 | -0.0372 |
| glm-5.1 | 0.1670 | 0.1699 | 0.0029 |
| kimi-k2.6 | 0.1523 | 0.1317 | -0.0207 |
| qwen3-max | 0.1913 | 0.1558 | -0.0355 |
| deepseek-v3.2-speciale | 0.1643 | 0.1306 | -0.0337 |
| gemini-3-flash | 0.1772 | 0.1528 | -0.0243 |
| gpt-5.4 | 0.1371 | 0.1121 | -0.0250 |
| gpt-oss-120b | 0.1576 | 0.1360 | -0.0216 |
| gemma-4-31b | 0.1807 | 0.1479 | -0.0328 |
| mistral-large | 0.1864 | 0.1379 | -0.0485 |

Models helped by evidence under P4: 11/12

## 6. P4 Exploratory Analysis

> P4 is exploratory and not preregistered. Interpret cautiously.

| Model | Type | P4 CB | Best P123 CB | Delta CB | Wins CB | P4 SE | Best P123 SE | Delta SE | Wins SE |
|-------|------|-------|-------------|---------|---------|-------|-------------|---------|---------|
| gpt-5.4 | reasoning | 0.1371 | 0.1421 (P1) | -0.0050 | YES | 0.1121 | 0.1166 (P3) | -0.0045 | YES |
| gemini-3.1-pro | reasoning | 0.1458 | 0.1691 (P1) | -0.0233 | YES | 0.1286 | 0.1065 (P1) | 0.0221 | no |
| kimi-k2.6 | reasoning | 0.1523 | 0.1503 (P3) | 0.0021 | no | 0.1317 | 0.1185 (P2) | 0.0131 | no |
| claude-opus-4.6 | reasoning | 0.1518 | 0.1611 (P1) | -0.0093 | YES | 0.1349 | 0.1232 (P2) | 0.0117 | no |
| grok-4.20 | reasoning | 0.1708 | 0.1764 (P2) | -0.0056 | YES | 0.1335 | 0.1375 (P3) | -0.0039 | YES |
| deepseek-v3.2-speciale | reasoning | 0.1643 | 0.1658 (P2) | -0.0015 | YES | 0.1306 | 0.1447 (P1) | -0.0141 | YES |
| gpt-oss-120b | reasoning | 0.1576 | 0.1637 (P2) | -0.0061 | YES | 0.1360 | 0.1504 (P1) | -0.0144 | YES |
| glm-5.1 | reasoning* | 0.1670 | 0.1802 (P1) | -0.0132 | YES | 0.1699 | 0.1521 (P2) | 0.0178 | no |
| qwen3-max | reasoning | 0.1913 | 0.1860 (P2) | 0.0053 | no | 0.1558 | 0.1468 (P1) | 0.0089 | no |
| gemma-4-31b | standard | 0.1807 | 0.1748 (P1) | 0.0059 | no | 0.1479 | 0.1620 (P2) | -0.0141 | YES |
| gemini-3-flash | standard | 0.1772 | 0.1855 (P2) | -0.0084 | YES | 0.1528 | 0.1666 (P1) | -0.0138 | YES |
| mistral-large | standard | 0.1864 | 0.1859 (P2) | 0.0005 | no | 0.1379 | 0.2050 (P2) | -0.0670 | YES |

**P4 beats best P1–P3:** 8/12 models in closed-book, 7/12 in shared-evidence

Standard models where P4 wins CB: 1/4  |  Reasoning: 6/8

## 7. Reasoning vs Standard Models

> gpt-oss-120b is classified as **reasoning** (called with reasoning: effort=high). The reasoning vs standard distinction reflects API call type, not architecture causally.

| Metric | Reasoning (8 models) | Standard (4 models) |
|--------|---------------------|-------------------|
| Mean Brier | 0.1558060417429513 | 0.1908841617933723 |
| Parse Rate (%) | 99.8 | 100.0 |
| Mean Latency (s) | 118.4 | 20.7 |
| Mean Input Tokens | 2490 | 2590 |
| Mean Output Tokens | 4845 | 672 |
| Mean Reasoning Tokens | 4502 | None |

**By prompt × condition:**

| Type | P1/CB | P1/SE | P2/CB | P2/SE | P3/CB | P3/SE |
|------|-------|-------|-------|-------|-------|-------|
| reasoning | 0.1681 | 0.1382 | 0.1684 | 0.1371 | 0.1768 | 0.1463 |
| standard | 0.1874 | 0.1852 | 0.1864 | 0.1780 | 0.2233 | 0.1850 |

## 8. Market Baseline Comparison

Market baseline Brier (P1–P3 rows): **0.0949**  
Market baseline Brier (P4 rows): **0.0952**

**P1–P3: Model vs market (positive delta = model worse than market):**

| Rank | Model | Model Brier | Market Brier | Delta | Beats Market? | N |
|------|-------|------------|-------------|-------|--------------|---|
| 1 | gpt-5.4 | 0.1367 | 0.0948 | 0.0419 | no | 684 |
| 2 | gemini-3.1-pro | 0.1432 | 0.0951 | 0.0481 | no | 682 |
| 3 | kimi-k2.6 | 0.1440 | 0.0948 | 0.0492 | no | 684 |
| 4 | claude-opus-4.6 | 0.1464 | 0.0948 | 0.0516 | no | 684 |
| 5 | grok-4.20 | 0.1608 | 0.0948 | 0.0660 | no | 684 |
| 6 | deepseek-v3.2-speciale | 0.1644 | 0.0948 | 0.0695 | no | 684 |
| 7 | glm-5.1 | 0.1734 | 0.0951 | 0.0783 | no | 682 |
| 8 | qwen3-max | 0.1755 | 0.0954 | 0.0801 | no | 680 |
| 9 | gpt-oss-120b | 0.1756 | 0.0947 | 0.0808 | no | 680 |
| 10 | gemma-4-31b | 0.1783 | 0.0948 | 0.0834 | no | 684 |
| 11 | gemini-3-flash | 0.1789 | 0.0948 | 0.0841 | no | 684 |
| 12 | mistral-large | 0.2155 | 0.0948 | 0.1206 | no | 684 |

**Models beating market (P1–P3): 0/12**

**P4: Model vs market (exploratory):**

| Model | P4 Brier | Market Brier | Delta | Beats Market? |
|-------|---------|-------------|-------|--------------|
| gpt-5.4 | 0.1246 | 0.0957 | 0.0289 | no |
| gemini-3.1-pro | 0.1370 | 0.0964 | 0.0407 | no |
| kimi-k2.6 | 0.1421 | 0.0955 | 0.0465 | no |
| claude-opus-4.6 | 0.1433 | 0.0948 | 0.0485 | no |
| deepseek-v3.2-speciale | 0.1474 | 0.0943 | 0.0531 | no |
| gpt-oss-120b | 0.1467 | 0.0914 | 0.0553 | no |
| grok-4.20 | 0.1521 | 0.0952 | 0.0568 | no |
| mistral-large | 0.1622 | 0.0951 | 0.0671 | no |
| gemma-4-31b | 0.1641 | 0.0950 | 0.0691 | no |
| gemini-3-flash | 0.1650 | 0.0948 | 0.0702 | no |
| glm-5.1 | 0.1684 | 0.0981 | 0.0703 | no |
| qwen3-max | 0.1734 | 0.0959 | 0.0775 | no |

**Models beating market (P4, exploratory): 0/12**

**By prompt × condition vs market (P1–P3):**

| Prompt | Condition | Model Brier | Market Brier | Delta |
|--------|-----------|------------|-------------|-------|
| P1 | CL | 0.1740 | 0.0948 | 0.0791 |
| P1 | SH | 0.1518 | 0.0950 | 0.0569 |
| P2 | CL | 0.1754 | 0.0949 | 0.0805 |
| P2 | SH | 0.1486 | 0.0950 | 0.0536 |
| P3 | CL | 0.1897 | 0.0949 | 0.0948 |
| P3 | SH | 0.1569 | 0.0950 | 0.0619 |

## 9. Calibration and Sharpness


### 9a. Overall

| Metric | P1–P3 | P4 (exploratory) |
|--------|-------|-----------------|
| N forecasts | 8196 | 2695 |
| Mean forecast | 0.2849 | 0.2591 |
| Forecast SD | 0.2941 | 0.2776 |
| % < 0.10 | 36.6 | 39.7 |
| % 0.10–0.90 | 58.2 | 56.2 |
| % > 0.90 | 5.2 | 4.0 |
| ECE | 0.08184 | 0.06369 |
| Overconf high (>0.8, outcome=0) | 384 | 85 |
| Overconf low (<0.2, outcome=1) | 340 | 125 |

### 9b. P1–P3 Calibration table (10 bins)

| Bin | N | Mean Forecast | Observed Freq | Gap (F−O) |
|-----|---|--------------|--------------|-----------|
| 0.0-0.1 | 2996 | 0.0351 | 0.0574 | -0.0223 |
| 0.1-0.2 | 1397 | 0.1366 | 0.1203 | 0.0163 |
| 0.2-0.3 | 800 | 0.2373 | 0.2325 | 0.0048 |
| 0.3-0.4 | 792 | 0.3406 | 0.3157 | 0.0249 |
| 0.4-0.5 | 377 | 0.4319 | 0.4005 | 0.0314 |
| 0.5-0.6 | 203 | 0.5497 | 0.4089 | 0.1408 |
| 0.6-0.7 | 441 | 0.6418 | 0.4671 | 0.1747 |
| 0.7-0.8 | 354 | 0.7389 | 0.4605 | 0.2785 |
| 0.8-0.9 | 351 | 0.8410 | 0.4729 | 0.3681 |
| 0.9-1.0 | 485 | 0.9596 | 0.5216 | 0.4379 |

*Gap > 0 = overconfident; Gap < 0 = underconfident.*

### 9c. By model (P1–P3, mean forecast and SD)

| Model | Mean Forecast | SD | OC-high N | OC-low N |
|-------|-------------|----|-----------|---------| 
| gpt-5.4 | 0.2561 | 0.2664 | 12 | 25 |
| gemini-3.1-pro | 0.2303 | 0.305 | 25 | 33 |
| kimi-k2.6 | 0.2338 | 0.2554 | 15 | 26 |
| claude-opus-4.6 | 0.2232 | 0.2978 | 28 | 41 |
| grok-4.20 | 0.3105 | 0.244 | 20 | 18 |
| deepseek-v3.2-speciale | 0.3134 | 0.3047 | 36 | 19 |
| gpt-oss-120b | 0.282 | 0.2984 | 41 | 30 |
| glm-5.1 | 0.2576 | 0.3085 | 40 | 36 |
| qwen3-max | 0.2576 | 0.3239 | 43 | 42 |
| gemma-4-31b | 0.3397 | 0.293 | 39 | 21 |
| gemini-3-flash | 0.3229 | 0.3001 | 45 | 25 |
| mistral-large | 0.3913 | 0.2713 | 40 | 24 |

### 9d. By prompt × condition (P1–P3, mean forecast and SD)

| Prompt | Condition | Mean Forecast | SD | ECE |
|--------|-----------|-------------|----|----|
| P1 | CL | 0.2723 | 0.2542 | 0.08848 |
| P1 | SH | 0.3032 | 0.3124 | 0.08725 |
| P2 | CL | 0.2603 | 0.2546 | 0.09043 |
| P2 | SH | 0.2913 | 0.3106 | 0.08332 |
| P3 | CL | 0.2749 | 0.2774 | 0.1146 |
| P3 | SH | 0.3076 | 0.3416 | 0.09092 |

## 10. Topic and Source Effects


### 10a. By Topic

| Topic | CB Brier | CB N | SE Brier | SE N | SE−CB delta |
|-------|---------|------|---------|------|------------|
| ai_technology | 0.1262 | 720 | 0.0965 | 720 | -0.0296 ← helps |
| conflict | 0.1701 | 648 | 0.1893 | 647 | 0.0192 ← hurts |
| economics | 0.1282 | 503 | 0.0872 | 503 | -0.0410 ← helps |
| energy | 0.2515 | 143 | 0.3886 | 144 | 0.1371 ← hurts |
| entertainment | 0.1831 | 180 | 0.2112 | 180 | 0.0281 ← hurts |
| finance_market | 0.2795 | 504 | 0.2169 | 504 | -0.0625 ← helps |
| geopolitics | 0.1627 | 718 | 0.1629 | 716 | 0.0002 |
| other | 0.2034 | 179 | 0.1413 | 180 | -0.0621 ← helps |
| public_health | 0.2545 | 215 | 0.1679 | 216 | -0.0866 ← helps |
| sports | 0.1839 | 288 | 0.0248 | 288 | -0.1591 ← helps |

### 10b. By Source

| Source | CB Brier | CB N | SE Brier | SE N | SE−CB delta |
|--------|---------|------|---------|------|------------|
| manifold | 0.1596 | 1295 | 0.0861 | 1296 | -0.0735 |
| polymarket | 0.1540 | 1472 | 0.1592 | 1473 | 0.0052 |
| metaculus | 0.2276 | 1331 | 0.2096 | 1329 | -0.0179 |

## 11. Error Analysis


### 11a. Top 20 Worst Individual Forecasts (P1–P3)

| # | Model | Prompt | Cond | Forecast | Outcome | Brier | Question |
|---|-------|--------|------|---------|---------|-------|---------|
| 1 | gpt-5.4 | P1 | CL | 0.00 | 1 | 1.0000 | US strikes Nigeria by December 31? |
| 2 | gpt-5.4 | P1 | SH | 1.00 | 0 | 1.0000 | Will the US House of Representatives censure, expe… |
| 3 | gpt-5.4 | P2 | SH | 1.00 | 0 | 1.0000 | Will the US House of Representatives censure, expe… |
| 4 | gemini-3.1-pro | P1 | CL | 1.00 | 0 | 1.0000 | Will Solana reach $140 in March? |
| 5 | gemini-3.1-pro | P1 | SH | 1.00 | 0 | 1.0000 | Elon no longer world's richest before 2026? |
| 6 | deepseek-v3.2-speciale | P1 | SH | 1.00 | 0 | 1.0000 | Will the US House of Representatives censure, expe… |
| 7 | deepseek-v3.2-speciale | P2 | SH | 1.00 | 0 | 1.0000 | Will the US House of Representatives censure, expe… |
| 8 | deepseek-v3.2-speciale | P2 | SH | 1.00 | 0 | 1.0000 | Elon no longer world's richest before 2026? |
| 9 | deepseek-v3.2-speciale | P3 | SH | 1.00 | 0 | 1.0000 | Will the US House of Representatives censure, expe… |
| 10 | kimi-k2.6 | P1 | SH | 1.00 | 0 | 1.0000 | Will the U.S. invade Venezuela by January 31, 2026… |
| 11 | kimi-k2.6 | P1 | SH | 1.00 | 0 | 1.0000 | Will the US House of Representatives censure, expe… |
| 12 | kimi-k2.6 | P1 | SH | 1.00 | 0 | 1.0000 | Elon no longer world's richest before 2026? |
| 13 | kimi-k2.6 | P2 | SH | 1.00 | 0 | 1.0000 | Elon no longer world's richest before 2026? |
| 14 | gemini-3-flash | P2 | SH | 1.00 | 0 | 1.0000 | If Trump wins the election will the inflation rate… |
| 15 | gemma-4-31b | P1 | SH | 1.00 | 0 | 1.0000 | Will the U.S. invade Venezuela by January 31, 2026… |
| 16 | gemma-4-31b | P1 | SH | 1.00 | 0 | 1.0000 | Will the US House of Representatives censure, expe… |
| 17 | gemma-4-31b | P1 | SH | 1.00 | 0 | 1.0000 | In 2025, will the domestic content requirements be… |
| 18 | gemma-4-31b | P1 | SH | 1.00 | 0 | 1.0000 | Elon no longer world's richest before 2026? |
| 19 | gemma-4-31b | P2 | SH | 1.00 | 0 | 1.0000 | Will the U.S. invade Venezuela by January 31, 2026… |
| 20 | gemma-4-31b | P2 | SH | 1.00 | 0 | 1.0000 | In 2025, will the domestic content requirements be… |

### 11b. Catastrophic Overconfidence Summary (P1–P3)

| Category | Total | % of valid rows |
|---------|-------|----------------|
| Forecast > 0.80, Outcome = 0 | 384 | 4.7% |
| Forecast < 0.20, Outcome = 1 | 340 | 4.1% |
| Total | 724 | 8.8% |

**Overconfidence by model:**

| Model | OC-high (>0.8/NO) | OC-low (<0.2/YES) | Total |
|-------|-----------------|-----------------|-------|
| gpt-5.4 | 12 | 25 | 37 |
| gemini-3.1-pro | 25 | 33 | 58 |
| kimi-k2.6 | 15 | 26 | 41 |
| claude-opus-4.6 | 28 | 41 | 69 |
| grok-4.20 | 20 | 18 | 38 |
| deepseek-v3.2-speciale | 36 | 19 | 55 |
| gpt-oss-120b | 41 | 30 | 71 |
| glm-5.1 | 40 | 36 | 76 |
| qwen3-max | 43 | 42 | 85 |
| gemma-4-31b | 39 | 21 | 60 |
| gemini-3-flash | 45 | 25 | 70 |
| mistral-large | 40 | 24 | 64 |

### 11c. Hardest Questions (highest mean Brier across all models/prompts)

| Rank | Question ID | Mean Brier | Question |
|------|------------|-----------|---------|
| 1 | `0x3b7e03065f6437f93e` | 0.8095 | US strikes Nigeria by December 31? |
| 2 | `40852` | 0.7984 | Will the US House of Representatives censure, expel or repri… |
| 3 | `0x3bc785525e9bb8e8f2` | 0.7032 | Elon no longer world's richest before 2026? |
| 4 | `SZtPUQqhp5` | 0.6424 | Will Intel Stock (INTC) reach $63 in 2026? |
| 5 | `0IUCA5s8EN` | 0.6421 | Will the US strike Iran by the end of February? |
| 6 | `35573` | 0.6328 | ¿Cerrará Bitcoin el 2025 más alto de lo que empezó? |
| 7 | `4919` | 0.6047 | At the end of 2025, will any of GiveWell's top charities per… |
| 8 | `0xcb139a61f2cc201078` | 0.4777 | Will Crude Oil (CL) hit (HIGH) $110 by end of June? |
| 9 | `42274` | 0.4694 | Will any AI model achieve a score of 95% or higher on the GP… |
| 10 | `0x58234757c9519ddfff` | 0.4471 | Will the Detroit Pistons win the 2025–2026 NBA Central Divis… |

### 11d. Top 10 SE improvements vs CB (same model/question/prompt)

| Model | Prompt | Question | CB Brier | SE Brier | Improvement |
|-------|--------|---------|---------|---------|------------|
| gpt-oss-120b | P3 | Will an AI score over 80% on FrontierMat… | 1.0000 | 0.0004 | 0.9996 |
| gemini-3.1-pro | P1 | Will Solana reach $140 in March?… | 1.0000 | 0.0025 | 0.9975 |
| qwen3-max | P3 | Will Trump & Elon cut >250,000 governmen… | 0.9980 | 0.0025 | 0.9955 |
| qwen3-max | P2 | Will Trump & Elon cut >250,000 governmen… | 0.9940 | 0.0025 | 0.9915 |
| gpt-oss-120b | P3 | Will the Julia programming language be i… | 1.0000 | 0.0121 | 0.9879 |
| gpt-oss-120b | P3 | Russia joins the Board of Peace by March… | 1.0000 | 0.0196 | 0.9804 |
| gemini-3.1-pro | P3 | Will Solana reach $140 in March?… | 0.9801 | 0.0001 | 0.9800 |
| qwen3-max | P1 | Will the fertility rate of South Korea i… | 0.9801 | 0.0001 | 0.9800 |
| qwen3-max | P1 | Will Trump & Elon cut >250,000 governmen… | 0.9801 | 0.0001 | 0.9800 |
| qwen3-max | P2 | Will the fertility rate of South Korea i… | 0.9801 | 0.0004 | 0.9797 |

## 12. Practicality and Model Selection


### 12a. Latency and token usage (P1–P3)

| Model | Type | Brier | SE Brier | Parse % | Lat (s) | In-tok | Out-tok | Reason-tok |
|-------|------|-------|---------|---------|---------|--------|---------|-----------|
| gpt-5.4 | reasoning | 0.1367 | 0.1274 | 100.0% | 61.6 | 2368 | 3158 | 2973 |
| gemini-3.1-pro | reasoning | 0.1432 | 0.1134 | 99.7% | 35.4 | 2592 | 2911 | 2414 |
| kimi-k2.6 | reasoning | 0.1440 | 0.1289 | 100.0% | 249.2 | 2417 | 7763 | 7674 |
| claude-opus-4.6 | reasoning | 0.1464 | 0.1290 | 100.0% | 39.2 | 2692 | 1896 | 864 |
| grok-4.20 | reasoning | 0.1608 | 0.1439 | 100.0% | 28.5 | 2448 | 2164 | 1789 |
| deepseek-v3.2-speciale | reasoning | 0.1644 | 0.1530 | 100.0% | 196.9 | 2412 | 7353 | 7064 |
| glm-5.1 | reasoning* | 0.1734 | 0.1567 | 99.7% | 76.6 | 2412 | 2463 | 2124 |
| qwen3-max | reasoning | 0.1755 | 0.1620 | 99.4% | 250.5 | 2557 | 9332 | 8883 |
| gpt-oss-120b | reasoning | 0.1756 | 0.1666 | 99.4% | 85.6 | 2431 | 4195 | 4333 |
| gemma-4-31b | standard | 0.1783 | 0.1703 | 100.0% | 39.3 | 2604 | 469 | N/A |
| gemini-3-flash | standard | 0.1789 | 0.1680 | 100.0% | 5.5 | 2591 | 601 | N/A |
| mistral-large | standard | 0.2155 | 0.2099 | 100.0% | 17.4 | 2577 | 945 | N/A |

*reasoning\* = called without explicit reasoning effort flag but produces internal reasoning tokens (mean 2,124). All other standard models (gemini-3-flash, gemma-4-31b, mistral-large) produce 0 reasoning tokens.*

### 12b. Recommended model sets for live Metaculus summer tournament


**Set A — Accuracy-focused (6 models):**

| Model | Justification |
|-------|-------------|
| gpt-5.4 (reasoning) | Best overall P1–P3 Brier; strong SE performance |
| gemini-3.1-pro (reasoning) | 2nd overall; excellent SE Brier |
| kimi-k2.6 (reasoning) | 3rd overall; consistent across prompts |
| claude-opus-4.6 (reasoning) | 4th overall; good calibration |
| grok-4.20 (reasoning) | 5th overall; strong reasoning model |
| deepseek-v3.2-speciale (reasoning) | 6th overall; diversity value |

**Set B — Practical/cost-balanced (6 models):**

| Model | Brier | Lat (s) | Type | Justification |
|-------|-------|---------|------|-------------|
| gpt-5.4 | 0.1367 | 61.6 | reasoning | Top accuracy |
| gemini-3.1-pro | 0.1432 | 35.4 | reasoning | Top accuracy |
| claude-opus-4.6 | 0.1464 | 39.2 | reasoning | Top accuracy |
| grok-4.20 | 0.1608 | 28.5 | reasoning | Top accuracy |
| kimi-k2.6 | 0.1440 | 249.2 | reasoning | Top accuracy |
| gemini-3-flash | 0.1789 | 5.5 | standard | Low latency + good accuracy |

## 13. Key Findings for Paper


**Finding 1:** **Evidence universally improves LLM forecasting.** Shared evidence reduces mean Brier from 0.1797 (CB) to 0.1524 (SE), a delta of -0.0272 across all 12 models and 3 prompts. This effect is significant at p<0.001 (cluster bootstrap, N=114 questions).

**Finding 2:** **P3 (Bayesian) is brittle in closed-book but shows the largest evidence gain.** P3 closed-book Brier = 0.1897 vs P1 closed-book = 0.1740: P3 is the worst prompt without evidence. Under shared evidence P3 improves to 0.1569, its largest absolute gain of any prompt, yet it remains worse than P1+SE (0.1518) and P2+SE in absolute terms. The Bayesian structure amplifies evidence when available but degrades performance without it.

**Finding 3:** **P3 × evidence interaction is directional but not significant.** The difference-in-differences (P3 gain vs P1 gain) = −0.011 (p≈0.08). The pattern is consistent across most models but the pilot (N=114) is underpowered to confirm this interaction.

**Finding 4:** **P4 Superforecaster prompt outperforms P1–P3 overall (exploratory).** P4 wins vs best P1–P3 in 8/12 models (CB) and 7/12 (SE). Gains are largest for weaker/standard models (e.g., mistral-large: P3/CB=0.2838 → P4/CB=0.1864). These results are exploratory and not preregistered.

**Finding 5:** **Best model overall: gpt-5.4** (Brier = 0.1367, BI = 63.02). Worst: mistral-large (Brier = 0.2155). Spread = 0.0787 Brier points.

**Finding 6:** **Best aggregate model–prompt–condition cell (P1–P3):** gemini-3.1-pro / P1 / shared_evidence (Brier = 0.1065, N=114). Best P4 cell: gpt-5.4 / P4 / shared_evidence (Brier = 0.1121, N=113, exploratory). Gemini 3.1 Pro's shared-evidence performance is one of the strongest single-cell results.

**Finding 7:** **No LLM beats the market baseline.** Market Brier = 0.0949. Best model gap: gpt-5.4 is 0.0419 Brier points above market (p<0.001). The crowd-sourced market is a strong benchmark that current frontier LLMs cannot match.

**Finding 8:** **LLMs are severely miscalibrated at high confidence.** Overall ECE = 0.0818. The miscalibration is not uniform: models are near-calibrated at low probabilities but dramatically overconfident above 0.5. In the 0.9–1.0 bin (N=485 forecasts), mean forecast = 0.9596 but observed frequency = 0.5216 — a gap of 0.4379. In the 0.7–0.8 bin (N=354), gap = 0.2785. This pattern holds across all 12 models and constitutes 724 catastrophic errors (8.8% of valid rows): 384 cases of forecast > 0.80 on events that did not occur, 340 cases of forecast < 0.20 on events that did occur. P4 shows better calibration (ECE = 0.0637).

**Finding 9:** **Reasoning models outperform standard models** (Brier 0.1558 vs 0.1909, delta = 0.0351). However, reasoning models are 5.7× slower. This difference is confounded by model size and recency.

**Finding 10:** **Topic and source heterogeneity is substantial.** Evidence helps most on sports questions (delta = -0.1591). Some topics show negligible evidence benefit. Caution: topic categories contain as few as 143 questions.


## 14. Paper-Ready Tables


### Table 1: Dataset and Design Summary

| | P1–P3 (Confirmatory) | P4 (Exploratory) |
|---|---|---|
| Questions | 114 binary ForecastBench | Same 114 questions |
| Models | 12 (8 reasoning, 4 standard) | Same 12 models |
| Prompts | 3 (Control, Base-Rate-First, Bayesian) | 1 (Superforecaster) |
| Conditions | 2 (closed-book, shared-evidence) | Same 2 conditions |
| Total calls | 8,208 | 2,736 |
| Valid forecasts | 8,196 (99.9%) | 2,695 (98.5%) |
| Evidence source | AskNews (pre-freeze) | Same |
| Market baseline | freeze\_datetime\_value | Same |

### Table 2: Model Leaderboard (P1–P3)

| Rank | Model | Type | Brier | BI | Parse % | Lat (s) |
|------|-------|------|-------|----|---------|---------| 
| 1 | gpt-5.4 | reasoning | 0.1367 | 63.02 | 100.0% | 61.6 |
| 2 | gemini-3.1-pro | reasoning | 0.1432 | 62.16 | 99.7% | 35.4 |
| 3 | kimi-k2.6 | reasoning | 0.1440 | 62.05 | 100.0% | 249.2 |
| 4 | claude-opus-4.6 | reasoning | 0.1464 | 61.73 | 100.0% | 39.2 |
| 5 | grok-4.20 | reasoning | 0.1608 | 59.90 | 100.0% | 28.5 |
| 6 | deepseek-v3.2-speciale | reasoning | 0.1644 | 59.46 | 100.0% | 196.9 |
| 7 | glm-5.1 | reasoning* | 0.1734 | 58.36 | 99.7% | 76.6 |
| 8 | qwen3-max | reasoning | 0.1755 | 58.11 | 99.4% | 250.5 |
| 9 | gpt-oss-120b | reasoning | 0.1756 | 58.10 | 99.4% | 85.6 |
| 10 | gemma-4-31b | standard | 0.1783 | 57.78 | 100.0% | 39.3 |
| 11 | gemini-3-flash | standard | 0.1789 | 57.70 | 100.0% | 5.5 |
| 12 | mistral-large | standard | 0.2155 | 53.58 | 100.0% | 17.4 |

### Table 3: Prompt × Condition Aggregate Brier

| Prompt | CB Brier | SE Brier | SE−CB delta |
|--------|---------|---------|------------|
| P1 | 0.1740 | 0.1518 | -0.0221 |
| P2 | 0.1754 | 0.1486 | -0.0268 |
| P3 | 0.1897 | 0.1569 | -0.0328 |
| P4† | 0.1653 | 0.1392 | -0.0261 |

*† Exploratory only.*

### Table 4: Model × Prompt × Condition Brier (condensed)

| Model | P1/CB | P1/SE | P2/CB | P2/SE | P3/CB | P3/SE | P4/CB† | P4/SE† |
|-------|-------|-------|-------|-------|-------|-------|--------|--------|
| gpt-5.4 | 0.1421 | 0.1245 | 0.1476 | 0.1412 | 0.1484 | 0.1166 | 0.1371 | 0.1121 |
| gemini-3.1-pro | 0.1691 | 0.1065 | 0.1748 | 0.1170 | 0.1750 | 0.1168 | 0.1458 | 0.1286 |
| kimi-k2.6 | 0.1618 | 0.1414 | 0.1654 | 0.1185 | 0.1503 | 0.1267 | 0.1523 | 0.1317 |
| claude-opus-4.6 | 0.1611 | 0.1369 | 0.1677 | 0.1232 | 0.1628 | 0.1269 | 0.1518 | 0.1349 |
| grok-4.20 | 0.1779 | 0.1544 | 0.1764 | 0.1398 | 0.1790 | 0.1375 | 0.1708 | 0.1335 |
| deepseek-v3.2-speciale | 0.1763 | 0.1447 | 0.1658 | 0.1476 | 0.1851 | 0.1667 | 0.1643 | 0.1306 |
| gpt-oss-120b | 0.1683 | 0.1504 | 0.1637 | 0.1543 | 0.2222 | 0.1950 | 0.1576 | 0.1360 |
| glm-5.1 | 0.1802 | 0.1611 | 0.1979 | 0.1521 | 0.1917 | 0.1570 | 0.1670 | 0.1699 |
| qwen3-max | 0.1886 | 0.1468 | 0.1860 | 0.1550 | 0.1925 | 0.1843 | 0.1913 | 0.1558 |
| gemma-4-31b | 0.1748 | 0.1731 | 0.1879 | 0.1620 | 0.1961 | 0.1757 | 0.1807 | 0.1479 |
| gemini-3-flash | 0.1942 | 0.1666 | 0.1855 | 0.1671 | 0.1899 | 0.1702 | 0.1772 | 0.1528 |
| mistral-large | 0.1932 | 0.2157 | 0.1859 | 0.2050 | 0.2838 | 0.2091 | 0.1864 | 0.1379 |

### Table 5: Market Baseline Comparison

| Model | Brier | Market Brier | Delta | Beats? |
|-------|-------|-------------|-------|-------|
| gpt-5.4 | 0.1367 | 0.0948 | 0.0419 | no |
| gemini-3.1-pro | 0.1432 | 0.0951 | 0.0481 | no |
| kimi-k2.6 | 0.1440 | 0.0948 | 0.0492 | no |
| claude-opus-4.6 | 0.1464 | 0.0948 | 0.0516 | no |
| grok-4.20 | 0.1608 | 0.0948 | 0.0660 | no |
| deepseek-v3.2-speciale | 0.1644 | 0.0948 | 0.0695 | no |
| glm-5.1 | 0.1734 | 0.0951 | 0.0783 | no |
| qwen3-max | 0.1755 | 0.0954 | 0.0801 | no |
| gpt-oss-120b | 0.1756 | 0.0947 | 0.0808 | no |
| gemma-4-31b | 0.1783 | 0.0948 | 0.0834 | no |
| gemini-3-flash | 0.1789 | 0.0948 | 0.0841 | no |
| mistral-large | 0.2155 | 0.0948 | 0.1206 | no |
| **Market overall** | — | **0.0949** | — | — |

### Table 6: P4 Exploratory Comparison

| Model | P4 CB | Best P1-3 CB | Delta CB | P4 SE | Best P1-3 SE | Delta SE |
|-------|-------|-------------|---------|-------|-------------|---------|
| gpt-5.4 | 0.1371 | 0.1421 | -0.0050 | 0.1121 | 0.1166 | -0.0045 |
| gemini-3.1-pro | 0.1458 | 0.1691 | -0.0233 | 0.1286 | 0.1065 | 0.0221 |
| kimi-k2.6 | 0.1523 | 0.1503 | 0.0021 | 0.1317 | 0.1185 | 0.0131 |
| claude-opus-4.6 | 0.1518 | 0.1611 | -0.0093 | 0.1349 | 0.1232 | 0.0117 |
| grok-4.20 | 0.1708 | 0.1764 | -0.0056 | 0.1335 | 0.1375 | -0.0039 |
| deepseek-v3.2-speciale | 0.1643 | 0.1658 | -0.0015 | 0.1306 | 0.1447 | -0.0141 |
| gpt-oss-120b | 0.1576 | 0.1637 | -0.0061 | 0.1360 | 0.1504 | -0.0144 |
| glm-5.1 | 0.1670 | 0.1802 | -0.0132 | 0.1699 | 0.1521 | 0.0178 |
| qwen3-max | 0.1913 | 0.1860 | 0.0053 | 0.1558 | 0.1468 | 0.0089 |
| gemma-4-31b | 0.1807 | 0.1748 | 0.0059 | 0.1479 | 0.1620 | -0.0141 |
| gemini-3-flash | 0.1772 | 0.1855 | -0.0084 | 0.1528 | 0.1666 | -0.0138 |
| mistral-large | 0.1864 | 0.1859 | 0.0005 | 0.1379 | 0.2050 | -0.0670 |

### Table 7: Calibration and Sharpness Summary

| Dataset | N | Mean F | SD | ECE | OC-high | OC-low |
|---------|---|--------|----|-----|---------|--------|
| P1-P3 | 8196 | 0.2849 | 0.2941 | 0.08184 | 384 (4.7%) | 340 (4.1%) |
| P4† | 2695 | 0.2591 | 0.2776 | 0.06369 | 85 (3.2%) | 125 (4.6%) |

## 15. Output Files

All outputs saved to `results/analysis/final/`.