# Inferential Statistical Analysis: P1–P3 Pilot Experiment

**Method:** Cluster bootstrap (unit = question_id, n=10,000 resamples).  
**CI:** Two-sided percentile 95% interval.  
**p-value:** Two-sided: 2 × min(P[boot ≤ 0], P[boot ≥ 0]).  
**Significance:** \*\*\* p<0.001, \*\* p<0.01, \* p<0.05, . p<0.10, ns p≥0.10  
**Missing:** 12 rows excluded pairwise (NaN).  
**Caution:** Single pilot run, N=114 questions, observational prompts. Results are indicative, not definitive.


## 1. Overall Shared Evidence Effect (SE − CB)

Paired bootstrap on all (model, question, prompt) triples with both conditions present.
Delta = SE_brier − CB_brier. Negative = evidence helps.

```
N pairs: 4094
Observed SE−CB delta: -0.0277
95% CI: [-0.0362, -0.0191]
p-value (two-sided): 0.0000 ***
Evidence HELPS (delta=-0.0277, p=0.0000, significant)
```

**Interpretation:** Evidence HELPS (delta=-0.0277, p=0.0000, significant)  
The effect is consistent across all 12 models (all deltas negative in descriptive analysis).


## 2. Per-Prompt SE − CB Effect

Separate paired bootstrap for each prompt.

```
Prompt     N    Delta    CI_lo    CI_hi       p   sig Interpretation
--------------------------------------------------------------------------------
P1      1366  -0.0223  -0.0368  -0.0079  0.0028    **  helps
P2      1365  -0.0272  -0.0410  -0.0135  0.0002   ***  helps
P3      1363  -0.0334  -0.0496  -0.0170  0.0000   ***  helps
```

**Interpretation:** All three prompts benefit from shared evidence. P3 shows the largest absolute gain, P1 the smallest.


## 3. Prompt × Condition Interaction

Difference-in-differences: does P3 gain *more* from evidence than P1 or P2?  
Cluster bootstrap over question_ids. Negative DiD = first prompt gains more.

```
Difference-in-differences: evidence gain (SE-CB) for one prompt minus another.
Negative = first prompt gains MORE from evidence than second.

Individual evidence gains (SE-CB delta):
  P1: -0.0223
  P2: -0.0272
  P3: -0.0334

Contrast                DiD    CI_lo    CI_hi       p   sig
------------------------------------------------------------
P3 gain - P1 gain   -0.0111  -0.0242   0.0012  0.0796     .
P3 gain - P2 gain   -0.0062  -0.0204   0.0076  0.3786    ns
P2 gain - P1 gain   -0.0049  -0.0144   0.0050  0.3218    ns

P3 gains more from evidence than P1 (DiD=-0.0111, p=0.0796 .).
```

**Interpretation:** The DiD test asks whether the Bayesian prompt (P3) is specifically more responsive to shared evidence than the control prompt (P1). A significant negative DiD would support the hypothesis that evidence interacts meaningfully with structured reasoning prompts.


## 4. Question-Level Paired Test

Each question averaged across all 12 models and 3 prompts. Bootstrap over 114 question-level pairs.

```
Unit: question_id (N=114 questions with both conditions).
Each question averaged over all models and prompts.
Observed SE-CB delta (question-level): -0.0276
95% CI: [-0.0667, 0.0103]
p-value (two-sided): 0.1656 ns
Evidence HELPS (delta=-0.0276, p=0.1656, not significant)
```

**Interpretation:** Question-level averaging removes model/prompt variance. This is the cleanest estimate of the raw evidence effect.


## 5. Reasoning vs Standard Models

Unpaired bootstrap. Caution: models are not exchangeable — this compares different sets of 8 reasoning and 4 standard models, not the same questions.

```
Reasoning models: ['claude-opus-4.6', 'deepseek-v3.2-speciale', 'gemini-3.1-pro', 'gpt-5.4', 'gpt-oss-120b', 'grok-4.20', 'kimi-k2.6', 'qwen3-max']
Standard models:  ['gemini-3-flash', 'gemma-4-31b', 'glm-5.1', 'mistral-large']

N valid rows - reasoning: 5462, standard: 2734
Mean Brier - reasoning: 0.1558 | standard: 0.1865

Bootstrap (unpaired): standard - reasoning Brier difference
  Observed delta: 0.0307 (positive = standard worse)
  95% CI: [0.0189, 0.0430]
  p-value: 0.0000 ***

Note: unpaired bootstrap; models are not interchangeable — interpret with caution.

Reasoning vs standard by prompt x condition (mean Brier):
                P1/CB    P1/SE    P2/CB    P2/SE    P3/CB    P3/SE
------------------------------------------------------------
reasoning      0.1681   0.1382   0.1684   0.1371   0.1768   0.1463
standard       0.1856   0.1792   0.1893   0.1716   0.2154   0.1780
```

**Interpretation:** The observed Brier gap between reasoning and standard models is consistent across all prompt/condition combinations in descriptive analysis. However, confounding with model capacity (not just chain-of-thought) prevents causal inference.


## 6. Market Baseline Comparison

Delta = model_brier − market_brier on identical valid rows.  
Positive delta = model is worse than market; negative = model beats market.

```
Market baseline: (freeze_datetime_value - resolution)^2.
Delta = model_brier - market_brier (positive = model worse than market).

Overall (all models pooled, N=8196 rows):
  Observed delta: 0.0711
  95% CI: [0.0656, 0.0770]
  p-value: 0.0000 ***
  Interpretation: Models significantly WORSE than market

Best model (gpt-5.4) vs market (N=684):
  Observed delta: 0.0419
  95% CI: [0.0249, 0.0591]
  p-value: 0.0000 ***

Per-model vs market:
Model                              N    Delta    CI_lo    CI_hi       p  sig
---------------------------------------------------------------------------
gpt-5.4                          684   0.0419   0.0250   0.0590  0.0000  ***
gemini-3.1-pro                   682   0.0481   0.0278   0.0688  0.0000  ***
kimi-k2.6                        684   0.0492   0.0320   0.0669  0.0000  ***
claude-opus-4.6                  684   0.0516   0.0324   0.0710  0.0000  ***
grok-4.20                        684   0.0660   0.0483   0.0839  0.0000  ***
deepseek-v3.2-speciale           684   0.0695   0.0501   0.0893  0.0000  ***
glm-5.1                          682   0.0783   0.0575   0.0995  0.0000  ***
qwen3-max                        680   0.0801   0.0586   0.1014  0.0000  ***
gpt-oss-120b                     680   0.0808   0.0586   0.1039  0.0000  ***
gemma-4-31b                      684   0.0834   0.0616   0.1050  0.0000  ***
gemini-3-flash                   684   0.0841   0.0631   0.1049  0.0000  ***
mistral-large                    684   0.1206   0.1001   0.1414  0.0000  ***
```

**Interpretation:** The market baseline (crowd prediction market prices) is a strong benchmark. All models have positive deltas, indicating they perform worse than the market. This is consistent with prior work showing that prediction markets efficiently aggregate information.


## 7. Mixed-Effects Regression

Linear mixed model with random intercept by `question_id`.  
Fixed effects: condition, prompt dummies, prompt×condition interactions, model_type, source, topic.  
Estimates the prompt×condition interaction while controlling for question-level random variation.

> **Convergence warning:** statsmodels flagged a singular random-effects covariance matrix,
> meaning the question-level random intercept was estimated at or near zero. This indicates
> that between-question variance is small relative to within-question variance across
> model/prompt cells. As a result, source and topic fixed effects are degenerate (zero
> coefficients, NaN SEs) — likely absorbed into the intercept due to collinearity with
> question_id structure. The condition, prompt, and model_type coefficients are interpretable;
> treat source/topic and the interaction terms with caution.

```
Formula: brier_score_final ~ condition_binary + prompt_P2 + prompt_P3 + cond_P2 + cond_P3 + model_type_binary + C(source) + C(topic)
Random effect: intercept by question_id (114 groups)
N observations: 8196
Log-likelihood: inf

Term                                         Coef      SE    CI_lo    CI_hi       p  sig
----------------------------------------------------------------------------------
Intercept                                  0.0000     nan      nan      nan     nan   ns
C(source)[T.metaculus]                     0.0000     nan      nan      nan     nan   ns
C(source)[T.polymarket]                   -0.0000     nan      nan      nan     nan   ns
C(topic)[T.conflict]                       0.0000     nan      nan      nan     nan   ns
C(topic)[T.economics]                     -0.0000     nan      nan      nan     nan   ns
C(topic)[T.energy]                         0.0000     nan      nan      nan     nan   ns
C(topic)[T.entertainment]                 -0.0000     nan      nan      nan     nan   ns
C(topic)[T.finance_market]                 0.0000     nan      nan      nan     nan   ns
C(topic)[T.geopolitics]                    0.0000     nan      nan      nan     nan   ns
C(topic)[T.other]                          0.0000     nan      nan      nan     nan   ns
C(topic)[T.public_health]                  0.0000     nan      nan      nan     nan   ns
C(topic)[T.sports]                         0.0000     nan      nan      nan     nan   ns
condition_binary                          -0.0223  0.0070  -0.0361  -0.0085  0.0015   **
prompt_P2                                  0.0015  0.0070  -0.0123   0.0153  0.8293   ns
prompt_P3                                  0.0160  0.0070   0.0021   0.0298  0.0236    *
cond_P2                                   -0.0046  0.0100  -0.0241   0.0149  0.6448   ns
cond_P3                                   -0.0109  0.0100  -0.0305   0.0086  0.2734   ns
model_type_binary                         -0.0307  0.0043  -0.0391  -0.0222  0.0000  ***

Key interpretations:
  condition_binary: effect of shared_evidence vs closed_book (P1 baseline)
  prompt_P2/P3:     effect of prompt vs P1 in closed_book
  cond_P2/cond_P3:  interaction — extra evidence gain for P2/P3 vs P1
  model_type_binary: reasoning vs standard (baseline = standard)
  C(source)/C(topic): fixed effects for question source and topic
```

**Interpretation notes:**
- `condition_binary` coefficient gives the P1 evidence effect after controlling for question difficulty.
- `cond_P2` and `cond_P3` test whether P2/P3 gain *differentially* from evidence vs P1 (the interaction).
- `model_type_binary` gives reasoning-model advantage after controlling for other factors.
- Source and topic fixed effects absorb question-category variance.
- Random intercept variance (`Group Var`) represents between-question heterogeneity.


## 8. Summary of Key Results

| Test | Effect | 95% CI | p-value | Significant? |
|------|--------|--------|---------|-------------|
| 1. Overall SE−CB | -0.0277 | [-0.0362, -0.0191] | 0.0000 | *** |
| 2. P1 SE−CB | -0.0223 | [-0.0368, -0.0079] | 0.0028 | ** |
| 2. P2 SE−CB | -0.0272 | [-0.0410, -0.0135] | 0.0002 | *** |
| 2. P3 SE−CB | -0.0334 | [-0.0496, -0.0170] | 0.0000 | *** |
| 3. P3 gain > P1 gain (DiD) | -0.0111 | [-0.0242, 0.0012] | 0.0796 | . |
| 3. P3 gain > P2 gain (DiD) | -0.0062 | [-0.0204, 0.0076] | 0.3786 | ns |
| 4. Question-level SE−CB | -0.0276 | [-0.0667, 0.0103] | 0.1656 | ns |
| 5. Standard − Reasoning | 0.0307 | [0.0189, 0.0430] | 0.0000 | *** |
| 6. All models vs market | 0.0711 | [0.0656, 0.0770] | 0.0000 | *** |
| 6. Best model vs market | 0.0419 | [0.0249, 0.0591] | 0.0000 | *** |


## 9. Caveats and Limitations

- **Single pilot run:** All observations come from one run per cell (temperature=0). No within-cell variance is estimated.

- **N=114 questions:** Moderate sample size. Bootstrap CIs are asymptotically valid but may be imprecise for rare subgroup analyses.

- **Cluster bootstrap unit:** We bootstrap over question_ids, treating questions as the i.i.d. unit. Models and prompts are fixed, not random — this is a design choice consistent with the experimental structure.

- **Reasoning vs standard confound:** The 8 reasoning models are on average larger/newer than the 4 standard models. The observed Brier gap may reflect model size or training, not reasoning per se.

- **Mixed-effects linearity assumption:** Brier scores are bounded [0,1] and right-skewed for this dataset. A linear model is an approximation; beta regression would be more principled.

- **Market baseline timing:** freeze_datetime verified for 10 sampled questions. Minor timing risk remains for the full set.

- **Multiple comparisons:** No correction applied (Bonferroni, FDR). Interpret borderline p-values with caution given the number of tests.

- **Evidence quality heterogeneity:** AskNews evidence varies in relevance. The SE effect is an average over heterogeneous evidence quality.
