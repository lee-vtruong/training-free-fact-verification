# M1.4.1 — D4 Mechanism Audit & Confirmatory Experiment Preparation

**Milestone:** M1.4.1  
**Project:** Training-Free Fact Verification (`lee-vtruong/training-free-fact-verification`)  
**Core Research Principle:** Strictly Training-Free, Quality-First (Zero parameter updates, zero learned routers)  
**Evaluator Status:** Scikit-Learn Verified, 10,000-Resample Paired Bootstrap, Exact Two-Sided McNemar Test  

---

## Executive Summary

Milestone M1.4 evaluated seven controlled conditions (D0–D6) on 46 admitted exploratory claims from Fold B (`train:50..99`) using an Error Memory Bank constructed strictly from 50 source claims in Fold A (`train:0..49`). In that pilot, condition **D4** (Similarity-Retrieved Error Demonstrations, $k=2$) produced an apparent surge in Macro-F1 (**0.5562**, +0.0679 over D0, +0.1095 over D3) while matching D3 in Accuracy (**69.57%**, 32/46 correct).

This milestone (M1.4.1) audits the mechanism behind D4, tests whether its superiority is statistically genuine or sample variance, and prepares a frozen confirmatory evaluation on an untouched independent population.

### Key Findings

1. **No Statistically Significant Superiority on $N=46$:**
   - **D4 vs D0:** Net gain of +4 claims (7 improved, 3 regressed). Accuracy delta $+8.70\%$ ($95\%$ CI: $[-4.35\%, +21.74\%]$), Macro-F1 delta $+0.0679$ ($95\%$ CI: $[-0.0812, +0.2180]$). Exact two-sided McNemar $p = 0.3438$. Both 95% confidence intervals cross zero.
   - **D4 vs D3:** Net gain of +0 claims (2 improved, 2 regressed, 30 unchanged correct). Accuracy delta $+0.00\%$ ($95\%$ CI: $[-8.70\%, +8.70\%]$), Macro-F1 delta $+0.1095$ ($95\%$ CI: $[-0.0454, +0.2713]$). Exact two-sided McNemar $p = 1.0000$. The Macro-F1 confidence interval crosses zero.
   - **D4 vs D5:** Net gain of +3 claims (4 improved, 1 regressed). Accuracy delta $+6.52\%$ ($95\%$ CI: $[-2.17\%, +15.22\%]$), Macro-F1 delta $+0.0486$ ($95\%$ CI: $[-0.0957, +0.1871]$). Exact two-sided McNemar $p = 0.3750$.
   - **Conclusion:** **D4 is NOT statistically significantly superior to D0 or D3 on this exploratory sample.** The observed Macro-F1 surge is a hypothesis-generating signal driven entirely by a net shift on just two claims.

2. **The Exact Mechanism Behind D4's Macro-F1 Surge:**
   - D4 and D3 achieved identical overall accuracy ($32/46$, $69.57\%$).
   - The $+0.1095$ Macro-F1 gap between D4 and D3 is explained entirely by class distribution skew in Fold B:
     * `Refuted`: 23 claims (majority)
     * `Supported`: 15 claims (majority)
     * `Conflicting Evidence/Cherrypicking`: 7 claims (minority)
     * `Not Enough Evidence`: 1 claim (extreme minority)
   - D3 (random exemplars) gravitated toward majority classes, correctly predicting 13/15 `Supported` and 18/23 `Refuted`, but failing on `Not Enough Evidence` (0/1, F1 = 0.0) and recovering only 1/7 `Conflicting Evidence` (F1 = 0.2222).
   - D4 (similarity-retrieved exemplars) sacrificed 2 majority claims (`train:69`, `train:82` in `Supported`), but correctly classified `train:99` (`Not Enough Evidence`, achieving 100% recall for that class, F1 = 0.3333) and `train:67` (`Conflicting Evidence`, F1 = 0.3636).
   - Because Macro-F1 weights all 4 classes equally ($0.25$ each), recovering 1 claim in `Not Enough Evidence` produces a substantial Macro-F1 jump without increasing overall accuracy.

3. **Exemplar Representation Finding (D4 vs D5):**
   - D4 and D5 retrieved the **exact same exemplar IDs** for all 46 claims.
   - D4 provided compact demonstrations with gold reasoning. D5 added the model's base erroneous verdict and an explicit error-correction narrative.
   - Adding explicit mistake text in D5 caused regression on 3 claims (`train:62`, `train:76`, `train:91`), dropping Accuracy from $69.57\%$ to $63.04\%$. Explicit mistake representation appears to introduce reasoning noise.

4. **Confirmatory Experiment Readiness (Fold C):**
   - Independent population selected: **Fold C** (`train:100..149`, 50 claims).
   - Near-duplicate quarantine between Fold A source memory and Fold C yields **50 admitted, 0 quarantined** (100% admission).
   - All parameters frozen: Qwen3-8B snapshot, greedy decoding, non-thinking, $k=2$, 3-gram lexical Jaccard similarity.
   - Pre-retrieval audit: Frozen R4 evidence currently exists only for pilot claims `train:0..99`. Evaluating Fold C requires extracting candidate stores `train-100` to `train-149` from `train_0_999.zip` and running R1-R4 retrieval.

---

## 1. Audited Metrics & Independent Validation (D0–D6)

All metrics were recomputed directly from sealed `predictions.jsonl` files in `outputs/m14-pilot-20261025-014111/` and cross-validated against `scikit-learn` v1.9.0. Numerical agreement is exact ($< 10^{-9}$ difference).

| Condition | Description | Valid Verdicts | Accuracy | Macro-F1 | Precision | Recall | Mean Latency | Mean Tokens (Prompt / Gen) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **D0** | Frozen Base Protocol v1.0 (Zero-shot B1) | 46/46 (100%) | **60.87%** (28/46) | **0.4883** | 0.5028 | 0.6453 | 4.70s | 2483 / 193.3 |
| **D1** | Concise Generic Verification Rules | 46/46 (100%) | **65.22%** (30/46) | **0.4457** | 0.4623 | 0.4337 | 1.75s | 2530 / 67.9 |
| **D2** | Concise Source-Derived Error Rules | 46/46 (100%) | **67.39%** (31/46) | **0.4640** | 0.4925 | 0.4504 | 1.84s | 2526 / 71.9 |
| **D3** | Random Error Demonstrations ($k=2$) | 46/46 (100%) | **69.57%** (32/46) | **0.4467** | 0.5006 | 0.4480 | 2.00s | 2736 / 78.0 |
| **D4** | Similarity-Retrieved Error Demonstrations ($k=2$) | 46/46 (100%) | **69.57%** (32/46) | **0.5562** | 0.5612 | 0.7004 | 1.98s | 2736 / 77.2 |
| **D5** | Similarity-Retrieved Demonstrations with Corrections ($k=2$) | 46/46 (100%) | **63.04%** (29/46) | **0.5076** | 0.4946 | 0.6562 | 2.12s | 2774 / 82.8 |
| **D6** | Correct/Wrong Contrastive Demonstrations ($k=2$) | 46/46 (100%) | **60.87%** (28/46) | **0.4479** | 0.4629 | 0.4484 | 2.08s | 2772 / 81.0 |

### Audited Confusion Matrices

#### D0 (Zero-Shot Base Protocol v1.0)
| Gold \ Pred | Conflicting Evidence | Not Enough Evidence | Refuted | Supported | Unparseable | Support |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Conflicting Evidence** | **2** | 2 | 2 | 1 | 0 | 7 |
| **Not Enough Evidence** | 0 | **1** | 0 | 0 | 0 | 1 |
| **Refuted** | 2 | 4 | **16** | 1 | 0 | 23 |
| **Supported** | 4 | 0 | 2 | **9** | 0 | 15 |

#### D3 (Random Error Demonstrations, $k=2$)
| Gold \ Pred | Conflicting Evidence | Not Enough Evidence | Refuted | Supported | Unparseable | Support |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Conflicting Evidence** | **1** | 1 | 2 | 3 | 0 | 7 |
| **Not Enough Evidence** | 0 | **0** | 1 | 0 | 0 | 1 |
| **Refuted** | 1 | 1 | **18** | 3 | 0 | 23 |
| **Supported** | 0 | 1 | 1 | **13** | 0 | 15 |

#### D4 (Similarity-Retrieved Error Demonstrations, $k=2$)
| Gold \ Pred | Conflicting Evidence | Not Enough Evidence | Refuted | Supported | Unparseable | Support |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Conflicting Evidence** | **2** | 2 | 1 | 2 | 0 | 7 |
| **Not Enough Evidence** | 0 | **1** | 0 | 0 | 0 | 1 |
| **Refuted** | 1 | 1 | **18** | 3 | 0 | 23 |
| **Supported** | 1 | 1 | 2 | **11** | 0 | 15 |

#### D5 (Similarity Demonstrations with Corrections, $k=2$)
| Gold \ Pred | Conflicting Evidence | Not Enough Evidence | Refuted | Supported | Unparseable | Support |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Conflicting Evidence** | **2** | 2 | 1 | 2 | 0 | 7 |
| **Not Enough Evidence** | 0 | **1** | 0 | 0 | 0 | 1 |
| **Refuted** | 2 | 1 | **17** | 3 | 0 | 23 |
| **Supported** | 3 | 1 | 2 | **9** | 0 | 15 |

---

## 2. Paired Comparisons & Direct Hypothesis Testing

Statistical testing used exact two-sided McNemar tests on discordant pairs and 10,000-resample paired bootstrap confidence intervals (percentile method).

| Comparison | CC | CW | WC | WW | Net Gain | Accuracy Delta | Accuracy 95% Bootstrap CI | Macro-F1 Delta | Macro-F1 95% Bootstrap CI | Exact McNemar $p$ | Significant ($\alpha=0.05$)? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **D4 vs D0** | 25 | 3 | 7 | 11 | **+4** | **+8.70%** | [-4.35%, +21.74%] | **+0.0679** | [-0.0812, +0.2180] | 0.3438 | **NO** |
| **D4 vs D3** | 30 | 2 | 2 | 12 | **+0** | **+0.00%** | [-8.70%, +8.70%] | **+0.1095** | [-0.0454, +0.2713] | 1.0000 | **NO** |
| **D4 vs D5** | 28 | 1 | 4 | 13 | **+3** | **+6.52%** | [-2.17%, +15.22%] | **+0.0486** | [-0.0957, +0.1871] | 0.3750 | **NO** |
| **D2 vs D1** | 30 | 0 | 1 | 15 | **+1** | **+2.17%** | [+0.00%, +6.52%] | **+0.0183** | [+0.0000, +0.0623] | 1.0000 | **NO** |

*Note: Unparsed counts were 0 across all conditions (100% parse success).*

### Scientific Interpretation
- In all four pairwise comparisons, the exact two-sided McNemar $p$-value exceeds $0.05$ ($0.3438$, $1.0000$, $0.3750$, $1.0000$).
- All paired bootstrap 95% confidence intervals cross zero.
- Under strict empirical standards, **no condition can be claimed to be statistically superior to D0 or D3 on this sample size ($N=46$)**.
- The positive signal of D4 remains a valuable, testable hypothesis that requires confirmatory testing on an independent fold.

---

## 3. Exemplar Retrieval & Representation Audit

### Exemplar Selection: D4 vs D3
- **Retriever:** Lexical character 3-gram Jaccard similarity against 50 Fold A source claims.
- **D4 Similarity Distribution:** Mean Jaccard similarity $= 0.1020$ (min: $0.0421$, max: $0.1644$).
- **D3 Similarity Distribution:** Mean Jaccard similarity $= 0.0520$ (min: $0.0000$, max: $0.1131$).
- **Exemplar Overlap:** Only **4 out of 92 exemplar slots (4.3%)** overlapped between D3 and D4 across the 46 evaluation claims.
- **Finding:** D4 systematically selected exemplars sharing lexical topic n-grams with the target claim, whereas D3 selected unrelated exemplars (e.g., claims about economics receiving exemplars about biology or celebrities).

### Exemplar Representation: D4 vs D5
- **Exemplar Identity:** D4 and D5 used **100% identical exemplar IDs** for every single claim ($46/46$ exact matches).
- **Format Difference:**
  * **D4:** Compact representation containing Claim, Evidence, and Gold Verdict + Gold Reasoning.
  * **D5:** Same as D4, but prepended with the model's erroneous Base Protocol prediction and an explicit correction label: `[Correction: Base model incorrectly predicted <wrong_verdict> due to misinterpreting the evidence. Correct verdict: <gold_verdict>]`.
- **Outcome:** D5 regressed on 3 claims that D4 got right (`train:62`, `train:76`, `train:91`).
- **Mechanism:** Presenting the incorrect label in-context appears to act as a distractor for frozen Qwen3-8B, occasionally priming the model to adopt the error mode rather than avoid it.

### Token Overhead & Latency
| Condition | Mean Prompt Tokens | Mean Generated Tokens | Mean Latency | Token Budget Adherence |
|:---|:---:|:---:|:---:|:---:|
| **D0** | 2,483 | 193.3 | 4.70s | Within 3,200 token budget |
| **D3** | 2,736.5 | 78.0 | 2.00s | Within 3,200 token budget |
| **D4** | 2,735.8 | 77.2 | 1.98s | Within 3,200 token budget |
| **D5** | 2,774.3 | 82.8 | 2.12s | Within 3,200 token budget |

The demonstration formats (D3/D4) reduced generation latency by more than half relative to D0 ($1.98$s vs $4.70$s) by providing compact reasoning templates that prevented rambling.

---

## 4. Qualitative Case Studies (D4 vs D3)

Out of 46 claims:
- **Both Correct:** 30 claims
- **Both Wrong:** 12 claims
- **D4 Correct & D3 Wrong:** 2 claims (`train:67`, `train:99`)
- **D3 Correct & D4 Wrong:** 2 claims (`train:69`, `train:82`)

### Case 1: D4 Correct & D3 Wrong — `train:67`
- **Claim:** `"US billionaires became $400 billion richer during the COVID-19 pandemic, while millions lost their $600 a week in extra unemployment."`
- **Gold Label:** `Conflicting Evidence/Cherrypicking`
- **D3 (Random Exemplars `train:47`, `train:17`):**
  * Predicted: `Supported` (Wrong)
  * Reasoning: Model accepted broad political statements in the retrieved evidence as confirming the entire claim without scrutinizing contradictory numbers.
- **D4 (Similarity Exemplars `train:24`, `train:35`):**
  * Predicted: `Conflicting Evidence/Cherrypicking` (Correct)
  * Generation: *"The claim is partially supported by evidence showing that US billionaires gained significant wealth during the pandemic... However, the claim that 'US billionaires became $400 billion richer' is not directly confirmed by any single source and is contradicted by evidence that notes gains of $565 billion... The evidence thus presents conflicting figures and lacks direct confirmation."*
  * Impact: Retrieved exemplars demonstrated how to identify statistical divergence across passages, enabling the model to assign `Conflicting Evidence`.

### Case 2: D4 Correct & D3 Wrong — `train:99`
- **Claim:** `"Cold drinks are the main cause of heart attacks."`
- **Gold Label:** `Not Enough Evidence`
- **D3 (Random Exemplars `train:47`, `train:17`):**
  * Predicted: `Refuted` (Wrong)
  * Reasoning: Model over-extrapolated from passages debunking cold drinks as causes of health issues, declaring the claim flatly refuted rather than recognizing the lack of definitive clinical trials.
- **D4 (Similarity Exemplars `train:48`, `train:49`):**
  * Predicted: `Not Enough Evidence` (Correct)
  * Generation: *"Verdict: Not Enough Evidence. Reasoning: The evidence discusses potential links between certain types of drinks (diet soda, energy drinks, sugary drinks) and increased risks of heart attacks or vascular events, but none of the passages directly state that 'cold drinks are the main cause of heart attacks.' The evidence also includes studies with limitations and does not establish cold drinks as the primary cause."*
  * Impact: Correctly recovered the sole `Not Enough Evidence` claim in Fold B, single-handedly raising that class's recall from 0% to 100%.

### Case 3: D3 Correct & D4 Wrong — `train:69`
- **Claim:** `"USA is the only major industrial economy to have it unemployment rate triple."`
- **Gold Label:** `Supported`
- **D3 (Random Exemplars `train:47`, `train:17`):**
  * Predicted: `Supported` (Correct)
- **D4 (Similarity Exemplars `train:35`, `train:31`):**
  * Predicted: `Refuted` (Wrong)
  * Reasoning: D4 over-scrutinized passages regarding modest unemployment rises in South Korea and Australia, incorrectly deducing that the claim was refuted.

### Case 4: D3 Correct & D4 Wrong — `train:82`
- **Claim:** `"Auto workers across Michigan could have lost their jobs if not for Barack Obama and Joe Biden"`
- **Gold Label:** `Supported`
- **D3 (Random Exemplars `train:47`, `train:17`):**
  * Predicted: `Supported` (Correct)
- **D4 (Similarity Exemplars `train:8`, `train:39`):**
  * Predicted: `Conflicting Evidence/Cherrypicking` (Wrong)
  * Reasoning: D4 observed broader economic context from FDIC reports and hedge-fund contributions, inappropriately downgrading a supported factual claim to conflicting evidence.

---

## 5. Anti-Leakage Safeguard Audit

1. **Source Isolation:**
   - Error Memory Bank was generated strictly from Fold A (`train:0..49`).
   - Zero overlap: $0 / 46$ evaluation claims appeared in the exemplar bank.
   - Guard assert passed: no target claim was ever retrieved as an in-context demonstration for itself.

2. **Near-Duplicate Quarantine:**
   - 4 candidate claims in Fold B (`train:50`, `train:52`, `train:53`, `train:54`) exhibited Jaccard similarity $\ge 0.60$ or 3-gram containment $\ge 0.50$ with source claims in Fold A (`train:48`, `train:45`, `train:49`).
   - All 4 claims were quarantined prior to inference, leaving 46 admitted claims.
   - Admitted claim list is identical across D0–D6.

3. **Gold-Label Isolation:**
   - Inference prompts were constructed strictly from claim text, retrieved R4 evidence, and Fold A historical records.
   - At no point did the pipeline inspect or leak the gold label, gold questions, or official justifications of the evaluation claims.

---

## 6. Preparation for Confirmatory Experiment (Fold C)

### Evaluation Population
- **Candidate Set:** **Fold C** (`train:100..149`, 50 claims from the official train split).
- **Quarantine Audit:** Scanning Fold C against Fold A (`train:0..49`) using identical thresholds ($0.60$ Jaccard, $0.50$ containment):
  * Total candidate claims: **50**
  * Quarantined claims: **0**
  * Admitted claims: **50** ($100\%$ admission rate)
- **Disjointness:** Fold C has zero overlap with Fold A (source memory) and Fold B (pilot tuning).

### Frozen Protocol Parameters
- **Backbone:** Frozen Qwen3-8B snapshot `b968826d9c46dd6066d109eabc6255188de91218`.
- **Decoding:** Greedy (`temperature=0.0`), Non-Thinking (`enable_thinking=False`), `max_new_tokens=512`.
- **Ontology & Parsing:** Strict 4-class AVeriTeC ontology with deterministic verdict extraction.
- **Exemplar Selection:** 3-gram lexical Jaccard similarity against Fold A memory bank, $k=2$, deterministic tie-breaking.
- **Confirmatory Family:**
  1. `D0`: Base Protocol v1.0 (Zero-shot baseline)
  2. `D3`: Random Error Demonstrations ($k=2$, control for exemplar presence)
  3. `D4`: Similarity-Retrieved Error Demonstrations ($k=2$, test condition)
  4. `D5`: Similarity Demonstrations with Error Corrections ($k=2$, representation variant)
  5. `D2`: Concise Source-Derived Error Rules (optional baseline control)

### Evidence & Store Pre-Check
- **Audit Result:** Frozen R4 evidence file (`outputs/m11-store-r14-pilot100-20261023-162716/R4.jsonl`) contains passages **only for claims 0..99**.
- **Requirement:** Before executing the confirmatory fact verification pipeline on Fold C (`train:100..149`), candidate stores `train-100` through `train-149` must be acquired from the remote knowledge store (`train_0_999.zip`) and indexed via the frozen R1–R4 pipeline to produce `outputs/m141-confirmatory-r4/R4.jsonl`.
- **Pipeline Implementation:** `src/tffv/verification/m141_confirmatory.py` and `scripts/run_m141_confirmatory.sh` have been implemented, tested, and validated.

---

## 7. Next Steps
1. User reviews the mechanism audit and statistical conclusion.
2. User executes the bounded Fold C retrieval or provides the pre-computed R4 evidence.
3. User runs `scripts/run_m141_confirmatory.sh` on `hvtham-server` to obtain the confirmatory result.
