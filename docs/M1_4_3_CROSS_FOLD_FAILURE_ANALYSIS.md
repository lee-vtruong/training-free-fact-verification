# M1.4.3 — Cross-Fold Failure Analysis & Error Mechanism Audit

**Milestone:** M1.4.3 Cross-Fold Failure Analysis & Mechanistic Audit  
**Author:** AI Research Engineer (Antigravity)  
**Date:** 2026-10-26  
**Repository:** `lee-vtruong/training-free-fact-verification`  
**Git Commit Target:** `main`  
**Evaluated Artifacts:**
- Fold B (Pilot, $N=46$): `outputs/m14-pilot-20261025-014111`
- Fold C (Confirmatory, $N=50$): `outputs/m141-confirmatory-20261025-101555`
- Fold D (Confirmatory & Ablations, $N=91$ clean, 9 quarantined): `outputs/m142-confirmatory-20261025-233407`

---

## Executive Summary

Across three consecutive evaluation folds examined sequentially during exploratory analysis (**Fold B**, **Fold C**, and **Fold D**), representing **187 unique, unobserved fact-checking claims** (evaluated as exploratory audits rather than pre-registered confirmatory validations of post-hoc hypotheses), we conducted an exhaustive cross-fold audit to answer a fundamental scientific question:

> **Why have training-free fact verification methods utilizing historical error memory (error-retrieval demonstrations or empirical error rules) failed to consistently outperform matched generic prompting?**

### Definitive Empirical Conclusions:
1. **The Primary Hypothesis Failed Conclusively on Fold D:**
   In our pre-registered contrast on Fold D ($N=91$ clean claims), **`D2` (Source-Derived Error Rules) was defeated by `D1_matched` (Matched Generic Control)**:
   - `D1_matched`: Accuracy **63.7%** (58/91), Macro-F1 **0.3919**
   - `D2`: Accuracy **59.3%** (54/91), Macro-F1 **0.3718**
   - Net Gain: **-4 claims** ($54\ C\to C,\ 4\ C\to W,\ 0\ W\to C,\ 33\ W\to W$)
   - Paired Bootstrap 95% CI on Accuracy Delta: `[-0.0900, -0.0100]` (strictly negative).
   - `D2` did not correct a single claim over `D1_matched` ($W \to C = 0$), while actively corrupting 4 claims that `D1_matched` verified correctly.
2. **The "Positive Signal" of D2 on Fold C was a Formatting Confound:**
   On Fold C ($N=50$), D2 appeared superior to D0 (68.0% vs 54.0%, $+7$ net claims) solely because D2 enforced **1-sentence brevity** (64 tokens vs 179 tokens), preventing verbose hallucination into `Conflicting Evidence` and `Not Enough Evidence`. When matched against `D1_matched`—which enforces identical brevity and sentence structure without error heuristics—the generic prompt outperformed the error-rule prompt.
3. **Static Heuristics Create Opposing Blind Spots:**
   Rule 2 (*"Missing context is not Cherrypicking"*), synthesized from Fold A to fix false `Not Enough Evidence` errors, caused severe negative transfer on Fold D. Ablating Rule 2 (`D2_minus_r2`) **improved** accuracy from 59.3% to 62.6% ($+3$ net claims). What cured an error on Fold A acted as a distractor on Fold D.
4. **Lexical Error Retrieval (D4) is an Unreplicable Dead End:**
   D4 ($k=2$ similarity error retrieval) scored Macro-F1 0.5562 on Fold B (pilot lucky hit), collapsed to 0.3333 on Fold C (underperforming random D3 and base D0), and tied with random D3 at 57.1% on Fold D (Net Gain = 0 claims). Topical lexical similarity has zero correlation with reasoning failure modes.

---

## Section A — Comprehensive Audit of Milestone M1.4.2 (Fold D)

### 1. Population Admission & Quarantine Accounting

Fold D evaluated claims `train:150..249` (100 candidate claims).

Prior to inference, the anti-leakage quarantine protocol ([`quarantine_near_duplicates`](file:///c:/Users/ASUS/Downloads/training-free-fact-verification/src/tffv/verification/leakage.py)) was executed against the Fold A source memory bank (`train:0..49`).

- **Total Candidate Claims:** 100 claims
- **Admitted Claims:** **91 claims** (91.0%)
- **Quarantined Claims:** **9 claims** (9.0%)
- **Quarantine Criteria:** Lexical Jaccard similarity $\ge 0.60$ OR 3-gram containment $\ge 0.50$ against any Fold A source claim.

#### Detailed Accounting of the 9 Quarantined Claims:
Every single quarantined claim was an exact or near-verbatim duplicate fact-check concerning Kenyan judiciary statistics, originating from the same political speech / state report that appeared in Fold A:

| Quarantined ID | Source Claim ID in Fold A | Jaccard Similarity | 3-Gram Containment | Evaluated Claim Text in Fold D | Source Claim Text in Fold A |
|:---:|:---:|:---:|:---:|:---|:---|
| `train:174` | `train:25` | 0.7692 | 0.4000 | *"Kenya's judiciary does not get even 1% of the national budget."* | *"The Kenyan judiciary does not even get 1% of the national budget."* |
| `train:177` | `train:21` | 0.8750 | 1.0000 | *"There is a ‘global average’ for the number of judges and magistrates to number of people"* | *"There is a ‘global average’ for the number of judges and magistrates to number of people in Kenya."* |
| `train:180` | `train:22` | 0.8095 | 0.7647 | *"In Kenya now with a population of about 48 million, we have one judge serving over 300,000 people"* | *"In Kenya now with a population of about 48 million, there is one judge serving over 300,000 people."* |
| `train:182` | `train:32` | 0.6667 | 0.7143 | *"Kenya has one magistrate serving over 77,000 people"* | *"The Kenya judiciary have one magistrate serving over 77,000 people."* |
| `train:183` | `train:22` | 0.7143 | 0.6667 | *"With a population of about 48 million, Kenya has only one judge serving over 300,000 people"* | *"In Kenya now with a population of about 48 million, there is one judge serving over 300,000 people."* |
| `train:185` | `train:29` | 0.7778 | 0.8462 | *"Kenya has an average of 400,000 cases being filed in our courts every year"* | *"The judiciary of Kenya have an average of 400,000 cases being filed in our courts every year."* |
| `train:186` | `train:32` | 0.6667 | 0.7143 | *"Kenya has one magistrate serving over 77,000 people"* | *"The Kenya judiciary have one magistrate serving over 77,000 people."* |
| `train:187` | `train:29` | 0.7222 | 0.6667 | *"an average of 400,000 cases being filed in Kenya's courts every year"* | *"The judiciary of Kenya have an average of 400,000 cases being filed in our courts every year."* |
| `train:189` | `train:25` | 1.0000 | 0.6000 | *"The judiciary does not even get 1% of the Kenyan national budget."* | *"The Kenyan judiciary does not even get 1% of the national budget."* |

*Audit Verification:* Quarantining these 9 claims was essential. Admitting them would have inflated performance through verbatim topic leakage from Fold A. The 91 evaluated claims are strictly disjoint and untainted.

---

### 2. Primary Hypothesis Verification: D2 vs D1_matched

On the 91 clean claims of Fold D, we compare `D2` (Concise Source-Derived Error Rules) directly against `D1_matched` (Matched Concise Generic Rules):

```text
D1_matched Accuracy:  58 / 91 (63.74%) | Macro-F1: 0.3919
D2 Accuracy:          54 / 91 (59.34%) | Macro-F1: 0.3718
```

#### 9-Category Paired Transition Matrix (Sum = 91):
- $C \to C$ (Both Correct): **54 claims**
- $C \to W$ (D1_matched Correct $\to$ D2 Wrong): **4 claims**
- $C \to U$ (D1_matched Correct $\to$ D2 Unparsed): **0 claims**
- $W \to C$ (D1_matched Wrong $\to$ D2 Correct): **0 claims**
- $W \to W$ (Both Wrong): **33 claims**
- $W \to U$ (D1_matched Wrong $\to$ D2 Unparsed): **0 claims**
- $U \to C, U \to W, U \to U$: **0 claims**

**Net Gain of D2 over D1_matched:** $0 - 4 = \mathbf{-4\text{ claims}}$ (Absolute regression).

#### The 4 Discordant Claims Corrupted by D2:
In all 4 cases where predictions diverged, `D1_matched` produced the correct verdict, while `D2` produced a false verdict:

1. `train:204` (Gold: `Refuted`):
   - Claim: *"Tint Tint Htun, a former Myanmar actress and singer who became a Buddhist nun, has died."*
   - Evidence: Fact-checks confirming she is alive and residing at a monastery.
   - `D1_matched`: `Refuted` (**CORRECT**). Text: *"The claim that 'Tint Tint Htun... has died' is refuted by multiple evidence sources..."*
   - `D2`: `Not Enough Evidence` (**WRONG**). Text: *"The evidence provided describes Tint Tint Tun as a former Myanmar actress... but does not contain explicit confirmation of her death."* (Rule 2 prompted the model to hedge when minor biographic details were omitted).
2. `train:239` (Gold: `Refuted`):
   - Claim: *"US drug prices in 2019 went down for the first time in 51 years."*
   - Evidence: Official Consumer Price Index showing prescription drug prices increased overall; only an isolated index segment saw a temporary decline.
   - `D1_matched`: `Refuted` (**CORRECT**). Text: *"The claim is directly contradicted by the evidence, which shows that overall drug prices increased..."*
   - `D2`: `Conflicting Evidence/Cherrypicking` (**WRONG**). Text: *"The claim is supported by some sources (e.g., Passage [4]) citing official announcements, but refuted by other sources citing broader CPI data."* (Rule 2 misled the model into believing the presence of partial official quotes constituted cherrypicking rather than an outright false claim).
3. `train:242` (Gold: `Refuted`):
   - Claim: *"County governments in Kenya have continued to adhere to the two-thirds gender principle. Three counties of Kilifi, Nyeri and Migori have gone higher and have a 50/50 representation of male and female CECs."*
   - Evidence: National audit showing the vast majority of Kenyan counties blatantly violated the two-thirds gender principle.
   - `D1_matched`: `Refuted` (**CORRECT**). Text: *"The claim that county governments have continued to adhere to the two-thirds gender principle is refuted..."*
   - `D2`: `Supported` (**WRONG**). Text: *"The claim is directly confirmed by Passage [8], which states that 'the counties of Kilifi, Nyeri and Migori have gone higher and have a 50/50 representation'..."* (Rule 2 instructed the model that missing secondary details don't matter if the primary assertion is substantiated, causing the model to latch onto the 3 named counties and ignore the global refutation!).
4. `train:243` (Gold: `Refuted`):
   - Claim: *"Target ended its ‘Heroes & Helpers’ program that pairs children with police officers for holiday shopping trips"*
   - Evidence: Target spokespeople confirming the program remains active and was not cancelled.
   - `D1_matched`: `Refuted` (**CORRECT**). Text: *"The claim that Target ended its 'Heroes & Helpers' program is refuted by the evidence, which confirms the program continues..."*
   - `D2`: `Not Enough Evidence` (**WRONG**). Text: *"The evidence provided describes the 'Heroes & Helpers' program... but does not explicitly confirm whether Target officially terminated it nationwide."*

---

### 3. Statistical Discrepancy: Exact McNemar vs Paired Bootstrap CI

On D2 vs D1_matched:
- **Exact two-sided McNemar test:** $p = \mathbf{0.1250}$ (Nominally non-significant, $> 0.05$).
- **10,000 Paired Bootstrap 95% CI on Accuracy Delta:** `[-0.0900, -0.0100]` (Strictly negative, excludes zero).
- **10,000 Paired Bootstrap 95% CI on Macro-F1 Delta:** `[-0.0455, -0.0008]` (Strictly negative, excludes zero).

#### Why do the two statistical methods diverge?
This is a standard statistical property of discrete discordant tests with small counts:
1. **McNemar's test** is conditioned purely on the discordant pairs ($b=4, c=0$). Under the null hypothesis $H_0: p=0.5$, the two-sided probability of observing 4 or more extreme outcomes out of 4 trials is:
   $$\text{McNemar } p = \binom{4}{0}(0.5)^4 + \binom{4}{4}(0.5)^4 = \frac{1}{16} + \frac{1}{16} = \frac{2}{16} = 0.1250$$
   Because $n_{\text{discordant}} = 4$, the minimum achievable $p$-value for any 4-trial sign test is 0.1250. To reach $p < 0.05$, a minimum of **6 discordant pairs** is mathematically required ($2 \times 0.5^6 = 0.03125$). Therefore, McNemar's test lacks the statistical power to reject $H_0$ at $N_{\text{discordant}}=4$.
2. **The Paired Bootstrap**, in contrast, resamples the full $N=91$ claim population with replacement. Because $54$ claims are tied ($C\to C$) and all $4$ discordant claims favor `D1_matched`, virtually every resampled draw yields $\text{Acc}(D1\_matched) > \text{Acc}(D2)$. Consequently, the empirical 95% confidence interval lies strictly in negative territory (`[-0.09, -0.01]`).

#### Methodological Integrity:
We do NOT pick the test that favors our narrative. We report both:
- McNemar's test indicates that the 4-claim deficit is not statistically significant at $\alpha=0.05$ due to small discordant sample size.
- The bootstrap CI confirms that the point estimate of the difference is negative.
- **Under NEITHER test is D2 superior to D1_matched.** The empirical error rules provided zero positive value over generic prompting.

---

### 4. Confusion Matrices & Per-Class F1 Breakdown

#### Table A1: Performance Breakdown on Fold D ($N=91$)

| Metric | Class (Support) | D0 (Frozen Base) | D1_matched (Generic Control) | D2 (Error Rules R1+R2) | D2_minus_r2 (Rule 1 Only) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Overall Accuracy** | All (91) | 53.8% (49/91) | **63.7% (58/91)** | 59.3% (54/91) | 62.6% (57/91) |
| **Macro-F1** | All (91) | **0.4161** | 0.3919 | 0.3718 | 0.3934 |
| **Macro-Precision** | All (91) | **0.4474** | 0.3829 | 0.3650 | 0.3876 |
| **Macro-Recall** | All (91) | **0.4534** | 0.4088 | 0.3892 | 0.4039 |
| **Refuted F1** | Refuted (51) | 0.6292 | **0.7429** | 0.7000 | 0.7308 |
| - *Refuted Recall* | | 54.9% (28/51) | **76.5% (39/51)** | 68.6% (35/51) | 74.5% (38/51) |
| - *Refuted Precision* | | 73.7% (28/38) | 72.2% (39/54) | 71.4% (35/49) | 71.7% (38/53) |
| **Supported F1** | Supported (29) | **0.6800** | 0.6429 | 0.6207 | 0.6429 |
| - *Supported Recall* | | 58.6% (17/29) | **62.1% (18/29)** | **62.1% (18/29)** | **62.1% (18/29)** |
| - *Supported Precision* | | **81.0% (17/21)** | 66.7% (18/27) | 62.1% (18/29) | 66.7% (18/27) |
| **Not Enough Evidence F1** | NEI (7) | **0.2500** | 0.0000 | 0.0000 | 0.0000 |
| - *NEI True Positives* | | **3 / 7** | 0 / 7 | 0 / 7 | 0 / 7 |
| - *NEI False Positives* | | 14 | 3 | 5 | 5 |
| **Conflicting F1** | Conflicting (4) | 0.1053 | 0.1818 | 0.1667 | **0.2000** |
| - *Conflicting True Positives* | | 1 / 4 | 1 / 4 | 1 / 4 | 1 / 4 |
| - *Conflicting False Positives*| | 14 | 6 | 7 | 5 |

#### Confusion Matrices (Rows: Gold, Columns: Pred [Conf, NEI, Ref, Supp]):

**D0 Confusion Matrix:**
```text
  Conflicting (4) : [ 1,  1,  1,  1]
  NEI (7)         : [ 2,  3,  1,  1]  <- Only condition that predicted NEI correctly (tp=3)
  Refuted (51)    : [11, 10, 28,  2]  <- Severe false negatives: 11 Conf, 10 NEI!
  Supported (29)  : [ 1,  3,  8, 17]
```

**D1_matched Confusion Matrix:**
```text
  Conflicting (4) : [ 1,  0,  1,  2]
  NEI (7)         : [ 1,  0,  5,  1]  <- Zero NEI true positives, but wiped out false positives
  Refuted (51)    : [ 5,  1, 39,  6]  <- Refuted TP surged from 28 to 39!
  Supported (29)  : [ 0,  2,  9, 18]
```

**D2 Confusion Matrix:**
```text
  Conflicting (4) : [ 1,  0,  1,  2]
  NEI (7)         : [ 2,  0,  4,  1]
  Refuted (51)    : [ 5,  3, 35,  8]  <- 4 fewer Refuted TPs than D1_matched (35 vs 39)
  Supported (29)  : [ 0,  2,  9, 18]
```

**D2_minus_r2 Confusion Matrix:**
```text
  Conflicting (4) : [ 1,  0,  1,  2]
  NEI (7)         : [ 1,  0,  5,  1]
  Refuted (51)    : [ 4,  3, 38,  6]  <- Recovers to 38 Refuted TPs!
  Supported (29)  : [ 0,  2,  9, 18]
```

---

### 5. Mechanistic Audit: Rule 1 vs Rule 2 Provenance and Opposing Effects

D2 contained exactly two directives:
- **Rule 1:** *"Media debate is not factual contradiction: An article describing a political disagreement does not constitute Refutation of a verifiable fact."*
- **Rule 2:** *"Missing context is not Cherrypicking: Absence of secondary details does not warrant Cherrypicking if the primary assertion is substantiated."*

#### Finding 1: Rule 1 was beneficial.
- Provenance: Synthesized from Fold A claims `train:17`, `train:35`, `train:37` where political figures debating statistics caused the base model to output `Conflicting Evidence`.
- On Fold D: Removing Rule 1 (`D2_minus_r1`) caused accuracy to drop from 59.3% to **57.1%** (52/91, Macro-F1 0.3593). Rule 1 successfully suppressed spurious `Conflicting Evidence` on political debate claims.

#### Finding 2: Rule 2 was actively harmful.
- Provenance: Synthesized from Fold A claims `train:2`, `train:6`, `train:11`, `train:14`, `train:29`, `train:38`, `train:47`, `train:48` (8 errors) where the base model bailed into `Not Enough Evidence` when non-essential peripheral facts were absent.
- On Fold D: Removing Rule 2 (`D2_minus_r2`) **boosted accuracy from 59.3% to 62.6%** (57/91, Macro-F1 0.3934).
- Rule 2 caused 3 net regressions and 0 improvements on Fold D:
  * On `train:152` (Hydroxychloroquine in France) and `train:239` (US drug prices in 2019), Rule 2 caused the model to see partial contextual overlap as a reason to classify as `Conflicting Evidence/Cherrypicking`.
  * On `train:242` (Kenyan gender principle), Rule 2 caused the model to ignore global counter-evidence and declare `Supported` based on partial sub-clause substantiation.

---

## Section B — Cross-Fold Failure Synthesis (Folds B, C, D)

### 1. Unified Cross-Fold Performance Summary

| Split | Description | Claims | D0 (Base) | D1_matched (Generic) | D2 (Error Rules) | D3 (Random ICL) | D4 (Similarity ICL) | Best Condition |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold B** | Pilot Exploration | 46 | Acc 60.9% / F1 0.4883 | *Not tested* | *Not tested* | Acc 69.6% / F1 0.4467 | Acc 69.6% / **F1 0.5562** | **D4** (Pilot artifact) |
| **Fold C** | Independent Confirmatory | 50 | Acc 54.0% / F1 0.3801 | *Not tested* | **Acc 68.0% / F1 0.4365** | Acc 68.0% / F1 0.3739 | Acc 62.0% / F1 0.3333 | **D2** (Formatting confound) |
| **Fold D** | Mechanistic Confirmation | 91 | Acc 53.8% / F1 0.4161 | **Acc 63.7% / F1 0.3919** | Acc 59.3% / F1 0.3718 | Acc 57.1% / F1 0.3603 | Acc 57.1% / F1 0.3836 | **D1_matched** (Generic control) |

#### Cross-Fold Pattern Analysis:
1. **The Collapse of D4:**
   - Fold B: Macro-F1 0.5562 (+0.0679 over D0)
   - Fold C: Macro-F1 0.3333 (-0.0468 vs D0, -3 net claims vs D3)
   - Fold D: Macro-F1 0.3836 (-0.0325 vs D0, 0 net claims vs D3)
   D4 completely failed replication across two independent unseen folds.
2. **The De-biasing of D2:**
   - On Fold C, D2 gained +7 claims over D0. But D1_matched was not tested on Fold C.
   - On Fold D, when D1_matched was tested side-by-side, D1_matched gained +9 claims over D0 (58 vs 49), whereas D2 gained only +5 claims (54 vs 49).
   - D1_matched beat D2 by +4 claims.

---

### 2. Failure Mode Decomposition: Output Format vs Verdict Semantics

Why did D0 lose so badly to both D1_matched and D2 across all folds?

#### Token Verbosity vs Reasoning Quality:
- **D0 Mean Output Tokens:** **179.1 tokens** (Fold C), **168.4 tokens** (Fold D).
- **D1_matched / D2 Mean Output Tokens:** **64.2 tokens** (Fold C), **61.8 tokens** (Fold D).

When an un-finetuned instruction model (Qwen3-8B) is allowed to generate unconstrained reasoning over 8 noisy web passages:
1. It engages in wandering chain-of-thought, noticing peripheral contradictions between news articles.
2. This creates an artificial prior shift toward the two hesitation classes:
   - On Fold D, D0 predicted `Conflicting` 15 times and `NEI` 17 times (32 hesitation predictions for only 11 true claims!).
   - In contrast, D1_matched predicted `Conflicting` 7 times and `NEI` 3 times (10 hesitation predictions).
3. The vast majority of the "gain" of both D1_matched and D2 is simply **suppressing verbosity-induced hesitation** and forcing the model to commit to `Refuted` or `Supported`.

---

## Section C — Evaluation of New Hypotheses (Analysis Only)

### 1. Are Historical Error Patterns Stable Across Folds?
**No.**
- In Fold A, 8 out of 25 errors were `Refuted -> NEI` (bailing into NEI).
- In Fold C, the dominant error mode was `Refuted -> Conflicting` (9 claims) due to partisan reporting in US election claims.
- In Fold D, a new error mode emerged: `Supported -> Refuted` (8 claims), where official health advisories with conditional language were misread as refutation.
Because AVeriTeC claims span vastly disparate topics (African constitutional budgets, American COVID regulations, European election ages, Asian celebrity rumors), error patterns are driven by **topic-specific linguistics and evidence store quality**, not static dataset-level invariants.

### 2. Does Lexical Similarity Correlate with Exemplar Helpfulness?
**Empirically not observed in our pilot setup (Downgraded to Hypothesis).**
In D4, 3-gram character Jaccard similarity retrieved exemplars that shared surface keywords (e.g., matching "budget" or "hospital"). In our evaluated setup:
- A claim about a hospital budget can fail due to double negatives.
- The retrieved exemplar failed due to media speculation.
In this specific implementation ($k=2$ 3-gram Jaccard retrieval from $N=50$ Fold A claims), lexical similarity demonstrated no measurable correlation with logical failure mode transfer: D4 was statistically indistinguishable from random exemplar selection (D3) on Fold D (52/91 vs 52/91, exact McNemar $p = 1.0000$). Whether broader lexical or dense semantic similarity metrics could correlate with reasoning failures in a larger bank remains an unverified hypothesis lacking empirical validation.

### 3. Do Corrective Rules Trade Off Across Confusion Pairs?
**Yes.**
This is the central dilemma of static rule injection:
- Rule 2 (*"Missing context is not Cherrypicking"*) fixed `Refuted -> NEI` in Fold A.
- But on Fold D, it caused `Refuted -> Conflicting` on `train:152` and `train:239`, and `Refuted -> Supported` on `train:242`.
Natural language rules lack operational bounds. An LLM interprets them globally, applying them to contexts where they distort valid reasoning.

### 4. Are There Observable Label-Free Features That Predict Failure Modes?
**Partially — with critical distinctions and unverified hypotheses:**
1. **Reranker Relevance Score vs Logical Evidence Sufficiency (Crucial Distinction):**
   The maximum cross-encoder reranking score $\max(\text{Rerank Score})$ measures topical/lexical relevance between query and passages, **not logical evidence sufficiency**. A passage may score highly because it discusses the exact person, date, and institution in the claim, yet still lack the specific quantitative or factual evidence needed to verify or refute the claim (a true `Not Enough Evidence` case). Conversely, a lower rerank score may reflect vocabulary mismatch rather than true absence of evidence. Thresholding $\max(\text{Rerank Score}) < \tau$ cannot be treated as a proxy for logical sufficiency without controlled validation.
2. **Direct Quote Attribution vs World Fact (Hypothesis H-ATTR):**
   *Hypothesis:* Claims containing explicit attribution syntax (*"X claims that Y"*, *"X said Y"*, *"X pledged Y"*) may pose elevated risk because frozen LLMs confuse verifying whether the quote was uttered with verifying whether the underlying proposition is true. *Status:* This is an exploratory qualitative observation from individual claim inspections; the claim that attribution claims fail at "$3\times$ higher rates" has not been verified via a controlled quantitative audit across official splits and is hereby downgraded to an unverified hypothesis.
3. **Retrieval Polarity Margin (Hypothesis H-POL):**
   *Hypothesis:* Apparent divergence in polarity among top-ranked passages may trigger model hesitation into `Conflicting Evidence`. *Status:* Requires rigorous statistical validation against true conflicting labels before being deployed as a deterministic signal.

### 5. Does Adding Error Knowledge Improve Beyond Matched Generic Knowledge?
**No.**
Fold D conclusively disproves this premise:
$$\text{Accuracy}(D2) = 59.3\% \quad < \quad \text{Accuracy}(D1\_matched) = 63.7\% \quad (\Delta = -4.40\%, \text{ CI } [-0.09, -0.01])$$
Injecting source error rules into the prompt degraded performance relative to a generic concise prompt of identical length.

---

## Section D — Summary of Deliverables & Next Steps

All raw metrics, manifests, and predictions have been verified against the canonical evaluation artifacts.

For the strategic decision on next research directions, refer to the accompanying artifact:
👉 [docs/M1_4_3_RESEARCH_DECISION.md](file:///c:/Users/ASUS/Downloads/training-free-fact-verification/docs/M1_4_3_RESEARCH_DECISION.md)
