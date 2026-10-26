# M1.5 — Source-Grounded Failure Mechanism Feasibility Study

**Milestone:** M1.5 Source-Grounded Failure Mechanism Feasibility Study  
**Author:** AI Research Engineer (Antigravity)  
**Date:** 2026-10-26  
**Repository:** `lee-vtruong/training-free-fact-verification`  
**Git Commit Target:** `main`  
**Permitted Historical Source:** Fold A (`train:0..49`, $N=50$ claims, 25 errors)  
**Evaluation Scope:** Strictly TRAINING-FREE (frozen Qwen3-8B, zero parameter tuning, zero learned routers)  

---

## Executive Summary & Final Research Decision

This study conducts a rigorous feasibility investigation to answer whether historical verification errors contain identifiable failure mechanisms that can be reliably recognized from unlabeled claims and evidence, and whether condition-specific prompt guidance can provide incremental value over matched generic prompting.

### Answers to the Three Core Research Questions:

1. **Do historical errors reveal failure mechanisms with sufficient stability and support?**  
   **NO.** In our strictly permitted source error memory bank (Fold A: $N=50$ claims, 25 errors), only **one mechanism** (*Scope, Quantifier & Denominator Mismatch*, $n=8$) has more than 5 historical support instances. Crucial categories such as *Attribution vs Proposition Confusion* ($n=2$) and *Negation / Polarity Inversion* ($n=2$) have negligible representation. Furthermore, cross-fold failure distributions shift drastically between folds (African fiscal governance in Fold A, American electoral/judicial disputes in Fold C, medical/scientific advisories in Fold D), demonstrating that error patterns are dominated by topic-specific vocabulary and retrieval store quality rather than stable dataset-level invariants.

2. **Can those mechanisms be recognized from observable inputs without looking at target labels?**  
   **POORLY / LOW PRECISION.** Label-free observable signals (regex matching for speech attribution, negation words, quantifier symbols) display severe false-alarm rates when applied to source claims. Specifically:
   - **Negation Signal:** Fired on 10 claims in Fold A, but 7 of those claims were **already verified correctly** by the base model (signal precision = 30.0%, risk of regressing correct claims = 28.0%).
   - **Attribution Signal:** Fired on 2 claims in Fold A (1 correct, 1 wrong; precision = 50.0%), capturing only 1 of the 2 attribution errors in Fold A.
   - **Scope / Quantifier Signal:** Fired on 24 claims (8 correct, 16 wrong; precision = 66.7%), posing a 32.0% risk of disrupting correctly predicted claims.

3. **Is there a grounded basis to expect mechanism-specific correction to exceed matched generic guidance?**  
   **NO.** Empirical evidence from Fold D ($N=91$ clean claims) conclusively disproved static error-rule injection: `D2` (empirical error rules) was defeated by `D1_matched` (matched generic directives) by $-4$ net claims (59.3% vs 63.7%, paired bootstrap 95% CI `[-0.09, -0.01]`, $W\to C = 0$). Injecting rules derived from source errors without high-precision triggers introduces negative transfer and corrupts valid reasoning on non-error claims.

---

### Final Research Decision: NO-GO on M2 Deployment; MODIFY Path Forward

Based strictly on empirical evidence, we issue a definitive **NO-GO on deploying M2 (Source-Derived Mechanism Recognition & Conditional Guidance) for confirmatory evaluation at this time**. Forcing the deployment of M2 with an underpowered source memory ($N=50$) and low-precision observable triggers would repeat the exact negative transfer observed in `D2` and `D4`.

#### Approved Modified Research Path:
1. **Prioritize M4 (Permutation Consistency Baseline):** A strictly training-free, zero-error-memory calibration baseline that aggregates multiple evidence permutations via majority voting, targeting LLM sensitivity to passage presentation order without risking negative transfer from ungrounded error rules.
2. **Prioritize M1 (Generic Claim-Type Clarification):** A unified, concise prompt that provides general principles for distinguishing statistical scope, attribution acts, and compound claims without historical error heuristics.
3. **Condition for Future M2 Re-evaluation:** M2 shall only be re-considered if the permitted source error memory bank is expanded to at least $N \ge 200$ claims under a pre-registered data partitioning plan with validated signal precision $\ge 80\%$.

---

## Part A — Scientific Claim Repairs & Methodological Clarifications

In accordance with project integrity standards, we rectify previous imprecise statements across project documentation:

### 1. Correction of the `Not Enough Evidence` (NEI) Metric
- **Incorrect Prior Statement:** Several earlier working summaries asserted that *"all LLM prompts achieved NEI F1 = 0.0000 across all folds"*.
- **Empirical Correction:** On Fold D ($N=91$ clean claims, 7 true NEI claims), the unconstrained baseline prompt **`D0` achieved NEI F1 = 0.2500** (Precision: $3/17 = 17.65\%$, Recall: $3/7 = 42.86\%$, 3 True Positives, 14 False Positives, 4 False Negatives).
- **Mechanistic Finding:** The complete collapse to $\text{NEI F1} = 0.0000$ was **strictly confined to brevity-constrained prompts** (`D1_matched`, `D2`, `D2_minus_r2`, `D3`, `D4`). The concise 1-sentence formatting directive successfully suppressed false hesitations (reducing NEI false positives from 14 down to 3), but simultaneously suppressed legitimate uncertainty, forcing all predictions into binary `Supported` or `Refuted` verdicts.

### 2. Disentangling Reranker Relevance Score from Logical Evidence Sufficiency
- **Fundamental Distinction:** The cross-encoder reranking score $\max_{p \in \text{Top-k}} \text{Score}_{\text{R4}}(p)$ measures query-passage topical and lexical relevance, **not logical evidence sufficiency**.
- A retrieved passage may receive a top reranker score (e.g., > 0.85) because it discusses the exact person, date, and institution named in the claim, yet still fail to provide the decisive fact or quantitative figure needed to verify the assertion (a genuine `Not Enough Evidence` case).
- Conversely, a lower rerank score may reflect lexical divergence or terminology mismatch despite the passage containing a logically decisive rebuttal.
- Deterministic gating of NEI based solely on $S_{\text{max}} < \tau$ conflates information retrieval ranking with truth-value completeness.

### 3. Downgrading Unverified Quantitative Assertions to Hypotheses
- **"Attribution claims fail at $3\times$ higher rates":** Downgraded to **Hypothesis H-ATTR**. Qualitative claim inspections revealed that models often confuse the truth of a quote's proposition with the truth of the utterance act; however, no statistically powered, controlled audit across official splits has established a fixed $3\times$ failure ratio.
- **"Lexical similarity has zero correlation with exemplar helpfulness":** Clarified as an **empirical finding specific to our pilot setup**. In our evaluated $k=2$ 3-gram character Jaccard implementation on Fold D, retrieved error exemplars were statistically indistinguishable from random exemplars (D4 vs D3: 52/91 vs 52/91, exact McNemar $p = 1.0000$). Whether broader dense semantic embeddings or reasoning-graph similarity could correlate with failure modes in a larger error bank remains an unverified hypothesis.

### 4. Methodological Distinction: Exploratory Audits vs Confirmatory Validation
- **Clarification on Folds B, C, and D:** Folds B ($N=46$), C ($N=50$), and D ($N=91$) were evaluated sequentially during exploratory research and failure mode diagnostics. Any hypotheses or insights formulated after observing outcomes on Folds B, C, and D cannot claim those same folds as independent confirmatory validations.
- Any future confirmatory evaluation must be executed on a strictly held-out partition (e.g., Fold E: `train:250..349`) under pre-registered hypotheses and frozen protocols.

---

## Part B — Source-Only Operational Failure Taxonomy

An exhaustive audit of all 25 errors committed by the baseline model (`D0`) on the permitted historical source split (Fold A: `train:0..49`, $N=50$ claims) was conducted using official gold justifications. Every error was categorized into one of six operational failure mechanisms:

```
Fold A (N=50 Claims):
├── Base Correct: 25 claims (50.0%)
└── Base Wrong:   25 claims (50.0%)
    ├── Scope, Quantifier & Denominator Mismatch: 8 errors (32.0%)
    ├── Partial Truth vs Cherrypicking:           5 errors (20.0%)
    ├── Evidence Sufficiency Confusion:           5 errors (20.0%)
    ├── Apparent vs Genuine Evidence Conflict:    3 errors (12.0%)
    ├── Attribution vs Proposition Confusion:     2 errors (8.0%)
    └── Negation / Polarity Inversion:            2 errors (8.0%)
```

### Detailed Operational Mechanism Profiles

#### 1. Scope, Quantifier and Denominator Mismatch
- **Operational Definition:** Veracity hinges on exact numerical thresholds, percentages, per-capita versus aggregate measures, or geographic/administrative jurisdictions, and the model confuses differing denominators or subsets.
- **Inclusion Criteria:** Claim contains explicit percentages, ratios, or quantifier phrases; evidence provides data for a related but non-identical denominator or scope; model verifies without reconciling base populations.
- **Exclusion Criteria:** Qualitative claims with no numerical or jurisdictional constraints.
- **Fold A Historical Examples ($n=8$):**
  - `train:2`: *"20% goes to local government (Nigeria)"* (Gold: Supported; Pred: NEI)
  - `train:6`: *"35% of revenue goes to states (Nigeria)"* (Gold: Refuted [26.72%]; Pred: NEI)
  - `train:11`: *"45% of Nigeria's revenue goes to federal government"* (Gold: Refuted [>50%]; Pred: NEI)
  - `train:21`: *"25 million Nigerians will get solar"* (Gold: Supported; Pred: NEI)
  - `train:27`: *"Counties striving to reach 10% Malabo allocation"* (Gold: NEI; Pred: Supported)
  - `train:47`: *"South Africans consume more alcohol per person than anywhere else in Africa"* (Gold: Refuted; Pred: NEI)
  - `train:48`: *"South Africans drink more per person than anywhere else in Africa"* (Gold: Refuted; Pred: NEI)
  - `train:49`: *"Only 31% of South Africans drink alcohol"* (Gold: NEI; Pred: Supported)
- **Gold Error Characterization:** Model repeatedly bails into NEI when percentages differ slightly or confuses drinkers vs whole population.
- **Observable Label-Free Signals:** Matches `\d+[%％]`, `per capita`, `per person`, `average`, `proportion`.
- **Ambiguous Cases:** Rhetorical percentage use (e.g., *"100% committed"*).
- **Support Count:** **$n=8$ / 25 errors (32.0%)**. Adequately supported in Fold A.

#### 2. Partial Truth vs Cherrypicking
- **Operational Definition:** Claim asserts multiple distinct propositions joined by conjunctions where only a subset is verified, and the model collapses the compound statement into binary Supported or Refuted instead of identifying Cherrypicking.
- **Inclusion Criteria:** Compound claims with multiple independent factual assertions; evidence substantiates only a subset.
- **Exclusion Criteria:** Single-assertion claims; compound subjects sharing identical predicates.
- **Fold A Historical Examples ($n=5$):**
  - `train:8`: *"Joe Biden voted for Iraq War and supported war in Serbia, Syria, Libya"* (Gold: Cherrypicking [supported 2 of 3]; Pred: Supported)
  - `train:12`: *"Donald Trump inherited a stagnant economy and then rebuilt it"* (Gold: Cherrypicking; Pred: Refuted)
  - `train:18`: *"16 year olds vote in Scotland, Wales, Austria"* (Gold: Cherrypicking [true for Scotland, not Wales]; Pred: NEI)
  - `train:19`: Duplicate variant of `train:18` (Gold: Cherrypicking; Pred: NEI)
  - `train:24`: *"Biden: No new taxes for anyone under $400k"* (Gold: Cherrypicking; Pred: Refuted)
- **Gold Error Characterization:** Inability to balance partial truth, oscillating between over-credulity and total refutation.
- **Observable Label-Free Signals:** Conjunctions connecting multiple clauses (`and`, `as well as`), multiple verbs.
- **Support Count:** **$n=5$ / 25 errors (20.0%)**. Sample-insufficient.

#### 3. Evidence Sufficiency Confusion
- **Operational Definition:** Retrieved passages are topically related to the subject but logically lack the specific factual data needed to confirm or refute the claim, yet the model hallucinates a definitive verdict instead of predicting Not Enough Evidence.
- **Inclusion Criteria:** Evidence covers claim entities; decisive quantitative or factual premise is absent; model predicts Supported or Refuted.
- **Exclusion Criteria:** Complete textual refutation or entailment present in evidence.
- **Fold A Historical Examples ($n=5$):**
  - `train:26`: *"No new taxes for people making less than $400,000"* (Gold: NEI; Pred: Supported)
  - `train:29`: *"Judiciary of Kenya have average 400,000 cases filed yearly"* (Gold: Supported; Pred: NEI)
  - `train:31`: *"There is a 'global average' for number of judges in Kenya"* (Gold: NEI; Pred: Refuted)
  - `train:38`: *"Kenya settles about 300,000 court cases each year"* (Gold: Supported; Pred: NEI)
  - `train:39`: *"There is a 'global average' for number of judges"* (Gold: NEI; Pred: Supported)
- **Gold Error Characterization:** Model treats broad discussions of Kenyan court backlogs as proof/refutation of specific annual filing statistics.
- **Observable Label-Free Signals:** High lexical entity overlap with absence of exact numerical metrics in passages.
- **Support Count:** **$n=5$ / 25 errors (20.0%)**. Sample-insufficient.

#### 4. Apparent vs Genuine Evidence Conflict
- **Operational Definition:** Multiple retrieved passages report divergent statistics due to different survey years or methodological definitions, and the model takes a unilateral stance rather than recognizing conflict.
- **Inclusion Criteria:** Top retrieved sources present irreconcilable factual accounts; model predicts Supported or Refuted.
- **Exclusion Criteria:** Media reporting on subjective political arguments.
- **Fold A Historical Examples ($n=3$):**
  - `train:17`: *"US Police 20.8 times more likely to kill than be killed"* (Gold: Supported [checks out from FBI data]; Pred: Cherrypicking)
  - `train:35`: *"Ghana unemployment declined from 11.9% to 7.3%"* (Gold: NEI [differing baseline years]; Pred: Cherrypicking)
  - `train:37`: *"Ghana GDP grew at 3.4% then 7%"* (Gold: Supported; Pred: Cherrypicking)
- **Gold Error Characterization:** Disagreement between government statistics and opposition claims caused model hesitation.
- **Observable Label-Free Signals:** Opposing directional terms (`increased` vs `decreased`) across passages.
- **Support Count:** **$n=3$ / 25 errors (12.0%)**. Severely sample-insufficient.

#### 5. Attribution vs Proposition Confusion
- **Operational Definition:** Claim asserts that a speaker uttered, promised, or signed an act, and the model evaluates whether the underlying proposition is true rather than whether the speech act occurred.
- **Inclusion Criteria:** Claim contains speech/pledge attribution verbs; evidence confirms the quote was made.
- **Fold A Historical Examples ($n=2$):**
  - `train:3`: *"Biden has pledged to stop border wall construction and give amnesty and health care"* (Gold: Supported [he did say it]; Pred: Refuted)
  - `train:14`: *"President Buhari signed new Police bill to allow arrest without warrant"* (Gold: Supported; Pred: NEI)
- **Support Count:** **$n=2$ / 25 errors (8.0%)**. Severely sample-insufficient.

#### 6. Negation / Polarity Inversion
- **Operational Definition:** Syntactic negation, double negatives, or restrictive terms in the claim or evidence lead to an inverted truth assessment.
- **Inclusion Criteria:** Explicit syntactic negation particles (`not`, `never`, `does not even`); verdict inverted.
- **Fold A Historical Examples ($n=2$):**
  - `train:20`: *"State government has begun project"* (Gold: Refuted [private consortium, not state]; Pred: Supported)
  - `train:25`: *"Does not even get 1% of funding"* (Gold: Refuted; Pred: NEI)
- **Support Count:** **$n=2$ / 25 errors (8.0%)**. Severely sample-insufficient.

---

## Part C — Observable Signal Feasibility Table

To evaluate whether mechanisms can be recognized without labels, we executed label-free extractors across all $N=50$ Fold A claims:

| Observable Signal | Regex / Heuristic Trigger | Total Fired | Correct Fired (False Alarm) | Wrong Fired (Hits) | Signal Precision | Risk of Regression |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Scope / Quantifier** | `%`, `percent`, `per capita`, `average` | 24 / 50 | 8 | 16 | **66.7%** | **32.0%** (8 correct claims) |
| **Partial Truth** | `and`, `as well as` + multi-clause (>10 words) | 9 / 50 | 2 | 7 | **77.8%** | **8.0%** (2 correct claims) |
| **Evidence Sufficiency** | Sparse text (<50 total evidence words) | 0 / 50 | 0 | 0 | **N/A** | **0.0%** |
| **Attribution** | `said`, `pledged`, `announced`, quotes `"..."` | 2 / 50 | 1 | 1 | **50.0%** | **4.0%** (1 correct claim) |
| **Negation** | `not`, `never`, `no`, `without`, `refuse` | 10 / 50 | **7** | **3** | **30.0%** | **28.0%** (7 correct claims) |

### Key Findings from Signal Feasibility:
1. **The Negation Paradox:** The negation signal fires on 10 claims, but **7 of those claims were already verified correctly by the base model**. Applying conditional negation guidance risks destabilizing 7 correct claims to address at most 3 errors.
2. **Attribution Under-Detection:** Regex captures only 2 claims in Fold A, missing subtle attribution phrasing while failing on 50% of matches.
3. **High Regression Risk for Scope Guidance:** While Scope/Quantifier has the highest hit count (16 errors), it also matches 8 claims that the model already answered correctly. As demonstrated by the failure of Rule 2 in `D2` on Fold D, applying broad statistical rules risks causing more regressions than corrections.

---

## Part D — Data Leakage, Near-Duplicate Quarantine & Group Safety

To maintain research validity and prevent circular evaluation:
1. **Near-Duplicate Quarantining:** Any claim in future evaluation splits with character 3-gram containment $\ge 0.50$ or word Jaccard similarity $\ge 0.60$ against Fold A must be quarantined prior to inference.
2. **Cross-Fold Topic Drift:**
   - Fold A (`train:0..49`): Dominated by Nigerian public finance and South African health statistics.
   - Fold C (`train:100..149`): Dominated by American presidential election controversies and European social policy.
   - Fold D (`train:150..249`): Dominated by international medical advisories, judicial salaries, and infrastructure contracts.
   - Because topics drift completely between folds, error rules derived from Fold A fail to transfer.

---

## Part E — Strategic Recommendation

| Option | Description | Evidence Basis | Recommendation |
|---|---|---|---|
| **M2 Pilot (Deploy Rules)** | Deploy conditional mechanism rules on Fold E | Defeated on Fold D ($-4$ net claims, CI `[-0.09, -0.01]`). Sample counts in Fold A ($n=2..5$) are insufficient. | **NO-GO** |
| **M4 Pilot (Permutation)** | Evaluate evidence order permutation majority voting | Zero error memory, zero parameter tuning, directly targets LLM presentation sensitivity. | **APPROVED (GO)** |
| **M1 Pilot (Generic Clarification)** | Deploy unified claim-type clarification prompt | Does not rely on empirical error memory; maintains format parity with `D1_matched`. | **APPROVED (GO)** |

For the complete experimental protocol and pre-registered hypotheses for M1 and M4, refer to:  
👉 [docs/M1_5_EXPERIMENT_PROTOCOL.md](file:///c:/Users/ASUS/Downloads/training-free-fact-verification/docs/M1_5_EXPERIMENT_PROTOCOL.md)
