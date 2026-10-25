# M1.4.2 — D2 Mechanism Audit & Error-Rule Study Report

**Milestone:** M1.4.2 — Controlled Verification Mechanism Analysis & Statistical Audit  
**Author:** AI Research Engineer (Antigravity)  
**Date:** 2026-10-25  
**Repository:** `lee-vtruong/training-free-fact-verification`  
**Git Commit Target:** `main`  
**Audited Run:** `outputs/m141-confirmatory-20261025-101555` ($N=50$, Fold C: `train:100..149`)  
**Core Constraint:** Strictly training-free. Frozen models only. Zero gold label contamination.

---

## Executive Summary

Following the confirmatory failure of **D4** (Similarity-Retrieved Error Demonstrations) on independent Fold C ($N=50$), where D4 collapsed to Macro-F1 0.3333 (falling below the D0 baseline of 0.3801 and underperforming D3 at 0.3739, $p=0.4531$), attention shifted to the secondary finding: **D2 (Concise Source-Derived Error Rules)**.

On Fold C, D2 posted **Accuracy 68.0%** (34/50) and **Macro-F1 0.4365**, achieving a nominal gain over D0 (+14.0% Acc, +0.0564 Macro-F1) with raw exact McNemar $p = 0.0156$.

### Key Findings of this Audit:
1. **Multiplicity Correction Nullifies Statistical Significance:** When applying standard family-wise error rate (FWER) control via the Holm-Bonferroni step-down procedure across the family of 5 planned comparisons evaluated on Fold C, D2 vs D0 yields an adjusted $p$-value of **$p_{\text{adj}} = \mathbf{0.0781} > 0.05$**. D2 is **NOT** statistically significant under formal error control.
2. **Selection Bias & Post-Hoc Risk:** The primary pre-registered confirmatory hypothesis for Fold C was **D4 vs D3/D5**. D2 was included merely as an exploratory concise rule control. Declaring D2 "confirmed" based on an unadjusted post-hoc finding would constitute textbook HARKing (Hypothesizing After Results are Known).
3. **Mechanism Confound (Brevity vs Domain Heuristics):** D0 produced verbose rationales averaging 179.1 tokens, leading the model to hallucinate nuances into `Conflicting Evidence` (12 claims) and `Not Enough Evidence` (7 claims). D2 enforced a strict 1-sentence constraint (averaging 64.2 tokens) alongside two empirical heuristics. Whether D2's lift stems from **empirical error knowledge** or simply **verbosity suppression / prior realignment** cannot be resolved without a matched generic control (`D1_matched`).
4. **Resolution via Pre-Registered Fold D Evaluation ($N=100$):** We establish a rigorous confirmatory protocol on fresh **Fold D (`train:150..249`)**, featuring double the sample size ($2\times$ statistical power), an unobserved population, pre-frozen prompts (`D1_matched`, `D2`, leave-one-out ablations `D2_minus_r1`/`D2_minus_r2`, and paraphrase variants), and a pre-registered primary hypothesis contrast of **D2 vs D1_matched**.

---

## Part A — Integrity & Statistical Audit (Fold C, $N=50$)

### 1. Recomputed Performance Summary

All metrics were recomputed directly from canonical per-claim output JSONL files (`outputs/m141-confirmatory-20261025-101555/*/predictions.jsonl`) and verified against `scikit-learn` with strict machine epsilon precision ($|\Delta| < 10^{-9}$).

Official Fold C ground truth distribution ($N=50$):
- **Refuted**: 35 claims (70.0%)
- **Supported**: 11 claims (22.0%)
- **Not Enough Evidence**: 3 claims (6.0%)
- **Conflicting Evidence/Cherrypicking**: 1 claim (2.0%)

#### Table 1: Independent Performance Metrics on Fold C ($N=50$)

| Condition | Description | Correct / 50 | Accuracy | Macro-F1 | Macro-P | Macro-R | Output Class Distribution (Ref / Supp / NEI / Conf / Unp) | Mean Output Tokens |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **D0** | Frozen Base Protocol v1.0 | 27 / 50 | 54.0% | 0.3801 | 0.4312 | 0.5448 | 22 / 9 / 7 / 12 / 0 | 179.1 |
| **D2** | Concise Source-Derived Error Rules | 34 / 50 | 68.0% | 0.4365 | 0.4187 | 0.6104 | 28 / 13 / 3 / 6 / 0 | 64.2 |
| **D3** | Random Error Demonstrations ($k=2$) | 34 / 50 | 68.0% | 0.3739 | 0.3833 | 0.3675 | 30 / 12 / 4 / 4 / 0 | 59.8 |
| **D4** | Similarity Error Demonstrations ($k=2$) | 31 / 50 | 62.0% | 0.3333 | 0.3415 | 0.3305 | 29 / 13 / 4 / 4 / 0 | 58.4 |
| **D5** | Corrected Error Demonstrations ($k=2$) | 26 / 50 | 52.0% | 0.3603 | 0.4226 | 0.5221 | 22 / 8 / 7 / 13 / 0 | 158.3 |

*Note: Parse failure rate was exactly 0.0% (0 unparseable verdicts across all 250 predictions).*

---

### 2. Paired Transition Analysis & Multiplicity Correction

We evaluate all paired transitions across the 9 mutually exclusive categories ($C\to C$, $C\to W$, $C\to U$, $W\to C$, $W\to W$, $W\to U$, $U\to C$, $U\to W$, $U\to U$, where $C=\text{Correct}$, $W=\text{Wrong}$, $U=\text{Unparseable}$).

We execute:
- **Exact two-sided McNemar test** on discordant pairs ($b = C\to W + C\to U$, $c = W\to C + U\to C$).
- **10,000 paired bootstrap resamples** (seed=42) for both Accuracy difference and Macro-F1 difference.
- **Holm-Bonferroni FWER step-down adjustment** across the family of 5 evaluated comparisons.

#### Table 2: Paired Transitions, McNemar Tests, Bootstrap CIs, and Multiplicity Adjustment

| Rank | Comparison | Net Gain | $C \to C$ | $C \to W$ | $W \to C$ | $W \to W$ | Accuracy $\Delta$ [95% CI] | Macro-F1 $\Delta$ [95% CI] | Raw $p$ | Holm $p_{\text{adj}}$ | Stat. Sig. ($\alpha=0.05$) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **D2 vs D0** | **+7** | 27 | 0 | 7 | 16 | +14.00% `[+0.0600, +0.2400]` | +0.0564 `[-0.0145, +0.1361]` | **0.0156** | **0.0781** | **NO** |
| 2 | **D4 vs D5** | +5 | 24 | 2 | 7 | 17 | +10.00% `[-0.0200, +0.2200]` | -0.0269 `[-0.1296, +0.0694]` | 0.1797 | 0.7188 | NO |
| 3 | **D4 vs D0** | +4 | 25 | 2 | 6 | 17 | +8.00% `[-0.0200, +0.2000]` | -0.0468 `[-0.1332, +0.0246]` | 0.2891 | 0.8672 | NO |
| 4 | **D4 vs D3** | -3 | 29 | 5 | 2 | 14 | -6.00% `[-0.1600, +0.0400]` | -0.0406 `[-0.0954, +0.0038]` | 0.4531 | 0.9062 | NO |
| 5 | **D2 vs D3** | 0 | 31 | 3 | 3 | 13 | 0.00% `[-0.1000, +0.1000]` | +0.0626 `[-0.0361, +0.1768]` | 1.0000 | 1.0000 | NO |

#### Statistical Interpretation:
- **D2 vs D0 loses significance after Holm-Bonferroni correction:** While the unadjusted McNemar $p = 0.0156$ is nominally below 0.05, the multiplicity-adjusted $p$-value is **$0.0781$**. Under formal statistical standards, D2 cannot be claimed as statistically confirmed.
- **Macro-F1 95% Confidence Interval crosses zero:** The paired bootstrap 95% CI for D2 vs D0 Macro-F1 delta is `[-0.0145, +0.1361]`. We cannot reject the null hypothesis of equal Macro-F1.
- **D2 vs D3 shows zero accuracy gain:** Both D2 and D3 achieve exactly 34/50 correct (68.0%). McNemar $p = 1.0000$. While D2 has a higher nominal Macro-F1 (+0.0626), its 95% CI `[-0.0361, +0.1768]` broadly spans zero.

---

### 3. Per-Class Analysis & Transition Audit

#### Table 3: Per-Class Breakdown on Fold C ($N=50$)

| Class | Support | Metric | D0 (Base) | D2 (Error Rules) | D3 (Random ICL) | D4 (Similarity ICL) |
|:---|:---:|:---|:---:|:---:|:---:|:---:|
| **Supported** | 11 | Precision / Recall / F1 | 0.7778 / 0.6364 / 0.7000 | 0.6154 / **0.7273** / 0.6667 | 0.6667 / **0.7273** / 0.6957 | 0.6364 / 0.6364 / 0.6364 |
| **Refuted** | 35 | Precision / Recall / F1 | 0.8636 / 0.5429 / 0.6667 | **0.8929** / 0.7143 / **0.7937** | 0.8667 / **0.7429** / 0.8000 | 0.8276 / 0.6857 / 0.7500 |
| **Not Enough Evidence** | 3 | Precision / Recall / F1 | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 |
| **Conflicting Evidence** | 1 | Precision / Recall / F1 | 0.0833 / **1.0000** / 0.1538 | 0.1667 / **1.0000** / **0.2857** | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 |

#### Confusion Matrices:
- **D0 Confusion Matrix** (Rows: Gold, Cols: Supp, Ref, NEI, Conf):
  - Supported (11): `[7, 1, 1, 2]`
  - Refuted (35): `[1, 19, 6, 9]`  $\leftarrow$ **15 / 35 false negatives!** (6 NEI, 9 Conflicting)
  - NEI (3): `[1, 2, 0, 0]`
  - Conflicting (1): `[0, 0, 0, 1]`
- **D2 Confusion Matrix**:
  - Supported (11): `[8, 1, 0, 2]`
  - Refuted (35): `[4, 25, 3, 3]`  $\leftarrow$ **Refuted recall jumped from 54.3% to 71.4%!**
  - NEI (3): `[1, 2, 0, 0]`
  - Conflicting (1): `[0, 0, 0, 1]`

#### Detailed Case Audit: The 7 Improved Claims in D2 vs D0
In D2 vs D0, there were **7 improvements and 0 regressions**:
1. `train:107` (Gold: `Supported`):
   - Claim: *"The United States Postal Service advises customers to never..."*
   - D0: `Refuted` (model over-interpreted a conditional disclaimer as refutation).
   - D2: `Supported` (corrected: concise evaluation directly confirmed the core advisory).
2. `train:109` (Gold: `Refuted`):
   - Claim: *"Former U.S. President George W. Bush supports the Democratic..."*
   - D0: `Not Enough Evidence` (model hedged because articles did not explicitly mention every campaign event).
   - D2: `Refuted` (corrected: spokespeople explicitly denied the endorsement).
3. `train:116` (Gold: `Refuted`):
   - Claim: *"Former president Pranab Mukherjee from India has passed away..."*
   - D0: `Conflicting Evidence/Cherrypicking` (model saw conflicting Twitter rumors vs hospital statement).
   - D2: `Refuted` (corrected: applied Rule 1, identifying media rumors as non-substantive).
4. `train:119` (Gold: `Refuted`):
   - Claim: *"Creativity is contagious. Pass it on was said by Albert Einstein"*
   - D0: `Conflicting Evidence/Cherrypicking` (model confused attributions in opinion blogs).
   - D2: `Refuted` (corrected: evidence explicitly stated Einstein never said this).
5. `train:124` (Gold: `Refuted`):
   - Claim: *"Sarah Huckabee Sanders said 'it's not the government's job to provide healthcare'..."*
   - D0: `Not Enough Evidence` (model bailed because full briefing transcript was partial).
   - D2: `Refuted` (corrected: fact-checks proved the quote was fabricated).
6. `train:140` (Gold: `Refuted`):
   - Claim: *"According to Harry Roque, even if 10,000 people die, 10 million..."*
   - D0: `Not Enough Evidence` (model hesitated on context).
   - D2: `Refuted` (corrected: Roque's actual quote directly contradicted the claim).
7. `train:145` (Gold: `Refuted`):
   - Claim: *"330,000 people a day are being tested for Covid-19 under the..."*
   - D0: `Conflicting Evidence/Cherrypicking` (model cited varying daily testing statistics).
   - D2: `Refuted` (corrected: official CDC data showed capacity was nowhere near 330k/day).

**Mechanism Diagnosis:** 6 out of the 7 improved claims were Gold `Refuted` claims that D0 had misclassified into `Conflicting Evidence` (3 claims) or `Not Enough Evidence` (3 claims). D2's rules specifically suppress spurious `Conflicting` and `NEI` classifications.

---

### 4. Integrity, Provenance & Quarantine Review

- **Isolation of Rules:** D2's prompt text was synthesized during Milestone M1.3 and formalized in M1.4 (`src/tffv/verification/m14_prompting.py`) **exclusively using failure modes observed on Fold A (`train:0..49`)**.
- **Frozen State:** The D2 prompt string was committed prior to Fold C acquisition and was not modified during or after Fold C execution.
- **Zero Evaluation Contamination:** No claim from Fold C (`train:100..149`) was present in the source error memory bank. All Fold C claims passed near-duplicate quarantine (Jaccard $<0.60$, 3-gram $<0.50$ against Fold A).

---

## Part B — Error-Rule Mechanism Study

### 1. Source Error Memory & Rule Provenance

In Fold A ($N=50$), the base model made **25 errors (50% error rate)** across 11 confusion categories. From these 25 errors, 4 empirical verification rules were synthesized:

1. **Rule 1: Media Debate vs Factual Contradiction**
   - *Provenance:* Fold A claims `train:17` (police fatality rates), `train:35` (unemployment rates), `train:37` (GDP growth rate).
   - *Failure Mode:* Retrieved articles quoted political candidates arguing over statistics. The base model mistook political argumentation for empirical contradiction, erroneously predicting `Conflicting Evidence/Cherrypicking`.
   - *Checklist Text:* *"If retrieved articles quote political opponents arguing about a claim, verify whether the underlying factual statement itself is substantiated. Do NOT classify political disagreement as Conflicting Evidence if official records confirm the fact."*
2. **Rule 2: Evidence Sufficiency Threshold**
   - *Provenance:* Fold A claims `train:2` (Nigeria revenue), `train:6` (state revenue), `train:11` (federal revenue), `train:14` (Police bill), `train:29` (judiciary cases), `train:38` (settled cases), `train:47` (alcohol consumption), `train:48` (alcohol per person) — **8 out of 25 errors!**
   - *Failure Mode:* The base model defaulted to `Not Enough Evidence` whenever retrieved evidence confirmed the core factual proposition but omitted secondary details (e.g., date of legislation, author of report).
   - *Checklist Text:* *"If evidence directly confirms or contradicts the core proposition, do not default to Not Enough Evidence simply because minor peripheral details are unmentioned."*
3. **Rule 3: Cherrypicking Precision**
   - *Provenance:* Fold A claims `train:8` (Biden voting record), `train:12` (inherited economy), `train:19` (promises kept), `train:24` (teen anxiety).
   - *Failure Mode:* The base model confused claims that quote true facts out of context with claims that are outright false or fully true.
   - *Checklist Text:* *"Choose 'Conflicting Evidence/Cherrypicking' only when the claim quotes true facts out of context to convey a demonstrably false implication. Outright false claims are Refuted."*
4. **Rule 4: Polarity Check**
   - *Provenance:* Fold A claims `train:20` (modular refinery), `train:25` (judiciary budget $<1\%$).
   - *Failure Mode:* Double negatives and conditional denials misled the model's entailment reasoning.
   - *Checklist Text:* *"Carefully evaluate conditional denials and double negatives to ensure the direction of entailment matches the claim."*

### 2. Distillation into Condition D2

To fit within standard zero-shot prompt budgets and avoid prompt bloat, M1.4 distilled these 4 rules into 2 concise directives paired with a 1-sentence generation constraint:

```text
Empirical Error Directives (derived from observed verification pitfalls):
1. Media debate is not factual contradiction: An article describing a political disagreement does not constitute Refutation of a verifiable fact.
2. Missing context is not Cherrypicking: Absence of secondary details does not warrant Cherrypicking if the primary assertion is substantiated.
State a concise 1-sentence factual reason, then output Verdict: <Label>.
```

Notice:
- Directive 1 addresses **spurious Refutation/Conflicting** triggered by media debates.
- Directive 2 addresses **spurious Cherrypicking/NEI** triggered by missing peripheral context.
- The closing directive enforces **1-sentence brevity**.

### 3. Comparison with Generic D1 and Identification of Confounds

In M1.4, **D1 (Concise Generic Rules)** was specified as:
```text
Core Verification Directives:
- If the provided evidence directly substantiates the core claim, select Supported.
- If the provided evidence directly contradicts the core claim, select Refuted.
- If the evidence is insufficient to verify or refute the claim, select Not Enough Evidence.
- If reputable sources present contradictory factual accounts, select Conflicting Evidence/Cherrypicking.
State a concise 1-sentence factual reason, then output Verdict: <Label>.
```

**The Confound:**
- D1 merely restates the 4 ontology definitions in generic terms, actively prompting the model to consider `Conflicting Evidence` and `Not Enough Evidence`.
- D2, by contrast, gives **asymmetric negative instructions** against choosing `Conflicting Evidence` and `Not Enough Evidence`.
- Furthermore, both D1 and D2 include `"State a concise 1-sentence factual reason"`, whereas D0 allows unbounded reasoning. D0's average output length was **179.1 tokens**, whereas D2's was **64.2 tokens**.
- When an LLM produces lengthy chain-of-thought on 8 noisy web passages without instruction tuning, it tends to discover spurious nuances, naturally inflating `Conflicting` and `NEI`.

**Scientific Question:** Does D2 succeed because of **domain-specific error rules**, or simply because **concise formatting + general focus** curtails overthinking?

### 4. Matched Generic Control: `D1_matched`

To isolate this confound, we design `D1_matched`. It has:
- Identical character count (422 chars vs 423 chars in D2).
- Identical 2-numbered directive structure.
- Identical tone, reading grade level, and closing 1-sentence constraint.
- **ZERO historical error memory heuristics.**

```text
Standard Verification Directives (applicable across general claims):
1. Primary assertion evaluation: Assess whether the central factual proposition is directly confirmed or denied by the provided sources.
2. Substantive content focus: Base the determination on the primary factual statement rather than peripheral commentary or stylistic phrasing.
State a concise 1-sentence factual reason, then output Verdict: <Label>.
```

### 5. Leave-One-Rule-Out Ablation Design

To determine the individual necessity and sufficiency of D2's two directives:
1. **`D2_minus_r1` (Keep Rule 2 only):**
   ```text
   Empirical Error Directives (derived from observed verification pitfalls):
   1. Missing context is not Cherrypicking: Absence of secondary details does not warrant Cherrypicking if the primary assertion is substantiated.
   State a concise 1-sentence factual reason, then output Verdict: <Label>.
   ```
2. **`D2_minus_r2` (Keep Rule 1 only):**
   ```text
   Empirical Error Directives (derived from observed verification pitfalls):
   1. Media debate is not factual contradiction: An article describing a political disagreement does not constitute Refutation of a verifiable fact.
   State a concise 1-sentence factual reason, then output Verdict: <Label>.
   ```

### 6. Paraphrase Robustness Design

To verify that D2's performance is not a brittle prompt-engineering artifact sensitive to exact wording:
1. **`D2_para1` (Action/Guideline style):**
   ```text
   Empirical Verification Guidelines (derived from past error analysis):
   1. Distinguish reporting from refutation: When news outlets quote opposing political arguments, verify the underlying factual claim rather than treating political dispute as refutation.
   2. Separate context omission from selective framing: Do not mark a claim as cherrypicked solely because peripheral context is absent, provided the main factual claim holds.
   State a concise 1-sentence factual reason, then output Verdict: <Label>.
   ```
2. **`D2_para2` (Formal Criteria style):**
   ```text
   Empirical Error Guidelines (based on observed verification pitfalls):
   1. Disagreement does not equal contradiction: Quotes showing political figures arguing about a statement do not disprove verified factual records.
   2. Unmentioned details do not equal cherrypicking: Missing non-essential facts do not justify a cherrypicking label when the core assertion is accurate.
   State a concise 1-sentence factual reason, then output Verdict: <Label>.
   ```

### 7. Sample Size & Rule Stability Critique

Fold A consists of only 50 claims, of which 25 resulted in base model errors. When stratified across 11 confusion pairs:
- Confusion pairs like `Supported -> Conflicting` had only **2 examples**.
- `Refuted -> Supported` had only **1 example**.
- Deriving generalizable rules from $n=1$ or $n=2$ instances carries severe risk of overfitting to the specific political or geographic topics of Fold A (e.g., Nigerian revenue allocations, Kenyan judiciary budgets).
- Expanding the error memory bank to larger source pools or validating across a fresh, unobserved population is essential before concluding that error rules are stable.

---

## Part C — Pre-Registered Independent Evaluation Protocol (Fold D, $N=100$)

To resolve whether D2 provides genuine empirical lift over generic prompting, we design an independent confirmatory evaluation on **Fold D (`train:150..249`)**.

### 1. Target Population: Fold D ($N=100$)
- **Population:** Official AVeriTeC train claims with indices `150..249` (100 claims).
- **Provenance:** Never previously loaded, audited, prompted, or evaluated in M1.1, M1.2, M1.3, M1.4, or M1.4.1.
- **Statistical Power:** Double the sample size of Fold C ($N=50$) and Fold B ($N=46$), reducing standard error by $\approx 29\%$ and narrowing bootstrap confidence intervals.
- **Gold Label Distribution:**
  - `Refuted`: 52 claims (52.0%)
  - `Supported`: 37 claims (37.0%)
  - `Not Enough Evidence`: 7 claims (7.0%)
  - `Conflicting Evidence/Cherrypicking`: 4 claims (4.0%)
  - *Balanced representation with both majority and minority classes.*

### 2. Pre-Frozen Evaluation Suite Conditions
We evaluate 8 conditions simultaneously under strictly identical execution environments:
1. `D0`: Frozen Base Protocol v1.0 (Zero-shot baseline).
2. `D1_matched`: Matched Generic Rules (Controlled format/length, zero error knowledge).
3. `D2`: Source-Derived Error Rules (Full empirical rules: R1 + R2).
4. `D2_minus_r1`: Ablation without Rule 1 (Rule 2 only).
5. `D2_minus_r2`: Ablation without Rule 2 (Rule 1 only).
6. `D2_para1`: Paraphrase Robustness Variant 1.
7. `D3`: Random Error Demonstrations ($k=2$, from Fold A bank).
8. `D4`: Similarity-Retrieved Error Demonstrations ($k=2$, from Fold A bank).

### 3. Pre-Registered Hypotheses & Statistical Contrasts

#### Primary Pre-Registered Hypothesis:
- **Contrast:** **`D2 vs D1_matched`**
- **Scientific Question:** Does empirical error memory derived from historical failures provide genuine verification performance gains beyond an equally concise, format-matched generic prompt?
- **Null Hypothesis ($H_0$):** $\Delta \text{Macro-F1} = 0$ and $\Delta \text{Accuracy} = 0$.
- **Alternative Hypothesis ($H_1$):** D2 achieves statistically significantly higher Macro-F1 and Accuracy than D1_matched.
- **Statistical Criteria:**
  - Two-sided exact McNemar test on discordant pairs ($p < 0.05$).
  - 10,000 paired bootstrap 95% CI for $\Delta \text{Macro-F1}$ strictly excluding zero.

#### Secondary Pre-Registered Hypotheses:
1. **Replication of Fold C Signal:** `D2 vs D0`. Does D2 replicate its nominal gain on 100 fresh claims?
2. **Component Necessity:** `D2 vs D2_minus_r1` and `D2 vs D2_minus_r2`. Is each rule necessary?
3. **Syntactic Invariance:** `D2 vs D2_para1`. Does phrasing affect the verdict?
4. **Exemplar Cross-Validation:** `D4 vs D3` and `D3 vs D0`. Does similarity error retrieval fail again or recover on $N=100$?
5. **Multiplicity Control:** Holm-Bonferroni correction applied across the family of primary and secondary contrasts.

### 4. Methodological Invariants & Safeguards
- **Backbone:** Frozen Qwen3-8B snapshot (`b968826d9c46dd6066d109eabc6255188de91218`), `bfloat16`, `device_map="cuda"`.
- **Decoding:** Strictly greedy (`temperature=0.0`, `do_sample=False`), non-thinking (`enable_thinking=False`), `max_new_tokens=512`.
- **Evidence Pipeline:** Standard frozen M1.1 pipeline (R1 BM25 + R2 Dense Qwen3-Embedding-0.6B + R3 RRF + R4 Qwen3-Reranker-0.6B), Top-8 deduplicated evidence passages.
- **Strict Anti-Leakage:** Fold D claims quarantined against Fold A source bank. No ground truth labels, QA, or justifications exposed to retrieval or prompts.
- **Content-Addressed Provenance:** All predictions, manifests, and metrics sealed with SHA-256 hashes.
