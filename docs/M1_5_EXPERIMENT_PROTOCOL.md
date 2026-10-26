# M1.5 — Controlled Verification & Mechanism Evaluation Protocol

**Milestone:** M1.5 Controlled Verification Evaluation Protocol  
**Author:** AI Research Engineer (Antigravity)  
**Date:** 2026-10-26  
**Repository:** `lee-vtruong/training-free-fact-verification`  
**Git Commit Target:** `main`  
**Protocol Status:** PRE-REGISTERED & FROZEN  
**Target Evaluation Partition:** Fold E (`train:250..349`, $N=100$ candidate claims)  
**Source Memory Partition:** Strictly Fold A (`train:0..49`, $N=50$ claims, 25 errors)  
**Quarantine Baseline:** All claims in Fold E with lexical Jaccard similarity $\ge 0.60$ or 3-gram containment $\ge 0.50$ against Fold A are quarantined prior to inference.  

---

## 1. Experimental Conditions & Prompt Contracts

All conditions run on the **frozen Qwen3-8B backbone** in `bfloat16` with deterministic greedy decoding (`temperature=0.0`, `do_sample=False`, `max_new_tokens=64`). All conditions enforce identical token budget parity (~150–180 prompt tokens) and output constraints (1-sentence rationale followed by `Verdict: <Label>`).

```
                              [Incoming Claim + Top-8 R4 Evidence]
                                               │
               ┌───────────────────────┬───────┴───────────────┬───────────────────────┐
               ▼                       ▼                       ▼                       ▼
            Condition M0            Condition M1            Condition M2            Condition M4
         Matched Generic        Generic Claim-Type       Source-Derived          Permutation
             Control               Clarification       Mechanism Guidance        Consistency
        (CONCISE_MATCHED)         (Broad Taxonomy)      (Signal Triggered)     (3-Permutation Vote)
               │                       │                       │                       │
               │                       │            [No Signal? Fallback]              │
               │                       │                       │                       │
               ▼                       ▼                       ▼                       ▼
          Single Pass             Single Pass             Single Pass             3 Forward Passes
         (64 new tokens)         (64 new tokens)         (64 new tokens)         (Majority Vote)
```

### Condition Specifications:

- **Condition M0 (Matched Generic Control):**
  - **Definition:** Single-pass verification using the concise matched generic prompt ([`CONCISE_MATCHED_GENERIC_RULES`](file:///c:/Users/ASUS/Downloads/training-free-fact-verification/src/tffv/verification/m14_prompting.py#L50-L55)).
  - **Purpose:** Primary reference baseline. Provides formatting and length parity without historical error knowledge.
  - **Token Budget:** ~160 prompt tokens, max 64 new tokens. 1 model call per claim.

- **Condition M1 (Generic Claim-Type Clarification):**
  - **Definition:** Single-pass verification with generic prompt clarifications distinguishing statistical scope, quote attribution, and compound claims, derived from general fact-checking principles without empirical error memory.
  - **Purpose:** Tests whether structuring verification by claim type improves accuracy over M0 without relying on historical error memory.
  - **Token Budget:** ~175 prompt tokens, max 64 new tokens. 1 model call per claim.

- **Condition M2 (Source-Derived Mechanism Guidance — Exploratory / Diagnostic):**
  - **Definition:** Conditional prompt guidance triggered exclusively when a label-free observable signal fires:
    - *Scope / Quantifier Signal:* Injects Scope & Denominator consistency directives.
    - *Attribution Signal:* Injects Speech Act vs Proposition directives.
    - *Partial Truth Signal:* Injects Compound Clause breakdown directives.
    - *Fallback:* If no observable signal fires, deterministically falls back to Condition M0.
  - **Purpose:** Tests whether conditional mechanism guidance improves accuracy on signal-active claims without regressing other claims.
  - **Token Budget:** ~175 prompt tokens, max 64 new tokens. 1 model call per claim.

- **Condition M3 (Negative Control — Random Guidance):**
  - **Definition:** Same mechanism guidance directives as M2, but randomly assigned to claims using a fixed random seed (`seed=42`), ignoring claim features.
  - **Purpose:** Establishes the null hypothesis: determines whether any observed difference in M2 is driven by semantic mechanism specificity or arbitrary prompt perturbation.
  - **Token Budget:** ~175 prompt tokens, max 64 new tokens. 1 model call per claim.

- **Condition M4 (Permutation Consistency Baseline):**
  - **Definition:** Training-free, zero-error-memory inference aggregation. Evaluates the claim under three deterministic evidence order permutations:
    1. *Order 1 (Canonical):* Top-8 passages ranked by R4 score $[1, 2, 3, 4, 5, 6, 7, 8]$.
    2. *Order 2 (Reverse):* Top-8 passages in reverse order $[8, 7, 6, 5, 4, 3, 2, 1]$.
    3. *Order 3 (Interleaved):* Top-8 passages interleaved $[1, 8, 2, 7, 3, 6, 4, 5]$.
    - Aggregates verdicts via majority voting; ties broken by Order 1 canonical verdict.
  - **Purpose:** Tests whether inference-time consistency resolves presentation sensitivity without any error memory.
  - **Token Budget:** 3 model calls per claim ($\approx 3 \times 64 = 192$ new tokens).

---

## 2. Pre-Registered Hypotheses & Primary Comparisons

To prevent p-hacking and multiple testing artifacts, all comparisons are pre-registered and partitioned into Primary, Secondary, and Diagnostic tests:

### Primary Hypothesis 1: Permutation Consistency (M4 vs M0)
- **Hypothesis:** Evidence presentation ordering creates artificial verification errors in single-pass LLM inference; aggregating over three permutations (`M4`) improves Accuracy and Macro-F1 over single-pass generic prompting (`M0`).
- **Statistical Contrast:** Paired McNemar test on clean admitted claims of Fold E.
- **Success Threshold:** Net Gain $\ge +3$ claims over M0, with paired bootstrap 95% CI lower bound $> 0.00$.

### Secondary Hypothesis 2: Generic Claim-Type Clarification (M1 vs M0)
- **Hypothesis:** Broad structural definitions of claim types (scope, attribution, conjunction) improve reasoning without empirical error memory.
- **Success Threshold:** Accuracy(M1) $\ge$ Accuracy(M0) AND paired McNemar $p < 0.05$.

### Diagnostic Hypothesis 3: Mechanism Specificity (M2 vs M3 on Active Claims)
- **Hypothesis:** Source-derived mechanism guidance provides positive transfer only when matched to the appropriate observable signal; M2 must outperform random guidance M3 on the active subset.
- **Success Threshold:** Accuracy(M2) $-$ Accuracy(M3) $> 0$ on the signal-triggered subpopulation.

---

## 3. Evaluation Metrics & Statistical Protocol

For every evaluated condition on Fold E, the following metrics must be computed from raw canonical per-claim prediction files:

1. **Primary Metrics:**
   - Classification Accuracy (\%)
   - Macro-F1 across all 4 official AVeriTeC classes (`Supported`, `Refuted`, `Not Enough Evidence`, `Conflicting Evidence/Cherrypicking`)
2. **Per-Class Metrics:**
   - Precision, Recall, and F1 for each of the 4 classes
   - Valid Verdict Rate (fraction of outputs successfully parsing into the 4 classes)
3. **Paired Difference Analysis:**
   - Paired transition matrix against M0:
     $$N_{C \to C}, \quad N_{C \to W} \text{ (Regressions)}, \quad N_{W \to C} \text{ (Corrections)}, \quad N_{W \to W}$$
   - Net Claim Gain: $\Delta = N_{W \to C} - N_{C \to W}$
4. **Hypothesis Testing:**
   - Two-tailed exact McNemar test on discordant pairs ($N_{C \to W}$ vs $N_{W \to C}$)
   - Paired bootstrap 95% Confidence Interval on Accuracy Delta (10,000 resamples with replacement)
   - Holm-Bonferroni family-wise error rate control across the pre-registered hypothesis suite.

---

## 4. Confound Isolation & Anti-Leakage Invariants

1. **Separation of Inference Consistency from Historical Error Memory:**
   - Condition M4 uses **zero historical error memory**. Any performance improvement in M4 is strictly an inference aggregation gain, and must NEVER be reported as evidence for the efficacy of historical error memory.
2. **Strict Split Quarantine:**
   - Fold E (`train:250..349`) is strictly held out. No Fold E claims or evidence were inspected during the formulation of prompts, taxonomy profiles, or observable signals.
   - Any Fold E claim exhibiting near-duplicate similarity to Fold A is excluded prior to metric calculation.
3. **Deterministic Reproducibility:**
   - All random seeds, prompt templates, and execution parameters are frozen in `taxonomy.py` and git-committed prior to any server execution.

---

## 5. Implementation & Execution CLI

- **Evidence Preparation Pipeline:** [`src/tffv/retrieval/prepare_fold_evidence.py`](file:///c:/Users/ASUS/Downloads/training-free-fact-verification/src/tffv/retrieval/prepare_fold_evidence.py)
  - Prepares R4 evidence for Fold E (`train:250..349`) into `outputs/m15-folde-r4/` using frozen BM25 + Qwen3-Embedding + Qwen3-Reranker-0.6B.
- **Evaluation Runner:** [`src/tffv/verification/m15_runner.py`](file:///c:/Users/ASUS/Downloads/training-free-fact-verification/src/tffv/verification/m15_runner.py)
  - Executes Conditions `M0`, `M1`, and `M4` (with optional `M2`, `M3` diagnostics).
  - Enforces anti-leakage quarantine against Fold A (`train:0..49`).
  - Computes exact McNemar tests, bootstrap CIs, Holm-Bonferroni corrections, and outputs [`M1_5_EVALUATION_REPORT.md`](file:///c:/Users/ASUS/Downloads/training-free-fact-verification/outputs/m15-folde-evaluation/M1_5_EVALUATION_REPORT.md).

