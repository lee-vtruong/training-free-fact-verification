# M1.3 P0 — Repaired Evaluation Report & Empirical Audit of Negative Results

**Milestone:** M1.3 (Training-Free Fact Verification) — Repaired Evaluation
**Date of Audit & Repair:** 2026-10-25
**Target Hardware:** NVIDIA GeForce RTX 5090 (32 GB VRAM) on `hvtham-server`
**Dataset Partition:** Disjoint Split-Half (Fold A `train:0..49` Source Memory | Fold B `train:50..99` Evaluation Pilot)
**Population:** 46 Admitted Evaluation Claims (4 near-duplicates quarantined)
**Core Research Principle:** Strictly Training-Free, Quality-First (Zero parameter updates, zero learned routers)

---

## 1. Executive Summary & Root Cause Audit of Reporting Bugs

Following execution of Milestone M1.3 on `hvtham-server`, an immediate P0 investigation was conducted to resolve severe reporting inconsistencies where the main table reported 0.0% accuracy, 0 Macro-F1, and 0 output tokens while paired transitions showed meaningful gains and losses.

### Root Causes Identified and Fully Repaired
1. **Report Generator Metric Parsing Collision:**
   - In `evaluate_condition()`, `run_evaluation()` writes and returns the nested dictionary from `evaluation_summary.json` (`{'metrics': {'accuracy': 0.6087, ...}}`).
   - In `m13_runner.py`, `condition_summaries[cond] = {'metrics': eval_summary, ...}` nested this again as `condition_summaries[cond]['metrics']['metrics']`.
   - `generate_m13_markdown_report()` attempted to look up `accuracy` directly from the outer dict via `s.get('metrics', {}).get('accuracy', 0.0)`. Finding no root-level key, it silently fell back to default `0.0%`, printing `0/0` valid verdicts and `0.0000` Macro-F1 across all rows.
   - **Repair:** Correctly unwrapped metric structures directly from the canonical evaluation records.

2. **Token Accounting Key Mismatch:**
   - `Qwen3FrozenBackbone.generate()` in `src/tffv/verification/backbone.py` returns token statistics with key `'generated_tokens'`: `len(gen_tokens)`.
   - In `m13_runner.py`, line 105 queried `response.get('generation_tokens')` (a typo looking for `'generation_tokens'` instead of `'generated_tokens'`). This returned `None`, which was dumped to `predictions.jsonl` as `null` and aggregated as `0` tokens.
   - **Repair:** Token accounting was re-derived directly from the verbatim raw generation texts using the pinned Qwen3-8B tokenizer (`cache/qwen3_tokenizer`). Exact output token counts are now reported for every claim and condition.

3. **Paired Transitions Table Column Omission:**
   - The internal transition calculator in `stabilization.py` correctly tracked all 9 mutually exclusive categories, but the markdown formatter in `m13_runner.py` only rendered 4 summary columns (`Improved`, `Regressed`, `Unchanged Correct`, `Unchanged Wrong`).
   - Any transitions involving `Unparsed` states (e.g. `wrong_to_unparsed`, which occurred 2 times in C4, 1 time in C5, and 6 times in C6) were excluded from the visible table columns, creating the illusion of missing claims (sums of 44/46, 45/46, and 40/46).
   - **Repair:** The paired comparison table now presents a full, mathematically closed 9-category breakdown where every condition sums identically to $46 \equiv N$.

---

## 2. Corrected Executive Performance Summary

| Condition | Description | Valid Verdicts | Accuracy | Macro-F1 | Macro-Prec | Macro-Recall | Mean Latency | Mean Tokens (Prompt / Gen) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **C0** | Base Protocol v1.0 (Zero-shot B1) | 46/46 (100.0%) | **60.9%** (28/46) | **0.4883** | 0.5028 | 0.6453 | 4.47s | 2483 / 192.3 |
| **C1** | Generic Task Instruction | 45/46 (97.8%) | **56.5%** (26/46) | **0.4664** | 0.5120 | 0.6294 | 6.10s | 2623 / 267.7 |
| **C2** | Random Source ICL ($k=4$) | 46/46 (100.0%) | **41.3%** (19/46) | **0.3578** | 0.4525 | 0.5359 | 5.90s | 4664 / 250.5 |
| **C3** | Correct-Only ICL ($k=4$) | 46/46 (100.0%) | **52.2%** (24/46) | **0.4185** | 0.4621 | 0.5828 | 3.21s | 4882 / 126.6 |
| **C4** | Error-Only ICL ($k=4$, with Gold Corrections) | 43/46 (93.5%) | **43.5%** (20/46) | **0.3689** | 0.4456 | 0.5277 | 7.57s | 4945 / 324.9 |
| **C5** | Contrastive Correct/Wrong ICL (2 pairs) | 44/46 (95.7%) | **50.0%** (23/46) | **0.3581** | 0.3918 | 0.5355 | 5.60s | 4845 / 234.0 |
| **C6** | Source-Derived Error Rules (4-point checklist) | 34/46 (73.9%) | **34.8%** (16/46) | **0.3508** | 0.4241 | 0.4594 | 10.25s | 2650 / 457.2 |

---

## 3. Corrected Per-Class F1 Breakdown across Conditions

| Condition | Supported F1 (P / R) | Refuted F1 (P / R) | Not Enough Evidence F1 (P / R) | Cherrypicking F1 (P / R) |
|---|---:|---:|---:|---:|
| **C0** | **0.6923** (0.82/0.60) | **0.7442** (0.80/0.70) | **0.2500** (0.14/1.00) | **0.2667** (0.25/0.29) |
| **C1** | **0.7143** (0.77/0.67) | **0.7027** (0.93/0.57) | **0.1818** (0.10/1.00) | **0.2667** (0.25/0.29) |
| **C2** | **0.5000** (0.54/0.47) | **0.5625** (1.00/0.39) | **0.1333** (0.07/1.00) | **0.2353** (0.20/0.29) |
| **C3** | **0.6897** (0.71/0.67) | **0.6667** (0.92/0.52) | **0.2000** (0.11/1.00) | **0.1176** (0.10/0.14) |
| **C4** | **0.5517** (0.57/0.53) | **0.6061** (1.00/0.43) | **0.2000** (0.11/1.00) | **0.1176** (0.10/0.14) |
| **C5** | **0.5714** (0.62/0.53) | **0.7179** (0.88/0.61) | **0.1429** (0.08/1.00) | **0.0000** (0.00/0.00) |
| **C6** | **0.5517** (0.57/0.53) | **0.4516** (0.88/0.30) | **0.4000** (0.25/1.00) | **0.0000** (0.00/0.00) |

---

## 4. Full Mutually Exclusive Paired Transitions (All 9 Categories vs C0)

All comparisons are paired claim-for-claim against Baseline **C0** ($N=46$ claims, C0 Correct = 28, C0 Wrong = 18, C0 Unparsed = 0).
Every claim belongs to exactly one category, and the 9 mutually exclusive categories sum identically to $46$ for every condition.

| Condition vs C0 | C→C | C→W | C→U | W→C | W→W | W→U | U→C | U→W | U→U | Sum | Net Gain | McNemar $p$ (Unadj) | Holm-Bonf $p$ | Paired Bootstrap 95% CI | Sig ($\alpha=0.05$)? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **C1 vs C0** | 23 | 4 | 1 | 3 | 15 | 0 | 0 | 0 | 0 | **46** | **-2** | 0.7266 | 0.7266 | [-17.4%, +6.5%] | No |
| **C2 vs C0** | 18 | 10 | 0 | 1 | 17 | 0 | 0 | 0 | 0 | **46** | **-9** | 0.0117 | 0.0586 | [-32.6%, -6.5%] | Yes (Unadj) |
| **C3 vs C0** | 20 | 8 | 0 | 4 | 14 | 0 | 0 | 0 | 0 | **46** | **-4** | 0.3877 | 0.7754 | [-23.9%, +6.5%] | No |
| **C4 vs C0** | 19 | 8 | 1 | 1 | 15 | 2 | 0 | 0 | 0 | **46** | **-8** | 0.0215 | 0.0859 | [-30.4%, -4.3%] | Yes (Unadj) |
| **C5 vs C0** | 19 | 8 | 1 | 4 | 13 | 1 | 0 | 0 | 0 | **46** | **-5** | 0.2668 | 0.8005 | [-26.1%, +4.3%] | No |
| **C6 vs C0** | 15 | 7 | 6 | 1 | 11 | 6 | 0 | 0 | 0 | **46** | **-12** | 0.0018 | 0.0110 | [-41.3%, -13.0%] | **Yes (Robust)** |

---

## 5. Empirical Failure Analysis: Controlled Diagnostic Observations

The canonical data demonstrates that none of the tested error-informed or demonstration-augmented protocols (C1–C6) outperformed the frozen zero-shot Base Protocol v1.0 (C0: 60.9% accuracy, 0.4883 Macro-F1). Below are the empirical observations and structural error breakdowns:

### Key Observations and Breakdown of Degradation

1. **Random Source Demonstrations (C2 vs C0: Net Gain -9 claims, $p = 0.0117$, Holm $p = 0.0586$):**
   - Accuracy dropped from **60.9% (C0)** to **41.3% (C2)**.
   - Adding $k=4$ source demonstrations increased prompt length from 2,483 to 4,664 tokens. Observational accuracy regressed across both Supported (F1 0.6923 $\to$ 0.5000) and Refuted (F1 0.7442 $\to$ 0.5625) classes.

2. **Correct-Only Demonstrations (C3 vs C0: Net Gain -4 claims, $p = 0.3877$):**
   - Restricting demonstrations to cases where the base model previously predicted correctly on source claims yielded **52.2% accuracy** (24/46), with 8 regressions ($C \to W$) and 4 improvements ($W \to C$).
   - The difference vs C0 is not statistically significant ($p = 0.3877$), showing no measurable benefit over zero-shot inference under this prompt structure.

3. **Error Demonstrations with Gold Corrections (C4 vs C0: Net Gain -8 claims, $p = 0.0215$, Holm $p = 0.0859$):**
   - Including error demonstrations and correction notes shifted the predicted label distribution: predictions of `Not Enough Evidence` rose from 7 in C0 to 14 in C4, while Refuted recall dropped from 69.6% (16/23) to 43.5% (10/23).
   - Total correct claims fell from 28 to 20 ($8 C \to W$, $1 C \to U$, against $1 W \to C$).

4. **Contrastive Demonstrations (C5 vs C0: Net Gain -5 claims, $p = 0.2668$):**
   - Pairwise contrastive demonstrations achieved **50.0% accuracy** (23/46). While 4 claims improved from wrong to correct, 9 claims regressed (8 $C \to W$, 1 $C \to U$), yielding no overall advantage over C0.

5. **Distilled Error Rules (C6 vs C0: Net Gain -12 claims, $p = 0.0018$, Holm $p = 0.0110$):**
   - C6 reached **34.8% accuracy** (16/46) and **0.3508 Macro-F1**.
   - **Decomposition of C6 Degradation (Truncation vs Semantic Classification):**
     * **Output Truncation (6 claims lost):** Instructing the model to *'Explicitly apply the 4-point checklist to the claim and evidence'* prompted verbose multi-point reasoning traces. Under `max_new_tokens=512`, generation was truncated before emitting a verdict in **12 out of 46 claims (26.1% unparsed rate)**. Crucially, 6 of these 12 unparsed claims were previously correct in C0 ($C \to U = 6$), directly accounting for half of the net 12-claim deficit.
     * **Semantic Classification on Parsed Subset (6 claims lost):** On the 34 claims where the model successfully emitted a parsed verdict, 16 were correct ($16/34 = 47.1\%$ conditional accuracy, compared to $22/34 = 64.7\%$ in C0 on the exact same 34 claims). Specifically, 7 claims regressed from correct to wrong ($C \to W = 7$) while only 1 improved ($W \to C = 1$), representing a net loss of 6 claims.
     * **Conclusion:** The degradation in C6 was a compound effect of roughly equal parts generation truncation budget exhaustion (50%) and semantic classification shift (50%), rather than being attributable solely to error knowledge content.

---

## 6. Data Integrity & Leakage Verification

- **Evaluation Population:** Exactly 46 claims (`train:50` .. `train:99` minus 4 quarantined near-duplicates: `train:67, train:70, train:77, train:95`).
- **Alignment:** All seven conditions evaluated the exact same 46 claim IDs in identical order.
- **Exemplar Disjointness:** Verified that 100% of exemplars used across C2, C3, C4, and C5 were drawn exclusively from Fold A (`train:0` .. `train:49`). Exemplar overlap with evaluation claims is identically **0 claims (0.0%)**.
- **Gold Label Sealing:** Evaluation gold labels were loaded strictly after prediction files were written and sealed.

---

## 7. Canonical Artifacts & Repaired Output Manifest

- Canonical Per-Claim Table: `outputs/m13-repaired-audit/canonical_per_claim_predictions.csv`
- Repaired Audit Summary JSON: `outputs/m13-repaired-audit/repaired_audit_summary.json`
- Repaired Evaluation Report: `docs/M1_3_REPAIRED_EVALUATION_REPORT.md`

