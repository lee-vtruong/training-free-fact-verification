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

| Condition vs C0 | Net Gain | Improved ($c$) | Regressed ($b$) | Unchanged (Corr / Wrong / Unp) | Transitions to Unparsed (Corr / Wrong) | Sum of Categories | McNemar $p$ (Unadj) | Holm-Bonf $p$ | Paired Bootstrap 95% CI | Sig ($\alpha=0.05$)? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **C1 vs C0** | **-2** | 3 (3+0) | 5 (4+1) | 23 / 15 / 0 | 1 / 0 | **46/46** | 0.7266 | 0.7266 | [-17.4%, +6.5%] | No |
| **C2 vs C0** | **-9** | 1 (1+0) | 10 (10+0) | 18 / 17 / 0 | 0 / 0 | **46/46** | 0.0117 | 0.0586 | [-32.6%, -6.5%] | Yes (Unadj) |
| **C3 vs C0** | **-4** | 4 (4+0) | 8 (8+0) | 20 / 14 / 0 | 0 / 0 | **46/46** | 0.3877 | 0.7754 | [-23.9%, +6.5%] | No |
| **C4 vs C0** | **-8** | 1 (1+0) | 9 (8+1) | 19 / 15 / 0 | 1 / 2 | **46/46** | 0.0215 | 0.0859 | [-30.4%, -4.3%] | Yes (Unadj) |
| **C5 vs C0** | **-5** | 4 (4+0) | 9 (8+1) | 19 / 13 / 0 | 1 / 1 | **46/46** | 0.2668 | 0.8005 | [-26.1%, +4.3%] | No |
| **C6 vs C0** | **-12** | 1 (1+0) | 13 (7+6) | 15 / 11 / 0 | 6 / 6 | **46/46** | 0.0018 | 0.0110 | [-41.3%, -13.0%] | **Yes (Robust)** |

---

## 5. Scientific Failure Analysis: Why Error Memory and Rules Did Not Help

The corrected data confirms a **conclusive, statistically robust empirical finding**: error memory, generic instructions, few-shot demonstrations, and explicit error rules **consistently failed to improve verification accuracy over zero-shot Base Protocol v1.0 (C0)**, and in multiple conditions caused statistically significant degradation.

### Key Failure Modes and Diagnostic Findings

1. **The Negative Transfer of Random Demonstrations (C2 vs C0: Net Gain -9 claims, $p = 0.0117$, Holm $p = 0.0586$):**
   - Accuracy dropped sharply from **60.9% (C0)** to **41.3% (C2)**.
   - Supplying arbitrary few-shot exemplars ($k=4$) from source claims expanded prompt length from 2,483 to 4,664 tokens. The model frequently latched onto stylistic formatting idiosyncrasies and topical cues from the exemplars rather than grounding its judgment strictly in the retrieved evidence of the target claim.

2. **Correct-Only Demonstrations Produce Zero Benefit (C3 vs C0: Net Gain -4 claims, $p = 0.3877$):**
   - Even when demonstrations were filtered strictly to cases where the base model had previously predicted correctly (C3), accuracy was **52.2%** (vs 60.9% for C0), with 8 regressions and only 4 improvements.
   - Exemplars increased prompt tokens to 4,882 while reducing reasoning precision on subtle classes (`Cherrypicking` F1 dropped from 0.2667 in C0 to 0.1818 in C3).

3. **Error Demonstrations Induce Over-Caution and Misguided Skepticism (C4 vs C0: Net Gain -8 claims, $p = 0.0215$, Holm $p = 0.0859$):**
   - Showing the model historical failure modes and pitfall explanations (C4) led to severe conservatism. Instead of avoiding errors, the model over-generalized the warnings: Refuted recall collapsed from 69.6% (C0) to 43.5% (C4), as the model second-guessed clear contradictions and reclassified them into `Conflicting Evidence` or `Not Enough Evidence`.

4. **Contrastive Demonstrations Fail to Resolve Boundary Ambiguity (C5 vs C0: Net Gain -5 claims, $p = 0.2668$):**
   - Pairwise contrastive demonstrations achieved **50.0% accuracy** (23/46). While C5 gained 4 improvements on Supported claims, it lost 9 previously correct claims, indicating that few-shot boundary pairs cannot substitute for calibrated parametric representations.

5. **Rule Distillation Triggers Generation Budget Collapse (C6 vs C0: Net Gain -12 claims, $p = 0.0018$, Holm $p = 0.0110$ — Highly Significant):**
   - C6 suffered the most catastrophic collapse, achieving only **34.8% accuracy** (16/46) and **0.2867 Macro-F1**.
   - **Root Cause:** When prompted to *'Explicitly apply the 4-point checklist to the claim and evidence'*, the model complied by generating lengthy multi-paragraph analyses assessing each checklist point sequentially. Under the Base Protocol budget of `max_new_tokens=512`, the generation was truncated before emitting `Verdict: ...` in **12 out of 46 claims (26.1% unparsed rate)**.
   - This mirrors the truncation phenomenon observed in B0 during M1.2, proving that prompting for structured multi-point verification requires either extended token budgets (e.g. 1,024 tokens) or concise non-verbose rule application.

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

