# Experiment Roadmap & Milestone Tracking

## Overview & Working Principles
- **Accuracy and Macro-F1** across official AVeriTeC splits and the 4-class ontology are primary.
- **Strictly Training-Free:** All models remain frozen; no parameter fine-tuning, no optimizer, no neural router, no gradient updates.
- **Label-Free Inference:** Gold labels, gold QA, and explanations never influence retrieval or verification inference.
- **Group-Safe Knowledge Isolation:** Historical demonstration memory must strictly exclude evaluation fold claims.

---

## Milestone Status & Pipeline Overview

| Milestone | Scope / Objective | Status | Key Deliverables & Artifacts |
|---|---|:---:|---|
| **M1.0** | Work rules, audit scaffold, sealed provenance | **COMPLETE** | `AGENTS.md`, `src/tffv/provenance.py` |
| **M1.1** | Claim-scoped retrieval benchmark (R1-R4) on 100 claims | **COMPLETE** | Hit@10: R1 5.0%, R2 8.0%, R3 9.0%, R4 14.0%; `docs/M1_1_RETRIEVAL_AUDIT.md` |
| **M1.2** | Frozen Base Verification & Error Profiling (`Qwen3-8B`) | **COMPLETE** | Accuracy 28.0%, Macro-F1 0.3076, `docs/M1_2_BASE_VERIFICATION.md`, `docs/M1_2_ERROR_ANALYSIS.md` |
| **M1.3** | Training-Free Error-Informed In-Context Learning (ICL) | **PLANNED** | Demonstration composition (B0, B4, B5, B6), fold-local error rules |
| **M1.4** | Multi-Turn Verification & Evidence Compression | **PLANNED** | Token-budget expansion, sub-claim decomposition, evidence summarization |

---

## Milestone M1.1 Summary (Completed)
- **Evaluated Population:** 100 official AVeriTeC train claims (`train:0` to `train:99`), 74,748 crawled web documents (~6.31M passages).
- **Core Findings:**
  1. *Oracle Coverage Ceiling:* 38.6% URL-level coverage, 54.0% claim-level oracle hit ceiling.
  2. *Reranker Dominance:* Frozen `Qwen3-Reranker-0.6B` (R4) achieved 14.0% Hit@10 (vs 5.0% BM25, 9.0% RRF), 0.081 Evidence Recall@8, and net rank shift +221.
  3. *Full Audit:* Verified in `docs/M1_1_RETRIEVAL_AUDIT.md`.

---

## Milestone M1.2: Frozen Base Verification & Error Profiling (Completed)

### Results & Measured Metrics
- **Backbone:** `Qwen/Qwen3-8B` (Apache-2.0, Revision `b968826d9c46dd6066d109eabc6255188de91218`).
- **Hardware:** NVIDIA GeForce RTX 5090 (32 GB VRAM). Peak VRAM: 17.02 GB. Mean latency: 11.31s/claim.
- **Inference Mode:** Strict greedy decoding (`temperature=0.0`), `torch.inference_mode()`, `bfloat16`. 100% frozen backbone.
- **Primary Metrics:**
  - Overall Accuracy: **28.0%** (28/100)
  - Accuracy on Parsed Claims: **43.1%** (28/65)
  - Macro-F1: **0.3076** (Macro Precision: 0.5023, Macro Recall: 0.2804)
  - Class Precision: `Refuted` **87.5%**, `Supported` **80.0%**, `Cherrypicking` **21.4%**, `Not Enough Evidence` **12.0%**.
- **Key Behavioral Discoveries:**
  1. *Truncation Bottleneck:* 35 claims failed due to native `<think>` token budget exhaustion (`max_new_tokens=512`).
  2. *Conservative Evidence Drift:* 22 False Positives for `Not Enough Evidence` (model demands near-absolute proof).
  3. *Retrieval Discrepancies:* 9 claims failed despite having gold evidence in Top-8; 24 claims succeeded without gold URL due to wire/syndication redundancy.
  4. *Group Safety Contract Sealed:* 100 pilot claims designated as exploratory; future evaluation must draw strictly from held-out slices (`train:100` .. `train:3067` or dev split).
- **Artifacts:** `docs/M1_2_BASE_VERIFICATION.md`, `docs/M1_2_ERROR_ANALYSIS.md`.

---

## Milestone M1.3: Training-Free Error-Informed Verification (Proposed Next Experiments)

### Experiment 1: Output Budget & Generation Guidance Calibration
- Test concise rationale prompting (`"Provide a 2-sentence rationale directly; do not summarize every passage"`) and increase generation budget to 1,024 tokens to eliminate the 35-claim truncation bottleneck without any training.

### Experiment 2: Demonstration Composition from Historical Memory (ICL)
Compare zero-shot baseline against structured in-context exemplars drawn from group-safe historical memory:
- **B0:** Zero-shot Base (M1.2 baseline).
- **B4:** Correct-only demonstrations ($\mathcal{D}_{\text{correct}}$).
- **B5:** Error-only demonstrations with explicit gold corrections ($\mathcal{D}_{\text{error}}$).
- **B6:** Mixed contrastive demonstrations (correct case paired with corrected error case).

### Experiment 3: Contrastive Cherrypicking & Evidence Shortfall Checklists
- Inject explicit contrastive definitions and boundary examples specifically targeting the two dominant error modes:
  1. Distinguishing between `Not Enough Evidence` and `Supported`/`Refuted`.
  2. Identifying `Conflicting Evidence/Cherrypicking` (true premise + misleading framing).
