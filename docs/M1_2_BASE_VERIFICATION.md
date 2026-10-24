# M1.2 — End-to-End Frozen Base Fact Verification Report

**Milestone:** M1.2  
**Status:** COMPLETED  
**Execution Date:** 2026-10-24 13:48:00 UTC  
**Target Hardware:** NVIDIA GeForce RTX 5090 (32 GB VRAM) on `hvtham-server`  
**Dataset:** AVeriTeC Train Pilot 100 Claims (`train:0` to `train:99`)  
**Ontology:** Official 4-Class (`Supported`, `Refuted`, `Not Enough Evidence`, `Conflicting Evidence/Cherrypicking`)  
**Strict Training-Free Constraint:** Zero fine-tuning, no optimizer, no loss/gradient backpropagation, no learned router. All backbone weights remained 100% frozen.

---

## 1. Executive Summary

Milestone M1.2 executes end-to-end frozen base fact verification on the 100 AVeriTeC pilot claims, consuming Top-8 retrieved passages from the M1.1 R4 frozen reranking pipeline (`outputs/m11-store-r14-pilot100-20261023-162716/R4.jsonl`).

Inference was conducted in complete isolation from ground truth: predictions were materialized, hashed, and sealed before gold labels were accessed from `/home/stackops/whale/ERRORFlow/data/raw/averitec/train.json`.

### Primary Results

| Metric | Measured Value | Scope / Notes |
|---|---:|---|
| **Evaluated Claims** | 100 | Exact pilot slice `train:0` .. `train:99` |
| **Overall Accuracy** | **28.0%** (28/100) | Strictly penalized for unparsed/truncated outputs |
| **Accuracy on Parsed Claims** | **43.1%** (28/65) | 65 claims completed before token budget cutoff |
| **Macro-F1** | **0.3076** | Macro average across all 4 classes |
| **Macro Precision** | **0.5023** | High precision on binary claims (`Refuted` 87.5%, `Supported` 80.0%) |
| **Macro Recall** | **0.2804** | Suppressed by reasoning truncation and under-retrieval |
| **Mean Latency per Claim** | 11.31s | RTX 5090, greedy decoding, bfloat16 |
| **Total Inference Time** | 1,130.64s (~18.8 min) | 100 claims sequentially processed |
| **Peak GPU VRAM** | 17.02 GB | Well within 32 GB capacity |

---

## 2. Model & Verification Protocol

### Backbone Specification
- **Model ID:** `Qwen/Qwen3-8B`
- **Revision:** `b968826d9c46dd6066d109eabc6255188de91218`
- **License:** Apache-2.0
- **Storage Location:** `/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218`
- **Architecture Enforcement:**
  - `model.eval()`
  - `model.requires_grad_(False)`
  - `torch.inference_mode()`
  - `temperature = 0.0` (deterministic greedy decoding, `do_sample = False`)
  - Precision: `torch.bfloat16`

### Zero-Shot Neutral Prompting
The model received an impartial prompt containing:
1. Impartial role description (verify factual claims strictly against provided evidence).
2. The claim text.
3. Top-8 evidence passages formatted with numbered headers and source URLs (maximum 1,200 characters per passage).
4. Official AVeriTeC 4-class verdict definitions.
5. Exact output formatting instructions: conclude on the final line with `Verdict: <Class>`.

Gold labels, gold QA pairs, and reference explanations were strictly excluded.

---

## 3. Evaluation Metrics & Confusion Matrix

### Per-Class Performance Breakdown

| Class | Ground Truth Support | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Supported** | 37 | 8 | 2 | 29 | **0.800** (80.0%) | 0.216 (21.6%) | 0.3404 |
| **Refuted** | 41 | 14 | 2 | 27 | **0.875** (87.5%) | 0.341 (34.1%) | **0.4912** |
| **Not Enough Evidence** | 9 | 3 | 22 | 6 | 0.120 (12.0%) | 0.333 (33.3%) | 0.1765 |
| **Conflicting Evidence/Cherrypicking** | 13 | 3 | 11 | 10 | 0.214 (21.4%) | 0.231 (23.1%) | 0.2222 |
| **Macro Average** | 100 | 28 | 37 | 72 | **0.5023** | **0.2804** | **0.3076** |

### Confusion Matrix (Ground Truth Rows $\times$ Predicted Columns)

| Ground Truth \ Predicted | Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking | Truncated / Unparsed | Class Support |
|---|---:|---:|---:|---:|---:|---:|
| **Supported** | **8** | 0 | 12 | 5 | 12 | 37 |
| **Refuted** | 1 | **14** | 7 | 5 | 14 | 41 |
| **Not Enough Evidence** | 1 | 0 | **3** | 1 | 4 | 9 |
| **Conflicting Evidence/Cherrypicking** | 0 | 2 | 3 | **3** | 5 | 13 |
| **Total Predicted** | 10 | 16 | 25 | 14 | 35 | 100 |

---

## 4. Key Behavioral Findings

1. **High Precision on Definite Claims:**
   When the model explicitly predicted `Refuted` (16 times), it was correct 14 times (**87.5% precision**). When it predicted `Supported` (10 times), it was correct 8 times (**80.0% precision**). There was only **one polarity reversal** across the entire dataset (`train:20`, gold `Refuted` predicted as `Supported`).

2. **Reasoning Truncation Bottleneck (35 claims):**
   `Qwen3-8B` natively generates `<think> ... </think>` reasoning traces. When analyzing 8 dense passages (~2,300 prompt tokens), the model methodically steps through passages 1 through 8 inside the reasoning trace. With a generation ceiling of `max_new_tokens=512`, the generation was truncated before the model emitted `Verdict: <Class>`. This accounts for 35 unparsed claims.

3. **Conservative Evidence Drift towards `Not Enough Evidence` (22 False Positives):**
   The model frequently falls back to `Not Enough Evidence` (25 total predictions, precision 12.0%). When evidence is partial or requires subtle synthesis, the model conservatively defaults to declaring evidence insufficient rather than committing to `Supported` (12 false negatives) or `Refuted` (7 false negatives).

4. **Nuance Confusion in `Conflicting Evidence/Cherrypicking`:**
   Out of 13 gold cherrypicking claims, only 3 were correctly classified (23.1% recall). 5 were truncated, 3 fell into `Not Enough Evidence`, and 2 were misclassified as `Refuted`.

---

## 5. Artifact Hashes & Exact Paths

| Artifact Description | Server Path | SHA-256 Digest |
|---|---|---|
| **Claims Input** | `outputs/m11-source-claims-20261021-055413/claims.jsonl` | `f1382c032a1c5746deae74fed4c6b870263754da0380c76b20f9e020bbaae52d` |
| **R4 Top-8 Evidence** | `outputs/m11-store-r14-pilot100-20261023-162716/R4.jsonl` | `f5e47540925d0b7e635d5c906aa12723d2fab04eb704449e83686375b5c11cb1` |
| **Gold Ground Truth** | `/home/stackops/whale/ERRORFlow/data/raw/averitec/train.json` | `ae5eda7c42ddf1695ef185a7ba1bc716928f5adf57103e4f78aae5f9afe00f9c` |
| **Base Predictions** | `outputs/m12-base-qwen3-8b-20261024-132900/predictions.jsonl` | `f519b6db9aa93000dcc530465e38de39585f55dfb5b933c137b2c6287dac4dc1` |
| **Base Manifest** | `outputs/m12-base-qwen3-8b-20261024-132900/manifest.json` | `4859a72df9b7203e05167ea82ebc82c2ffab67160cb65f1710ee9cfb237db875` |
| **Evaluation Summary** | `outputs/m12-eval-base-20261024-132900/evaluation_summary.json` | `37f59d5dc012a6114eb3897bda86c3db5a8cb5d5ecfe880f7690656041ec6057` |
| **Per-Claim CSV** | `outputs/m12-eval-base-20261024-132900/per_claim_evaluation.csv` | `0ba56b0cfa0fc9001b6e4ae15fca5f29910e53a5cce0220cb959ddaa2c3f8751` |
| **Group Safety Plan** | `outputs/m12-group-safety-20261024-132900/fold_isolation_plan.json` | `1482eae18c5c6d666d4bb3d26a34e3e8a6001a9d9ede672102d671a2b14ede09` |
