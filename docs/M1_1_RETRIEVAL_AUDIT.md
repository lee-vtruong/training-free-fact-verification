# M1.1 Retrieval Quality & Protocol Audit Report

**Date:** 2026-10-24  
**Target Milestone:** Milestone M1.1 — Training-Free Retrieval (AVeriTeC 100-Claim Pilot)  
**Evaluated Run Path:** `outputs/m11-store-r14-pilot100-20261023-162716`  
**Evaluation Path:** `outputs/m11-eval-pilot100-20261023-162716`  
**Source Dataset:** Official AVeriTeC `train.json` (`ae5eda7c42ddf1695ef185a7ba1bc716928f5adf57103e4f78aae5f9afe00f9c`)  
**Corpus Admission Status:** `false` (exploratory candidate store retrieval evaluation; unadmitted)  

---

## 1. Executive Summary & Verified Metrics Overview

Across the 100 pilot claims (`train:0` to `train:99`), all four retrieval pipelines were executed strictly within each claim's isolated candidate store. No gold labels, gold QA pairs, or justifications were accessed during retrieval or ranking.

| Method | Architecture / Adapter | Hit@5 | Hit@10 | Hit@20 | Hit@50 | Hit@100 | Recall@10 | Recall@100 | MRR@100 | Evidence Rec@8 | URL Coverage | Claim Oracle Hit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **R1** | BM25 (`k1=1.2, b=0.75`) | 2.0% | 5.0% | 9.0% | 15.0% | 26.0% | 0.035 | 0.146 | 0.0147 | 0.023 | 38.6% | 54.0% |
| **R2** | Frozen Dense (`Qwen3-Embedding-0.6B`) | 4.0% | 8.0% | 12.0% | 18.0% | 25.0% | 0.051 | 0.145 | 0.0316 | 0.051 | 38.6% | 54.0% |
| **R3** | RRF Hybrid (`constant=60`) | 5.0% | 9.0% | 13.0% | 22.0% | 27.0% | 0.051 | 0.154 | 0.0265 | 0.046 | 38.6% | 54.0% |
| **R4** | **Frozen Reranker** (`Qwen3-Reranker-0.6B`) | **8.0%** | **14.0%** | **19.0%** | **25.0%** | **30.0%** | **0.091** | **0.184** | **0.0478** | **0.081** | **38.6%** | **54.0%** |

---

## 2. Metric Definitions, Denominators, and Formulas

All evaluation metrics are computed post-hoc by `src/tffv/retrieval/quality_eval.py` against the sealed ranking outputs.

1. **`Hit@K` (Macro Average over Claims):**
   $$\text{Hit@K} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left( \text{Rank}_{\text{first gold URL}}(i) \le K \right)$$
   - **Denominator ($N$):** Exactly **100 annotated claims**. Every claim in `train:0`..`train:99` contains at least one ground-truth gold evidence URL ($n_{\text{gold}} \ge 1$).
   - **Binary indicator:** 1.0 if at least one retrieved passage in Top-$K$ originates from a ground-truth evidence URL; 0.0 otherwise.

2. **`Recall@K` (Macro-averaged URL Recall):**
   $$\text{Recall@K} = \frac{1}{N} \sum_{i=1}^{N} \frac{|\text{Retrieved Gold URLs in Top-}K(i)|}{|\text{Total Gold URLs Annotated for Claim } i|}$$
   - **Denominator for per-claim recall:** $|\text{Gold URLs}(i)|$.
   - **Denominator for macro aggregation:** $N = 100$.

3. **`MRR@100` (Mean Reciprocal Rank):**
   $$\text{MRR@100} = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{1}{\text{Rank}_{\text{first gold URL}}(i)} \text{ if } \text{Rank}_{\text{first gold URL}}(i) \le 100 \text{ else } 0 \right)$$
   - **Denominator:** $N = 100$. If a claim has no gold URL retrieved within the Top-100 candidates, its reciprocal rank is strictly 0.0.

4. **`Evidence Recall@8`:**
   $$\text{Evidence Recall@8} = \frac{1}{N} \sum_{i=1}^{N} \frac{|\text{Gold URLs in Top-8 Evidence Slots}(i)|}{|\text{Total Gold URLs Annotated for Claim } i|}$$
   - **Definition:** Measures the percentage of gold URLs captured in the **final Top-8 evidence passages** selected by `select_evidence` (which applies semantic near-deduplication with token Jaccard threshold $0.95$). This is the exact context delivered to downstream verification LLMs.

5. **`Store Coverage (URL-level)` vs `Claim-level Oracle Hit Ceiling` (Item 5):**
   - **URL-level Store Coverage (38.63%):**
     $$\text{Store Gold Coverage} = \frac{1}{N} \sum_{i=1}^{N} \frac{|\text{Unique Gold URLs Present in Candidate Store } i|}{|\text{Total Gold URLs Annotated for Claim } i|} = 0.38626$$
     *Meaning:* Across the 100 claims, only **38.6% of the annotated gold URLs** actually exist within the downloaded candidate stores. The remaining 61.4% were dropped, uncrawled, or excluded during the authors' crawling.
   - **Claim-level Oracle Hit Ceiling (54.0%):**
     $$\text{Store Gold Hit} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left( |\text{Gold URLs in Candidate Store } i| \ge 1 \right) = 0.540$$
     *Meaning:* **Only 54 out of 100 claims** contain at least one gold evidence URL in their candidate store. For the other 46 claims (46.0%), **zero gold evidence URLs exist in the store**. Thus, no retrieval algorithm can ever hit gold for those 46 claims.
   - **Relative Oracle Efficiency:** Out of the 54 reachable claims, R4 achieves Hit@100 on 30 claims, reaching **55.56% of the theoretical Oracle Hit ceiling** ($30 / 54$).

6. **URL Normalization Pipeline:**
   - Handled by `src/tffv/retrieval/url_norm.py`:
     1. RFC-compliant URL split (`urllib.parse.urlsplit`).
     2. Lowercases scheme, netloc, and path; collapses multi-slashes; strips trailing slashes and port defaults (`:80`, `:443`).
     3. Normalizes both `http` and `https` schemes to `https` for uniform web entity comparison.
     4. Strips leading `www.` subdomains.
     5. Filters out tracking parameters (`utm_*`, `fbclid`, `gclid`, `ref`, etc.) and canonicalizes remaining query parameters alphabetically.
     6. Completely discards URL fragments (`#...`).

---

## 3. Discrepancy Investigation: Why R3 Hit@100 = 27% while R4 Hit@100 = 30%

### Initial Apparent Contradiction
If R4 is a reranker applied to R3's Top-100 candidate pool, R4 can reorder candidates but cannot introduce new candidates. Therefore, if R4's pool were strictly R3's Top-100, `Hit@100(R4)` would have to be $\le \text{Hit@100(R3)}$. Why did R4 achieve 30% vs R3's 27%?

### Root-Cause Analysis in Retrieval Architecture
Inspection of `configs/store_retrieval_m11.template.json` and `src/tffv/retrieval/store_retrieval.py` reveals the exact mechanism:

```json
  "candidate_depth": 200,
  "fusion_depth": 200,
  "rerank_depth": 200,
  "ranking_store_depth": 100
```

1. **Fusion Pool Depth:**
   In `store_retrieval.py` (line 354):
   `fused_hits = rrf({"bm25": b_hits, "dense": d_hits}, fusion_depth=200, rrf_constant=60)`
   The RRF fusion produces a ranked list of **200 candidate passages**.
2. **Persistence Truncation for R3:**
   In `benchmark.py` (lines 99-105):
   `store_depth = config.get("ranking_store_depth", 100)`
   `saved_hits = hits[:store_depth]`
   When writing `R3.jsonl`, only the **Top-100** candidates (ranks 1..100) are persisted in `ranked_candidates`. Ranks 101..200 are truncated from `R3.jsonl`.
3. **Reranker Pool Depth for R4:**
   In `store_retrieval.py` (lines 365, 389-398):
   R4 receives `fused_hits`, which contains **all 200 candidates**!
   `reranker.score(claim["claim"], candidate_texts)` scores all 200 passages.
   `r4_hits = ordered(..., rerank_depth=200)` reorders all 200 candidates.
   Then `result_record` saves the Top-100 of these newly ordered candidates into `R4.jsonl`.

### Empirical Verification on Per-Claim Artifacts
Comparing per-claim rankings between R3 and R4 across the 100 claims identifies exactly 7 claims with differing Hit@100 indicators:

- **5 Claims Promoted from Ranks 101–200 into Top-100 (+5):**
  1. `train:0`: Gold passage `c6ccba5c...` had BM25 rank 118, Dense rank 135, and RRF rank > 100 (absent from R3 Top-100). R4 scored it $0.9241$, placing it at **Rank 83** in R4!
  2. `train:6`: Gold passage `c3b92bd5...` had Dense rank 90, BM25 miss, RRF rank > 100 (absent from R3 Top-100). R4 scored it $0.9823$, placing it at **Rank 31** in R4!
  3. `train:34`: Gold passage `60b6ee1e...` had Dense rank 105, BM25 miss, RRF rank > 100 (absent from R3 Top-100). R4 scored it $0.9508$, placing it at **Rank 18** in R4!
  4. `train:59`: Gold passage `e6f953f3...` had Dense rank 115, BM25 miss, RRF rank > 100 (absent from R3 Top-100). R4 scored it $0.0907$, placing it at **Rank 81** in R4!
  5. `train:62`: Gold passage `80de021c...` had Dense rank 64, BM25 miss, RRF rank > 100 (absent from R3 Top-100). R4 scored it $0.8093$, placing it at **Rank 23** in R4!

- **2 Claims Demoted past Rank 100 (-2):**
  1. `train:49`: Gold passage was at Rank 63 in R3. R4 gave it a low probability score, dropping it to rank > 100 in R4.
  2. `train:95`: Gold passage was at Rank 85 in R3. R4 dropped it to rank > 100 in R4.

**Net Arithmetic Proof:**
$$\text{Hit@100}(R4) = \text{Hit@100}(R3) + 5 - 2 = 27 + 5 - 2 = 30 \text{ claims } (30.0\%)$$

*Conclusion:* The discrepancy is fully accounted for. R4 did not rerank a Top-100 pool; **R4 reranked a Top-200 RRF pool**, effectively rescuing 5 gold passages that hybrid fusion had placed between ranks 101 and 200.

---

## 4. Verification of Frozen State & Absence of Parameter Training

To ensure strict compliance with project working rules (`Accuracy and Macro-F1 are primary. All models/features remain frozen; no parameter fitting, optimizers, neural routers or learned rerankers`):

1. **Checkpoint Identity & Pinning:**
   - **Local Path:** `/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473`
   - **Revision:** `e61197ed45024b0ed8a2d74b80b4d909f1255473`
   - **Content Fingerprint (SHA-256):** `5219a20fa423e1054c34b93a5a8c0ffa640ca52320181603be32d7afe26de353`
   - Validated by `tffv.retrieval.frozen.validate_spec` before model instantiation.

2. **Parameter Freezing Guarantees in `src/tffv/retrieval/frozen.py`:**
   - **Evaluation Mode:** Line 82 executes `self.model.to(device).eval()`.
   - **Gradient Disabling:** Line 83 executes `self.model.requires_grad_(False)`.
   - **Assertion Check:** Line 84-85 verifies:
     ```python
     if any(p.requires_grad for p in self.model.parameters()):
         raise RuntimeError("Failed to freeze parameters")
     ```
   - **Inference Context:** Line 140 wraps scoring in `with self.torch.inference_mode():`.

3. **Codebase-Wide Verification:**
   - Zero occurrences of `torch.optim`, `loss.backward()`, `optimizer.step()`, `.train()`, or parameter update loops anywhere in `src/tffv/`.
   - Model weights remain completely untouched and identical to the pinned snapshot.

---

## 5. Per-Claim Artifact Audit: 22 Improved, 10 Degraded, Net Rank Shift +221

Direct audit of `per_claim_metrics.csv` (`57299730b9cf54c0e57c3c7492d7ddeeb1358175089c1edec1a0c3135d6d0d57`) verifies:

### A. The 22 Improved Claims
- 17 claims where gold URL was in Top-100 of both R3 and R4, and R4 strictly improved the rank:
  - `train:1`: Rank 9 $\to$ Rank 2 (+7)
  - `train:3`: Rank 99 $\to$ Rank 15 (+84)
  - `train:7`: Rank 42 $\to$ Rank 26 (+16)
  - `train:12`: Rank 43 $\to$ Rank 2 (**+41, promoted to Top-10**)
  - `train:13`: Rank 78 $\to$ Rank 13 (+65)
  - `train:20`: Rank 17 $\to$ Rank 6 (**+11, promoted to Top-10**)
  - `train:23`: Rank 39 $\to$ Rank 5 (**+34, promoted to Top-10**)
  - `train:36`: Rank 77 $\to$ Rank 46 (+31)
  - `train:52`: Rank 4 $\to$ Rank 2 (+2)
  - `train:56`: Rank 47 $\to$ Rank 9 (**+38, promoted to Top-10**)
  - `train:60`: Rank 6 $\to$ Rank 2 (+4)
  - `train:63`: Rank 15 $\to$ Rank 6 (**+9, promoted to Top-10**)
  - `train:66`: Rank 3 $\to$ Rank 2 (+1)
  - `train:72`: Rank 23 $\to$ Rank 13 (+10)
  - `train:74`: Rank 23 $\to$ Rank 4 (**+19, promoted to Top-10**)
  - `train:75`: Rank 16 $\to$ Rank 6 (**+10, promoted to Top-10**)
  - `train:79`: Rank 17 $\to$ Rank 8 (**+9, promoted to Top-10**)
- 5 claims where gold URL was rescued from ranks 101–200 into Top-100:
  - `train:0`: None $\to$ Rank 83
  - `train:6`: None $\to$ Rank 31
  - `train:34`: None $\to$ Rank 18
  - `train:59`: None $\to$ Rank 81
  - `train:62`: None $\to$ Rank 23

### B. The 10 Degraded Claims & 3 Top-10 Demotions
- 8 claims where gold URL remained in Top-100 but dropped in rank:
  - `train:17`: Rank 32 $\to$ Rank 51 (-19)
  - `train:19`: Rank 7 $\to$ Rank 27 (**-20, demoted from Top-10**)
  - `train:26`: Rank 2 $\to$ Rank 3 (-1)
  - `train:35`: Rank 9 $\to$ Rank 65 (**-56, demoted from Top-10**)
  - `train:37`: Rank 41 $\to$ Rank 46 (-5)
  - `train:64`: Rank 4 $\to$ Rank 6 (-2)
  - `train:67`: Rank 30 $\to$ Rank 82 (-52)
  - `train:93`: Rank 5 $\to$ Rank 20 (**-15, demoted from Top-10**)
- 2 claims where gold URL was pushed from Top-100 into ranks > 100:
  - `train:49`: Rank 63 $\to$ None
  - `train:95`: Rank 85 $\to$ None

### C. Net Rank Shift Arithmetic
Sum of positive shifts: $+7 + 84 + 16 + 41 + 65 + 11 + 34 + 31 + 2 + 38 + 4 + 9 + 1 + 10 + 19 + 10 + 9 = \mathbf{+392}$  
Sum of negative shifts: $-19 - 20 - 1 - 56 - 5 - 2 - 52 - 15 = \mathbf{-171}$  
$$\text{Net Rank Shift} = +392 - 171 = \mathbf{+221 \text{ ranks}}$$

---

## 6. Provenance, Exact Paths, and Cryptographic Hashes

| Artifact Description | Canonical Path | Size (Bytes) | SHA-256 Checksum |
|---|---|---:|---|
| AVeriTeC Ground Truth | `/home/stackops/whale/ERRORFlow/data/raw/averitec/train.json` | 10,184,813 | `ae5eda7c42ddf1695ef185a7ba1bc716928f5adf57103e4f78aae5f9afe00f9c` |
| Candidate Manifest (100) | `outputs/m11-candidates-pilot100-20261023-161557/candidate_manifest.json` | 50,270 | `d6178d3bb879aea0d05fc67232f681ff6671076fea19017f6843072b5c49e54c` |
| Retrieval Run Manifest | `outputs/m11-store-r14-pilot100-20261023-162716/manifest.json` | 15,876 | `868d785dea491115e0caab8765247735d120b70039d80c8c35a8540adce7520f` |
| Retrieval Run Summary | `outputs/m11-store-r14-pilot100-20261023-162716/summary.json` | 43,943 | `5f28431f60f51c6265db9de1a5ed86c8125a32f9558b639040313f9984abcf8d` |
| R1 BM25 Rankings | `outputs/m11-store-r14-pilot100-20261023-162716/R1.jsonl` | 3,216,879 | `2a1efd4fca1ff752c2fce35ad59b851599a76569e58e2a2075723c7488f37f87` |
| R2 Dense Rankings | `outputs/m11-store-r14-pilot100-20261023-162716/R2.jsonl` | 3,284,662 | `0d0e016a61dfb997f7432e9e7828312870d47430e6518e537f6860d51ca48474` |
| R3 Hybrid Rankings | `outputs/m11-store-r14-pilot100-20261023-162716/R3.jsonl` | 4,029,435 | `4be064ff9bed737ad987038492e16393ff753069cbf5b51a755d91085bdc239f` |
| R4 Reranked Rankings | `outputs/m11-store-r14-pilot100-20261023-162716/R4.jsonl` | 4,478,551 | `f5e47540925d0b7e635d5c906aa12723d2fab04eb704449e83686375b5c11cb1` |
| Evaluation Summary | `outputs/m11-eval-pilot100-20261023-162716/evaluation_summary.json` | 8,136 | `92541a260593c527e310f461b0df187caeb4ca9d60a5d6937f84729fac8c3408` |
| Evaluation Markdown Report | `outputs/m11-eval-pilot100-20261023-162716/evaluation_report.md` | 2,514 | `d4e17cf1d58e6d9e2c27602f521f44bbe770644e55240eae6fb577f86867a8d5` |
| Per-Claim Detailed Metrics | `outputs/m11-eval-pilot100-20261023-162716/per_claim_metrics.csv` | 35,173 | `57299730b9cf54c0e57c3c7492d7ddeeb1358175089c1edec1a0c3135d6d0d57` |
| Evaluation Manifest | `outputs/m11-eval-pilot100-20261023-162716/manifest.json` | 731 | `51688fbe2f26c1b24a2d7c53e66101d3769002c811a44309e2316c65b43adae2` |

---

## 7. Recommended Next Milestone: Base Error Profiling & Fact Verification (M1.2)

Per project working rules:
> **Do not turn this project into a retrieval-only or neural routing optimization exercise.** Fact verification accuracy and macro-F1 across the AVeriTeC 4-class ontology remain the ultimate research objectives.

### Scope and Objectives for Milestone M1.2:
1. **Base Verification Pipeline with Frozen Qwen3-8B:**
   - Consume the Top-8 evidence passages already retrieved and persisted in `outputs/m11-store-r14-pilot100-20261023-162716/R4.jsonl` (and R1/R2/R3 for ablation).
   - Construct prompt templates strictly in zero-shot / frozen inference mode.
   - Run label-free prediction across the 4 official AVeriTeC classes: `Supported`, `Refuted`, `Not Enough Info`, `Conflicting Evidence/Cherrypicking`.

2. **Base Correct/Wrong Error Profiling:**
   - Audit predictions against gold labels post-inference:
     - Compute official Macro-F1 and 4-class confusion matrix.
     - Isolate exact subsets: $\mathcal{D}_{\text{correct}}$ vs $\mathcal{D}_{\text{error}}$.
     - Characterize failure modes: Does the model hallucinate when evidence is missing (`Not Enough Info` confusion), or does it fail on subtle refutations (`Refuted` vs `Supported`)?

3. **Training-Free Error-Informed In-Context Learning (ICL):**
   - Use the profiled error memories to design structured demonstration exemplars.
   - Enforce strict exclusion of evaluation folds from historical demonstration memory.
   - Measure whether error-informed prompting resolves the specific failure modes identified in Base error profiling, without any parameter updates or prompt fine-tuning.
