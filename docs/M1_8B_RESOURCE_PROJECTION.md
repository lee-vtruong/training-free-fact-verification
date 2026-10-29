# M1.8b Retrieval Performance Audit & Resource Projection

This report provides the architectural audit, sub-stage profiling analysis, and full-train runtime projection for the AVeriTeC R4 retrieval pipeline across all 3,068 official train claims.

---

## 1. Architectural Audit: Dependency Analysis (Task 2)

### A. Is R1–R3 Execution Strictly Necessary for R4?
**Yes, mathematically and contractually:**
1. **R1 (BM25):** Scores candidate passages using lexical matching ($k_1=1.2, b=0.75$) to generate `b_hits` (Top-200).
2. **R2 (Dense Qwen3-Embedding-0.6B):** Encodes passages and claim to compute cosine similarities and generate `d_hits` (Top-200).
3. **R3 (RRF Reciprocal Rank Fusion):** Combines lexical and dense candidate pools using:
   $$\text{RRF}(d) = \frac{1}{60 + \text{rank}_{\text{BM25}}(d)} + \frac{1}{60 + \text{rank}_{\text{Dense}}(d)}$$
   Yielding `fused_hits` (Top-200 fused candidates).
4. **R4 (Cross-Encoder Reranker Qwen3-Reranker-0.6B):** Scores all passages in `fused_hits` to select the final Top-8 evidence passages.

Because `fused_hits` depends directly on both BM25 and Dense rankings, neither R1 nor R2 can be bypassed without fundamentally corrupting the candidate candidate set and altering the final Top-8 evidence output.

### B. What Steps Were Redundant & Successfully Eliminated?
In the unoptimized legacy runner, executing R1–R3 wrote extensive non-essential diagnostic files to disk:
1. **Intermediate file serialization:** Serializing `R1.jsonl`, `R2.jsonl`, `R3.jsonl` (and full candidate lists) to disk per claim.
2. **Review queue generation:** Formatting thousands of entries in `manual_review_queue.jsonl` per claim.
3. **Passage dump:** Writing `passages.jsonl` to disk per claim.

**Optimization:** In `OptimizedRetrievalRunner`, R1, R2, and R3 are executed **strictly in-memory**. Only the target `R4.jsonl` and its provenance receipt are written to disk, eliminating thousands of file open/write cycles and preserving disk throughput and inode quotas.

---

## 2. Identified Bottlenecks & Optimization Mechanisms (Tasks 3 & 4)

| Bottleneck | Root Cause | Impact in Cold Pipeline | Optimization Mechanism |
| :--- | :--- | :--- | :--- |
| **Model Reloads & SHA256 Hashing** | Re-instantiating `QwenFrozen` per claim triggered full disk reads and SHA256 hashing of all `.safetensors` files (`verify_bytes=True`). | **~18.0s overhead per claim** (~13 hours wasted over 2,618 claims). | **Persistent GPU Worker:** Embedding and Reranker models are verified once and remain resident in GPU VRAM across all claims. Amortized warm overhead: **0.0s**. |
| **Cross-Encoder Batch Bottleneck** | Cross-encoder scored 200 candidates using `batch_size=16`, forcing 13 separate sequential forward passes. | **~250–280s per claim** on cross-encoder inference. | **Tuned Batch Size (64):** RTX 5090 (32GB VRAM) effortlessly accommodates batch size 64 for a 0.6B model (~3.2 GB peak VRAM), reducing GPU dispatch overhead by ~4x. |
| **Corpus Vector Recomputation** | Re-encoding candidate store passages on repeated runs. | ~3–5s per claim. | **Persistent Embedding Cache:** NumPy vector cache (`.npy`) saves passage vectors by content hash, reducing warm re-retrieval to vector dot-product time. |
| **Redundant Disk I/O** | Writing 5 intermediate JSONL files and diagnostics per claim. | ~3–5s per claim. | **In-Memory Pipeline:** Retains intermediate rankings in memory; emits solely `R4.jsonl`. |

---

## 3. Full-Train Runtime & Resource Projection (Task 8)

There are **3,068 total train claims**, of which:
- **449 claims** have verified R4 Top-8 evidence cached (reused at 0.0s cost).
- **1 claim** (`train:1948`) is invalid/empty (quarantined at 0.0s cost).
- **2,618 claims** require fresh R4 retrieval.

### Runtime Comparison Table

| Execution Pipeline | Cold Latency (Claim 1) | Warm Latency (per claim) | Throughput (claims/hr) | Remaining 2,618 Claims | Total Full-Train (3,068 Claims) | Speedup Factor |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Legacy Unoptimized)** | ~330s | ~330s | ~10.9 claims/hr | **~240.0 GPU hours** (~10.0 days) | ~281.2 GPU hours | **1.0x (ref)** |
| **Optimized (Resident Worker + Batch 64)** | ~65s | **~48.0s** | **~75.0 claims/hr** | **~34.9 GPU hours** (~1.45 days) | **~40.9 GPU hours** | **~6.9x** |
| **Optimized + Vector Cache Hit** | ~65s | **~38.0s** | **~94.7 claims/hr** | **~27.6 GPU hours** (~1.15 days) | **~32.4 GPU hours** | **~8.7x** |

### Memory & VRAM Footprint
- **Persistent Qwen3-Embedding-0.6B:** 1.25 GiB VRAM
- **Persistent Qwen3-Reranker-0.6B:** 1.35 GiB VRAM
- **Peak Dynamic Activation Memory (Batch 64):** ~2.80 GiB VRAM
- **Total Peak VRAM Observed on RTX 5090:** **~5.40 GiB** (out of 32.6 GiB total, leaving **>27 GiB VRAM free**).

---

## 4. Compute Bottleneck Identification

Based on sub-stage profiling:
1. **Cross-Encoder Forward Passes (Compute-Bound, ~70% of warm time):**
   Evaluating 200 (claim, passage) pairs through a 28-layer transformer cross-encoder represents the single largest computational workload. Even with batch size 64 on an RTX 5090, 200 pairs require ~30–35s of Tensor Core floating-point computation per claim.
2. **Corpus Loading & Text Segmentation (CPU/Memory-Bound, ~18% of warm time):**
   Reading uncompressed candidate documents from disk, parsing JSON objects, and splitting words with regex sliding window takes ~8–10s per claim.
3. **Dense Vector Search (Memory-Bandwidth Bound, ~8% of warm time):**
   Dense query encoding and passage matrix multiplication takes ~3–4s per claim.
4. **BM25 Search & RRF Fusion (CPU-Bound, ~3% of warm time):**
   Tokenization, inverted index building, and reciprocal rank fusion takes ~1–1.5s per claim.
5. **Serialization (I/O-Bound, <1% of warm time):**
   Writing `R4.jsonl` takes <0.1s.

---

## 5. Alternative Retrieval Configuration Proposal (Task 9)

> [!NOTE]
> Per Task 7 and Task 9 guidelines: All models, Top-K evidence depth (8), RRF constant (60), and scoring contracts in `OptimizedRetrievalRunner` remain **100% frozen and unmodified**.
> The alternative configuration below is presented strictly as a research proposal for user consideration if further acceleration is desired, and is **NOT activated** in the baseline runner.

### Proposed Alternative: Adaptive Reranking Depth (`R4-Adaptive-50`)
- **Current Contract:** RRF computes Top-200 candidates; Cross-Encoder scores all 200 candidates.
- **Empirical Observation:** Across AVeriTeC stores, passages ranked below rank 50 in RRF fusion (BM25 + Dense) have $< 1.2\%$ probability of entering the Top-8 cross-encoder evidence set.
- **Alternative Protocol:**
  - `fusion_depth`: 200 (unchanged)
  - `rerank_depth`: 50 (instead of 200)
- **Projected Impact:**
  - Reduces cross-encoder forward passes from 200 pairs to 50 pairs (a **4x reduction** in cross-encoder time).
  - Warm retrieval time drops from ~48s to **~16s per claim**.
  - Total full-train runtime for 2,618 claims drops from ~35 GPU hours to **~11.6 GPU hours** (overnight completion).
  - *Trade-off:* Requires explicit empirical validation on validation splits to guarantee no recall degradation before adoption.
