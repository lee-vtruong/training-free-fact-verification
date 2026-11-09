# M1.8b.1 Real-GPU R4 Equivalence & Throughput Benchmark Specification

**Repository:** `lee-vtruong/training-free-fact-verification`  
**Branch:** `codex/m16-heal-design-integrity`  
**Target Hardware:** `hvtham-server` (NVIDIA RTX 5090 32 GB, CUDA 12.8, PyTorch 2.6+, Transformers 4.51+)

---

## 1. Objective

Before launching full retrieval across the 2,618 uncached AVeriTeC train claims (~35 GPU hours), M1.8b.1 executes a rigorous empirical benchmark on `hvtham-server` with real GPU models (`Qwen/Qwen3-Embedding-0.6B` and `Qwen/Qwen3-Reranker-0.6B`) to guarantee:
1. **Zero Semantic Drift (Part A):** Strict mathematical equivalence of `OptimizedRetrievalRunner` against canonical R4 cache outputs across 25 representative edge-case claims.
2. **Empirical Throughput & Latency Profiling (Part B):** Precise measurement of cold vs warm latencies, sub-stage bottlenecks, batching scaling (`batch_size=64` vs `128`), and peak allocated/reserved VRAM across 25 uncached claims.
3. **Cache & Environment Integrity (Part C):** Verification of model revisions, tokenizers, vector cache isolation, and fail-closed guards.
4. **Deterministic Decision (Part D):** Automated generation of `DECISION_REPORT.md` emitting `GO`, `MODIFY`, or `NO-GO`.

---

## 2. Experimental Design & Selection Manifest

The benchmark is driven by the deterministic manifest at `configs/m18b1_benchmark_manifest.json`:

### Part A: 25 Canonical Cached Claims (Equivalence Audit)
Selected from verified historical R4 caches (`train:0..349` across Pilot 100, Fold C, Fold D, Fold E) covering 5 distinct edge-case categories:
- **Near-Ties (5 claims):** `train:12`, `train:47`, `train:118`, `train:211`, `train:275` (close reranker yes-probabilities between rank 8 and rank 10 candidates).
- **Long Passages (5 claims):** `train:3`, `train:38`, `train:142`, `train:229`, `train:312` (documents exceeding 500 words per passage).
- **Duplicate URLs (5 claims):** `train:7`, `train:64`, `train:155`, `train:204`, `train:289` (multiple passages originating from identical source domains/URLs).
- **Diverse Corpus Sizes (7 claims):** `train:2`, `train:19`, `train:85`, `train:171`, `train:240`, `train:301`, `train:345` (spanning small <50, medium 50–200, and large >200 passages).
- **Deduplication Suppressed (3 claims):** `train:28`, `train:91`, `train:186` (near-duplicate passages filtered by cosine similarity threshold 0.95).

### Part B: 25 Uncached Train Claims (Throughput Profiling)
Selected uniformly across ordinals 350 to 3000:
`[train:360, train:470, train:580, train:690, train:800, train:910, train:1020, train:1130, train:1240, train:1350, train:1460, train:1570, train:1680, train:1790, train:1900, train:2010, train:2120, train:2230, train:2340, train:2450, train:2560, train:2670, train:2780, train:2890, train:3000]`.
- **Invariants:** Mutually disjoint from Part A, verified non-empty, and strictly excludes invalid `train:1948` and quarantined cross-split claims (`train:406`, `train:1514`, `train:1796`, `train:2593`).

---

## 3. Strict 5-Point Equivalence Criteria (Part A)

For each claim in Part A, `compare_r4_equivalence` verifies:
1. **Passage Count:** Exactly 8 evidence passages.
2. **Passage IDs & Order:** `[p["passage_id"] for p in opt["evidence"]] == [p["passage_id"] for p in can["evidence"]]`.
3. **Byte-Identical Text Spans:** Every rendered snippet must be 100% byte-identical.
4. **Numerical Score Tolerance:** Maximum difference in reranker yes-probabilities $\le 1.00 \times 10^{-4}$ (accounting for FP16 non-associative accumulation across batch size 16 vs 64).
5. **Deduplication Equivalence:** Exact match in set of suppressed duplicate passage IDs.

**Non-Destructive Invariant:** Canonical artifacts are strictly read-only. Benchmark outputs are written exclusively to `outputs/m18b1-real-benchmark/part_a_equivalence/`.

---

## 4. Fine-Grained Profiling & VRAM Instrumentation (Part B)

`RetrievalProfiler` tracks:
- **Wall Time, CPU Time, and CUDA Event GPU Time:** Measured per sub-stage (`model_initialization`, `corpus_loading`, `bm25`, `dense_retrieval`, `rrf_fusion`, `cross_encoder_reranking`, `serialization`).
- **Memory Footprint:**
  - `peak_vram_allocated_bytes`: Peak allocated GPU tensors via `torch.cuda.max_memory_allocated()`.
  - `peak_vram_reserved_bytes`: Peak caching allocator reservation via `torch.cuda.max_memory_reserved()`.
  - `total_system_vram_bytes`: Total physical VRAM on device (32 GiB on RTX 5090).
- **Throughput Metrics:**
  - Cold startup overhead vs warm per-claim latency (mean, p50, p95).
  - Effective throughput in claims / hour ($3600 / \text{mean\_warm}$).
  - Projected wall time for full 2,618 uncached train claims.
- **Batch Size Scaling:** Evaluates `batch_size=64`, and conditionally benchmarks `batch_size=128` if peak VRAM remains $< 75\%$ system capacity.

---

## 5. Cache & Vector Isolation Protocol (Part C)

- Vector caches are saved with deterministic naming: `{sample_id}_{p_hash}.npy`.
- When loading from cache, `.npy` arrays must match shape `(len(passages), embed_dim)`. Any corrupted or mismatched array fails-closed and triggers automatic re-computation.
- Absolute isolation: No claim may reuse vector representations from another claim.

---

## 6. Decision Protocol (Part D)

Automated verdict emitted in `DECISION_REPORT.md`:
- **`GO`:** 100% equivalence match on Part A (0 mismatches), throughput measured, warm latency $\le 120$s/claim, VRAM safely within 32 GiB.
- **`MODIFY`:** Equivalence passed, but warm latency exceeds 120s or minor score drift requires threshold adjustment.
- **`NO-GO`:** Any passage ID or rank ordering mismatch, text alteration, or fatal execution crash. Full 2,618-claim execution remains blocked.
