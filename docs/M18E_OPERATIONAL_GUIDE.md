# Milestone M1.8e: Full-Train Historical Memory Construction & Unattended GPU Execution Guide

## 1. Overview & Research Contracts

Milestone M1.8e constructs the comprehensive Historical Error Memory for HEAL-B from all **3,068 AVeriTeC Official Train claims** under strict scientific integrity:
- **Total claims:** 3,068.
- **Valid processable claims:** 3,067.
- **Quarantined invalid claims:** `train:1948` (empty claim text `""`).
- **Dev leakage exclusions:** Four exact normalized train-dev claim overlaps identified in M1.8a audit (`train:1340`, `train:1341`, `train:1536`, `train:1539`) are strictly excluded from selectable memory.
- **Official Dev 500:** Strictly withheld; zero Dev gold labels or Dev text are accessed during memory construction.

### Frozen Component Specifications

1. **Frozen Retrieval Pipeline (`R4-PF1000-v1`):**
   - Per-claim knowledge store segmentation: 180 words, 30 overlap.
   - BM25 pre-filter: Top-1,000 candidate passages ($K=1000$).
   - Dense retrieval: Qwen3-Embedding-0.6B Top-200 within filtered pool.
   - Hybrid fusion: Reciprocal Rank Fusion (RRF, constant 60) between BM25 Top-200 and Dense Top-200.
   - Cross-Encoder reranking: Qwen3-Reranker-0.6B on Top-200 fused candidates.
   - Final evidence selection: Top-8 passages with deduplication threshold 0.95.
   - *Empirical efficiency:* 10.12x faster than canonical R0, reducing train processing from 7.7 days down to ~18 hours.
2. **Frozen Base Verifier:**
   - Model: `Qwen/Qwen3-8B` (BF16, greedy decoding, `enable_thinking=False`, `max_new_tokens=512`).
   - Prompt: Standard D0 prompt format.
   - *Gold isolation:* Strictly gold-blind inference; predictions persisted before accessing gold labels.
3. **Stronger LLM Judge:**
   - Model: Frozen `Qwen/Qwen3-14B-AWQ`.
   - Two-stage verification:
     * Stage 1: Gold-blind evidence adequacy & failure mechanism assessment.
     * Stage 2: Gold-aware correction validation with exact-quote substring matching against displayed evidence.
   - Provenance label: `LLM_VALIDATED` (never `HUMAN_VERIFIED`).

---

## 2. Unattended Server Execution & Automation

The pipeline is fully automated for background execution under process supervision, file-locking, and crash recovery.

### Launching the Pipeline
To launch the complete unattended pipeline in the background on `hvtham-server`:
```bash
bash scripts/launch_m18e_full.sh
```
*Features of launcher:*
- Resolves project root and activates `.venv`.
- Sets `HF_HOME="/home/stackops/whale/cache/huggingface"` and `HF_HUB_OFFLINE=1`.
- Acquires process lock via `.m18e.pid` to prevent concurrent colliding runs.
- Executes under `nohup` redirecting stdout/stderr to timestamped log files in `outputs/m18e-full-history-pf1000-v1/logs/`.
- Returns immediately, printing the active PID and monitoring commands.

### Monitoring Progress
To check live phase progression, claim counts, and GPU telemetry:
```bash
bash scripts/status_m18e_full.sh
```
Live tail of the execution log:
```bash
tail -f outputs/m18e-full-history-pf1000-v1/logs/m18e_*.log
```

### Stopping Execution Gracefully
To pause or stop the job without corrupting state:
```bash
bash scripts/stop_m18e_full.sh
```
Sends `SIGTERM` to the active PID, allows current claim to flush atomically to disk, and removes the lockfile.

### Resuming After Interruption ("Nhớ chạy có resum")
To resume a paused, interrupted, or restarted job:
```bash
bash scripts/resume_m18e_full.sh
```
The orchestrator reads existing `.jsonl` files (`retrieval_receipts.jsonl`, `base_predictions.jsonl`, `candidate_corrections.jsonl`, `judge_decisions.jsonl`) and `state.json`. Any claim already processed is skipped with zero redundant computation.

---

## 3. GPU VRAM Management & Phase Lifecycle

To prevent CUDA Out-of-Memory (OOM) on the RTX 5090 (32GB VRAM), the orchestrator strictly isolates GPU model residency across sequential phases:

```
[Phase A: Preflight] (CPU-only, scans manifests & cache inventory)
        │
[Phase B: PF1000 Retrieval] (Loads Qwen3-Embedding & Reranker ~3.5 GiB VRAM)
        │  ──> Unloads retrieval models & empties CUDA cache
[Phase C: Base Verifier] (Loads Qwen3-8B BF16 ~16 GiB VRAM)
        │  ──> Unloads Qwen3-8B & empties CUDA cache
[Phase D: Outcome Bank] (CPU-only, evaluates Accuracy & Macro-F1)
        │
[Phase E: Candidate Generation] (CPU-only, drafts grounded rationales)
        │
[Phase F: Stronger Judge] (Loads Qwen3-14B-AWQ ~9.5 GiB VRAM)
        │  ──> Unloads Judge model & empties CUDA cache
[Phase G: Memory Assembly] (CPU-only, excludes overlaps, labels LLM_VALIDATED)
        │
[Phase H: Validation & Reporting] (CPU-only, asserts completeness & exports Markdown)
```

At each phase boundary, models are deleted, `gc.collect()` is invoked, and `torch.cuda.empty_cache()` clears the PyTorch caching allocator. Peak allocated memory never exceeds ~16 GiB.

---

## 4. Deliverables & Output Inventory

All generated artifacts reside in `outputs/m18e-full-history-pf1000-v1/`:

| Artifact | Format | Description |
| :--- | :---: | :--- |
| `config_lock.json` | JSON | Frozen retrieval, verifier, judge, and exclusion hyperparameters |
| `cache_inventory.json` | JSON | Audit of reusable compatible PF1000 retrieval and Base predictions |
| `state.json` | JSON | Global execution state, phase statuses, and latency trackers |
| `retrieval_receipts.jsonl` | JSONL | Immutable Top-8 evidence receipts per claim with content hash |
| `base_predictions.jsonl` | JSONL | Gold-blind Base Qwen3-8B zero-shot prediction outputs |
| `outcome_bank.json` | JSON | Partitioned outcomes (CORRECT, WRONG, UNPARSEABLE), Accuracy, Macro-F1 |
| `candidate_corrections.jsonl` | JSONL | Grounded candidate rationales for WRONG claims |
| `judge_decisions.jsonl` | JSONL | Two-stage Qwen3-14B-AWQ Judge decisions with exact quote verification |
| `memory.json` | JSON | Final expanded LLM-validated memory bank excluding train-dev overlaps |
| `summary.json` | JSON | High-level execution metrics, latency, and sample counts |
| `INTEGRITY_AUDIT.md` | Markdown | Comprehensive audit certifying zero dev contamination and exact-quote validity |
| `FINAL_REPORT.md` | Markdown | Executive summary of full-train memory construction |
