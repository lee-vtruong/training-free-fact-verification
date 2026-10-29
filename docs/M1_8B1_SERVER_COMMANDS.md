# M1.8b.1 Server Execution Guide — Real-GPU Equivalence & Throughput Benchmark

**Branch:** `codex/m16-heal-design-integrity`  
**Host:** `hvtham-server`  
**Working Directory:** `/home/stackops/whale/training-free-fact-verification`  
**Python Environment:** `/home/stackops/whale/venvs/errorflow/bin/python`

---

## Step 1: Commit & Push Changes Locally (On Laptop)

Review the staged files and push the M1.8b.1 implementation to GitHub:

```bash
git add configs/m18b1_benchmark_manifest.json \
        src/tffv/retrieval/profiler.py \
        src/tffv/retrieval/optimized_runner.py \
        scripts/benchmark_m18b1_real_gpu.py \
        tests/test_m18b1_integrity.py \
        docs/M1_8B1_BENCHMARK_SPEC.md \
        docs/M1_8B1_SERVER_COMMANDS.md \
        docs/WORK_LOG.md

git commit -m "feat(m18b1): add real-GPU R4 equivalence and throughput benchmark suite"
git push origin codex/m16-heal-design-integrity
```

---

## Step 2: Pull and Verify on `hvtham-server`

SSH into `hvtham-server` and verify the repository state:

```bash
cd /home/stackops/whale/training-free-fact-verification
git fetch origin codex/m16-heal-design-integrity
git checkout codex/m16-heal-design-integrity
git pull --ff-only origin codex/m16-heal-design-integrity

# Run unit tests in the server environment (CPU-only, ~1 second)
/home/stackops/whale/venvs/errorflow/bin/python -m pytest tests/test_m18b1_integrity.py tests/test_m18b_equivalence.py -v
```

---

## Step 3: Launch Benchmark in Background (`nohup`)

Execute the complete benchmark suite on the RTX 5090 GPU:

```bash
cd /home/stackops/whale/training-free-fact-verification
mkdir -p logs outputs/m18b1-real-benchmark

export PYTHONPATH=src
nohup /home/stackops/whale/venvs/errorflow/bin/python scripts/benchmark_m18b1_real_gpu.py \
    --manifest configs/m18b1_benchmark_manifest.json \
    --config configs/m161_source_batch25.json \
    --acq-plan configs/m161_acquisition_plan.json \
    --output-dir outputs/m18b1-real-benchmark \
    --batch-sizes 64,128 \
    --score-tolerance 1e-4 \
    > logs/m18b1_real_benchmark.log 2>&1 &

echo "Benchmark launched with PID: $!"
```

---

## Step 4: Monitor Progress & GPU Status

Monitor execution live:

```bash
# Follow logs
tail -f logs/m18b1_real_benchmark.log

# In another terminal: monitor GPU utilization and VRAM
watch -n 2 nvidia-smi
```

### Estimated Runtime
- **Part A (25 Canonical Equivalence Claims):** ~15–20 minutes.
- **Part B (25 Uncached Claims @ Batch 64):** ~20–25 minutes.
- **Part B2 (10 Claims @ Batch 128 Scaling Comparison):** ~6–8 minutes.
- **Total Expected Run Time:** **~40–50 minutes**.

---

## Step 5: Inspect Decision Report & Logs

Once the process finishes:

```bash
# Check overall exit status
tail -n 30 logs/m18b1_real_benchmark.log

# View the generated Decision Report
cat outputs/m18b1-real-benchmark/DECISION_REPORT.md

# View summary JSON metrics
cat outputs/m18b1-real-benchmark/decision.json
```

---

## Step 6: Return Artifacts to Assistant

Send the following output back to the assistant:
1. `outputs/m18b1-real-benchmark/DECISION_REPORT.md` (or copy its text).
2. The final 30 lines of `logs/m18b1_real_benchmark.log`.
3. If decision is `GO`, we will immediately proceed to launch full retrieval across all 2,618 uncached train claims.
