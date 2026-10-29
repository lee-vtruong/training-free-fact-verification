# M1.8b Benchmark Execution Commands for Server

This document contains the exact commands for the user to execute the empirical R4 performance benchmark on `hvtham-server` (RTX 5090).

---

## 1. Connect & Update Server Codebase

```bash
# SSH to server
ssh stackops@hvtham-server

# Navigate to repository and pull the latest optimized code
cd /home/stackops/whale/training-free-fact-verification
git fetch origin codex/m16-heal-design-integrity
git checkout codex/m16-heal-design-integrity
git pull origin codex/m16-heal-design-integrity

# Activate project environment
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
```

---

## 2. Verify Equivalence & Test Suite (CPU-Only, ~4s)

Run both the equivalence tests and M1.8a integrity tests to confirm environment readiness:

```bash
python -m pytest tests/test_m18a.py tests/test_m18b_equivalence.py -v
```

---

## 3. Run Representative Benchmark (25 Uncached Claims)

Execute the instrumented benchmark across 25 representative uncached claims (spanning ordinals 350 to 3067):

```bash
# Run 25 representative claims with batch_size=64 cross-encoder
python scripts/benchmark_m18b.py \
  --claims 25 \
  --reranker-batch-size 64 \
  --output outputs/m18b-benchmark-v1
```

### To run in background with logging:

```bash
nohup python scripts/benchmark_m18b.py \
  --claims 25 \
  --reranker-batch-size 64 \
  --output outputs/m18b-benchmark-v1 > outputs/m18b-benchmark.log 2>&1 &

# Monitor progress
tail -f outputs/m18b-benchmark.log

# Monitor GPU VRAM and Utilization
watch -n 1 nvidia-smi
```

---

## 4. Inspect Generated Profiling Reports

Once the benchmark finishes, view the generated statistics:

```bash
# Summary JSON
cat outputs/m18b-benchmark-v1/summary.json

# Markdown Report
cat outputs/m18b-benchmark-v1/BENCHMARK_REPORT.md
```
