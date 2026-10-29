# M1.8a Server Execution Commands & Runbook

This runbook provides the exact bash commands to execute, monitor, and verify Milestone M1.8a on `hvtham-server`.

---

## 1. Connect & Update Codebase

```bash
# Connect to server
ssh stackops@hvtham-server

# Navigate to project repository and update branch
cd /home/stackops/whale/training-free-fact-verification
git fetch origin codex/m16-heal-design-integrity
git checkout codex/m16-heal-design-integrity
git pull origin codex/m16-heal-design-integrity

# Activate virtual environment
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
```

---

## 2. Preflight Audits & Verification (CPU Only, ~5 seconds)

Run the full integrity, cache scan, and preflight pipeline:

```bash
# 1. Audit full 3,068 train claims and train-dev overlap
python -m tffv.experiments.m18a audit-train

# 2. Scan server cache inventory for reusable R4 evidence and Base predictions
python -m tffv.experiments.m18a scan-cache

# 3. Build Historical Outcome Bank (CORRECT vs WRONG pools)
python -m tffv.experiments.m18a build-outcome-bank

# 4. Generate candidate corrections for WRONG pool
python -m tffv.experiments.m18a generate-candidates

# 5. Build initial expanded LLM-validated memory
python -m tffv.experiments.m18a build-memory

# 6. Verify Official Dev 500 readiness
python -m tffv.experiments.m18a verify-dev-readiness

# 7. Run automated CPU-only unit test suite
python -m pytest tests/test_m18a.py -v
```

---

## 3. Background Batch Execution (R4 Retrieval + Base Verifier)

To run a batch unattended in the background (e.g. claims 350 to 400):

```bash
# Launch batch in background with nohup
nohup bash scripts/run_m18a_batch.sh 350 400 outputs/m18a-batch-run > outputs/m18a-batch-run.log 2>&1 &

# Inspect background PID
echo "Launched batch runner PID: $!"
```

### Monitoring Progress & Hardware

```bash
# Tail live batch progress
tail -f outputs/m18a-batch-run.log

# Check completed claim count from state.json
python -c "import json; s=json.load(open('outputs/m18a-batch-run/state.json')); print('Completed:', len(s.get('completed', [])), 'Spent min:', round(s.get('spent_seconds', 0)/60, 1))"

# Check GPU VRAM and temperature
watch -n 2 nvidia-smi
```

---

## 4. Running Stronger LLM Judge (Qwen3-14B-AWQ)

Once candidates are prepared or batches complete:

```bash
# Step 1: Launch frozen Qwen3-14B-AWQ Judge server on port 17171
nohup python scripts/m17_qwen14b_judge.py --serve --port 17171 > outputs/m18a_judge_server.log 2>&1 &
JUDGE_PID=$!
echo "Judge server PID: $JUDGE_PID"

# Step 2: Verify server is listening
curl -s http://127.0.0.1:17171/ || echo "Server online"

# Step 3: Run two-stage LLM judge runner
python -m tffv.experiments.m18a run-judge --port 17171

# Step 4: Stop Judge server when evaluation finishes
kill $JUDGE_PID
```

---

## 5. Build Expanded LLM-Validated Memory

```bash
# Assemble certified memory bank and summarize coverage
python -m tffv.experiments.m18a build-memory

# Inspect summary
cat outputs/m18a-fulltrain-memory-v1/summary.json
```
