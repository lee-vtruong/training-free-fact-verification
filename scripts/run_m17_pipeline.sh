#!/usr/bin/env bash
# ==============================================================================
# Milestone M1.7 — End-to-End LLM-Judged Corrective Memory Pipeline
#
# Model: Qwen/Qwen3-14B-AWQ (AWQ 4-bit, revision 55182c306332ddb9f47ade5de8d8fdb53b8bcb03)
# Target: 21 SOURCE candidates (Gold-blind stage 1 + Grounded correction stage 2)
# ==============================================================================
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -f "$PWD/.venv/bin/activate" ]]; then
  source "$PWD/.venv/bin/activate"
fi
export PYTHONPATH="$PWD/src"

MODEL_ID="Qwen/Qwen3-14B-AWQ"
MODEL_REVISION="55182c306332ddb9f47ade5de8d8fdb53b8bcb03"
PORT=17171
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Stopping Judge Server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "================================================================================"
echo ">>> [M1.7] Step 1: Start Qwen3-14B-AWQ Judge Server"
echo "================================================================================"
python scripts/m17_qwen14b_judge.py --serve --port "$PORT" &
SERVER_PID=$!

echo "Waiting for Judge Server on port $PORT..."
for i in {1..30}; do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=1)" 2>/dev/null || \
     python -c "import socket; s=socket.socket(); s.connect(('127.0.0.1', $PORT)); s.close()" 2>/dev/null; then
    echo "Judge Server is online and ready!"
    break
  fi
  sleep 1
done

echo "================================================================================"
echo ">>> [M1.7] Step 2: Stage 1 — Gold-blind Evidence Assessment"
echo "================================================================================"
if [[ ! -d "outputs/m17-stage1-requests" ]] || [[ -z "$(ls -A outputs/m17-stage1-requests 2>/dev/null)" ]]; then
  python -m tffv.experiments.m17 make-requests \
    --evidence-bank "$PWD/outputs/m162a-inputs-v1/memory75.json" \
    --output outputs/m17-stage1-requests
fi

python scripts/m17_judge_adapter.py \
  --requests outputs/m17-stage1-requests/evidence_assessment_requests.jsonl \
  --output outputs/m17-stage1-raw.jsonl \
  --command "python scripts/m17_qwen14b_judge.py --port $PORT" \
  --model-id "$MODEL_ID" --model-revision "$MODEL_REVISION" \
  --temperature 0.0 --max-new-tokens 768 --resume

if [[ ! -d "outputs/m17-stage1-validated" ]]; then
  python -m tffv.experiments.m17 validate \
    --requests outputs/m17-stage1-requests/evidence_assessment_requests.jsonl \
    --outputs outputs/m17-stage1-raw.jsonl \
    --output outputs/m17-stage1-validated
fi

echo "================================================================================"
echo ">>> [M1.7] Step 3: Stage 2 — Correction Validation"
echo "================================================================================"
if [[ ! -d "outputs/m17-stage2-requests" ]] || [[ -z "$(ls -A outputs/m17-stage2-requests 2>/dev/null)" ]]; then
  python -m tffv.experiments.m17 make-requests \
    --evidence-bank "$PWD/outputs/m162a-inputs-v1/memory75.json" \
    --stage-one-outputs outputs/m17-stage1-validated/validated_outputs.jsonl \
    --output outputs/m17-stage2-requests
fi

python scripts/m17_judge_adapter.py \
  --requests outputs/m17-stage2-requests/correction_validation_requests.jsonl \
  --output outputs/m17-stage2-raw.jsonl \
  --command "python scripts/m17_qwen14b_judge.py --port $PORT" \
  --model-id "$MODEL_ID" --model-revision "$MODEL_REVISION" \
  --temperature 0.0 --max-new-tokens 768 --resume

if [[ ! -d "outputs/m17-stage2-validated" ]]; then
  python -m tffv.experiments.m17 validate \
    --requests outputs/m17-stage2-requests/correction_validation_requests.jsonl \
    --outputs outputs/m17-stage2-raw.jsonl \
    --output outputs/m17-stage2-validated
fi

echo "================================================================================"
echo ">>> [M1.7] Step 4: Build LLM-Validated Corrective Memory Bank"
echo "================================================================================"
if [[ ! -d "outputs/m17-llm-validated-memory-v1" ]]; then
  python -m tffv.experiments.m17 build-memory \
    --evidence-bank "$PWD/outputs/m162a-inputs-v1/memory75.json" \
    --stage-one outputs/m17-stage1-validated/validated_outputs.jsonl \
    --stage-two outputs/m17-stage2-validated/validated_outputs.jsonl \
    --output outputs/m17-llm-validated-memory-v1
fi

echo "================================================================================"
echo ">>> [M1.7] Summary Report:"
echo "================================================================================"
cat outputs/m17-llm-validated-memory-v1/summary.json
echo ""
echo "M1.7 Pipeline Completed Successfully!"
