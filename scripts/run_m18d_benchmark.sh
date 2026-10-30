#!/usr/bin/env bash
# ==============================================================================
# Milestone M1.8d — Automated Retrieval Benchmark & Model Selection
#
# ONE-COMMAND REPRODUCIBLE BENCHMARK RUNNER
# Target Hardware: hvtham-server (NVIDIA RTX 5090, 32GB VRAM)
#
# Usage:
#   bash scripts/run_m18d_benchmark.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

export PYTHONPATH="src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1

mkdir -p logs outputs/m18d-retrieval-benchmark-v1

LOG_FILE="logs/m18d_benchmark_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "================================================================================"
echo "STARTING MILESTONE M1.8d: AUTOMATED RETRIEVAL BENCHMARK & MODEL SELECTION"
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "Host: $(hostname)"
echo "Git Commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
echo "Python: $(python3 --version)"
echo "================================================================================"

# Hardware pre-check
if command -v nvidia-smi &> /dev/null; then
    echo "--- GPU Hardware Status ---"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,temperature.gpu --format=csv,noheader
    echo "---------------------------"
else
    echo "⚠️ Warning: nvidia-smi not detected. Running without GPU telemetry."
fi

# Ensure stratified manifests exist
if [ ! -f "configs/m18d_manifests/pilot_manifest.json" ] || [ ! -f "configs/m18d_manifests/holdout_manifest.json" ]; then
    echo "Building stratified manifests..."
    python3 src/tffv/retrieval/m18d_manifest.py
fi

echo ""
echo "Executing unified benchmark pipeline (Pilot N=50, Holdout N=200, Qwen3-8B verifier)..."
echo "Log file: ${LOG_FILE}"
echo ""

python3 scripts/run_m18d_benchmark.py \
    --config configs/m161_source_batch25.json \
    --acq-plan configs/m161_acquisition_plan.json \
    --raw-train data/raw/averitec/train.json \
    --pilot-manifest configs/m18d_manifests/pilot_manifest.json \
    --holdout-manifest configs/m18d_manifests/holdout_manifest.json \
    --output-dir outputs/m18d-retrieval-benchmark-v1 \
    --dense-batch-size 64 \
    --rerank-batch-size 16

BENCHMARK_STATUS=$?

echo ""
echo "================================================================================"
if [ ${BENCHMARK_STATUS} -eq 0 ]; then
    echo "✅ M1.8d BENCHMARK COMPLETED SUCCESSFULLY!"
    echo "Decision Report: outputs/m18d-retrieval-benchmark-v1/DECISION_REPORT.md"
    echo "Decision Manifest: outputs/m18d-retrieval-benchmark-v1/decision_manifest.json"
    echo "================================================================================"
    
    # Display the final decision summary
    if [ -f "outputs/m18d-retrieval-benchmark-v1/DECISION_REPORT.md" ]; then
        echo ""
        head -n 25 "outputs/m18d-retrieval-benchmark-v1/DECISION_REPORT.md"
        echo ""
    fi
else
    echo "❌ M1.8d BENCHMARK FAILED (Exit Code: ${BENCHMARK_STATUS})"
    echo "Check logs at: ${LOG_FILE}"
    echo "================================================================================"
fi

exit ${BENCHMARK_STATUS}
