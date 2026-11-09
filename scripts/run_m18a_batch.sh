#!/usr/bin/env bash
# ==============================================================================
# Milestone M1.8a — Full-Train Batch Retrieval & Base Inference Runner
# ==============================================================================
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -f "$PWD/.venv/bin/activate" ]]; then
  source "$PWD/.venv/bin/activate"
fi
export PYTHONPATH="$PWD/src"

START=${1:-350}
END=${2:-400}
BATCH_DIR=${3:-"outputs/m18a-batch-run"}

echo "================================================================================"
echo ">>> [M1.8a] Starting Batch Execution for Claims ${START}..${END}"
echo ">>> Output Directory: ${BATCH_DIR}"
echo ">>> Timestamp: $(date --iso-8601=seconds)"
echo "================================================================================"

python -m tffv.experiments.m18a retrieve-and-base \
  --start "$START" \
  --end "$END" \
  --batch-dir "$BATCH_DIR"

echo "================================================================================"
echo ">>> [M1.8a] Batch ${START}..${END} Finished at $(date --iso-8601=seconds)"
echo "================================================================================"
