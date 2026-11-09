#!/usr/bin/env bash
# ==============================================================================
# Full Factorial Ablation Execution: Milestone M2.0f
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m20f-factorial-ablation-v1}"
mkdir -p "$OUTPUT_DIR"

PYTHON_BIN="python"
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
fi

export HF_HOME="${HF_HOME:-/home/stackops/whale/cache/huggingface}"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

echo "======================================================================"
echo " Executing Milestone M2.0f Factorial Ablation (6 Arms x 550 Claims)"
echo " Output Directory: $OUTPUT_DIR"
echo "======================================================================"

"$PYTHON_BIN" "$PROJECT_ROOT/src/tffv/experiments/m20f_runner.py" \
    --output-dir "$OUTPUT_DIR" \
    --run-gpu \
    "$@"
