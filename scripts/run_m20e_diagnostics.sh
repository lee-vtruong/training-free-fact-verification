#!/usr/bin/env bash
# ==============================================================================
# Diagnostics Script: Milestone M2.0e Baseline Reproducibility & Selective Feasibility
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m20e-selective-feasibility-v1}"
M20C_DIR="${M20C_DIR:-$PROJECT_ROOT/outputs/m20c-independent-holdout-v1}"
M20D_DIR="${M20D_DIR:-$PROJECT_ROOT/outputs/m20d-contrastive-verification-v1}"

mkdir -p "$OUTPUT_DIR"

PYTHON_BIN="python"
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
fi

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

echo "======================================================================"
echo " Running Milestone M2.0e Diagnostics & Reproducibility Suite (CPU-only)"
echo " Output Directory: $OUTPUT_DIR"
echo "======================================================================"

"$PYTHON_BIN" "$PROJECT_ROOT/src/tffv/experiments/m20e_runner.py" \
    --m20c-dir "$M20C_DIR" \
    --m20d-dir "$M20D_DIR" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
