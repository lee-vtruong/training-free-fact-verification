#!/usr/bin/env bash
# ==============================================================================
# Execution Script: Milestone M1.9h CPU Diagnostics & Feasibility Pipeline
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT_DIR="${INPUT_DIR:-$PROJECT_ROOT/outputs/m19g-evidence-complete-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m19h-error-diagnostics-v1}"

echo "======================================================================"
echo " Launching Milestone M1.9h CPU Diagnostics Pipeline"
echo " Input Directory:  $INPUT_DIR"
echo " Output Directory: $OUTPUT_DIR"
echo " Mode: CPU-Only Deterministic Execution"
echo "======================================================================"

if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

python "$SCRIPT_DIR/run_m19h_diagnostics.py" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    "$@"

echo "[SUCCESS] Milestone M1.9h diagnostics pipeline finished successfully."
