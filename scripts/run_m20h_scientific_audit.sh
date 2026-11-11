#!/usr/bin/env bash
# ==============================================================================
# Milestone M2.0h: Scientific Validation & Historical Error Knowledge Audit
# CPU-Only Deterministic Forensic & Verification Pipeline
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m20h-scientific-validation-v1}"
M20G_DIR="${M20G_DIR:-$PROJECT_ROOT/outputs/m20g-robustness-attribution-v1}"
TRAIN_RAW="${TRAIN_RAW:-$PROJECT_ROOT/data/raw/averitec/train.json}"

PYTHON_BIN="python"
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
fi

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

echo "======================================================================"
echo " Running Milestone M2.0h Scientific Validation Audit (CPU-Only)"
echo " M2.0g Directory: $M20G_DIR"
echo " Output Directory: $OUTPUT_DIR"
echo "======================================================================"

"$PYTHON_BIN" "$PROJECT_ROOT/src/tffv/experiments/m20h_scientific_audit.py" \
    --m20g-dir "$M20G_DIR" \
    --train-raw "$TRAIN_RAW" \
    --output-dir "$OUTPUT_DIR" \
    "$@"

echo "[SUCCESS] Milestone M2.0h scientific audit completed successfully."
