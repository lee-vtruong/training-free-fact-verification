#!/usr/bin/env bash
# ==============================================================================
# Execution Script: Milestone M1.8f Correction Yield, Judge Rejection & Memory Coverage Audit
# Strictly CPU-only. Inspects existing M1.8e artifacts without GPU invocation.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT_DIR="${INPUT_DIR:-$PROJECT_ROOT/outputs/m18e-full-history-pf1000-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m18f-correction-yield-audit-v1}"
TRAIN_JSON="${TRAIN_JSON:-$PROJECT_ROOT/data/raw/averitec/train.json}"
DEV_JSON="${DEV_JSON:-$PROJECT_ROOT/data/raw/averitec/dev.json}"

# Try server paths if project root files are absent
if [ ! -f "$TRAIN_JSON" ] && [ -f "/home/stackops/whale/training-free-fact-verification/data/raw/averitec/train.json" ]; then
    TRAIN_JSON="/home/stackops/whale/training-free-fact-verification/data/raw/averitec/train.json"
fi
if [ ! -f "$DEV_JSON" ] && [ -f "/home/stackops/whale/training-free-fact-verification/data/raw/averitec/dev.json" ]; then
    DEV_JSON="/home/stackops/whale/training-free-fact-verification/data/raw/averitec/dev.json"
fi

# Environment activation
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -f "/home/stackops/whale/training-free-fact-verification/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "/home/stackops/whale/training-free-fact-verification/.venv/bin/activate"
fi

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

echo "======================================================================"
echo " Launching Milestone M1.8f CPU Audit"
echo " Project root : $PROJECT_ROOT"
echo " Input dir    : $INPUT_DIR"
echo " Output dir   : $OUTPUT_DIR"
echo " Train data   : $TRAIN_JSON"
echo " Dev data     : $DEV_JSON"
echo "======================================================================"

python "$SCRIPT_DIR/run_m18f_audit.py" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --train-json "$TRAIN_JSON" \
    --dev-json "$DEV_JSON" \
    "$@"
