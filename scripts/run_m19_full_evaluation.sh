#!/usr/bin/env bash
# Production runner for Milestone M1.9 Historical Outcome Evaluation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

M18E_DIR="${M18E_DIR:-$PROJECT_ROOT/outputs/m18e-full-history-pf1000-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m19-historical-outcome-eval-v1}"
TRAIN_PATH="${TRAIN_PATH:-$PROJECT_ROOT/data/raw/averitec/train.json}"
DEV_PATH="${DEV_PATH:-$PROJECT_ROOT/data/raw/averitec/dev.json}"
K_VAL="${K_VAL:-2}"
SEED="${SEED:-42}"

# Fallback paths on server if needed
if [ ! -f "$DEV_PATH" ] && [ -f "/home/stackops/whale/ERRORFlow/data/raw/averitec/dev.json" ]; then
    DEV_PATH="/home/stackops/whale/ERRORFlow/data/raw/averitec/dev.json"
fi

echo "======================================================================"
echo " Starting Milestone M1.9 Historical Outcome Full Evaluation"
echo " Host: $(hostname)"
echo " Output Directory: $OUTPUT_DIR"
echo " Time: $(date)"
echo "======================================================================"

mkdir -p "$OUTPUT_DIR"

PYTHON_BIN="python"
if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
fi

CMD=(
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_m19_evaluation.py"
    --m18e-dir "$M18E_DIR"
    --output-dir "$OUTPUT_DIR"
    --train-path "$TRAIN_PATH"
    --dev-path "$DEV_PATH"
    --k "$K_VAL"
    --seed "$SEED"
)

# Append any arguments passed into the script (e.g. --run-gpu, --pilot-only)
CMD+=("$@")

"${CMD[@]}"

echo "======================================================================"
echo " M1.9 Execution Complete!"
echo " Results available in: $OUTPUT_DIR"
echo "======================================================================"
