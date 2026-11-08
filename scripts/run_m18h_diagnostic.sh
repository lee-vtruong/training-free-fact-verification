#!/usr/bin/env bash
# Milestone M1.8h: Rejection Root-Cause Analysis & Evidence Recoverability Runner
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

M18E_DIR="${M18E_DIR:-$PROJECT_ROOT/outputs/m18e-full-history-pf1000-v1}"
M18G_DIR="${M18G_DIR:-$PROJECT_ROOT/outputs/m18g-correction-recovery-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m18h-rejection-diagnostic-v1}"
TRAIN_PATH="${TRAIN_PATH:-$PROJECT_ROOT/data/raw/averitec/train.json}"
DEV_PATH="${DEV_PATH:-$PROJECT_ROOT/data/raw/averitec/dev.json}"
SAMPLE_SIZE="${SAMPLE_SIZE:-150}"
SEED="${SEED:-42}"

# Fallback paths on server if needed
if [ ! -f "$DEV_PATH" ] && [ -f "/home/stackops/whale/ERRORFlow/data/raw/averitec/dev.json" ]; then
    DEV_PATH="/home/stackops/whale/ERRORFlow/data/raw/averitec/dev.json"
fi

echo "======================================================================"
echo " Starting Milestone M1.8h Diagnostic & Recoverability Runner"
echo " Host: $(hostname)"
echo " Output Directory: $OUTPUT_DIR"
echo " Sample Size: $SAMPLE_SIZE (Seed: $SEED)"
echo "======================================================================"

mkdir -p "$OUTPUT_DIR"

PYTHON_BIN="python"
if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
fi

CMD=(
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_m18h_diagnostic.py"
    --m18e-dir "$M18E_DIR"
    --m18g-dir "$M18G_DIR"
    --output-dir "$OUTPUT_DIR"
    --train-path "$TRAIN_PATH"
    --dev-path "$DEV_PATH"
    --sample-size "$SAMPLE_SIZE"
    --seed "$SEED"
)

# Pass optional GPU intervention if requested
if [[ "${1:-}" == "--run-gpu-intervention" ]]; then
    CMD+=("--run-gpu-intervention")
fi

"${CMD[@]}"

echo "======================================================================"
echo " M1.8h Execution Completed Successfully!"
echo " Artifacts available in: $OUTPUT_DIR"
echo "======================================================================"
