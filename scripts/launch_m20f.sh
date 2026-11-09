#!/usr/bin/env bash
# ==============================================================================
# Background Launcher: Milestone M2.0f Factorial Ablation
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m20f-factorial-ablation-v1}"
PID_FILE="$OUTPUT_DIR/.m20f.pid"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$LOG_DIR"

if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" || echo "")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[ERROR] M2.0f is already running under PID $OLD_PID."
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

PYTHON_BIN="python"
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
fi

export HF_HOME="${HF_HOME:-/home/stackops/whale/cache/huggingface}"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/m20f_${TIMESTAMP}.log"

echo "======================================================================"
echo " Launching Milestone M2.0f Factorial Ablation across 6 Arms"
echo " Output Directory: $OUTPUT_DIR"
echo " Log File:         $LOG_FILE"
echo "======================================================================"

nohup "$PYTHON_BIN" "$PROJECT_ROOT/src/tffv/experiments/m20f_runner.py" \
    --output-dir "$OUTPUT_DIR" \
    --run-gpu \
    "$@" > "$LOG_FILE" 2>&1 &

NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

echo "[SUCCESS] Milestone M2.0f launched in background under PID: $NEW_PID"
echo "Monitor progress with: bash scripts/status_m20f.sh"
