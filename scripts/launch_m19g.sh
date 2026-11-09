#!/usr/bin/env bash
# ==============================================================================
# Launch Script: Milestone M1.9g Evidence-Complete Official Dev 500 Evaluation
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m19g-evidence-complete-v1}"
PID_FILE="$OUTPUT_DIR/.m19g.pid"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$LOG_DIR"

if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" || echo "")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[ERROR] M1.9g is already running under PID $OLD_PID."
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

export HF_HOME="${HF_HOME:-/home/stackops/whale/cache/huggingface}"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/m19g_${TIMESTAMP}.log"

echo "======================================================================"
echo " Launching Milestone M1.9g (Evidence-Complete Official Dev 500)"
echo " Output Directory: $OUTPUT_DIR"
echo " Log File: $LOG_FILE"
echo "======================================================================"

nohup python "$SCRIPT_DIR/run_m19g_evaluation.py" \
    --output-dir "$OUTPUT_DIR" \
    --run-gpu \
    "$@" > "$LOG_FILE" 2>&1 &

NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

echo "[SUCCESS] Evaluation pipeline launched in background under PID: $NEW_PID"
echo "Check status with: bash scripts/status_m19g.sh"
