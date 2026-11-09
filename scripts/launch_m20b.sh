#!/usr/bin/env bash
# ==============================================================================
# Launch Script: Milestone M2.0b Instance-Grounded Error Intervention
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m20b-instance-grounded-v1}"
PID_FILE="$OUTPUT_DIR/.m20b.pid"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$LOG_DIR"

if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" || echo "")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[ERROR] M2.0b is already running under PID $OLD_PID."
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
LOG_FILE="$LOG_DIR/m20b_${TIMESTAMP}.log"

echo "======================================================================"
echo " Launching Milestone M2.0b (Instance-Grounded Error Intervention)"
echo " Output Directory: $OUTPUT_DIR"
echo " Log File:         $LOG_FILE"
echo "======================================================================"

nohup python "$SCRIPT_DIR/run_m20b_experiment.py" \
    --output-dir "$OUTPUT_DIR" \
    --run-gpu \
    "$@" > "$LOG_FILE" 2>&1 &

NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

echo "[SUCCESS] Milestone M2.0b launched in background under PID: $NEW_PID"
echo "Monitor progress with: bash scripts/status_m20b.sh"
