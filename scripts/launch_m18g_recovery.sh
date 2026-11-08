#!/usr/bin/env bash
# ==============================================================================
# Launch Script: Milestone M1.8g Substantive Correction Recovery & Memory Revalidation
# Background execution with sequential GPU isolation and crash-resumption support.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

M18E_DIR="${M18E_DIR:-$PROJECT_ROOT/outputs/m18e-full-history-pf1000-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m18g-correction-recovery-v1}"
PID_FILE="$OUTPUT_DIR/.m18g.pid"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$LOG_DIR"

# 1. Concurrency guard via lockfile
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" || echo "")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[ERROR] M1.8g recovery is already running under PID $OLD_PID."
        echo "Check status with: bash scripts/status_m18g_recovery.sh"
        echo "To stop it: bash scripts/stop_m18g_recovery.sh"
        exit 1
    else
        echo "[INFO] Stale lockfile found (PID $OLD_PID not active). Overwriting lock."
        rm -f "$PID_FILE"
    fi
fi

# 2. Environment activation
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -f "/home/stackops/whale/training-free-fact-verification/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "/home/stackops/whale/training-free-fact-verification/.venv/bin/activate"
fi

export HF_HOME="${HF_HOME:-/home/stackops/whale/cache/huggingface}"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/m18g_${TIMESTAMP}.log"

echo "======================================================================"
echo " Launching Milestone M1.8g (Substantive Correction Recovery)"
echo " Project root: $PROJECT_ROOT"
echo " M1.8e input: $M18E_DIR"
echo " Output dir : $OUTPUT_DIR"
echo " Log file   : $LOG_FILE"
echo "======================================================================"

nohup python "$SCRIPT_DIR/run_m18g_recovery.py" \
    --m18e-dir "$M18E_DIR" \
    --output-dir "$OUTPUT_DIR" \
    "$@" > "$LOG_FILE" 2>&1 &

NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

echo "[SUCCESS] Recovery pipeline launched in background under PID: $NEW_PID"
echo "Monitor live logs with: tail -f \"$LOG_FILE\""
echo "Check progress with: bash \"$SCRIPT_DIR/status_m18g_recovery.sh\""
echo "Stop execution with: bash \"$SCRIPT_DIR/stop_m18g_recovery.sh\""
