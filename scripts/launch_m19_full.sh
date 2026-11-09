#!/usr/bin/env bash
# Unattended background launcher for Milestone M1.9 Full Evaluation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m19-historical-outcome-eval-v1}"
PID_FILE="$OUTPUT_DIR/.m19.pid"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" || echo "")
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo "[Error] M1.9 evaluation is already running under PID $PID!"
        exit 1
    fi
    rm -f "$PID_FILE"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/m19_${TIMESTAMP}.log"

echo "======================================================================"
echo " Launching Milestone M1.9 Full Evaluation in Background"
echo " Host: $(hostname)"
echo " Log file: $LOG_FILE"
echo " Output directory: $OUTPUT_DIR"
echo "======================================================================"

nohup bash "$PROJECT_ROOT/scripts/run_m19_full_evaluation.sh" --run-gpu "$@" > "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"
echo "Process launched successfully with PID: $PID"
echo "Use 'bash scripts/status_m19_full.sh' to monitor execution."
