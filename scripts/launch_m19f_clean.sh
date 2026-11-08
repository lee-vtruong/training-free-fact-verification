#!/usr/bin/env bash
# Unattended background launcher for Milestone M1.9f Clean Official Dev 500 Evaluation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m19f-clean-official-dev-v1}"
PID_FILE="$OUTPUT_DIR/.m19f.pid"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" || echo "")
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo "[Error] M1.9f evaluation is already running under PID $PID!"
        exit 1
    fi
    rm -f "$PID_FILE"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/m19f_${TIMESTAMP}.log"

echo "======================================================================"
echo " Launching Milestone M1.9f Clean Official Dev Evaluation in Background"
echo " Host: $(hostname)"
echo " Log file: $LOG_FILE"
echo " Output directory: $OUTPUT_DIR"
echo "======================================================================"

PYTHON_BIN="python"
if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
fi

nohup "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_m19f_evaluation.py" \
    --output-dir "$OUTPUT_DIR" \
    --run-gpu "$@" > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"
echo "Process launched successfully with PID: $PID"
echo "Use 'bash scripts/status_m19f_clean.sh' to monitor execution."
