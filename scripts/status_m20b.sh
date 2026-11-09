#!/usr/bin/env bash
# ==============================================================================
# Status Monitor: Milestone M2.0b Instance-Grounded Error Intervention
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m20b-instance-grounded-v1}"
PID_FILE="$OUTPUT_DIR/.m20b.pid"
PREDS_FILE="$OUTPUT_DIR/predictions.jsonl"
LOG_DIR="$OUTPUT_DIR/logs"

echo "=== M2.0b Job Status ==="
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Process is RUNNING under PID: $PID"
    else
        echo "Process with PID $PID is NOT running (finished or killed)."
    fi
else
    echo "No active PID file found at $PID_FILE."
fi

if [ -f "$PREDS_FILE" ]; then
    TOTAL_PREDS=$(wc -l < "$PREDS_FILE")
    echo "Predictions written so far: $TOTAL_PREDS"
else
    echo "No predictions file found yet."
fi

if [ -d "$LOG_DIR" ]; then
    LATEST_LOG=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -n 1 || true)
    if [ -n "$LATEST_LOG" ]; then
        echo "--- Recent Log Output ($LATEST_LOG) ---"
        tail -n 25 "$LATEST_LOG"
    fi
fi
