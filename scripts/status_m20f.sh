#!/usr/bin/env bash
# ==============================================================================
# Status Monitor: Milestone M2.0f Factorial Ablation
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m20f-factorial-ablation-v1}"

echo "=== M2.0f Factorial Ablation Job Status ==="
PID=$(pgrep -f "m20f_runner.py" || true)

if [ -n "$PID" ]; then
    echo "Process is RUNNING under PID: $PID"
else
    echo "Process is NOT running (finished, idle, or killed)."
fi

PREDS_FILE="$OUTPUT_DIR/predictions.jsonl"
if [ -f "$PREDS_FILE" ]; then
    COUNT=$(wc -l < "$PREDS_FILE")
    echo "Predictions written so far: $COUNT / 3300 (6 arms x 550 claims)"
else
    echo "Predictions file not created yet."
fi

LATEST_LOG=$(ls -t "$OUTPUT_DIR"/logs/m20f_*.log 2>/dev/null | head -n 1 || true)
if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
    echo "--- Recent Log Output ($LATEST_LOG) ---"
    tail -n 25 "$LATEST_LOG"
else
    echo "No log files found in $OUTPUT_DIR/logs/."
fi
