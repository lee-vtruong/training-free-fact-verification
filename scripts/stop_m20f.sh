#!/usr/bin/env bash
# ==============================================================================
# Process Stopper: Milestone M2.0f Factorial Ablation
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m20f-factorial-ablation-v1}"
PID_FILE="$OUTPUT_DIR/.m20f.pid"

echo "=== Terminating M2.0f Factorial Ablation Job ==="

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Killing PID $PID from $PID_FILE..."
        kill -15 "$PID" 2>/dev/null || true
        sleep 2
        if kill -0 "$PID" 2>/dev/null; then
            echo "Force killing PID $PID..."
            kill -9 "$PID" 2>/dev/null || true
        fi
    fi
    rm -f "$PID_FILE"
fi

PIDS=$(pgrep -f "m20f_runner.py" || true)
if [ -n "$PIDS" ]; then
    echo "Killing remaining m20f_runner.py processes: $PIDS"
    kill -15 $PIDS 2>/dev/null || true
    sleep 2
    kill -9 $PIDS 2>/dev/null || true
fi

echo "[SUCCESS] M2.0f processes stopped."
