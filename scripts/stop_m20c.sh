#!/usr/bin/env bash
# ==============================================================================
# Stop Script: Milestone M2.0c
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m20c-independent-holdout-v1}"
PID_FILE="$OUTPUT_DIR/.m20c.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Terminating M2.0c process $PID..."
        kill -TERM "$PID"
        sleep 2
        if kill -0 "$PID" 2>/dev/null; then
            echo "Process still active; sending SIGKILL..."
            kill -KILL "$PID"
        fi
        echo "Terminated."
    else
        echo "Process $PID is not running."
    fi
    rm -f "$PID_FILE"
else
    echo "No PID file found at $PID_FILE."
fi
