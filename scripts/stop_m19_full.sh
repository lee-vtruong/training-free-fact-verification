#!/usr/bin/env bash
# Graceful stop script for Milestone M1.9 Evaluation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m19-historical-outcome-eval-v1}"
PID_FILE="$OUTPUT_DIR/.m19.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo "Stopping M1.9 evaluation process $PID..."
        kill -TERM "$PID" 2>/dev/null || kill -9 "$PID" 2>/dev/null || true
        rm -f "$PID_FILE"
        echo "M1.9 evaluation process terminated."
    else
        echo "Process $PID is not running."
        rm -f "$PID_FILE"
    fi
else
    echo "No active M1.9 lockfile found."
fi
