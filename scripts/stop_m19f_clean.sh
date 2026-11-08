#!/usr/bin/env bash
# Safe process stopper for Milestone M1.9f Clean Official Dev 500 Evaluation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m19f-clean-official-dev-v1}"
PID_FILE="$OUTPUT_DIR/.m19f.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "[Info] No active PID file found at $PID_FILE."
    exit 0
fi

PID=$(cat "$PID_FILE")
if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    echo "Stopping M1.9f evaluation process (PID: $PID)..."
    kill "$PID" || true
    sleep 2
    if kill -0 "$PID" 2>/dev/null; then
        echo "Process did not terminate gracefully, sending SIGKILL..."
        kill -9 "$PID" || true
    fi
    echo "Process $PID terminated."
else
    echo "[Info] Process $PID is not running."
fi

rm -f "$PID_FILE"
echo "PID file removed."
