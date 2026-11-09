#!/usr/bin/env bash
# Graceful stop script for Milestone M1.8g Recovery
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m18g-correction-recovery-v1}"
PID_FILE="$OUTPUT_DIR/.m18g.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "[INFO] No active PID file found at $PID_FILE."
    exit 0
fi

PID=$(cat "$PID_FILE" || echo "")
if [ -z "$PID" ]; then
    echo "[INFO] PID file is empty."
    rm -f "$PID_FILE"
    exit 0
fi

if kill -0 "$PID" 2>/dev/null; then
    echo "[INFO] Sending SIGTERM to PID $PID for graceful shutdown..."
    kill -TERM "$PID"

    # Wait up to 30 seconds for graceful checkpoint flush
    COUNT=0
    while kill -0 "$PID" 2>/dev/null && [ "$COUNT" -lt 30 ]; do
        sleep 1
        COUNT=$((COUNT + 1))
        echo "Waiting for process $PID to exit... ($COUNT/30)"
    done

    if kill -0 "$PID" 2>/dev/null; then
        echo "[WARNING] Process did not exit in 30s. Sending SIGKILL..."
        kill -KILL "$PID" || true
    fi
    echo "[SUCCESS] Process $PID stopped."
else
    echo "[INFO] Process $PID is not running."
fi

rm -f "$PID_FILE"
echo "[INFO] Lockfile removed."
