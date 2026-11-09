#!/usr/bin/env bash
# ==============================================================================
# Graceful Stop Script: Milestone M2.0-A Error-Derived Verification Rules
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m20a-error-rules-v1}"
PID_FILE="$OUTPUT_DIR/.m20a.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "[INFO] Sending SIGTERM to M2.0-A process (PID $PID)..."
        kill -TERM "$PID"
        sleep 2
        if kill -0 "$PID" 2>/dev/null; then
            echo "[INFO] Process still alive; sending SIGKILL..."
            kill -KILL "$PID" || true
        fi
        echo "[SUCCESS] Process terminated."
    else
        echo "[INFO] Process $PID is not running."
    fi
    rm -f "$PID_FILE"
else
    echo "[INFO] No PID file found at $PID_FILE."
fi
