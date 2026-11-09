#!/usr/bin/env bash
# ==============================================================================
# Stop Script: Gracefully Terminate Milestone M2.0d Job
# ==============================================================================
set -euo pipefail

PID=$(pgrep -f "m20d_runner.py" || true)

if [ -n "$PID" ]; then
    echo "Terminating M2.0d process (PID: $PID)..."
    kill -TERM "$PID"
    sleep 3
    if kill -0 "$PID" 2>/dev/null; then
        echo "Process still alive. Sending SIGKILL..."
        kill -9 "$PID"
    fi
    echo "Process terminated."
else
    echo "No running M2.0d process found."
fi
