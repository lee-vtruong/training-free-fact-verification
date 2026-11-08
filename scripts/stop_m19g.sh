#!/usr/bin/env bash
# ==============================================================================
# Graceful Stop Script: Milestone M1.9g Evaluation
# ==============================================================================
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-outputs/m19g-evidence-complete-v1}"
PID_FILE="$OUTPUT_DIR/.m19g.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" || echo "")
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo "[INFO] Sending SIGTERM to PID $PID..."
        kill -15 "$PID" 2>/dev/null || true
        sleep 2
        if kill -0 "$PID" 2>/dev/null; then
            echo "[INFO] Force killing PID $PID..."
            kill -9 "$PID" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
        echo "[SUCCESS] M1.9g process terminated."
    else
        echo "[INFO] No active process for PID $PID."
        rm -f "$PID_FILE"
    fi
else
    echo "[INFO] No PID file found at $PID_FILE."
fi
