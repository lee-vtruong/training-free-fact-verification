#!/usr/bin/env bash
# ==============================================================================
# Stop Script: Graceful Termination of Milestone M1.8e
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m18e-full-history-pf1000-v1}"
PID_FILE="$OUTPUT_DIR/.m18e.pid"
ROOT_PID_FILE="$PROJECT_ROOT/.m18e.pid"

echo "[M1.8e] Stopping background pipeline..."

PID=""
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" || echo "")
elif [ -f "$ROOT_PID_FILE" ]; then
    PID=$(cat "$ROOT_PID_FILE" || echo "")
fi

if [ -z "$PID" ]; then
    echo "[INFO] No active PID file found. Nothing to stop."
    exit 0
fi

if ! kill -0 "$PID" 2>/dev/null; then
    echo "[INFO] Process with PID $PID is not running. Cleaning up stale lockfiles..."
    rm -f "$PID_FILE" "$ROOT_PID_FILE"
    exit 0
fi

echo "[M1.8e] Sending SIGTERM to process PID $PID for graceful state flush..."
kill -TERM "$PID"

# Wait up to 30 seconds for graceful shutdown
TIMEOUT=30
for ((i=1; i<=TIMEOUT; i++)); do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "[SUCCESS] Process $PID exited cleanly."
        rm -f "$PID_FILE" "$ROOT_PID_FILE"
        exit 0
    fi
    sleep 1
done

echo "[WARNING] Process did not terminate within $TIMEOUT seconds. Sending SIGKILL..."
kill -KILL "$PID" 2>/dev/null || true
rm -f "$PID_FILE" "$ROOT_PID_FILE"
echo "[M1.8e] Process terminated and lockfiles removed."
