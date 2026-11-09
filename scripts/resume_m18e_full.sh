#!/usr/bin/env bash
# ==============================================================================
# Resume Script: Seamlessly Resume Milestone M1.8e Background Pipeline
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m18e-full-history-pf1000-v1}"
PID_FILE="$OUTPUT_DIR/.m18e.pid"

echo "======================================================================"
echo " Resuming Milestone M1.8e Pipeline"
echo " Output Directory: $OUTPUT_DIR"
echo "======================================================================"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" || echo "")
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo "[ERROR] Pipeline is already running under PID $PID. Cannot launch another instance."
        echo "Check status with: bash scripts/status_m18e_full.sh"
        exit 1
    fi
fi

echo "[INFO] Existing outputs will be detected and skipped without recomputation."
bash "$SCRIPT_DIR/launch_m18e_full.sh" "$@"
