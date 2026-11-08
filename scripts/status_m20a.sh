#!/usr/bin/env bash
# ==============================================================================
# Status Monitor: Milestone M2.0-A Error-Derived Verification Rules
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m20a-error-rules-v1}"
PID_FILE="$OUTPUT_DIR/.m20a.pid"
LOG_DIR="$OUTPUT_DIR/logs"

echo "======================================================================"
echo " Milestone M2.0-A Error-Derived Verification Rules Evaluation Status"
echo " Time: $(date -u)"
echo " Output Directory: $OUTPUT_DIR"
echo "======================================================================"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" || echo "")
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo "Process Status: RUNNING (PID: $PID)"
    else
        echo "Process Status: NOT RUNNING (Stale PID: $PID)"
    fi
else
    echo "Process Status: NO PID FILE FOUND"
fi

echo ""
echo "--- Pipeline Artifacts ---"
[ -f "$OUTPUT_DIR/rules_manifest.json" ] && echo "1. Rules Manifest     : SEALED (5 arms)" || echo "1. Rules Manifest     : PENDING"
[ -f "$OUTPUT_DIR/prompt_receipts.jsonl" ] && echo "2. Prompt Receipts    : $(wc -l < "$OUTPUT_DIR/prompt_receipts.jsonl") receipts" || echo "2. Prompt Receipts    : PENDING"

if [ -f "$OUTPUT_DIR/predictions.jsonl" ]; then
    N_PREDS=$(wc -l < "$OUTPUT_DIR/predictions.jsonl")
    echo "3. Predictions Sealed : $N_PREDS / 2500 calls"
else
    echo "3. Predictions Sealed : 0 / 2500 calls"
fi

[ -f "$OUTPUT_DIR/FINAL_REPORT.md" ] && echo "4. Final Report       : AVAILABLE" || echo "4. Final Report       : PENDING"
[ -f "$OUTPUT_DIR/DECISION_REPORT.md" ] && echo "5. Decision Report    : AVAILABLE" || echo "5. Decision Report    : PENDING"

echo ""
echo "--- GPU Telemetry ---"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader || echo "GPU Query Failed"
fi

LATEST_LOG=$(find "$LOG_DIR" -type f -name "*.log" 2>/dev/null | sort | tail -n 1)
if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
    echo ""
    echo "--- Recent Log Output ($LATEST_LOG) ---"
    tail -n 25 "$LATEST_LOG"
fi
echo "======================================================================"
