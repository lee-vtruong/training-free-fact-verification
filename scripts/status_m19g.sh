#!/usr/bin/env bash
# ==============================================================================
# Status Monitor: Milestone M1.9g Evidence-Complete Official Dev Evaluation
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m19g-evidence-complete-v1}"
PID_FILE="$OUTPUT_DIR/.m19g.pid"
LOG_DIR="$OUTPUT_DIR/logs"

echo "======================================================================"
echo " Milestone M1.9g Evidence-Complete Official Dev 500 Evaluation Status"
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
[ -f "$OUTPUT_DIR/dataset_manifest.json" ] && echo "1. Dataset Manifest   : $(jq -r '.total_claims // 0' "$OUTPUT_DIR/dataset_manifest.json" 2>/dev/null || echo "EXISTS") claims" || echo "1. Dataset Manifest   : PENDING"
[ -f "$OUTPUT_DIR/dev_retrieval_receipts.jsonl" ] && echo "2. Dev Retrieval      : $(wc -l < "$OUTPUT_DIR/dev_retrieval_receipts.jsonl") / 500 receipts" || echo "2. Dev Retrieval      : PENDING"
[ -f "$OUTPUT_DIR/smoke_test_report.md" ] && echo "3. Smoke Test Report  : PASSED" || echo "3. Smoke Test Report  : PENDING"

if [ -f "$OUTPUT_DIR/predictions.jsonl" ]; then
    N_PREDS=$(wc -l < "$OUTPUT_DIR/predictions.jsonl")
    echo "4. Predictions Sealed : $N_PREDS / 3000 calls"
else
    echo "4. Predictions Sealed : 0 / 3000 calls"
fi

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
