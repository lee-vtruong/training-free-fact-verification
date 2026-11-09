#!/usr/bin/env bash
# Status monitor for Milestone M1.9f Clean Official Dev 500 Evaluation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m19f-clean-official-dev-v1}"
PID_FILE="$OUTPUT_DIR/.m19f.pid"

echo "======================================================================"
echo " Milestone M1.9f Clean Official Dev 500 Evaluation Status"
echo " Time: $(date)"
echo " Output Directory: $OUTPUT_DIR"
echo "======================================================================"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Process Status: RUNNING (PID: $PID)"
    else
        echo "Process Status: NOT RUNNING (Stale PID: $PID)"
    fi
else
    echo "Process Status: NO ACTIVE PID FILE"
fi

echo ""
echo "--- Pipeline Artifacts ---"
[ -f "$OUTPUT_DIR/dataset_manifest.json" ] && echo "1. Dataset Manifest   : $(python3 -c "import json; print(len(json.load(open('$OUTPUT_DIR/dataset_manifest.json'))))" 2>/dev/null || echo 'EXISTS') claims" || echo "1. Dataset Manifest   : MISSING"
[ -f "$OUTPUT_DIR/memory_eligibility.json" ] && echo "2. Memory Eligibility : READY" || echo "2. Memory Eligibility : MISSING"
[ -f "$OUTPUT_DIR/smoke_test_report.md" ] && echo "3. Smoke Test Report  : PASSED" || echo "3. Smoke Test Report  : PENDING"
[ -f "$OUTPUT_DIR/predictions.jsonl" ] && echo "4. Predictions Sealed : $(wc -l < "$OUTPUT_DIR/predictions.jsonl") / 3000 calls" || echo "4. Predictions Sealed : 0 calls"
[ -f "$OUTPUT_DIR/DECISION_REPORT.md" ] && echo "5. Decision Report    : AVAILABLE" || echo "5. Decision Report    : PENDING"

echo ""
if command -v nvidia-smi &>/dev/null; then
    echo "--- GPU Telemetry ---"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
fi

LATEST_LOG=$(ls -t "$OUTPUT_DIR/logs"/m19f_*.log 2>/dev/null | head -n 1 || true)
if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
    echo ""
    echo "--- Recent Log Output ($LATEST_LOG) ---"
    tail -n 20 "$LATEST_LOG"
fi
echo "======================================================================"
