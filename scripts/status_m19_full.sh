#!/usr/bin/env bash
# Status monitor for Milestone M1.9 Evaluation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m19-historical-outcome-eval-v1}"
PID_FILE="$OUTPUT_DIR/.m19.pid"

echo "======================================================================"
echo " Milestone M1.9 Historical Outcome Evaluation Status"
echo " Time: $(date)"
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
    echo "Process Status: NO ACTIVE LOCKFILE"
fi

echo ""
echo "--- Pipeline Progress & Artifact Counts ---"
[ -f "$OUTPUT_DIR/historical_store.jsonl" ] && echo "1. Historical Store    : $(wc -l < "$OUTPUT_DIR/historical_store.jsonl") records" || echo "1. Historical Store    : 0 records"
if [ -f "$OUTPUT_DIR/predictions.jsonl" ]; then
    PRED_COUNT=$(wc -l < "$OUTPUT_DIR/predictions.jsonl")
    TARGET_CALLS=3000
    echo "2. Predictions Sealed  : $PRED_COUNT / $TARGET_CALLS calls ($(( PRED_COUNT * 100 / TARGET_CALLS ))%)"
    if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
        "$PROJECT_ROOT/.venv/bin/python" - "$OUTPUT_DIR/predictions.jsonl" << 'PYEOF'
import json, sys
from collections import Counter
counts = Counter()
try:
    with open(sys.argv[1]) as f:
        for line in f:
            d = json.loads(line)
            counts[d.get("arm", "UNKNOWN")] += 1
    print("   Completed per arm:", dict(counts))
except Exception as e:
    pass
PYEOF
    fi
else
    echo "2. Predictions Sealed  : 0 calls"
fi
[ -f "$OUTPUT_DIR/DECISION_REPORT.md" ] && echo "3. Final Decision Report: AVAILABLE" || echo "3. Final Decision Report: PENDING"

echo ""
echo "--- GPU Telemetry ---"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
else
    echo "nvidia-smi not available"
fi

echo ""
LATEST_LOG=$(ls -t "$OUTPUT_DIR/logs"/m19_*.log 2>/dev/null | head -n 1 || true)
if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
    echo "--- Recent Log Output ($LATEST_LOG) ---"
    tail -n 15 "$LATEST_LOG"
fi
echo "======================================================================"
