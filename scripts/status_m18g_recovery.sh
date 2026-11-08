#!/usr/bin/env bash
# Status script for Milestone M1.8g Recovery
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m18g-correction-recovery-v1}"
PID_FILE="$OUTPUT_DIR/.m18g.pid"

echo "======================================================================"
echo " Milestone M1.8g Recovery & Revalidation Status"
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
echo "--- Pipeline Artifact Counts ---"
[ -f "$OUTPUT_DIR/generation_results.jsonl" ] && echo "1. Generated Candidates : $(wc -l < "$OUTPUT_DIR/generation_results.jsonl") records" || echo "1. Generated Candidates : 0 records"
if [ -f "$OUTPUT_DIR/judge_decisions.jsonl" ]; then
    JUDGED_COUNT=$(wc -l < "$OUTPUT_DIR/judge_decisions.jsonl")
    echo "2. Judge Decisions      : $JUDGED_COUNT / 1190 records ($(( JUDGED_COUNT * 100 / 1190 ))%)"
    if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
        "$PROJECT_ROOT/.venv/bin/python" - "$OUTPUT_DIR/judge_decisions.jsonl" << 'PYEOF'
import json, sys
from collections import Counter
counts = Counter()
try:
    with open(sys.argv[1]) as f:
        for line in f:
            d = json.loads(line)
            counts[d.get("decision", "UNKNOWN")] += 1
    print("   Decisions breakdown:", dict(counts))
except Exception as e:
    print("   Decisions error:", e)
PYEOF
    fi
else
    echo "2. Judge Decisions      : 0 records"
fi
[ -f "$OUTPUT_DIR/memory.json" ] && echo "3. Final Recovered Memory: $(grep -o '"sample_id"' "$OUTPUT_DIR/memory.json" | wc -l) records" || echo "3. Final Recovered Memory: 0 records"

echo ""
echo "--- GPU Telemetry ---"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
else
    echo "nvidia-smi not available"
fi

echo ""
LATEST_LOG=$(ls -t "$OUTPUT_DIR/logs"/m18g_*.log 2>/dev/null | head -n 1 || true)
if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
    echo "--- Recent Log Output ($LATEST_LOG) ---"
    tail -n 15 "$LATEST_LOG"
fi
echo "======================================================================"
