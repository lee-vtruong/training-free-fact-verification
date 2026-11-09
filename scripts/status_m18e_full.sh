#!/usr/bin/env bash
# ==============================================================================
# Status Script: Milestone M1.8e Progress & Telemetry Monitor
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activate virtualenv if available
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m18e-full-history-pf1000-v1}"
PID_FILE="$OUTPUT_DIR/.m18e.pid"
STATE_FILE="$OUTPUT_DIR/state.json"
RECEIPTS_FILE="$OUTPUT_DIR/retrieval_receipts.jsonl"
PREDS_FILE="$OUTPUT_DIR/base_predictions.jsonl"
JUDGE_FILE="$OUTPUT_DIR/judge_decisions.jsonl"
MEMORY_FILE="$OUTPUT_DIR/memory.json"
LOG_DIR="$OUTPUT_DIR/logs"

echo "======================================================================"
echo " Milestone M1.8e Execution & Progress Status"
echo " Time: $(date)"
echo " Output Directory: $OUTPUT_DIR"
echo "======================================================================"

# 1. Process Status
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" || echo "")
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo -e "Process Status: \e[32mRUNNING\e[0m (PID: $PID)"
    else
        echo -e "Process Status: \e[33mSTOPPED / INACTIVE\e[0m (PID in lock: $PID)"
    fi
else
    echo -e "Process Status: \e[34mNO ACTIVE LOCKFILE\e[0m"
fi

# 2. Phase & Progress State
if [ -f "$STATE_FILE" ]; then
    echo ""
    echo "--- Phase Progression ---"
    python -c "
import json
try:
    with open('$STATE_FILE', 'r', encoding='utf-8') as f:
        s = json.load(f)
    print(f'Current Phase: {s.get(\"current_phase\", \"UNKNOWN\")}')
    print(f'Elapsed Seconds: {s.get(\"spent_seconds\", 0.0):.1f}s ({s.get(\"spent_seconds\", 0.0)/3600:.2f} hrs)')
    print('Phases Detail:')
    for p, st in s.get('phase_status', {}).items():
        print(f'  - {p:32s}: {st}')
except Exception as e:
    print(f'Could not parse state.json: {e}')
"
fi

# 3. Artifact Metrics (Claim counts)
echo ""
echo "--- Pipeline Artifact Counts (Target: 3,067 Valid Claims) ---"
N_REC=0
N_PRED=0
N_JUDGE=0
N_MEM=0

if [ -f "$RECEIPTS_FILE" ]; then N_REC=$(wc -l < "$RECEIPTS_FILE"); fi
if [ -f "$PREDS_FILE" ]; then N_PRED=$(wc -l < "$PREDS_FILE"); fi
if [ -f "$JUDGE_FILE" ]; then N_JUDGE=$(wc -l < "$JUDGE_FILE"); fi
if [ -f "$MEMORY_FILE" ]; then
    N_MEM=$(python -c "import json; d=json.load(open('$MEMORY_FILE')); print(d.get('total_records', len(d.get('records', []))))" 2>/dev/null || echo "0")
fi

P_REC=$(awk "BEGIN {printf \"%.1f\", ($N_REC/3067)*100}")
P_PRED=$(awk "BEGIN {printf \"%.1f\", ($N_PRED/3067)*100}")

echo "1. Retrieval Receipts (PF1000) : $N_REC / 3,067 (${P_REC}%)"
echo "2. Base Predictions (Qwen3-8B) : $N_PRED / 3,067 (${P_PRED}%)"
echo "3. Judge Decisions (Qwen3-14B) : $N_JUDGE evaluated"
echo "4. Assembled Memory Size       : $N_MEM records (LLM_VALIDATED)"

# 4. GPU Telemetry
if command -v nvidia-smi &>/dev/null; then
    echo ""
    echo "--- GPU Telemetry ---"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader || true
fi

# 5. Recent Log Snippet
if [ -d "$LOG_DIR" ]; then
    LATEST_LOG=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -n 1 || true)
    if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
        echo ""
        echo "--- Recent Log Output ($LATEST_LOG) ---"
        tail -n 12 "$LATEST_LOG"
    fi
fi
echo "======================================================================"
