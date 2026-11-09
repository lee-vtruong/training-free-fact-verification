#!/usr/bin/env bash
# ==============================================================================
# Preflight Isolation Audit: Milestone M2.0g
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/m20g-robustness-attribution-v1}"
mkdir -p "$OUTPUT_DIR"

PYTHON_BIN="python"
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
fi

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

echo "======================================================================"
echo " Running Milestone M2.0g Preflight Isolation & Integrity Audit"
echo " Output Directory: $OUTPUT_DIR"
echo "======================================================================"

"$PYTHON_BIN" "$PROJECT_ROOT/src/tffv/experiments/m20g_runner.py" \
    --output-dir "$OUTPUT_DIR" \
    --preflight-only
