#!/usr/bin/env bash
# ==============================================================================
# Milestone M1.4.1 — Confirmatory Fact Verification Pipeline (Fold C)
#
# Strictly Training-Free, Quality-First (Zero parameter updates, zero learned routers)
# Independent Confirmatory Set: Fold C (train:100..149)
# Source Memory Bank: Strictly Fold A (train:0..49)
# Confirmatory Conditions: D0, D3, D4, D5, D2
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 1. Environment and Cache Setup
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/home/stackops/whale/cache/huggingface}"
export PYTHONUNBUFFERED=1

echo "================================================================================"
echo ">>> [M1.4.1] Confirmatory Experiment Suite: D0, D3, D4, D5, D2"
echo "================================================================================"

# 2. Frozen Artifact Paths
MODEL_PATH="/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
TRAIN_JSON="/home/stackops/whale/ERRORFlow/data/raw/averitec/train.json"
B1_PREDS="outputs/m121-b1-preds-20261024-152746/predictions.jsonl"
R4_SOURCE_EVIDENCE="outputs/m11-store-r14-pilot100-20261023-162716/R4.jsonl"

# 3. Preflight Dependency Verification
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Frozen Qwen3-8B snapshot not found at: $MODEL_PATH"
    exit 1
fi

if [[ ! -f "$TRAIN_JSON" ]]; then
    echo "ERROR: Official AVeriTeC train.json not found at: $TRAIN_JSON"
    exit 1
fi

if [[ ! -f "$B1_PREDS" ]]; then
    echo "ERROR: Source B1 predictions not found at: $B1_PREDS"
    exit 1
fi

if [[ ! -f "$R4_SOURCE_EVIDENCE" ]]; then
    echo "ERROR: Source R4 evidence not found at: $R4_SOURCE_EVIDENCE"
    exit 1
fi

# 4. Check Fold C R4 Evidence Availability
PREPARE_EVIDENCE=false
if [[ "${1:-}" == "--prepare-evidence" ]]; then
    PREPARE_EVIDENCE=true
    shift
fi

FOLD_C_R4="${1:-outputs/m141-confirmatory-r4/R4.jsonl}"
FOLD_C_CLAIMS="${2:-outputs/m141-confirmatory-r4/claims.jsonl}"

if [[ "$PREPARE_EVIDENCE" == "true" || (! -f "$FOLD_C_R4" && -z "${1:-}") ]]; then
    if [[ "$PREPARE_EVIDENCE" == "true" || -t 0 ]]; then
        echo ">>> Fold C R4 evidence missing. Triggering automated preparation pipeline..."
        chmod +x scripts/prepare_fold_c_evidence.sh
        bash scripts/prepare_fold_c_evidence.sh "outputs/m141-confirmatory-r4"
    fi
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="outputs/m141-confirmatory-$TIMESTAMP"

if [[ ! -f "$FOLD_C_R4" || ! -f "$FOLD_C_CLAIMS" ]]; then
    echo "--------------------------------------------------------------------------------"
    echo "AUDIT NOTICE: Fold C (train:100..149) R4 evidence file is NOT yet generated."
    echo "Target R4 path checked: $FOLD_C_R4"
    echo "Target claims path checked: $FOLD_C_CLAIMS"
    echo "--------------------------------------------------------------------------------"
    echo "Run the preparation script first, or pass --prepare-evidence:"
    echo "  bash scripts/prepare_fold_c_evidence.sh"
    echo "  bash scripts/run_m141_confirmatory.sh --prepare-evidence"
    echo "--------------------------------------------------------------------------------"
    exit 2
fi

echo ">>> Verified Fold C R4 Evidence: $FOLD_C_R4"
echo ">>> Verified Fold C Claims: $FOLD_C_CLAIMS"
echo ">>> Output Directory: $OUTPUT_DIR"

# 5. Execute Confirmatory Fact Verification
python -m tffv.verification.m141_confirmatory \
  --r4-evidence "$FOLD_C_R4" \
  --claims-jsonl "$FOLD_C_CLAIMS" \
  --train-json "$TRAIN_JSON" \
  --b1-predictions "$B1_PREDS" \
  --r4-source-evidence "$R4_SOURCE_EVIDENCE" \
  --output-dir "$OUTPUT_DIR" \
  --model-path "$MODEL_PATH" \
  --conditions D0 D3 D4 D5 D2 \
  --eval-start 100 \
  --eval-end 150 \
  --seed 42

echo ""
echo "================================================================================"
echo ">>> [M1.4.1] Confirmatory Experiment Finished Successfully!"
echo ">>> Report: $OUTPUT_DIR/M1_4_1_CONFIRMATORY_REPORT.md"
echo "================================================================================"
cat "$OUTPUT_DIR/M1_4_1_CONFIRMATORY_REPORT.md"
