#!/usr/bin/env bash
# ==============================================================================
# Milestone M1.5 — Controlled Verification Evaluation Suite (Fold E, 100 claims)
#
# Strictly Training-Free, Quality-First (Zero parameter updates, zero learned routers)
# Target Population: Fold E (train:250..349, 100 claims; 99 admitted, 1 quarantined)
# Source Error Memory: Strictly Fold A (train:0..49, N=50 claims, 25 errors)
# Pre-registered Conditions:
#   M0: Matched generic control
#   M1: Generic claim-type clarification
#   M4: Permutation consistency (Order 1, Order 2, Order 3, majority voting)
# Primary Hypothesis: M4 vs M0
# Secondary Hypothesis: M1 vs M0
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Environment
export HF_HOME="${HF_HOME:-/home/stackops/whale/cache/huggingface}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="src"
PYTHON_BIN="${PYTHON_BIN:-/home/stackops/miniconda3/bin/python}"

MODEL_PATH="${MODEL_PATH:-/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
TRAIN_JSON="${TRAIN_JSON:-data/raw/averitec/train.json}"
FOLD_E_R4="${FOLD_E_R4:-outputs/m15-folde-r4/R4.jsonl}"
FOLD_E_CLAIMS="${FOLD_E_CLAIMS:-outputs/m15-folde-r4/claims.jsonl}"

RUN_TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-outputs/m15-evaluation-${RUN_TIMESTAMP}}"

echo "================================================================================"
echo ">>> [M1.5] Controlled Verification Evaluation Suite (Fold E)"
echo ">>> Target Population: Fold E (train:250..349)"
echo ">>> Output Directory: $OUTPUT_DIR"
echo "================================================================================"

$PYTHON_BIN src/tffv/verification/m15_runner.py \
    --r4-evidence "$FOLD_E_R4" \
    --claims-jsonl "$FOLD_E_CLAIMS" \
    --train-json "$TRAIN_JSON" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --conditions M0 M1 M4 \
    --eval-start 250 \
    --eval-end 350 \
    --source-start 0 \
    --source-end 50 \
    --seed 42

echo "================================================================================"
echo ">>> [M1.5] Evaluation completed! Report available at:"
echo ">>> $OUTPUT_DIR/M1_5_EVALUATION_REPORT.md"
echo "================================================================================"
