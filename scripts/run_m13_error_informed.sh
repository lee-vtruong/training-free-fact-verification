#!/usr/bin/env bash
set -euo pipefail

# 1. Environment and Project Root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

source .venv/bin/activate
export HF_HOME=/home/stackops/whale/cache/huggingface

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
echo "================================================================================"
echo ">>> M1.3 — Error-Informed Training-Free Fact Verification Runner (C0–C6)"
echo ">>> Timestamp: $TIMESTAMP"
echo "================================================================================"

# 2. Verify GPU & Resources
if command -v nvidia-smi &> /dev/null; then
    echo ">>> GPU Detected:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
else
    echo "WARNING: nvidia-smi not available in PATH"
fi

# 3. Locate inputs
R4_PATH="$(find outputs -maxdepth 2 -type f -path 'outputs/m11-store-r14-*/R4.jsonl' | sort | tail -n 1)"
if [[ ! -f "$R4_PATH" ]]; then
    echo "ERROR: R4.jsonl not found: $R4_PATH"
    exit 1
fi
echo ">>> Using R4 Evidence: $R4_PATH"

CLAIMS_PATH="$(find outputs -maxdepth 2 -type f -path 'outputs/m11-source-claims-*/claims.jsonl' | sort | tail -n 1)"
if [[ ! -f "$CLAIMS_PATH" ]]; then
    echo "ERROR: claims.jsonl not found: $CLAIMS_PATH"
    exit 1
fi
echo ">>> Using Claims: $CLAIMS_PATH"

TRAIN_JSON="/home/stackops/whale/ERRORFlow/data/raw/averitec/train.json"
if [[ ! -f "$TRAIN_JSON" ]]; then
    echo "ERROR: train.json not found: $TRAIN_JSON"
    exit 1
fi
echo ">>> Using Ground Truth: $TRAIN_JSON"

MODEL_PATH="/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Qwen3-8B snapshot directory not found: $MODEL_PATH"
    exit 1
fi
echo ">>> Using Frozen Backbone: $MODEL_PATH"

# Locate latest B1 predictions (from M1.2.1 stabilization)
B1_PREDICTIONS="$(find outputs -type f \( -path '*/B1/predictions.jsonl' -o -path '*b1*preds*/predictions.jsonl' -o -path '*b1*/predictions.jsonl' \) | sort | tail -n 1)"
if [[ -z "$B1_PREDICTIONS" || ! -f "$B1_PREDICTIONS" ]]; then
    echo "ERROR: B1 predictions not found for historical error memory construction!"
    exit 1
fi
echo ">>> Using Historical B1 Predictions: $B1_PREDICTIONS"

# 4. Target output directory
OUT_DIR="outputs/m13-error-informed-$TIMESTAMP"
mkdir -p "$OUT_DIR"

echo ">>> Output Directory: $OUT_DIR"
echo ">>> Commencing M1.3 Evaluation across C0–C6..."

# 5. Run M1.3 Runner across C0..C6
python -m tffv.verification.m13_runner \
  --r4-path "$R4_PATH" \
  --claims-path "$CLAIMS_PATH" \
  --train-json "$TRAIN_JSON" \
  --model-path "$MODEL_PATH" \
  --b1-predictions "$B1_PREDICTIONS" \
  --output-dir "$OUT_DIR" \
  --condition ALL \
  --split-mode fold_b \
  --device cuda \
  --dtype bfloat16 \
  --seed 42

echo "================================================================================"
echo ">>> M1.3 Complete! Output files in: $OUT_DIR"
echo "================================================================================"
cat "$OUT_DIR/M1_3_EVALUATION_REPORT.md"
