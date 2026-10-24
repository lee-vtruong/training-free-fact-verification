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
echo ">>> M1.2.1 — Frozen Base Stabilization Runner (B0 vs B1 vs B2)"
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
REVISION="b968826d9c46dd6066d109eabc6255188de91218"
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Qwen3-8B snapshot directory not found: $MODEL_PATH"
    exit 1
fi
echo ">>> Using Frozen Backbone: $MODEL_PATH"

# 4. [B0] Check existing M1.2 Baseline evaluation
B0_EVAL_DIR="$(find outputs -maxdepth 2 -type d -name 'm12-eval-base-*' | sort | tail -n 1)"
if [[ -z "$B0_EVAL_DIR" || ! -f "$B0_EVAL_DIR/evaluation_summary.json" ]]; then
    echo ">>> B0 evaluation not found in outputs/. Re-running evaluation on existing M1.2 predictions..."
    B0_PRED_DIR="$(find outputs -maxdepth 2 -type d -name 'm12-base-qwen3-8b-*' | sort | tail -n 1)"
    B0_EVAL_DIR="outputs/m121-b0-eval-$TIMESTAMP"
    python -m tffv.verification.evaluate \
      --predictions-dir "$B0_PRED_DIR" \
      --train-json "$TRAIN_JSON" \
      --output "$B0_EVAL_DIR"
fi
echo ">>> [B0] Baseline Evaluation Directory: $B0_EVAL_DIR"

# 5. [B1] Non-Thinking Mode (max_new_tokens = 512)
B1_PRED_DIR="outputs/m121-b1-preds-$TIMESTAMP"
B1_EVAL_DIR="outputs/m121-b1-eval-$TIMESTAMP"
echo ""
echo "================================================================================"
echo ">>> [1/3] Running B1 (Non-thinking, max_new_tokens=512) in $B1_PRED_DIR..."
echo "================================================================================"
python -m tffv.verification.predict \
  --r4-path "$R4_PATH" \
  --claims-path "$CLAIMS_PATH" \
  --model-path "$MODEL_PATH" \
  --revision "$REVISION" \
  --config "configs/m121_b1_nonthinking_512.json" \
  --device cuda \
  --dtype bfloat16 \
  --output "$B1_PRED_DIR"

echo ">>> [1/3] Evaluating B1 in $B1_EVAL_DIR..."
python -m tffv.verification.evaluate \
  --predictions-dir "$B1_PRED_DIR" \
  --train-json "$TRAIN_JSON" \
  --output "$B1_EVAL_DIR"

# 6. [B2] Thinking Mode (max_new_tokens = 1024)
B2_PRED_DIR="outputs/m121-b2-preds-$TIMESTAMP"
B2_EVAL_DIR="outputs/m121-b2-eval-$TIMESTAMP"
echo ""
echo "================================================================================"
echo ">>> [2/3] Running B2 (Thinking, max_new_tokens=1024) in $B2_PRED_DIR..."
echo "================================================================================"
python -m tffv.verification.predict \
  --r4-path "$R4_PATH" \
  --claims-path "$CLAIMS_PATH" \
  --model-path "$MODEL_PATH" \
  --revision "$REVISION" \
  --config "configs/m121_b2_thinking_1024.json" \
  --device cuda \
  --dtype bfloat16 \
  --output "$B2_PRED_DIR"

echo ">>> [2/3] Evaluating B2 in $B2_EVAL_DIR..."
python -m tffv.verification.evaluate \
  --predictions-dir "$B2_PRED_DIR" \
  --train-json "$TRAIN_JSON" \
  --output "$B2_EVAL_DIR"

# 7. [Comparison] Run Cross-Experiment Stabilization Evaluation
COMP_DIR="outputs/m121-comparison-$TIMESTAMP"
echo ""
echo "================================================================================"
echo ">>> [3/3] Generating B0 vs B1 vs B2 Comparison Report in $COMP_DIR..."
echo "================================================================================"
python -m tffv.verification.stabilization \
  --b0-eval "$B0_EVAL_DIR" \
  --b1-eval "$B1_EVAL_DIR" \
  --b2-eval "$B2_EVAL_DIR" \
  --output "$COMP_DIR"

echo ""
echo "================================================================================"
echo ">>> M1.2.1 Stabilization Execution Complete!"
echo ">>> Results saved to: $COMP_DIR/comparison_report.md"
echo "================================================================================"
cat "$COMP_DIR/comparison_report.md"
