#!/usr/bin/env bash
set -euo pipefail

# 1. Environment and Root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

source .venv/bin/activate
export HF_HOME=/home/stackops/whale/cache/huggingface

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

# 2. Locate inputs
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

RETRIEVAL_METRICS="$(find outputs -maxdepth 2 -type f -path 'outputs/m11-eval-*/per_claim_metrics.csv' | sort | tail -n 1)"
echo ">>> Using Retrieval Metrics: $RETRIEVAL_METRICS"

# 3. Model Checkpoint
MODEL_PATH="/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
REVISION="b968826d9c46dd6066d109eabc6255188de91218"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Qwen3-8B snapshot directory not found: $MODEL_PATH"
    exit 1
fi
echo ">>> Using Frozen Backbone: $MODEL_PATH"

# 4. Phase 1: Prediction
PRED_DIR="outputs/m12-base-qwen3-8b-$TIMESTAMP"
echo ">>> [1/4] Running Frozen Base Verification Predictions in $PRED_DIR..."
python -m tffv.verification.predict \
  --r4-path "$R4_PATH" \
  --claims-path "$CLAIMS_PATH" \
  --model-path "$MODEL_PATH" \
  --revision "$REVISION" \
  --device cuda \
  --dtype bfloat16 \
  --output "$PRED_DIR"

echo ">>> Prediction Manifest Sealed:"
head -n 25 "$PRED_DIR/manifest.json"

# 5. Phase 2: Evaluation
EVAL_DIR="outputs/m12-eval-base-$TIMESTAMP"
echo ">>> [2/4] Evaluating Accuracy & Macro-F1 in $EVAL_DIR..."
python -m tffv.verification.evaluate \
  --predictions-dir "$PRED_DIR" \
  --train-json "$TRAIN_JSON" \
  --output "$EVAL_DIR"

echo ">>> Evaluation Complete. Report preview:"
cat "$EVAL_DIR/evaluation_report.md"

# 6. Phase 3: Error Profiling
ERR_DIR="outputs/m12-errors-base-$TIMESTAMP"
echo ">>> [3/4] Running Error Profiling & Taxonomy in $ERR_DIR..."
python -m tffv.verification.error_profiling \
  --eval-dir "$EVAL_DIR" \
  --predictions "$PRED_DIR/predictions.jsonl" \
  --retrieval-csv "$RETRIEVAL_METRICS" \
  --output "$ERR_DIR"

echo ">>> Error Profiling Report Preview:"
cat "$ERR_DIR/error_profiling_report.md"

# 7. Phase 4: Group Safety & Fold Partitioning Plan
SAFETY_PLAN="outputs/m12-group-safety-$TIMESTAMP/fold_isolation_plan.json"
echo ">>> [4/4] Creating Group-Safety and Fold Isolation Plan in $SAFETY_PLAN..."
python -c "
import json
from pathlib import Path
from tffv.verification.group_safety import create_pilot_group_safety_plan

r4_rows = [json.loads(line) for line in open('$R4_PATH', encoding='utf-8')]
sids = [r['sample_id'] for r in r4_rows]
create_pilot_group_safety_plan(sids, Path('$SAFETY_PLAN'))
print('Group safety plan sealed successfully.')
"

echo ">>> All M1.2 tasks completed successfully!"
