#!/usr/bin/env bash
set -euo pipefail

MODEL_SHARD="/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/model-00003-of-00005.safetensors"
LOG_FILE="outputs/m12_run.log"

echo "Waiting for $MODEL_SHARD to complete..."
while [[ ! -f "$MODEL_SHARD" ]]; do
    sleep 10
done

echo "Model shard 3 detected! Checking file integrity..."
sleep 5
echo "Triggering M1.2 Base Verification Pipeline..."
bash scripts/run_m12_base_verification.sh 2>&1 | tee "$LOG_FILE"
echo "M1.2 Base Verification Pipeline finished!"
