#!/usr/bin/env bash
# M1.7 Exploratory Feasibility 4-Arm Evaluation Runner (24 Development Claims)
# Evaluates Arms A, B, C, D using frozen Qwen3-8B verifier and LLM-validated memory bank.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_DIR="outputs/m17-eval-24dev"
MEMORY_BANK="outputs/m17-llm-validated-memory-v1/memory.json"
LOG_FILE="outputs/m17-eval-run.log"

source .venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

mkdir -p outputs
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================================"
echo ">>> [M1.7] Step 1: Prepare 24 Clean Development Targets & Batch Manifest"
echo "================================================================================"
python -m tffv.experiments.m17_eval prepare --output "$RUN_DIR" --memory "$MEMORY_BANK" --targets 24

echo "================================================================================"
echo ">>> [M1.7] Step 2: Retrieve Frozen R4 Top-8 Evidence for 24 Claims"
echo "================================================================================"
python -m tffv.experiments.m17_eval retrieve --run "$RUN_DIR"

echo "================================================================================"
echo ">>> [M1.7] Step 3: Run 4-Arm Inference (A, B, C, D) on Qwen3-8B (RTX 5090)"
echo "================================================================================"
python -m tffv.experiments.m17_eval predict --run "$RUN_DIR"

echo "================================================================================"
echo ">>> [M1.7] Step 4: Evaluate Against Gold Labels & Compute McNemar / Bootstrap CI"
echo "================================================================================"
python -m tffv.experiments.m17_eval evaluate --run "$RUN_DIR" --gold /home/stackops/whale/ERRORFlow/data/raw/averitec/train.json

echo "================================================================================"
echo ">>> [M1.7] Evaluation Finished Successfully!"
echo "================================================================================"
cat "$RUN_DIR/REPORT.md"
