#!/usr/bin/env bash
# User-run only. No daemon, scheduler, automatic expansion or HEAL ablations.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -f "$PWD/.venv/bin/activate" ]]; then
  source "$PWD/.venv/bin/activate"
fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN="$PWD/outputs/m161-source-batch25-v1"
LEGACY="$PWD/outputs/m161-legacy-source-v1"
PLAN="$PWD/configs/m161_acquisition_plan.json"
CLAIMS="$PWD/outputs/m11-source-claims-20261021-055413/claims_manifest.json"
GOLD="/home/stackops/whale/ERRORFlow/data/raw/averitec/train.json"
mkdir -p "$PWD/outputs/m161-logs"
LOG="$PWD/outputs/m161-logs/batch25-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
echo "M1.6.1 EXPLORATORY only; fixed new source count=25; no HEAL evaluation"
echo "UTC=$(date -u +%FT%TZ) branch=$(git branch --show-current) head=$(git rev-parse HEAD)"
echo "LOG=$LOG RUN=$RUN LEGACY=$LEGACY"
python --version
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
python scripts/m16_validate_seal.py docs/partitions/m16-v2
# Offline CPU audit of own-project source predictions; gold is not a worker input.
if [[ ! -f "$LEGACY/memory.json" ]]; then
  python -m tffv.experiments.m161 legacy-audit \
    --predictions outputs/m121-b1-preds-20261024-152746/predictions.jsonl \
    --prediction-manifest outputs/m121-b1-preds-20261024-152746/manifest.json \
    --evidence outputs/m11-store-r14-pilot100-20261023-162716/R4.jsonl \
    --retrieval-manifest outputs/m11-store-r14-pilot100-20261023-162716/manifest.json \
    --gold-source "$GOLD" --output "$LEGACY"
fi
python -m tffv.experiments.m161 validate-memory --memory "$LEGACY/memory.json"
if [[ ! -d "$LEGACY/reviews" ]]; then
  python -m tffv.experiments.m161 export-reviews --memory "$LEGACY/memory.json" --output "$LEGACY/reviews"
fi
python -m tffv.experiments.m161 prepare-batch --output "$RUN" --plan "$PLAN" --claims-manifest "$CLAIMS"
# Failure stops the script BEFORE any acquisition/retrieval/inference.
python -m tffv.experiments.m161 preflight --run "$RUN"
python -m tffv.experiments.m161 run-batch --run "$RUN" --execute --resume
# A separate post-prediction process joins SOURCE gold, never dev/test gold.
if [[ ! -f "$RUN/memory/memory.json" ]]; then
  python -m tffv.experiments.m161 build-batch-memory --run "$RUN" --gold-source "$GOLD" --output "$RUN/memory"
fi
python -m tffv.experiments.m161 validate-memory --memory "$RUN/memory/memory.json"
if [[ ! -d "$RUN/memory/reviews" ]]; then
  python -m tffv.experiments.m161 export-reviews --memory "$RUN/memory/memory.json" --output "$RUN/memory/reviews"
fi
echo "DONE source_count=25; no automatic next batch; all corrections remain pending human review"
echo "LOG=$LOG RAW_OUTPUTS=$RUN MEMORY=$RUN/memory/memory.json REVIEWS=$RUN/memory/reviews"
echo "Resume: rerun bash scripts/run_m161_batch25.sh (same checkout/config/inputs required)."
