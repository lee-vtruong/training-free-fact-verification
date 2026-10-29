#!/usr/bin/env bash
# User-run only: one new cache-only mixed condition, exactly 50 calls when empty.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ "$(git branch --show-current)" == "codex/m16-heal-design-integrity" ]]
git diff HEAD --quiet -- src/tffv configs schemas docs/partitions
source "$PWD/.venv/bin/activate"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
CANONICAL="$PWD/outputs/m162a-dev50-v1"
INDEX="$PWD/outputs/m16-cache-audit-v1/canonical-index.json"
MEMORY="$PWD/outputs/m162a-inputs-v1/memory75.json"
TOKENIZER="/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
RUN="$PWD/outputs/m162b-mixed-v1"
GOLD="/home/stackops/whale/ERRORFlow/data/raw/averitec/train.json"
mkdir -p outputs/m162b-logs
LOG="$PWD/outputs/m162b-logs/mixed-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
echo "M1.6.2b EXPLORATORY: one cache-only mixed k=2 gold-only arm; no retrieval."
echo "UTC=$(date -u +%FT%TZ) HEAD=$(git rev-parse HEAD) LOG=$LOG"
test -f "$INDEX"
if [[ ! -d "$RUN" ]]; then
  python -m tffv.experiments.m162b prepare --canonical-run "$CANONICAL" --cache-index "$INDEX" --memory "$MEMORY" --tokenizer "$TOKENIZER" --output "$RUN"
fi
python -m tffv.experiments.m162b check --run "$RUN"
# The runner verifies cached R4 hashes at preparation; it has no retrieval command.
python -m tffv.experiments.m162b run --run "$RUN" --execute
if [[ ! -f "$RUN/evaluation-v1/results.json" ]]; then
  python -m tffv.experiments.m162b evaluate --run "$RUN" --gold-source "$GOLD" --output "$RUN/evaluation-v1"
fi
echo "DONE. New model calls are exactly 50 on first run; resumptions execute only missing receipts."
echo "RESULT=$RUN/evaluation-v1/results.json LOG=$LOG"
