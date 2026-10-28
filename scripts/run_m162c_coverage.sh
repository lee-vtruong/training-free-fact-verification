#!/usr/bin/env bash
# User-run only: three cache-only coverage arms, exactly 150 new calls when empty.
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
MIXED="$PWD/outputs/m162b-mixed-v1"
GOLD="/home/stackops/whale/ERRORFlow/data/raw/averitec/train.json"
mkdir -p outputs/m162c-logs
LOG="$PWD/outputs/m162c-logs/coverage-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
echo "M1.6.2c EXPLORATORY: three cache-only k=3 gold-only coverage arms; no retrieval."
echo "UTC=$(date -u +%FT%TZ) HEAD=$(git rev-parse HEAD) LOG=$LOG"
test -f "$INDEX"
for ARM in COV_PLACEBO_G3 COV_NEI_G3; do
  RUN="$PWD/outputs/m162c-$ARM-v1"
  if [[ ! -d "$RUN" ]]; then
    python -m tffv.experiments.m162c prepare --canonical-run "$CANONICAL" \
      --cache-index "$INDEX" --memory "$MEMORY" --tokenizer "$TOKENIZER" \
      --arm "$ARM" --output "$RUN"
  fi
  python -m tffv.experiments.m162c check --run "$RUN"
  python -m tffv.experiments.m162c run --run "$RUN" --execute
  if [[ ! -f "$RUN/evaluation-v1/results.json" ]]; then
    python -m tffv.experiments.m162c evaluate --run "$RUN" \
      --canonical-run "$CANONICAL" --mixed-run "$MIXED" \
      --gold-source "$GOLD" --output "$RUN/evaluation-v1"
  fi
done
echo "COV_CONF_G3 is NO-GO in this release: its only usable Conflicting-correct exemplar (train:7) shares a leakage block with the frozen pair exemplars; re-run only after a separately authorized source expansion supplies a non-colliding exemplar."
echo "DONE. Exactly 50 new inference calls per GO arm on first execution; resumptions run only missing receipts."
echo "RESULTS=$PWD/outputs/m162c-COV_*-v1/evaluation-v1/results.json LOG=$LOG"
