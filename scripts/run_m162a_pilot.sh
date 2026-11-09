#!/usr/bin/env bash
# User-run only: exactly DEV50; no expansion or automatic prompt optimization.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ "$(git branch --show-current)" == "codex/m16-heal-design-integrity" ]]
git diff HEAD --quiet -- src/tffv configs schemas docs/partitions
source "$PWD/.venv/bin/activate"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
RUN="$PWD/outputs/m162a-dev50-v1"
MEMORY="$PWD/outputs/m162a-inputs-v1/memory75.json"
TOKENIZER="/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
CLAIMS="$PWD/outputs/m11-source-claims-20261021-055413/claims_manifest.json"
GOLD="/home/stackops/whale/ERRORFlow/data/raw/averitec/train.json"
mkdir -p outputs/m162a-logs
LOG="$PWD/outputs/m162a-logs/pilot-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
echo "EXPLORATORY DEV50 only; k=1 seed=42; P4 BLOCKED in this unreviewed bank."
echo "UTC=$(date -u +%FT%TZ) head=$(git rev-parse HEAD) LOG=$LOG RUN=$RUN"
python --version
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
printf '%s  %s\n' b3f7275112aa49b71595d94c7313a4ae101b3ed263c277bb277f97def2565581 "$MEMORY" | sha256sum -c -
python scripts/m16_validate_seal.py docs/partitions/m16-v2
python -m tffv.experiments.m161 validate-memory --memory "$MEMORY"
python -m tffv.experiments.m162a prepare --output "$RUN" --memory "$MEMORY" --claims-manifest "$CLAIMS" --tokenizer "$TOKENIZER"
python -m tffv.experiments.m162a check --run "$RUN"
# Each explicit GPU command runs environment/model/disk/VRAM checks before work.
# Same sealed receipts are reused; unfinished attempts get a new directory.
python -m tffv.experiments.m162a retrieve --run "$RUN" --execute
# Fail closed if any Top8 evidence, token-length, or context-limit check fails.
python -m tffv.experiments.m162a seal-prompts --run "$RUN"
python -m tffv.experiments.m162a run-pilot --run "$RUN" --execute
# Evaluator verifies completion of ALL conditions/cases before reading DEV labels.
if [[ ! -f "$RUN/evaluation-v1/results.json" ]]; then
  python -m tffv.experiments.m162a evaluate --run "$RUN" --gold-source "$GOLD" --output "$RUN/evaluation-v1"
else
  python - "$RUN" <<'PY'
from pathlib import Path
import sys
from tffv.memory.heal import read
from tffv.provenance import file_hash
run=Path(sys.argv[1]);result=read(run/'evaluation-v1/results.json')
assert result['batch_sha256']==file_hash(run/'batch.json')
assert result['completed_sha256']==file_hash(run/'completed.json')
print('Existing scoring belongs to the same completed pilot; retained.')
PY
fi
cat "$RUN/state.json"
echo "DONE DEV50. Results=$RUN/evaluation-v1/results.json Per_claim=$RUN/evaluation-v1/per_claim.jsonl"
echo "LOG=$LOG; resume with the same checkout/inputs: bash scripts/run_m162a_pilot.sh"
echo "No automatic extra claims, seeds, k, CoT or confirmatory experiment."
