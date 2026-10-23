#!/usr/bin/env bash
set -euo pipefail

# 1. Paths and environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

source .venv/bin/activate

# 2. Check latest audit summary
AUDIT_SUMMARY="${1:-$(find outputs -maxdepth 2 -type f -path 'outputs/m11-audit-pilot100-*/audit_summary.json' | sort | tail -n 1)}"
if [[ ! -f "$AUDIT_SUMMARY" ]]; then
    echo "ERROR: Audit summary not found: $AUDIT_SUMMARY"
    exit 1
fi
echo ">>> Using Audit Summary: $AUDIT_SUMMARY"

CLAIMS_MANIFEST="$(find outputs -maxdepth 2 -type f -path 'outputs/m11-source-claims-*/claims_manifest.json' | sort | tail -n 1)"
if [[ ! -f "$CLAIMS_MANIFEST" ]]; then
    echo "ERROR: Claims manifest not found: $CLAIMS_MANIFEST"
    exit 1
fi
echo ">>> Using Claims Manifest: $CLAIMS_MANIFEST"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

# 3. Candidate Preparation (Quarantine Policy)
PILOT_CAND_OUT="outputs/m11-candidates-pilot100-$TIMESTAMP"
echo ">>> [1/3] Preparing candidate stores in $PILOT_CAND_OUT..."
python -m tffv.retrieval.prepare_store_candidates \
  --audit-summary "$AUDIT_SUMMARY" \
  --claims-manifest "$CLAIMS_MANIFEST" \
  --policy quarantine_review_flags_v1 \
  --output "$PILOT_CAND_OUT"

CANDIDATE_MANIFEST="$PILOT_CAND_OUT/candidate_manifest.json"
echo ">>> Candidate stores prepared: $(grep '"n_claims"' "$CANDIDATE_MANIFEST" || true)"

# 4. GPU Retrieval Benchmark (R1-R4 Top-200 with batch_size 16 & index caching)
export TFFV_EMBED_MODEL="/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
export TFFV_EMBED_REVISION="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
export TFFV_EMBED_SHA256="667106c1965ebad2038f612af424b1990ab6e6dabcee5cf20f45ed5c4dee071a"

export TFFV_RERANK_MODEL="/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473"
export TFFV_RERANK_REVISION="e61197ed45024b0ed8a2d74b80b4d909f1255473"
export TFFV_RERANK_SHA256="5219a20fa423e1054c34b93a5a8c0ffa640ca52320181603be32d7afe26de353"

mkdir -p outputs/index_cache
PILOT_RUN_OUT="outputs/m11-store-r14-pilot100-$TIMESTAMP"
echo ">>> [2/3] Executing GPU Benchmark R1-R4 across 100 stores in $PILOT_RUN_OUT..."
python -m tffv.retrieval.store_retrieval \
  --candidate-manifest "$CANDIDATE_MANIFEST" \
  --methods R1 R2 R3 R4 \
  --config configs/store_retrieval_m11.template.json \
  --index-cache outputs/index_cache \
  --output "$PILOT_RUN_OUT" \
  --execute

echo ">>> GPU Benchmark completed. Summary:"
head -n 35 "$PILOT_RUN_OUT/summary.json"

# 5. Quality Evaluation (Against AVeriTeC train.json)
PILOT_EVAL_OUT="outputs/m11-eval-pilot100-$TIMESTAMP"
TRAIN_JSON="/home/stackops/whale/ERRORFlow/data/raw/averitec/train.json"
echo ">>> [3/3] Evaluating Retrieval Quality in $PILOT_EVAL_OUT..."
python -m tffv.retrieval.quality_eval \
  --run-dir "$PILOT_RUN_OUT" \
  --train-json "$TRAIN_JSON" \
  --output "$PILOT_EVAL_OUT"

echo ">>> Evaluation complete! Report preview:"
cat "$PILOT_EVAL_OUT/evaluation_report.md"
