#!/usr/bin/env bash
# ==============================================================================
# Milestone M1.4.2 — Fold D (train:150..249, 100 claims) R4 Evidence Preparation
#
# Bounded HTTP range acquisition, schema audit, quarantine review flags v1,
# frozen R1-R4 retrieval (BM25 + Qwen3-Embedding-0.6B + RRF + Qwen3-Reranker-0.6B).
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 1. Environment and Checkpoints
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/home/stackops/whale/cache/huggingface}"
export PYTHONUNBUFFERED=1

export TFFV_EMBED_MODEL="/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
export TFFV_EMBED_REVISION="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
export TFFV_EMBED_SHA256="667106c1965ebad2038f612af424b1990ab6e6dabcee5cf20f45ed5c4dee071a"

export TFFV_RERANK_MODEL="/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473"
export TFFV_RERANK_REVISION="e61197ed45024b0ed8a2d74b80b4d909f1255473"
export TFFV_RERANK_SHA256="5219a20fa423e1054c34b93a5a8c0ffa640ca52320181603be32d7afe26de353"

OUTPUT_DIR="${1:-outputs/m142-foldd-r4}"

echo "================================================================================"
echo ">>> [M1.4.2] Preparing Fold D R4 Evidence (train:150..249, 100 claims)"
echo ">>> Output Directory: $OUTPUT_DIR"
echo "================================================================================"

python -m tffv.retrieval.prepare_fold_evidence \
  --start-claim 150 \
  --end-claim 250 \
  --tag foldd \
  --output-dir "$OUTPUT_DIR" \
  --execute

echo ">>> Fold D Evidence preparation completed successfully!"
