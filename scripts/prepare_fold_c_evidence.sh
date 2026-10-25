#!/usr/bin/env bash
# ==============================================================================
# Milestone M1.4.1 — Fold C Evidence Preparation Pipeline (R1–R4 Top-200)
#
# Strictly Training-Free, Quality-First (Zero parameter updates, zero learned routers)
# Target Population: Fold C (train:100..149, 50 claims)
# Extraction Source: train_0_999.zip from HuggingFace chenxwh/AVeriTeC
# Retrieval: BM25 (R1) + Qwen3-Embedding-0.6B (R2) + RRF (R3) + Qwen3-Reranker-0.6B (R4)
# Delivery Target: outputs/m141-confirmatory-r4/ (R4.jsonl and claims.jsonl)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 1. Environment and Caches
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/home/stackops/whale/cache/huggingface}"
export PYTHONUNBUFFERED=1

echo "================================================================================"
echo ">>> [M1.4.1] Fold C R4 Evidence Preparation Pipeline (train:100..149)"
echo "================================================================================"

# 2. Pin Frozen Retrieval Models & Hashes (Exact M1.1 Setup)
export TFFV_EMBED_MODEL="/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
export TFFV_EMBED_REVISION="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
export TFFV_EMBED_SHA256="667106c1965ebad2038f612af424b1990ab6e6dabcee5cf20f45ed5c4dee071a"

export TFFV_RERANK_MODEL="/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473"
export TFFV_RERANK_REVISION="e61197ed45024b0ed8a2d74b80b4d909f1255473"
export TFFV_RERANK_SHA256="5219a20fa423e1054c34b93a5a8c0ffa640ca52320181603be32d7afe26de353"

# 3. Preflight Dependency & File Verification
PLAN_FILE="$(find outputs -maxdepth 2 -type f -path 'outputs/m11-corpus-plan-*/acquisition_plan.json' | sort | tail -n 1)"
if [[ ! -f "$PLAN_FILE" ]]; then
    echo "ERROR: Acquisition plan not found: outputs/m11-corpus-plan-*/acquisition_plan.json"
    exit 1
fi
echo ">>> Verified Acquisition Plan: $PLAN_FILE"

CLAIMS_MANIFEST="$(find outputs -maxdepth 2 -type f -path 'outputs/m11-source-claims-*/claims_manifest.json' | sort | tail -n 1)"
if [[ ! -f "$CLAIMS_MANIFEST" ]]; then
    echo "ERROR: Claims manifest not found: outputs/m11-source-claims-*/claims_manifest.json"
    exit 1
fi
echo ">>> Verified Claims Manifest: $CLAIMS_MANIFEST"

if [[ ! -d "$TFFV_EMBED_MODEL" ]]; then
    echo "ERROR: Frozen Qwen3-Embedding snapshot not found at: $TFFV_EMBED_MODEL"
    exit 1
fi

if [[ ! -d "$TFFV_RERANK_MODEL" ]]; then
    echo "ERROR: Frozen Qwen3-Reranker snapshot not found at: $TFFV_RERANK_MODEL"
    exit 1
fi

mkdir -p outputs/index_cache
OUTPUT_DIR="${1:-outputs/m141-confirmatory-r4}"

# 4. Execute End-to-End Fold C Evidence Pipeline
echo ">>> Executing Fold C extraction, candidate preparation, and R1-R4 retrieval..."
python -m tffv.retrieval.prepare_fold_c \
  --plan "$PLAN_FILE" \
  --claims-manifest "$CLAIMS_MANIFEST" \
  --start-claim 100 \
  --end-claim 150 \
  --output-dir "$OUTPUT_DIR" \
  --config configs/store_retrieval_m11.template.json \
  --index-cache outputs/index_cache \
  --execute

echo ""
echo "================================================================================"
echo ">>> [M1.4.1] Fold C R4 Evidence Preparation Completed Successfully!"
echo ">>> R4 Evidence: $OUTPUT_DIR/R4.jsonl"
echo ">>> Claims File: $OUTPUT_DIR/claims.jsonl"
echo ">>> Manifest:    $OUTPUT_DIR/evidence_manifest.json"
echo "================================================================================"
cat "$OUTPUT_DIR/evidence_manifest.json"
