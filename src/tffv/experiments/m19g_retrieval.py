"""Milestone M1.9g: Frozen R4-PF1000 Retrieval Pipeline for Official Dev 500.

Executes:
1. Strict pre-flight knowledge store verification.
2. Frozen R4-PF1000-v1 retrieval (BM25 K=1000 + Dense Top-200 + RRF 60 + Reranker Top-200 + Top-8 dedup 0.95).
3. Emission of dev_retrieval_receipts.jsonl with hash and passage integrity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tffv.experiments.m19g_guardrails import (
    KnowledgeStoreUnverifiedError,
    RetrievalReceiptsMissingError,
    verify_knowledge_store,
    verify_retrieval_receipts,
)


def load_dev_knowledge_documents(
    sid: str,
    store_dir: Path,
) -> List[Dict[str, Any]]:
    """Loads documents for claim from verified uncompressed dev knowledge store."""
    ordinal = sid.split(":")[1] if ":" in sid else sid
    claim_file = store_dir / f"{ordinal}.json"
    
    if not claim_file.exists():
        raise KnowledgeStoreUnverifiedError(f"Missing knowledge store file: {claim_file}")

    documents = []
    with open(claim_file, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                url = row.get("url") or row.get("source_url") or f"https://example.org/{sid}/{line_no}"
                text = ""
                if row.get("text"):
                    text = str(row.get("text"))
                elif row.get("url2text"):
                    u2t = row.get("url2text")
                    text = "\n".join(u2t) if isinstance(u2t, list) else str(u2t)

                if not text.strip():
                    continue

                doc_id = row.get("document_id") or hashlib.sha256(f"{sid}:{line_no}:{url}".encode("utf-8")).hexdigest()[:16]
                documents.append({
                    "document_id": doc_id,
                    "source_url": url,
                    "source_kind": "web_document",
                    "retrieved_at": "unknown",
                    "acquisition_id": f"acq_{sid}",
                    "text": text,
                })
            except Exception:
                continue

    return documents


class M19gRetrievalPipeline:
    def __init__(
        self,
        dev_path: str | Path = "data/raw/averitec/dev.json",
        dev_store_dir: str | Path = "data_store/knowledge_store/dev/uncompressed",
        output_dir: str | Path = "outputs/m19g-evidence-complete-v1",
        run_gpu: bool = False,
    ):
        self.dev_path = Path(dev_path)
        self.dev_store_dir = Path(dev_store_dir)
        self.output_dir = Path(output_dir)
        self.run_gpu = run_gpu
        self.receipts_file = self.output_dir / "dev_retrieval_receipts.jsonl"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, limit: Optional[int] = None) -> Dict[str, Any]:
        print("======================================================================")
        print(" Milestone M1.9g: Frozen R4-PF1000 Retrieval for Official Dev 500")
        print(f" Target Knowledge Store: {self.dev_store_dir}")
        print(f" Output Receipts: {self.receipts_file}")
        print(f" Execution Mode: {'GPU (Frozen Qwen3-Embedding & Reranker)' if self.run_gpu else 'CPU Mock Simulation'}")
        print("======================================================================")

        t0 = time.time()

        # Step 1: Load raw dev claims
        if not self.dev_path.exists():
            raise FileNotFoundError(f"Canonical dev.json not found: {self.dev_path}")
        dev_data = json.loads(self.dev_path.read_text(encoding="utf-8"))
        claims = [
            {"sample_id": f"dev:{idx}", "claim": item["claim"], "idx": idx}
            for idx, item in enumerate(dev_data)
        ]
        if limit:
            claims = claims[:limit]
        sample_ids = [c["sample_id"] for c in claims]

        # Step 2: Strict Pre-flight Knowledge Store Verification
        print(f"\n[M1.9g Retrieval] >>> Step 1: Verifying Knowledge Store for {len(sample_ids)} claims...")
        ks_summary = verify_knowledge_store(sample_ids, self.dev_store_dir)
        print(f"[M1.9g Retrieval] Knowledge Store VERIFIED: {ks_summary.get('total_docs', 0)} documents indexed.")

        # Step 3: Check existing receipts for resumption
        existing_receipts: Dict[str, Dict[str, Any]] = {}
        if self.receipts_file.exists():
            with open(self.receipts_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        existing_receipts[r["sample_id"]] = r

        pending_claims = [c for c in claims if c["sample_id"] not in existing_receipts]
        print(f"[M1.9g Retrieval] Total claims: {len(claims)} | Cached: {len(existing_receipts)} | Pending: {len(pending_claims)}")

        # Step 4: Execute retrieval if any pending
        if pending_claims:
            arm_runner = self._init_retrieval_worker()
            try:
                for idx_p, c in enumerate(pending_claims):
                    sid = c["sample_id"]
                    claim_text = c["claim"]
                    documents = load_dev_knowledge_documents(sid, self.dev_store_dir)

                    if not documents:
                        raise KnowledgeStoreUnverifiedError(f"Zero documents found for claim {sid} during retrieval!")

                    res, timings = arm_runner.retrieve({"sample_id": sid, "claim": claim_text}, documents)

                    receipt_entry = {
                        "sample_id": sid,
                        "method": "R4-PF1000-v1",
                        "evidence": res.get("evidence", []),
                        "latency_seconds": res.get("latency_seconds", 0.0),
                        "source": "DEV_PF1000_RUN",
                    }

                    with open(self.receipts_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(receipt_entry, ensure_ascii=False) + "\n")

                    existing_receipts[sid] = receipt_entry

                    if (idx_p + 1) % 25 == 0 or idx_p == len(pending_claims) - 1:
                        elapsed = time.time() - t0
                        rate = (idx_p + 1) / elapsed if elapsed > 0 else 0
                        print(f"[M1.9g Retrieval] Progress: [{len(existing_receipts)}/{len(claims)}] retrieved ({rate:.1f} claims/s)...")
            finally:
                arm_runner.close()

        # Step 5: Post-retrieval Verification
        print("\n[M1.9g Retrieval] >>> Step 5: Verifying Generated Retrieval Receipts...")
        receipts_summary = verify_retrieval_receipts(sample_ids, self.receipts_file, require_all_claims=True)
        print(f"[M1.9g Retrieval] Retrieval receipts successfully verified: {receipts_summary['total_receipts']} receipts, {receipts_summary['total_passages']} passages.")

        return {
            "status": "COMPLETED",
            "total_claims": len(claims),
            "receipts_path": str(self.receipts_file),
            "elapsed_seconds": time.time() - t0,
            "receipts_summary": receipts_summary,
        }

    def _init_retrieval_worker(self) -> Any:
        from tffv.retrieval.m18d_arms import ArmRunner

        retrieval_cfg = {
            "candidate_depth": 1000,
            "fusion_depth": 200,
            "rerank_depth": 200,
            "evidence_depth": 8,
            "dedup_threshold": 0.95,
            "rrf_constant": 60,
            "bm25": {"k1": 0.9, "b": 0.4},
            "segmentation": {"words": 300, "overlap": 60},
            "device": "cuda" if self.run_gpu else "cpu",
            "dtype": "float16",
        }

        if not self.run_gpu:
            from tffv.retrieval.synthetic import SyntheticEncoder, SyntheticReranker
            return ArmRunner(
                arm_id="R4_BM25_PF1000",
                retrieval_cfg=retrieval_cfg,
                reranker=SyntheticReranker(),
                qwen_encoder=SyntheticEncoder(),
                synthetic=True,
            )

        from tffv.retrieval.frozen import QwenFrozen
        emb_revision = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
        rerank_revision = "e61197ed45024b0ed8a2d74b80b4d909f1255473"

        emb_spec = {
            "model_id": "Qwen/Qwen3-Embedding-0.6B",
            "revision": emb_revision,
            "snapshot_sha256": "667106c1965ebad2038f612af424b1990ab6e6dabcee5cf20f45ed5c4dee071a",
            "local_path": f"/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/{emb_revision}",
        }
        rerank_spec = {
            "model_id": "Qwen/Qwen3-Reranker-0.6B",
            "revision": rerank_revision,
            "snapshot_sha256": "5219a20fa423e1054c34b93a5a8c0ffa640ca52320181603be32d7afe26de353",
            "local_path": f"/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/{rerank_revision}",
        }

        print("[M1.9g Retrieval] Loading frozen Qwen3-Embedding and Reranker onto GPU...")
        qwen_enc = QwenFrozen(emb_spec, device="cuda", dtype="float16", batch_size=32)
        reranker = QwenFrozen(rerank_spec, causal=True, device="cuda", dtype="float16", batch_size=32)

        return ArmRunner(
            arm_id="R4_BM25_PF1000",
            retrieval_cfg=retrieval_cfg,
            reranker=reranker,
            qwen_encoder=qwen_enc,
            device="cuda",
            synthetic=False,
        )


def main():
    parser = argparse.ArgumentParser(description="M1.9g Official Dev Retrieval Pipeline")
    parser.add_argument("--dev-path", default="data/raw/averitec/dev.json")
    parser.add_argument("--dev-store-dir", default="data_store/knowledge_store/dev/uncompressed")
    parser.add_argument("--output-dir", default="outputs/m19g-evidence-complete-v1")
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pipeline = M19gRetrievalPipeline(
        dev_path=args.dev_path,
        dev_store_dir=args.dev_store_dir,
        output_dir=args.output_dir,
        run_gpu=args.run_gpu,
    )
    pipeline.run(limit=args.limit)


if __name__ == "__main__":
    main()
