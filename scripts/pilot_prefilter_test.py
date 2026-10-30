#!/usr/bin/env python3
"""Pilot Study: BM25 Pre-filtering for Dense Retrieval Acceleration.

Compares standard R4 retrieval against BM25 Pre-filtered Dense R4 across
the 25 official Part A canonical claims to verify Top-8 evidence equivalence,
score difference, and speedup factor.
"""
import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from tffv.provenance import canonical, write_json
from tffv.retrieval.bm25 import BM25
from tffv.retrieval.dense import dense_search
from tffv.retrieval.fusion import Hit, ordered, rrf
from tffv.retrieval.optimized_runner import OptimizedRetrievalRunner
from scripts.benchmark_m18b1_real_gpu import compare_r4_equivalence, load_canonical_r4_cache, prepare_claim_input_and_candidates


class FilteredDenseRunner(OptimizedRetrievalRunner):
    """Retrieval runner with configurable BM25 candidate pre-filtering for Dense search."""

    def __init__(self, *args, bm25_prefilter_k: int = 2000, **kwargs):
        super().__init__(*args, **kwargs)
        self.bm25_prefilter_k = bm25_prefilter_k

    def retrieve_claim(self, claim: Dict[str, Any], documents: List[Dict[str, Any]], output_dir: Path = None):
        from tffv.retrieval.data import segment
        from tffv.retrieval.store_retrieval import result_record

        self.profiler.start_claim()
        is_cold = not self.models_initialized
        if not self.models_initialized:
            with self.profiler.measure("model_initialization"):
                self.initialize_models()

        with self.profiler.measure("corpus_loading"):
            passages = segment(documents, **self.config["segmentation"])
            passage_map = {p.passage_id: p for p in passages}

        # 1. R1 BM25
        with self.profiler.measure("bm25"):
            bm25 = BM25(passages, **self.config["bm25"])
            b_hits = bm25.search(claim["claim"], self.config.get("candidate_depth", 200))
            
            # Prefilter top-K candidate passages for Dense search if corpus exceeds prefilter_k
            if len(passages) > self.bm25_prefilter_k:
                prefilter_hits = bm25.search(claim["claim"], self.bm25_prefilter_k)
                dense_target_passages = [passage_map[h.passage_id] for h in prefilter_hits]
            else:
                dense_target_passages = passages

        # 2. R2 Dense Search on BM25 pre-filtered subset
        with self.profiler.measure("dense_retrieval"):
            if self.encoder:
                self.encoder.reset_peak()
            doc_texts = [p.text for p in dense_target_passages]
            corpus_matrix = self.encoder.encode(doc_texts)
            query_vec = self.encoder.encode([claim["claim"]], query=True)[0]
            dense_peak = self.encoder.peak_bytes() if self.encoder else None
            p_ids = [p.passage_id for p in dense_target_passages]
            d_hits = dense_search(query_vec, corpus_matrix, p_ids, self.config.get("candidate_depth", 200))

        # 3. R3 RRF Fusion
        with self.profiler.measure("rrf_fusion"):
            fusion_depth = self.config.get("fusion_depth", self.config.get("candidate_depth", 200))
            fused_hits = rrf({"bm25": b_hits, "dense": d_hits}, fusion_depth, self.config.get("rrf_constant", 60))

        if self.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        # 4. R4 Cross-Encoder Reranking
        with self.profiler.measure("cross_encoder_reranking"):
            if self.reranker:
                self.reranker.reset_peak()
            candidate_texts = [passage_map[h.passage_id].text for h in fused_hits]
            scores = self.reranker.score(claim["claim"], candidate_texts)
            rerank_depth = self.config.get("rerank_depth", len(fused_hits))
            r4_hits = ordered(
                [Hit(h.passage_id, float(s), dict(h.component_scores, reranker_yes_probability=float(s)))
                 for h, s in zip(fused_hits, scores)],
                rerank_depth,
            )
            peaks = [p for p in (dense_peak, self.reranker.peak_bytes() if self.reranker else None) if p is not None]
            r4_peak = max(peaks) if peaks else None

        with self.profiler.measure("serialization"):
            r4_record = result_record(
                claim=claim,
                method="R4",
                hits=r4_hits,
                passages=passage_map,
                config=self.config,
                timing=self.profiler.stages.get("cross_encoder_reranking", None).wall_seconds if "cross_encoder_reranking" in self.profiler.stages else 0.0,
                gpu_peak=r4_peak,
            )
            if output_dir:
                out = Path(output_dir).resolve()
                out.mkdir(parents=True, exist_ok=True)
                with (out / "R4.jsonl").open("w", encoding="utf-8") as f:
                    f.write(canonical(r4_record) + "\n")

        prof = self.profiler.end_claim()
        prof["sample_id"] = claim["sample_id"]
        prof["passages_count"] = len(passages)
        prof["dense_evaluated_passages"] = len(dense_target_passages)
        r4_record["latency_seconds"] = prof["wall_time_seconds"]
        return r4_record, prof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=2000, help="BM25 pre-filter depth for Dense (default 2000)")
    parser.add_argument("--claims-count", type=int, default=10, help="Number of claims to pilot (default 10)")
    parser.add_argument("--output-dir", default="outputs/pilot_prefilter", help="Output directory")
    args = parser.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path("configs/m18b1_benchmark_manifest.json").read_text(encoding="utf-8"))
    part_a_claims = manifest["part_a_canonical_equivalence"]["claims"][:args.claims_count]
    canonical_cache = load_canonical_r4_cache()

    raw_train = json.loads(Path("data/raw/averitec/train.json").read_text(encoding="utf-8"))
    cfg = json.loads(Path("configs/m161_source_batch25.json").read_text(encoding="utf-8"))["retrieval"]

    runner = FilteredDenseRunner(
        config=cfg,
        reranker_batch_size=16,
        dense_batch_size=64,
        bm25_prefilter_k=args.k,
    )

    results = []
    print("=" * 80)
    print(f"PILOT EVALUATION: BM25 Pre-filtering Top-{args.k} on {len(part_a_claims)} Canonical Claims")
    print("=" * 80)

    try:
        for idx, item in enumerate(part_a_claims, 1):
            sid = item["sample_id"]
            claim_work_dir = out / sid.replace(":", "-")

            claim_meta, documents = prepare_claim_input_and_candidates(
                sid=sid,
                raw_train=raw_train,
                acq_plan_path=Path("configs/m161_acquisition_plan.json"),
                claim_work_dir=claim_work_dir,
            )

            rec, prof = runner.retrieve_claim(claim_meta, documents, claim_work_dir / "retrieval")
            can_rec = canonical_cache[sid]["record"]

            comp = compare_r4_equivalence(rec, can_rec, score_tolerance=1e-4)
            comp["wall_time_seconds"] = prof["wall_time_seconds"]
            comp["dense_time_seconds"] = prof.get("dense_retrieval_seconds", 0.0)
            comp["total_passages"] = prof["passages_count"]
            comp["dense_passages"] = prof["dense_evaluated_passages"]
            results.append(comp)

            status = "✅ MATCH" if comp["status"] == "MATCH" else "❌ MISMATCH"
            print(
                f"[{idx}/{len(part_a_claims)}] {sid}: {status} | total={prof['wall_time_seconds']:.2f}s "
                f"(dense={prof.get('dense_retrieval_seconds', 0.0):.2f}s, "
                f"passages={prof['passages_count']} -> dense_k={prof['dense_evaluated_passages']}) "
                f"max_diff={comp['max_score_diff']:.2e}",
                flush=True,
            )
    finally:
        runner.close()

    matched = sum(1 for r in results if r["status"] == "MATCH")
    total = len(results)
    avg_lat = sum(r["wall_time_seconds"] for r in results) / total
    avg_dense = sum(r["dense_time_seconds"] for r in results) / total

    summary = {
        "prefilter_k": args.k,
        "total_claims": total,
        "matched_claims": matched,
        "match_rate": matched / total,
        "avg_wall_time_seconds": avg_lat,
        "avg_dense_time_seconds": avg_dense,
        "results": results,
    }
    write_json(out / "pilot_summary.json", summary)

    print("\n" + "=" * 80)
    print(f"PILOT SUMMARY (k={args.k}):")
    print(f"- Match Rate against Canonical R4: {matched}/{total} ({matched/total*100:.1f}%)")
    print(f"- Average Latency per Claim: {avg_lat:.2f}s (Dense: {avg_dense:.2f}s)")
    print("=" * 80)


if __name__ == "__main__":
    main()
