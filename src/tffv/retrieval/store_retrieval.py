"""Claim-scoped retrieval for R1-R4 on isolated candidate knowledge stores.

Ensures that every method searches exclusively within the eligible corpus for
the respective claim, without merging knowledge stores into a global pool.
Supports preflight inspection with blocker detection, bounded smoke testing,
reproducible hashing, and strict label-free validation.
"""
import argparse
from collections import defaultdict
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import re
import time

from tffv.data import read_jsonl
from tffv.provenance import canonical, digest, file_hash, write_json
from tffv.retrieval.benchmark import dump_rows, output_directory, result_record, validate_result
from tffv.retrieval.bm25_store_smoke import load_candidates
from tffv.retrieval.data import segment
from tffv.retrieval.frozen import TASK, QwenFrozen, validate_spec
from tffv.retrieval.metrics import proxies, summarize
from tffv.retrieval.ranking import BM25, Hit, dense_search, ordered, rrf
from tffv.retrieval.synthetic import SyntheticEncoder, SyntheticReranker


METHODS = ("R1", "R2", "R3", "R4")

DEFAULT_CONFIG = dict(
    segmentation=dict(words=180, overlap=30),
    bm25=dict(k1=1.2, b=0.75),
    rrf_constant=60,
    dedup_threshold=0.95,
    candidate_depth=50,
    evidence_depth=8,
    evaluation_depth=10,
    seed=42,
    device="cpu",
    dtype="float32",
    max_length=1024,
    batch_size=4,
    corpus_scope="per_claim",
    protocol="claim_scoped_retrieval_v1",
    embedding=dict(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        local_path=None,
        revision=None,
        snapshot_sha256=None,
    ),
    reranker=dict(
        model_id="Qwen/Qwen3-Reranker-0.6B",
        local_path=None,
        revision=None,
        snapshot_sha256=None,
    ),
)


def expand(value):
    import os
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    return value


def checked_config(config_dict=None):
    cfg = dict(DEFAULT_CONFIG)
    if config_dict:
        config_dict = expand(config_dict)
        for k, v in config_dict.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
    if cfg["rrf_constant"] <= 0 or not 0 < cfg["dedup_threshold"] <= 1:
        raise ValueError("Invalid fusion/deduplication threshold")
    return cfg


def check_prerequisites(config, methods, synthetic=False):
    """Check environment dependencies and model checkpoints; report blockers."""
    blockers = []
    needs_dense = bool(set(methods) & {"R2", "R3", "R4"})
    needs_rerank = "R4" in methods

    if synthetic:
        return blockers

    if needs_dense or needs_rerank:
        # Check transformers
        try:
            version = importlib.metadata.version("transformers")
            nums = tuple(int(n) for n in version.split(".")[:2] if n.isdigit())
            if nums < (4, 51):
                blockers.append(f"transformers {version} is incompatible with Qwen3 (requires >=4.51).")
        except Exception as e:
            blockers.append(f"Missing transformers dependency: {e}")

        # Check torch
        try:
            import torch
            if config.get("device") == "cuda" and not torch.cuda.is_available():
                blockers.append("CUDA requested but torch.cuda.is_available() is False.")
        except Exception as e:
            blockers.append(f"Missing PyTorch dependency: {e}")

        # Check embedding checkpoint
        if needs_dense:
            emb_spec = config.get("embedding", {})
            local_path = emb_spec.get("local_path")
            if not local_path or not Path(local_path).is_dir():
                blockers.append(f"Missing Qwen3-Embedding-0.6B checkpoint at path: {local_path}")
            else:
                try:
                    validate_spec(emb_spec, "Qwen/Qwen3-Embedding-0.6B", verify_bytes=False)
                except Exception as exc:
                    blockers.append(f"Invalid embedding spec: {exc}")

        # Check reranker checkpoint
        if needs_rerank:
            rrk_spec = config.get("reranker", {})
            local_path = rrk_spec.get("local_path")
            if not local_path or not Path(local_path).is_dir():
                blockers.append(f"Missing Qwen3-Reranker-0.6B checkpoint at path: {local_path}")
            else:
                try:
                    validate_spec(rrk_spec, "Qwen/Qwen3-Reranker-0.6B", verify_bytes=False)
                except Exception as exc:
                    blockers.append(f"Invalid reranker spec: {exc}")

    return blockers


def plan(manifest_path, methods=METHODS, config=None, synthetic=False, max_claims=None):
    """Preflight planning for claim-scoped retrieval without running models."""
    config = checked_config(config)
    valid_methods = [m for m in methods if m in METHODS]
    if not valid_methods or len(set(valid_methods)) != len(valid_methods):
        raise ValueError(f"Methods must be a unique subset of {METHODS}")

    _, manifest_hash, loaded = load_candidates(manifest_path)
    if max_claims and max_claims > 0:
        loaded = loaded[:max_claims]

    blockers = check_prerequisites(config, valid_methods, synthetic=synthetic)

    stores = []
    total_docs = 0
    total_est_passages = 0
    words_cfg = config["segmentation"]["words"]
    overlap_cfg = config["segmentation"]["overlap"]
    step = words_cfg - overlap_cfg

    for claim, documents, store, _ in loaded:
        n_passages = 0
        for doc in documents:
            words = sum(1 for _ in re.finditer(r"\S+", doc["text"]))
            n_passages += 1 + max(0, (words - words_cfg + step - 1) // step) if words else 0
        total_docs += len(documents)
        total_est_passages += n_passages
        stores.append(dict(
            sample_id=claim["sample_id"],
            documents=len(documents),
            estimated_passages=n_passages,
        ))

    needs_dense = bool(set(valid_methods) & {"R2", "R3", "R4"})
    needs_rerank = "R4" in valid_methods

    return dict(
        scope="claim_scoped_operational_smoke_not_admitted",
        corpus_scope="per_claim",
        candidate_manifest_sha256=manifest_hash,
        methods=valid_methods,
        synthetic=synthetic,
        n_claims=len(stores),
        total_documents=total_docs,
        total_estimated_passages=total_est_passages,
        stores=stores,
        corpus_embedding_rows=total_est_passages if needs_dense else 0,
        query_embedding_rows=len(stores) if needs_dense else 0,
        reranker_pairs_upper_bound=len(stores) * min(40, total_est_passages) if needs_rerank else 0,
        config=config,
        config_sha256=digest(config),
        corpus_admitted=False,
        corpus_status="complete_candidates_not_admitted",
        quality_benchmark_permitted=False,
        operational_smoke_permitted=True,
        ready_to_execute=len(blockers) == 0,
        blockers=blockers,
        limitations=[
            "Retrieval is isolated to each claim's own candidate store; no global corpus union.",
            "Candidate corpus is unadmitted; official quality metrics cannot be certified.",
            "Preflight validates configuration and input counts only; does not execute GPU inference.",
        ],
    )


def run(manifest_path, output, methods=METHODS, config=None, synthetic=False, max_claims=None):
    """Execute claim-scoped retrieval for requested methods across all candidate stores."""
    config = checked_config(config)
    valid_methods = [m for m in methods if m in METHODS]
    if not valid_methods or len(set(valid_methods)) != len(valid_methods):
        raise ValueError(f"Methods must be a unique subset of {METHODS}")

    preflight = plan(manifest_path, methods=valid_methods, config=config, synthetic=synthetic, max_claims=max_claims)
    if not preflight["ready_to_execute"]:
        raise RuntimeError(f"Cannot execute claim-scoped retrieval due to blockers: {preflight['blockers']}")

    manifest_path, manifest_hash, loaded = load_candidates(manifest_path)
    if max_claims and max_claims > 0:
        loaded = loaded[:max_claims]

    destination = output_directory(output)
    random.seed(config["seed"])
    started_total = time.perf_counter()

    needs_dense = bool(set(valid_methods) & {"R2", "R3", "R4"})
    needs_rerank = "R4" in valid_methods

    records = {m: [] for m in valid_methods}
    diagnostics = []
    store_timings = []
    claims_list = []
    pending_r4 = []
    all_passages_by_store = {}

    encoder = None
    reranker = None
    gpu_info = None

    try:
        # 1. Setup dense encoder if needed
        if needs_dense:
            if synthetic:
                encoder = SyntheticEncoder()
            else:
                import torch
                torch.manual_seed(config["seed"])
                if config["device"] == "cuda":
                    torch.cuda.manual_seed_all(config["seed"])
                    gpu_info = dict(
                        name=torch.cuda.get_device_name(),
                        cuda_runtime=torch.version.cuda,
                        total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                    )
                encoder = QwenFrozen(
                    config["embedding"],
                    device=config["device"],
                    dtype=config["dtype"],
                    max_length=config["max_length"],
                    batch_size=config["batch_size"],
                )

        # 2. Iterate per claim: isolated segmentation, BM25, Dense, RRF
        for claim, documents, store, input_dir in loaded:
            claims_list.append(claim)
            store_dir = destination / store["directory"]
            store_dir.mkdir(parents=True, exist_ok=True)

            t0 = time.perf_counter()
            passages = segment(documents, **config["segmentation"])
            seg_time = time.perf_counter() - t0
            passage_map = {p.passage_id: p for p in passages}
            all_passages_by_store[claim["sample_id"]] = (passages, passage_map)
            dump_rows(store_dir / "passages.jsonl", [p.to_dict() for p in passages])

            timing_entry = dict(
                sample_id=claim["sample_id"],
                documents=len(documents),
                passages=len(passages),
                segmentation_seconds=seg_time,
            )

            # R1 (BM25)
            t0 = time.perf_counter()
            bm25 = BM25(passages, **config["bm25"])
            bm25_idx_time = time.perf_counter() - t0
            timing_entry["bm25_index_seconds"] = bm25_idx_time

            t0 = time.perf_counter()
            b_hits = bm25.search(claim["claim"], config["candidate_depth"])
            b_query_time = time.perf_counter() - t0
            timing_entry["bm25_query_seconds"] = b_query_time

            if "R1" in valid_methods:
                r1_record = result_record(claim, "R1", b_hits, passage_map, config, b_query_time)
                records["R1"].append(r1_record)

            d_hits = []
            d_query_time = 0.0
            dense_peak = None
            if needs_dense:
                t0 = time.perf_counter()
                if encoder:
                    encoder.reset_peak()
                doc_texts = [p.text for p in passages]
                corpus_matrix = encoder.encode(doc_texts)
                query_vec = encoder.encode([claim["claim"]], query=True)[0]
                dense_peak = encoder.peak_bytes() if encoder else None
                p_ids = [p.passage_id for p in passages]
                d_hits = dense_search(query_vec, corpus_matrix, p_ids, config["candidate_depth"])
                d_query_time = time.perf_counter() - t0
                timing_entry["dense_query_seconds"] = d_query_time
                timing_entry["dense_peak_gpu_bytes"] = dense_peak

                if "R2" in valid_methods:
                    r2_record = result_record(claim, "R2", d_hits, passage_map, config, d_query_time, dense_peak)
                    records["R2"].append(r2_record)

            fused_hits = []
            f_time = 0.0
            if "R3" in valid_methods or needs_rerank:
                t0 = time.perf_counter()
                fused_hits = rrf({"bm25": b_hits, "dense": d_hits}, 40, config["rrf_constant"])
                f_time = time.perf_counter() - t0
                timing_entry["rrf_fusion_seconds"] = f_time

                if "R3" in valid_methods:
                    r3_timing = b_query_time + d_query_time + f_time
                    r3_record = result_record(claim, "R3", fused_hits, passage_map, config, r3_timing, dense_peak)
                    records["R3"].append(r3_record)

            if needs_rerank:
                pending_r4.append((claim, fused_hits, b_query_time + d_query_time + f_time, dense_peak, passage_map))

            store_timings.append(timing_entry)

        # 3. Unload Dense Encoder before Reranker to conserve VRAM
        if encoder:
            encoder.close()
            encoder = None
            gc.collect()

        # 4. R4 Reranker Stage
        if needs_rerank:
            if synthetic:
                reranker = SyntheticReranker()
            else:
                reranker = QwenFrozen(
                    config["reranker"],
                    causal=True,
                    device=config["device"],
                    dtype=config["dtype"],
                    max_length=config["max_length"],
                    batch_size=config["batch_size"],
                )

            for claim, fused_hits, base_time, dense_peak, passage_map in pending_r4:
                reranker.reset_peak()
                candidate_texts = [passage_map[h.passage_id].text for h in fused_hits]
                t0 = time.perf_counter()
                scores = reranker.score(claim["claim"], candidate_texts)
                r_time = time.perf_counter() - t0
                if len(scores) != len(fused_hits):
                    raise ValueError("Reranker score count mismatch")
                r4_hits = ordered(
                    [Hit(h.passage_id, float(s), dict(h.component_scores, reranker_yes_probability=float(s)))
                     for h, s in zip(fused_hits, scores)],
                    40,
                )
                peaks = [p for p in (dense_peak, reranker.peak_bytes()) if p is not None]
                r4_peak = max(peaks) if peaks else None
                r4_record = result_record(claim, "R4", r4_hits, passage_map, config, base_time + r_time, r4_peak)
                records["R4"].append(r4_record)

            reranker.close()
            reranker = None
            gc.collect()

        # 5. Build diagnostics & review queue
        claim_texts = {c["sample_id"]: c["claim"] for c in claims_list}
        manual_review = []
        for method in valid_methods:
            for r in records[method]:
                for ev in r["evidence"]:
                    manual_review.append(dict(
                        sample_id=r["sample_id"],
                        method=method,
                        claim=claim_texts[r["sample_id"]],
                        evidence=ev,
                        assessment=None,
                        scope="claim_scoped_operational_evidence_review",
                    ))
                if method == valid_methods[0]:
                    diagnostics.append(dict(sample_id=r["sample_id"], **proxies(r["evidence"])))

        # 6. Dump all outputs
        dump_rows(destination / "claims.jsonl", claims_list)
        for method in valid_methods:
            dump_rows(destination / f"{method}.jsonl", records[method])
        dump_rows(destination / "diagnostics.jsonl", diagnostics)
        dump_rows(destination / "manual_review_queue.jsonl", manual_review)

        # 7. Summary JSON
        method_summaries = {}
        for method in valid_methods:
            m_metrics = [dict(proxies(r["evidence"]), latency_seconds=r["latency_seconds"],
                              peak_gpu_memory_bytes=r["peak_gpu_memory_bytes"]) for r in records[method]]
            method_summaries[method] = summarize(m_metrics)
            method_summaries[method]["relevance_metrics"] = {k: None for k in ("recall@5", "recall@8", "recall@10", "mrr@10", "ndcg@10")}

        write_json(destination / "summary.json", dict(
            scope="claim_scoped_operational_smoke_not_admitted",
            corpus_admitted=False,
            n_claims=len(claims_list),
            methods=method_summaries,
            store_timings=store_timings,
            proxies=summarize(diagnostics),
            elapsed_seconds=time.perf_counter() - started_total,
            gpu_information=gpu_info,
            limitations=[
                "Every retrieval method searched strictly within the respective claim's candidate store.",
                "Candidate corpus is unadmitted; Recall/MRR/nDCG are explicitly null (not measured).",
                "Coverage denotes filled evidence slots, not verified factual completeness.",
                "Manual review queue must be audited for residual answer leakage before considering corpus admission.",
            ],
        ))

        # 8. Manifest JSON
        pkg_root = Path(__file__).resolve().parents[1]
        code_hashes = {str(p.relative_to(pkg_root)).replace("\\", "/"): file_hash(p)
                       for p in sorted(pkg_root.rglob("*.py"))}
        output_hashes = {str(p.relative_to(destination)).replace("\\", "/"): file_hash(p)
                         for p in sorted(destination.rglob("*")) if p.is_file()}

        dep_versions = {}
        for dep in ("torch", "transformers", "numpy"):
            try:
                dep_versions[dep] = importlib.metadata.version(dep)
            except Exception:
                dep_versions[dep] = None

        write_json(destination / "manifest.json", dict(
            status="complete_operational_smoke_not_admitted",
            corpus_admitted=False,
            candidate_manifest_sha256=manifest_hash,
            methods=valid_methods,
            synthetic=synthetic,
            config=config,
            config_sha256=digest(config),
            implementation=code_hashes,
            outputs_sha256=output_hashes,
            python=platform.python_version(),
            dependencies=dep_versions,
            gold_labels_used=False,
            network_accessed=False,
            population="claim_scoped_candidate_stores_operational_only",
        ))

        return destination

    except Exception as exc:
        write_json(destination / "failure.json", dict(
            status="failed_unsealed",
            error_type=type(exc).__name__,
            message=str(exc),
        ))
        raise
    finally:
        if encoder:
            encoder.close()
        if reranker:
            reranker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", required=True, help="Path to candidate_manifest.json")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS), help="Methods to run (R1, R2, R3, R4)")
    parser.add_argument("--config", help="Optional JSON config file to override defaults")
    parser.add_argument("--max-claims", type=int, help="Optional bound on number of claims to process")
    parser.add_argument("--synthetic", action="store_true", help="Use non-neural synthetic test doubles")
    parser.add_argument("--execute", action="store_true", help="Execute retrieval; otherwise prints preflight plan only")
    parser.add_argument("--output", help="Fresh output directory under outputs/ (required with --execute)")
    args = parser.parse_args()

    cfg = None
    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    if args.execute:
        if not args.output:
            parser.error("--execute requires --output")
        out = run(args.candidate_manifest, args.output, methods=args.methods, config=cfg,
                  synthetic=args.synthetic, max_claims=args.max_claims)
        print(f"Run completed successfully: {out}")
    else:
        if args.output:
            parser.error("Preflight prints JSON only; omit --output unless using --execute")
        p = plan(args.candidate_manifest, methods=args.methods, config=cfg,
                 synthetic=args.synthetic, max_claims=args.max_claims)
        print(json.dumps(p, indent=2))


if __name__ == "__main__":
    main()
