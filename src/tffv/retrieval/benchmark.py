"""R1-R4 orchestration. Relevance annotations are intentionally not an input."""
import importlib.metadata
import json
import os
import platform
import random
import time
from pathlib import Path
from tffv.provenance import canonical, digest, file_hash, write_json
from tffv.retrieval.data import leakage_scan, load_inputs, segment
from tffv.retrieval.frozen import TASK, QwenFrozen, validate_spec
from tffv.retrieval.metrics import proxies, summarize
from tffv.retrieval.ranking import BM25, Hit, dense_search, ordered, rrf, select_evidence


METHODS = ("R1", "R2", "R3", "R4")


def output_directory(path, create=True):
    path = Path(path).expanduser().resolve()
    allowed = Path(__file__).resolve().parents[3] / "outputs"
    if path == allowed.resolve() or not path.is_relative_to(allowed.resolve()):
        raise ValueError("Use a fresh subdirectory of the NEW project's outputs/")
    if create:
        path.mkdir(parents=True, exist_ok=False)
    return path


def dump_rows(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")


def checked_config(config):
    required = {"claims_path", "corpus_path", "data_manifest_path", "segmentation", "bm25", "rrf_constant",
                "dedup_threshold", "seed", "device", "dtype", "max_length", "batch_size", "embedding", "reranker"}
    if set(config) != required:
        raise ValueError(f"Unknown/missing config fields: {set(config) ^ required}")
    if config["rrf_constant"] <= 0 or not 0 < config["dedup_threshold"] <= 1:
        raise ValueError("Invalid fusion/dedup settings")
    if set(config["segmentation"]) != {"words", "overlap"} or set(config["bm25"]) != {"k1", "b"}:
        raise ValueError("Unknown segmentation/BM25 parameters")
    return config


def prepare(config, synthetic=False):
    checked_config(config)
    manifest = json.loads(Path(config["data_manifest_path"]).read_text(encoding="utf-8"))
    claims, documents = load_inputs(config["claims_path"], config["corpus_path"], manifest, synthetic)
    passages = segment(documents, **config["segmentation"])
    if not passages:
        raise ValueError("No passages after segmentation")
    scan = leakage_scan(claims, documents)
    if scan["verdict_pattern_documents"]:
        raise ValueError("Verdict-pattern contamination: review/sanitize corpus upstream with provenance")
    return claims, passages, manifest, scan


def plan(config, methods=METHODS, max_claims=None, synthetic=False, max_passages=None):
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= set(METHODS):
        raise ValueError("Select unique method IDs R1-R4")
    claims, passages, manifest, scan = prepare(config, synthetic)
    if max_claims is not None and (type(max_claims) is not int or max_claims < 1):
        raise ValueError("max_claims must be positive")
    if max_claims:
        claims = claims[:max_claims]
    if max_passages is not None:
        if type(max_passages) is not int or max_passages < 1:
            raise ValueError("max_passages must be positive")
        passages = passages[:max_passages]
    needs_dense = bool(set(methods) & {"R2", "R3", "R4"})
    models = {}
    if not synthetic:
        for key, model_id in (("embedding", "Qwen/Qwen3-Embedding-0.6B"), ("reranker", "Qwen/Qwen3-Reranker-0.6B")):
            if (key == "embedding" and needs_dense) or (key == "reranker" and "R4" in methods):
                models[key] = str(validate_spec(config[key], model_id))
    n, p = len(claims), len(passages)
    return {
        "claims": n, "passages": p, "methods": list(methods), "split_counts": {s: sum(r["split"] == s for r in claims) for s in sorted({r["split"] for r in claims})},
        "synthetic": synthetic, "model_paths_verified_exist": models,
        "scope": "bounded_smoke_not_quality_benchmark" if max_passages or max_claims else "full_declared_population",
        "corpus_embedding_rows": p if needs_dense else 0,
        "query_embedding_rows": n if needs_dense else 0,
        "reranker_pairs_upper_bound": n * min(40, p) if "R4" in methods else 0,
        "embedding_storage_bytes_float32": p * 1024 * 4 if needs_dense else 0,
        "estimated_runtime_seconds": None, "estimated_peak_gpu_memory_bytes": None,
        "claim_substring_flags": len(scan["claim_text_in_documents"]),
        "note": "Preflight does not load models or infer. Resource capacity must be checked on the server.",
    }


def result_record(claim, method, hits, passages, config, timing, gpu_peak=None):
    final_top10, suppressed = select_evidence(hits, passages, 10, config["dedup_threshold"])
    selected = final_top10[:8]
    evidence = [dict(passages[h.passage_id].to_dict(), rank=i, score=h.score,
                     score_type={"R1": "bm25", "R2": "cosine", "R3": "rrf", "R4": "yes_probability"}[method],
                     component_scores=h.component_scores) for i, h in enumerate(selected, 1)]
    store_depth = config.get("ranking_store_depth", 100)
    saved_hits = hits[:store_depth] if store_depth else hits
    result = dict(schema_version=1, sample_id=claim["sample_id"], dataset=claim["dataset"], split=claim["split"],
                  claim_sha256=digest(claim["claim"]), method=method, requested_evidence=8,
                  evidence=evidence, shortfall=max(0, 8 - len(evidence)),
                  ranked_candidates=[dict(passage_id=h.passage_id, rank=i, score=h.score, component_scores=h.component_scores)
                                     for i, h in enumerate(saved_hits, 1)],
                  evaluation_ranking_top10=[dict(passage_id=h.passage_id, rank=i, score=h.score) for i, h in enumerate(final_top10, 1)],
                  dedup_suppressed=suppressed, latency_seconds=timing,
                  peak_gpu_memory_bytes=gpu_peak, gold_accessed=False)
    validate_result(result, passages)
    return result


def validate_result(row, passages):
    required = {"schema_version", "sample_id", "dataset", "split", "claim_sha256", "method", "requested_evidence",
                "evidence", "shortfall", "ranked_candidates", "evaluation_ranking_top10", "dedup_suppressed", "latency_seconds", "peak_gpu_memory_bytes", "gold_accessed"}
    if set(row) != required or row["method"] not in METHODS or row["gold_accessed"] is not False:
        raise ValueError("Invalid result schema")
    import math
    if row["schema_version"] != 1 or row["requested_evidence"] != 8:
        raise ValueError("Unsupported result version or evidence depth")
    for key in ("sample_id", "dataset", "split", "claim_sha256"):
        if not isinstance(row[key], str) or not row[key]:
            raise ValueError("Invalid result identity")
    peak = row["peak_gpu_memory_bytes"]
    if peak is not None and (type(peak) is not int or peak < 0):
        raise ValueError("Invalid peak GPU memory")
    if type(row["latency_seconds"]) not in (int, float) or not math.isfinite(row["latency_seconds"]) or row["latency_seconds"] < 0:
        raise ValueError("Invalid latency")
    evidence, ranking = row["evidence"], row["ranked_candidates"]
    if len(evidence) > 8 or row["shortfall"] != 8 - len(evidence):
        raise ValueError("Invalid Top-8 cardinality")
    ids = [r["passage_id"] for r in ranking]
    if len(set(ids)) != len(ids) or any(pid not in passages for pid in ids):
        raise ValueError("Unknown/duplicate candidate IDs")
    maximum = 200
    if len(ids) > maximum or [r["rank"] for r in ranking] != list(range(1, len(ranking) + 1)):
        raise ValueError("Invalid candidate depth/order")
    if any(not math.isfinite(r["score"]) for r in ranking) or any(a["score"] < b["score"] for a, b in zip(ranking, ranking[1:])):
        raise ValueError("Invalid candidate scores")
    final = row["evaluation_ranking_top10"]
    final_ids = [r["passage_id"] for r in final]
    if len(final) > 10 or len(set(final_ids)) != len(final_ids) or not set(final_ids) <= set(ids):
        raise ValueError("Invalid deduplicated evaluation ranking")
    if [r["rank"] for r in final] != list(range(1, len(final) + 1)) or [r["passage_id"] for r in evidence] != final_ids[:8]:
        raise ValueError("Top-8 evidence must be the prefix of the evaluation ranking")
    candidates_by_id = {r["passage_id"]: r for r in ranking}
    if any(r["score"] != candidates_by_id[r["passage_id"]]["score"] for r in final):
        raise ValueError("Evaluation/candidate score mismatch")
    if [ids.index(pid) for pid in final_ids] != sorted(ids.index(pid) for pid in final_ids):
        raise ValueError("Deduplication must preserve candidate order")
    seen = set()
    for rank, item in enumerate(evidence, 1):
        if item["passage_id"] not in ids or item["passage_id"] in seen or item["rank"] != rank:
            raise ValueError("Invalid evidence IDs/ranks")
        seen.add(item["passage_id"])
        p = passages[item["passage_id"]]
        if set(item) != set(p.to_dict()) | {"rank", "score", "score_type", "component_scores"}:
            raise ValueError("Unknown/missing evidence fields")
        if item["score"] != candidates_by_id[item["passage_id"]]["score"]:
            raise ValueError("Evidence/candidate score mismatch")
        if any(item.get(k) != v for k, v in p.to_dict().items()):
            raise ValueError("Evidence provenance/content mismatch")


def _timed(function, backend=None):
    if backend:
        backend.sync()
    start = time.perf_counter()
    value = function()
    if backend:
        backend.sync()
    return value, time.perf_counter() - start


def corpus_matrix(encoder, passages, config, cache_root, synthetic, implementation_hash):
    """Shared immutable embeddings; interrupted caches are rejected, never overwritten."""
    import numpy as np
    identity = dict(passages_sha256=digest([p.to_dict() for p in passages]),
                    model=config["embedding"], dtype=config["dtype"], max_length=config["max_length"],
                    batch_size=config["batch_size"], seed=config["seed"], synthetic=synthetic,
                    implementation_sha256=implementation_hash)
    directory = output_directory(cache_root, create=False) / digest(identity)
    metadata_path = directory / "index_manifest.json"
    if directory.exists():
        if not metadata_path.is_file():
            raise ValueError("Unsealed embedding index exists; choose a separate index cache directory")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["identity"] != identity or file_hash(directory / "vectors.npy") != metadata["vectors_sha256"]:
            raise ValueError("Embedding index integrity mismatch")
        matrix = np.load(directory / "vectors.npy", mmap_mode="r", allow_pickle=False)
        if matrix.shape != (len(passages), 1024):
            raise ValueError("Cached embedding shape mismatch")
        return matrix, dict(cache_hit=True, path=str(directory), **metadata)
    directory.mkdir(parents=True, exist_ok=False)
    matrix = np.lib.format.open_memmap(directory / "vectors.npy", mode="w+", dtype="float32", shape=(len(passages), 1024))
    start = time.perf_counter()
    for offset in range(0, len(passages), config["batch_size"]):
        vectors = encoder.encode([p.text for p in passages[offset:offset + config["batch_size"]]])
        if vectors.shape != (min(config["batch_size"], len(passages) - offset), 1024):
            raise ValueError("Expected 1024-dimensional embeddings")
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
            raise ValueError("Invalid/non-normalized corpus embeddings")
        matrix[offset:offset + len(vectors)] = vectors
    encoder.sync()
    matrix.flush()
    metadata = dict(identity=identity, vectors_sha256=file_hash(directory / "vectors.npy"),
                    build_seconds=time.perf_counter() - start, index_peak_gpu_memory_bytes=encoder.peak_bytes(),
                    passage_ids_sha256=digest([p.passage_id for p in passages]))
    write_json(metadata_path, metadata)
    return matrix, dict(cache_hit=False, path=str(directory), **metadata)


def run(config, output, methods=METHODS, max_claims=None, synthetic_backends=None, index_cache=None, max_passages=None):
    synthetic = synthetic_backends is not None
    workload = plan(config, methods, max_claims, synthetic, max_passages)
    claims, passages, data_manifest, scan = prepare(config, synthetic)
    if max_claims:
        claims = claims[:max_claims]
    if max_passages:
        passages = passages[:max_passages]
    destination = output_directory(output)
    dump_rows(destination / "claims.jsonl", claims)
    dump_rows(destination / "passages.jsonl", [p.to_dict() for p in passages])
    write_json(destination / "leakage_scan.json", scan)
    write_json(destination / "workload.json", workload)
    passage_map = {p.passage_id: p for p in passages}
    ids = [p.passage_id for p in passages]
    package = Path(__file__).resolve().parents[1]
    implementation = {str(p.relative_to(package)).replace("\\", "/"): file_hash(p) for p in sorted(package.rglob("*.py"))}
    sealed_config = dict(config=config, methods=list(methods), max_claims=max_claims, max_passages=max_passages, instruction=TASK,
                         synthetic=synthetic, implementation_sha256=digest(implementation))
    input_hashes = {key: file_hash(config[key]) for key in ("claims_path", "corpus_path", "data_manifest_path")}
    random.seed(config["seed"])
    needs_dense = bool(set(methods) & {"R2", "R3", "R4"})
    encoder = reranker = None
    setup = {}
    gpu_information = None
    records = {method: [] for method in methods}
    pending_r4 = []
    try:
        bm25, setup["bm25_index_seconds"] = _timed(lambda: BM25(passages, **config["bm25"]))
        if needs_dense:
            import numpy as np
            np.random.seed(config["seed"])
            if synthetic:
                encoder = synthetic_backends[0]
            else:
                import torch
                torch.manual_seed(config["seed"])
                if config["device"] == "cuda":
                    torch.cuda.manual_seed_all(config["seed"])
                    gpu_information = dict(name=torch.cuda.get_device_name(), cuda_runtime=torch.version.cuda,
                                           total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
                load_started = time.perf_counter()
                encoder = QwenFrozen(config["embedding"], device=config["device"], dtype=config["dtype"],
                                     max_length=config["max_length"], batch_size=config["batch_size"])
                setup["embedding_load_and_verify_seconds"] = time.perf_counter() - load_started
            encoder.reset_peak()
            matrix, index_metadata = corpus_matrix(encoder, passages, config,
                index_cache or destination.parent / "retrieval_index_cache", synthetic, digest(implementation))
            write_json(destination / "embedding_index_reference.json", index_metadata)
            setup["corpus_embedding_seconds_this_run"] = 0. if index_metadata["cache_hit"] else index_metadata["build_seconds"]
            setup["index_peak_gpu_memory_bytes"] = index_metadata["index_peak_gpu_memory_bytes"]
        for claim in claims:
            b_hits, b_time = _timed(lambda: bm25.search(claim["claim"], 50))
            if "R1" in methods:
                records["R1"].append(result_record(claim, "R1", b_hits, passage_map, config, b_time))
            if needs_dense:
                encoder.reset_peak()
                d_hits, d_time = _timed(lambda: dense_search(encoder.encode([claim["claim"]], query=True)[0], matrix, ids, 50), encoder)
                peak = encoder.peak_bytes()
                if "R2" in methods:
                    records["R2"].append(result_record(claim, "R2", d_hits, passage_map, config, d_time, peak))
                fused, f_time = _timed(lambda: rrf({"bm25": b_hits, "dense": d_hits}, 40, config["rrf_constant"]))
                if "R3" in methods:
                    records["R3"].append(result_record(claim, "R3", fused, passage_map, config, b_time + d_time + f_time, peak))
                if "R4" in methods:
                    pending_r4.append((claim, fused, b_time + d_time + f_time, peak))
        # Seal intermediate non-reranked results so a failed R4 does not erase R1-R3.
        for method in methods:
            if method != "R4":
                dump_rows(destination / f"{method}.jsonl", records[method])
        if encoder:
            setup["embedding_forward_passes"] = encoder.forward_passes
            setup["embedding_tokens"] = encoder.token_count
            setup["embedding_truncated_inputs"] = encoder.truncated_inputs
            encoder.close()
            encoder = None
            del matrix
        if "R4" in methods:
            load_started = time.perf_counter()
            reranker = synthetic_backends[1] if synthetic else QwenFrozen(config["reranker"], causal=True,
                device=config["device"], dtype=config["dtype"], max_length=config["max_length"], batch_size=config["batch_size"])
            setup["reranker_load_and_verify_seconds"] = time.perf_counter() - load_started if not synthetic else None
            for claim, fused, base_time, dense_peak in pending_r4:
                reranker.reset_peak()
                scores, r_time = _timed(lambda: reranker.score(claim["claim"], [passage_map[h.passage_id].text for h in fused]), reranker)
                if len(scores) != len(fused):
                    raise ValueError("Reranker score alignment failure")
                hits = ordered([Hit(h.passage_id, float(s), dict(h.component_scores, reranker_yes_probability=float(s))) for h, s in zip(fused, scores)], 40)
                values = [v for v in (dense_peak, reranker.peak_bytes()) if v is not None]
                records["R4"].append(result_record(claim, "R4", hits, passage_map, config, base_time + r_time, max(values) if values else None))
            setup["reranker_forward_passes"] = reranker.forward_passes
            setup["reranker_tokens"] = reranker.token_count
            setup["reranker_truncated_inputs"] = reranker.truncated_inputs
            dump_rows(destination / "R4.jsonl", records["R4"])
        summary = {}
        for method, rows in records.items():
            metrics = [dict(proxies(r["evidence"]), latency_seconds=r["latency_seconds"], peak_gpu_memory_bytes=r["peak_gpu_memory_bytes"]) for r in rows]
            summary[method] = summarize(metrics)
            summary[method]["relevance_metrics"] = None
        write_json(destination / "summary.json", {"synthetic": synthetic, "methods": summary, "setup": setup,
            "limitations": ["Coverage is filled evidence slots, not semantic completeness.",
                             "Latency components reused across R1-R4; model load/indexing reported separately; dedup/serialization not in query timing.",
                             "No qrels were accessed; relevance is evaluation-only."]})
        for key, expected in input_hashes.items():
            if file_hash(config[key]) != expected:
                raise ValueError("Input changed during retrieval; run is unsealed")
        versions = {}
        for name in ("torch", "transformers", "numpy"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        hashes = {p.name: file_hash(p) for p in sorted(destination.iterdir()) if p.is_file()}
        write_json(destination / "manifest.json", {"status": "complete", "experiment_id": destination.name,
            "config": sealed_config, "config_sha256": digest(sealed_config), "data_manifest": data_manifest,
            "input_sha256": input_hashes, "outputs_sha256": hashes, "implementation": implementation,
            "python": platform.python_version(), "dependencies": versions,
            "gpu_information": gpu_information,
            "training_free_verified": True, "source_labels_used": False,
            "official_validation_used": any(c["split"] == "dev" for c in claims),
            "test_split_used": any(c["split"] == "test" for c in claims), "gold_labels_used": False,
            "selected_claim_ids_sha256": digest([c["sample_id"] for c in claims]), "workload": workload})
        return destination
    except Exception as exc:
        write_json(destination / "failure.json", {"status": "failed_unsealed", "error_type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if encoder:
            encoder.close()
        if reranker:
            reranker.close()
