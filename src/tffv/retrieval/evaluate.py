"""Offline comparison of sealed evidence caches; optional validated passage qrels."""
import json
import re
from pathlib import Path
from tffv.data import read_jsonl
from tffv.provenance import file_hash, write_json
from tffv.retrieval.benchmark import dump_rows, output_directory, validate_result
from tffv.retrieval.data import Passage
from tffv.retrieval.metrics import proxies, relevance_metrics, summarize


def verify_run(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("gold_labels_used") is not False:
        raise ValueError("Only sealed label-free retrieval runs can be evaluated")
    for name, expected in manifest["outputs_sha256"].items():
        if Path(name).name != name or file_hash(directory / name) != expected:
            raise ValueError("Run artifact hash mismatch")
    return manifest


def load_qrels(path, metadata_path, directory, run_manifest, claims, passages):
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    for flag in ("validated", "no_inferred_labels"):
        if metadata.get(flag) is not True:
            raise ValueError("Qrels require validated human/official relevance, never inferred labels")
    expected = {"claims_sha256": file_hash(directory / "claims.jsonl"),
                "passages_sha256": file_hash(directory / "passages.jsonl"), "qrels_sha256": file_hash(path),
                "relevance_unit": "passage", "synthetic": run_manifest["config"]["synthetic"]}
    if any(metadata.get(k) != v for k, v in expected.items()) or not metadata.get("annotation_provenance"):
        raise ValueError("Qrels population/unit/provenance mismatch")
    if metadata.get("unjudged_policy") != "unjudged_as_nonrelevant":
        raise ValueError("Declare treatment of incomplete judgments explicitly")
    valid_claims = {r["sample_id"] for r in claims}
    result = {}
    for row in read_jsonl(path):
        if set(row) != {"sample_id", "passage_id", "relevance"}:
            raise ValueError("Unexpected qrels fields")
        sid, pid = row["sample_id"], row["passage_id"]
        if sid not in valid_claims or pid not in passages:
            raise ValueError("Qrels reference unknown claim/passage")
        if pid in result.setdefault(sid, {}):
            raise ValueError("Duplicate relevance judgment")
        result[sid][pid] = row["relevance"]
    for judgments in result.values():
        relevance_metrics([], judgments)
    return result, metadata


def evaluate(run_path, output, qrels_path=None, qrels_manifest_path=None):
    if bool(qrels_path) != bool(qrels_manifest_path):
        raise ValueError("Qrels and their manifest must be provided together")
    directory = Path(run_path).resolve()
    manifest = verify_run(directory)  # Before opening any annotations.
    claims = read_jsonl(directory / "claims.jsonl")
    passages = {r["passage_id"]: Passage(**r) for r in read_jsonl(directory / "passages.jsonl")}
    judgments, annotations = ({}, None) if not qrels_path else load_qrels(qrels_path, qrels_manifest_path, directory, manifest, claims, passages)
    destination = output_directory(output)
    methods, review = {}, []
    null_relevance = {k: None for k in ("recall@5", "recall@8", "recall@10", "mrr@10", "ndcg@10")}
    for method in manifest["config"]["methods"]:
        rows = read_jsonl(directory / f"{method}.jsonl")
        if len(rows) != len(claims) or [r["sample_id"] for r in rows] != [c["sample_id"] for c in claims]:
            raise ValueError("Methods must use identical claims/order")
        results = []
        for row, claim in zip(rows, claims):
            validate_result(row, passages)
            if row["method"] != method or row["dataset"] != claim["dataset"] or row["split"] != claim["split"]:
                raise ValueError("Method/claim protocol mismatch")
            from tffv.provenance import digest
            if row["claim_sha256"] != digest(claim["claim"]):
                raise ValueError("Claim hash mismatch")
            rel = relevance_metrics([r["passage_id"] for r in row["evaluation_ranking_top10"]], judgments[row["sample_id"]]) if row["sample_id"] in judgments else null_relevance
            results.append(dict(sample_id=row["sample_id"], split=claim["split"], **proxies(row["evidence"]), **rel,
                                latency_seconds=row["latency_seconds"], peak_gpu_memory_bytes=row["peak_gpu_memory_bytes"]))
            # Content-only tags for manual inspection, not verdicts or causal labels.
            tags = []
            if re.search(r"\d", claim["claim"]): tags.append("numerical_candidate")
            if re.search(r"\b(?:19|20)\d{2}\b|\b(?:before|after|since|during)\b", claim["claim"], re.I): tags.append("temporal_candidate")
            if re.search(r"\b(?:conflict|dispute|however|both|although)\w*\b", claim["claim"], re.I): tags.append("conflicting_language_candidate")
            if row["shortfall"]: tags.append("evidence_shortfall")
            if tags or len(review) < 3:
                review.append(dict(sample_id=claim["sample_id"], method=method, claim=claim["claim"], tags=tags,
                    relevance=rel, evidence=row["evidence"], assessment="manual review pending; tags are heuristics"))
        methods[method] = dict(overall=summarize(results), by_split={split: summarize([r for r in results if r["split"] == split]) for split in sorted({r["split"] for r in results})})
        dump_rows(destination / f"{method}_per_claim_metrics.jsonl", results)
    write_json(destination / "comparison.json", {"synthetic": manifest["config"]["synthetic"], "methods": methods,
        "annotation_manifest": annotations, "judged_claim_count": len(judgments),
        "limitations": ["Recall denominator is known positive judgments, not unknown exhaustive relevance.",
                         "No-positive and unjudged queries are excluded from relevance means and counted separately.",
                         "@5/8/10 use the same deduplicated ranking; exported Top-8 is its prefix.",
                         "Slot coverage/diversity are proxies, not factual evidence sufficiency."]})
    dump_rows(destination / "manual_review_queue.jsonl", review)
    table = ["# Retrieval comparison", "", "Synthetic validation only." if manifest["config"]["synthetic"] else "AVeriTeC retrieval; no automatic winner selection.", "",
             "| Method | Recall@5 | Recall@8 | Recall@10 | MRR@10 | nDCG@10 | Slots filled | Latency s |", "|---|---|---|---|---|---|---|---|"]
    for method, metrics in methods.items():
        values = metrics["overall"]["means"]
        def fmt(key):
            return "not measured" if values.get(key) is None else f"{values[key]:.6f}"
        table.append("| " + " | ".join([method] + [fmt(k) for k in ("recall@5", "recall@8", "recall@10", "mrr@10", "ndcg@10", "slot_coverage", "latency_seconds")]) + " |")
    (destination / "comparison.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    verify_run(directory)
    if qrels_path and file_hash(qrels_path) != annotations["qrels_sha256"]:
        raise ValueError("Qrels changed during evaluation; evaluation remains unsealed")
    write_json(destination / "evaluation_manifest.json", {"retrieval_manifest_sha256": file_hash(directory / "manifest.json"),
        "qrels_sha256": file_hash(qrels_path) if qrels_path else None,
        "qrels_manifest_sha256": file_hash(qrels_manifest_path) if qrels_manifest_path else None,
        "evaluation_implementation_sha256": file_hash(__file__),
        "outputs_sha256": {p.name: file_hash(p) for p in destination.iterdir() if p.is_file()},
        "selection_performed": False, "training_free_verified": True})
    return destination
