"""Replay integrity checks on existing operational caches; no retrieval/inference."""
import argparse
import json
from pathlib import Path

from tffv.data import read_jsonl
from tffv.provenance import digest, file_hash
from tffv.retrieval.benchmark import validate_result
from tffv.retrieval.bm25_store_smoke import CONFIG, load_candidates
from tffv.retrieval.data import segment


def verify(run_path, candidate_manifest):
    directory = Path(run_path).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete_operational_smoke_not_admitted"
            or manifest.get("corpus_admitted") is not False or manifest.get("gold_labels_used") is not False):
        raise ValueError("Require sealed unadmitted operational smoke")
    if manifest.get("config") != CONFIG or manifest.get("config_sha256") != digest(CONFIG):
        raise ValueError("Operational config mismatch")
    hashes = manifest["outputs_sha256"]
    required = {"claims.jsonl", "R1.jsonl", "summary.json", "diagnostics.jsonl", "manual_review_queue.jsonl"}
    if not required <= set(hashes):
        raise ValueError("Incomplete artifact inventory")
    for name, expected in hashes.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory) or path.resolve() == directory or file_hash(path) != expected:
            raise ValueError(f"Run artifact hash mismatch: {name}")
    _, candidate_hash, loaded = load_candidates(candidate_manifest)
    if candidate_hash != manifest["candidate_manifest_sha256"]:
        raise ValueError("Candidate manifest identity mismatch")
    claims = [claim for claim, _, _, _ in loaded]
    if read_jsonl(directory / "claims.jsonl") != claims:
        raise ValueError("Run/source claim mismatch")
    records = read_jsonl(directory / "R1.jsonl")
    if len(records) != len(claims):
        raise ValueError("Result population mismatch")
    profiles, expected_review = [], []
    for record, (claim, documents, store, _) in zip(records, loaded):
        filename = store["directory"] + "/passages.jsonl"
        if filename not in hashes:
            raise ValueError("Unsealed passage inventory")
        cached = read_jsonl(directory / filename)
        # Reconstruct unchanged segmentation, IDs, hashes, offsets and provenance
        # directly from the pinned candidate documents. No BM25 search is rerun.
        reconstructed = segment(documents, **CONFIG["segmentation"])
        if cached != [p.to_dict() for p in reconstructed]:
            raise ValueError(f"Passage replay mismatch for {claim['sample_id']}")
        passage_map = {p.passage_id: p for p in reconstructed}
        validate_result(record, passage_map)
        if (record["sample_id"] != claim["sample_id"] or record["dataset"] != claim["dataset"]
                or record["split"] != claim["split"] or record["claim_sha256"] != digest(claim["claim"])
                or record["method"] != "R1"):
            raise ValueError("Result/claim identity mismatch")
        for evidence in record["evidence"]:
            if digest(evidence["text"]) != evidence["text_sha256"]:
                raise ValueError("Selected evidence text hash mismatch")
            expected_review.append(dict(sample_id=record["sample_id"], claim=claim["claim"],
                evidence=evidence, assessment=None, scope="operational_unadmitted_source_review"))
        profiles.append(dict(sample_id=claim["sample_id"], passages_verified=len(cached),
                             evidence_verified=len(record["evidence"])))
    review = read_jsonl(directory / "manual_review_queue.jsonl")
    if review != expected_review:
        raise ValueError("Review queue differs from sealed results/source claims")
    return dict(status="verified_operational_cache_not_admitted", corpus_admitted=False,
                stores=profiles, evidence_verified=len(review), artifact_files_verified=len(hashes),
                review_queue_sha256=file_hash(directory / "manual_review_queue.jsonl"),
                candidate_manifest_sha256=candidate_hash, verifier_sha256=file_hash(__file__),
                inference_executed=False, retrieval_rerun=False, artifacts_modified=False,
                limitation="Byte/provenance replay is not relevance, answer-leakage or factual-correctness validation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.run, args.candidate_manifest), indent=2))


if __name__ == "__main__": main()
