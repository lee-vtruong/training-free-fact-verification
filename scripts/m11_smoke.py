"""Offline synthetic R1-R4 wiring test, not an AVeriTeC/Qwen experiment."""
import argparse
from pathlib import Path
from tffv.provenance import file_hash, write_json
from tffv.retrieval.benchmark import dump_rows, output_directory, run
from tffv.retrieval.data import ONTOLOGY
from tffv.retrieval.evaluate import evaluate
from tffv.retrieval.synthetic import SyntheticEncoder, SyntheticReranker


def fixture_config(destination):
    claims = [dict(sample_id="train:0", dataset="SYNTHETIC", split="train", claim="City population was 1200 in 2020"),
              dict(sample_id="dev:0", dataset="SYNTHETIC", split="dev", claim="City population after 2021 was 1300"),
              dict(sample_id="dev:1", dataset="SYNTHETIC", split="dev", claim="Conflicting city population reports in 2022")]
    documents = [dict(document_id=f"doc:{i:03d}", text=f"City population report {i} records {1200 + i} residents in {2020 + i % 4}. Archive source table {i}.",
        source_url=f"https://source{i % 8}.example.org/document/{i}", source_kind="web_document", retrieved_at="synthetic-fixture",
        acquisition_id="synthetic-independent-documents") for i in range(64)]
    dump_rows(destination / "claims.jsonl", claims)
    dump_rows(destination / "documents.jsonl", documents)
    manifest = dict(dataset="SYNTHETIC", ontology=ONTOLOGY, synthetic=True,
        claims_sha256=file_hash(destination / "claims.jsonl"), corpus_sha256=file_hash(destination / "documents.jsonl"),
        split_counts={"train": 1, "dev": 2}, official_splits_preserved=True, corpus_origin_reviewed=True,
        no_gold_labels=True, no_gold_justifications=True, no_ground_truth_qa=True, acquisition_label_free=True,
        answer_leakage_reviewed=True, corpus_provenance="Synthetic fixtures, not web acquisition",
        claim_id_provenance="Explicit synthetic IDs", population_status="synthetic_validation_only")
    write_json(destination / "data_manifest.json", manifest)
    config = dict(claims_path=str(destination / "claims.jsonl"), corpus_path=str(destination / "documents.jsonl"),
        data_manifest_path=str(destination / "data_manifest.json"), segmentation=dict(words=180, overlap=30),
        bm25=dict(k1=1.2, b=.75), rrf_constant=60, dedup_threshold=.95, seed=2026,
        device="cpu", dtype="float32", max_length=1024, batch_size=4,
        embedding={"model_id": "SYNTHETIC_TEST_DOUBLE_NOT_QWEN"}, reranker={"model_id": "SYNTHETIC_TEST_DOUBLE_NOT_QWEN"})
    write_json(destination / "config.json", config)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = output_directory(args.output)
    config = fixture_config(root)
    run(config, root / "run", synthetic_backends=(SyntheticEncoder(), SyntheticReranker()), index_cache=root / "index_cache")
    evaluate(root / "run", root / "evaluation")
    print(f"SYNTHETIC ONLY — not Qwen/AVeriTeC results: {root}")


if __name__ == "__main__": main()
