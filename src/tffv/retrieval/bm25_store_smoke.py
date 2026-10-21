"""CPU-only operational R1 on unadmitted claim-scoped candidates, not a benchmark."""
import argparse
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

from tffv.data import read_jsonl
from tffv.provenance import digest, file_hash, write_json
from tffv.retrieval.benchmark import dump_rows, output_directory, result_record
from tffv.retrieval.data import CLAIM_FIELDS, DOCUMENT_FIELDS, VERDICT, require_strings, segment
from tffv.retrieval.metrics import proxies, summarize
from tffv.retrieval.prepare_store_candidates import POLICY
from tffv.retrieval.ranking import BM25


CONFIG = dict(segmentation=dict(words=180, overlap=30), bm25=dict(k1=1.2, b=.75),
              dedup_threshold=.95, candidate_depth=50, evidence_depth=8, evaluation_depth=10,
              corpus_scope="per_claim", method="R1", protocol="operational_unadmitted_v1")


def load_candidates(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest_hash = file_hash(manifest_path)
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (metadata.get("status") != "complete_candidates_not_admitted"
            or metadata.get("corpus_admitted") is not False or metadata.get("policy") != POLICY):
        raise ValueError("Require unadmitted candidate manifest and declared policy")
    stores = metadata["stores"]
    if not 1 <= len(stores) <= 3 or len({r["sample_id"] for r in stores}) != len(stores):
        raise ValueError("Require unique bounded source stores")
    loaded = []
    directories = set()
    for store in stores:
        name = store["directory"]
        directory = manifest_path.parent / name
        if (Path(name).name != name or name in directories
                or not directory.resolve().is_relative_to(manifest_path.parent)):
            raise ValueError("Unsafe/duplicate candidate directory")
        directories.add(name)
        for filename, field in (("claims.jsonl", "claims_sha256"), ("corpus.jsonl", "corpus_sha256"),
                                ("record_ledger.jsonl", "ledger_sha256")):
            if file_hash(directory / filename) != store[field]:
                raise ValueError("Candidate file hash mismatch")
        claims = read_jsonl(directory / "claims.jsonl")
        if len(claims) != 1:
            raise ValueError("Exactly one claim per store required")
        claim = claims[0]
        require_strings(claim, CLAIM_FIELDS)
        if claim["sample_id"] != store["sample_id"] or claim["dataset"] != "AVeriTeC" or claim["split"] != "train":
            raise ValueError("Claim/store identity mismatch")
        ledger = read_jsonl(directory / "record_ledger.jsonl")
        if len(ledger) != store["source_records"] or any(r["sample_id"] != claim["sample_id"] for r in ledger):
            raise ValueError("Ledger scope/count mismatch")
        retained = {r["document_id"]: r for r in ledger if r["status"] == "retained"}
        documents = read_jsonl(directory / "corpus.jsonl")
        if not documents or len(documents) != store["counts"]["retained"] or len(documents) != len(retained):
            raise ValueError("Retained document count mismatch")
        ids = set()
        for doc in documents:
            require_strings(doc, DOCUMENT_FIELDS)
            url = urlsplit(doc["source_url"])
            if (doc["document_id"] in ids or doc["document_id"] not in retained
                    or retained[doc["document_id"]]["source_url"] != doc["source_url"]
                    or url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password
                    or doc["source_kind"] != "web_document"):
                raise ValueError("Invalid retained document provenance")
            if VERDICT.search(doc["text"]):
                raise ValueError("Residual verdict pattern; regenerate/review candidates")
            ids.add(doc["document_id"])
        loaded.append((claim, documents, store, directory))
    return manifest_path, manifest_hash, loaded


def plan(manifest_path):
    _, manifest_hash, loaded = load_candidates(manifest_path)
    stores = []
    for claim, documents, _, _ in loaded:
        n_passages = 0
        for doc in documents:
            words = sum(1 for _ in re.finditer(r"\S+", doc["text"]))
            n_passages += 1 + max(0, (words - 180 + 149) // 150) if words else 0
        stores.append(dict(sample_id=claim["sample_id"], documents=len(documents), passages=n_passages))
    return dict(scope="operational_smoke_not_quality_benchmark", candidate_manifest_sha256=manifest_hash,
                config=CONFIG, config_sha256=digest(CONFIG), stores=stores, n_claims=len(stores),
                gpu_required=False, llm_forward_passes=0, corpus_admitted=False,
                estimated_runtime_seconds=None, estimated_peak_cpu_memory_bytes=None,
                note="All retained passages; no arbitrary document/passage cap. Preflight reads candidate files only.")


def run(manifest_path, output):
    manifest_path, manifest_hash, loaded = load_candidates(manifest_path)
    destination = output_directory(output)
    started = time.perf_counter()
    records, timings, diagnostics = [], [], []
    try:
        for claim, documents, store, input_directory in loaded:
            store_output = destination / store["directory"]
            store_output.mkdir()
            start = time.perf_counter()
            passages = segment(documents, **CONFIG["segmentation"])
            segment_seconds = time.perf_counter() - start
            passage_map = {p.passage_id: p for p in passages}
            start = time.perf_counter()
            bm25 = BM25(passages, **CONFIG["bm25"])
            index_seconds = time.perf_counter() - start
            start = time.perf_counter()
            hits = bm25.search(claim["claim"], CONFIG["candidate_depth"])
            query_seconds = time.perf_counter() - start
            start = time.perf_counter()
            record = result_record(claim, "R1", hits, passage_map, CONFIG, query_seconds)
            dedup_seconds = time.perf_counter() - start
            records.append(record)
            diagnostics.append(dict(sample_id=claim["sample_id"], **proxies(record["evidence"])))
            dump_rows(store_output / "passages.jsonl", [p.to_dict() for p in passages])
            timings.append(dict(sample_id=claim["sample_id"], documents=len(documents), passages=len(passages),
                segmentation_seconds=segment_seconds, index_seconds=index_seconds,
                query_seconds=query_seconds, dedup_and_schema_seconds=dedup_seconds))
            for filename, field in (("claims.jsonl", "claims_sha256"), ("corpus.jsonl", "corpus_sha256"),
                                    ("record_ledger.jsonl", "ledger_sha256")):
                if file_hash(input_directory / filename) != store[field]:
                    raise ValueError("Candidate inputs changed during smoke")
            del bm25, passages, passage_map
        if file_hash(manifest_path) != manifest_hash:
            raise ValueError("Candidate manifest changed during smoke")
        dump_rows(destination / "R1.jsonl", records)
        dump_rows(destination / "claims.jsonl", [claim for claim, _, _, _ in loaded])
        dump_rows(destination / "diagnostics.jsonl", diagnostics)
        claim_text = {claim["sample_id"]: claim["claim"] for claim, _, _, _ in loaded}
        dump_rows(destination / "manual_review_queue.jsonl", [dict(sample_id=r["sample_id"],
            claim=claim_text[r["sample_id"]], evidence=e, assessment=None,
            scope="operational_unadmitted_source_review") for r in records for e in r["evidence"]])
        write_json(destination / "summary.json", dict(scope="operational_smoke_not_quality_benchmark",
            corpus_admitted=False, n_claims=len(records), stores=timings, proxies=summarize(diagnostics),
            relevance_metrics={key: None for key in ("recall@5", "recall@8", "recall@10", "mrr@10", "ndcg@10")},
            elapsed_seconds=time.perf_counter() - started, gpu_used=False, peak_cpu_memory_bytes=None,
            limitations=["Residual answer leakage unresolved; not a quality benchmark or method comparison.",
                "Query latency excludes segmentation/indexing/dedup/serialization, which are separate.",
                "Coverage means filled evidence slots, not semantic completeness. No qrels used.",
                "No full cross-split contamination or residual answer-leakage clearance.",
                "Only R1, three or fewer exploratory source claims; R2-R4 not run."]))
        package = Path(__file__).resolve().parents[1]
        code_hashes = {str(p.relative_to(package)).replace("\\", "/"): file_hash(p) for p in sorted(package.rglob("*.py"))}
        write_json(destination / "manifest.json", dict(status="complete_operational_smoke_not_admitted",
            corpus_admitted=False, candidate_manifest_sha256=manifest_hash,
            config=CONFIG, config_sha256=digest(CONFIG), implementation=code_hashes,
            outputs_sha256={str(p.relative_to(destination)).replace("\\", "/"): file_hash(p)
                            for p in sorted(destination.rglob("*")) if p.is_file()},
            gold_labels_used=False, network_accessed=False, llm_forward_passes=0))
    except Exception as exc:
        write_json(destination / "failure.json", dict(status="failed_unsealed", error_type=type(exc).__name__, message=str(exc)))
        raise
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.execute:
        if not args.output: parser.error("--execute requires a fresh --output")
        print(run(args.candidate_manifest, args.output))
    else:
        if args.output: parser.error("Preflight prints JSON only; omit --output")
        print(json.dumps(plan(args.candidate_manifest), indent=2))


if __name__ == "__main__": main()
