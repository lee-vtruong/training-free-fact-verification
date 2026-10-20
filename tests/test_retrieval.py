"""M1.1 synthetic correctness and leakage tests; no model downloads/inference."""
import importlib.util
import json
import math
import shutil
import uuid
from pathlib import Path
import numpy as np
import pytest
from tffv.provenance import digest, file_hash
from tffv.retrieval.benchmark import METHODS, output_directory, plan, run, validate_result
from tffv.retrieval.data import DOCUMENT_FIELDS, Passage, leakage_scan, load_inputs, segment
from tffv.retrieval.evaluate import evaluate, verify_run
from tffv.retrieval.frozen import runtime_check, snapshot_fingerprint, validate_spec
from tffv.retrieval.metrics import proxies, relevance_metrics
from tffv.retrieval.ranking import BM25, Hit, dense_search, near_duplicate, ordered, rrf, select_evidence
from tffv.retrieval.synthetic import SyntheticEncoder, SyntheticReranker


@pytest.fixture
def workspace():
    root = Path(__file__).resolve().parents[1] / "outputs" / ("pytest-m11-" + uuid.uuid4().hex)
    root.mkdir()
    try: yield root
    finally: shutil.rmtree(root)


@pytest.fixture
def config(workspace):
    path = Path(__file__).resolve().parents[1] / "scripts" / "m11_smoke.py"
    module_spec = importlib.util.spec_from_file_location("m11_smoke_fixture", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.fixture_config(workspace)


def doc(text, key="a"):
    return dict(document_id=key, text=text, source_url=f"https://example.org/{key}", source_kind="web_document",
                retrieved_at="fixture", acquisition_id="fixture")


def passage(text, key):
    return segment([doc(text, key)])[0]


def test_segmentation_exact_offsets_and_determinism():
    document = doc("one  two\nthree four five six seven")
    result = segment([document], words=4, overlap=1)
    assert len(result) == 2
    assert result[0].text == "one  two\nthree four"
    assert result[1].text == "four five six seven"
    for p in result:
        assert document["text"][p.start_char:p.end_char] == p.text
        assert p.text_sha256 == digest(p.text)
    assert [p.passage_id for p in segment([doc("x", "b"), document])] == [p.passage_id for p in segment([document, doc("x", "b")])]
    with pytest.raises(ValueError): segment([document], 4, 4)


def test_bm25_hand_computation_and_no_matches():
    a, b = passage("alpha alpha", "a"), passage("beta beta", "b")
    hits = BM25([a, b]).search("alpha")
    expected = math.log(2) * 2 * 2.2 / (2 + 1.2)
    assert len(hits) == 1
    assert hits[0].passage_id == a.passage_id
    assert hits[0].score == pytest.approx(expected)
    assert BM25([a, b]).search("unknown") == []
    with pytest.raises(ValueError): BM25([a, a])


def test_rrf_is_rank_based_and_top40():
    one = [Hit("a", 200), Hit("b", -100)]
    two = [Hit("b", 5), Hit("a", 1)]
    hits = rrf({"bm25": one, "dense": two}, 40, 60)
    assert hits[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert [h.passage_id for h in hits] == ["a", "b"]
    assert hits[0].component_scores["bm25_rank"] == 1
    assert len(rrf({"bm25": [Hit(str(i), 1) for i in range(50)]}, 40)) == 40
    with pytest.raises(ValueError): rrf({"x": [Hit("a", 1), Hit("a", 2)]})
    with pytest.raises(ValueError): rrf({"x": one}, constant=0)


def test_dedup_retains_provenance_and_numeric_differences():
    a, b, c = passage("Value 10 is recorded", "a"), passage("value 10 IS recorded", "b"), passage("Value 11 is recorded", "c")
    assert near_duplicate(a.text, b.text)
    assert not near_duplicate(a.text, c.text)
    assert not near_duplicate(" ".join(["word"] * 100) + " not allowed", " ".join(["word"] * 100) + " allowed")
    hits, suppressed = select_evidence([Hit(a.passage_id, 3), Hit(b.passage_id, 2), Hit(c.passage_id, 1)], {p.passage_id: p for p in (a, b, c)}, 8)
    assert [h.passage_id for h in hits] == [a.passage_id, c.passage_id]
    assert suppressed == [{"passage_id": b.passage_id, "duplicate_of": a.passage_id}]


def test_dense_cosine_and_stable_ties_across_blocks():
    matrix = np.asarray([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
    hits = dense_search(np.asarray([1, 0]), matrix, ["z", "b", "a"], 2, block_size=1)
    assert [h.passage_id for h in hits] == ["a", "z"]
    with pytest.raises(ValueError): dense_search(np.asarray([2, 0]), matrix, ["z", "b", "a"])
    with pytest.raises(ValueError): ordered([Hit("a", float("nan"))], 1)
    with pytest.raises(ValueError): ordered([Hit("a", 1)], 0)


def test_relevance_metrics_hand_computed():
    result = relevance_metrics(["x", "b", "a"], {"a": 2, "b": 1})
    assert result["recall@5"] == 1
    assert result["mrr@10"] == .5
    assert result["ndcg@10"] == pytest.approx((1 / math.log2(3) + 3 / math.log2(4)) / (3 + 1 / math.log2(3)))
    assert relevance_metrics(["a"], {"x": 1})["mrr@10"] == 0
    assert all(v is None for v in relevance_metrics(["a"], {"a": 0}).values())
    with pytest.raises(ValueError): relevance_metrics(["a"], {"a": -1})
    with pytest.raises(ValueError): relevance_metrics(["a", "a"], {"a": 1})


def test_recall_depth_is_not_truncated_to_export_top8():
    ranked = [str(i) for i in range(12)]
    metrics = relevance_metrics(ranked, {"9": 1})
    assert metrics["recall@8"] == 0
    assert metrics["recall@10"] == 1
    assert metrics["mrr@10"] == .1


def test_no_evidence_proxies_remain_unknown_for_semantics():
    metrics = proxies([])
    assert metrics["slot_coverage"] == 0
    assert metrics["semantic_evidence_coverage"] is None
    assert metrics["mean_pairwise_token_jaccard"] is None


@pytest.mark.parametrize("failure", ["gold_claim_field", "gold_qa", "source_kind", "source_url", "hash", "attestation", "split", "synthetic"])
def test_fail_closed_inputs(config, failure):
    manifest_path = Path(config["data_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claims_path = Path(config["claims_path"])
    corpus_path = Path(config["corpus_path"])
    claims = [json.loads(x) for x in claims_path.read_text(encoding="utf-8").splitlines()]
    corpus = [json.loads(x) for x in corpus_path.read_text(encoding="utf-8").splitlines()]
    if failure == "gold_claim_field": claims[0]["gold_label"] = "SUPPORTED"
    elif failure == "gold_qa": corpus[0]["questions"] = "must not reach retriever"
    elif failure == "source_kind": corpus[0]["source_kind"] = "gold_justification"
    elif failure == "source_url": corpus[0]["source_url"] = "https://user:secret@example.org/a"
    elif failure == "attestation": manifest["no_ground_truth_qa"] = False
    elif failure == "split": claims[0]["split"] = "random_train_fold"
    elif failure == "synthetic": manifest["synthetic"] = False
    claims_path.write_text("\n".join(json.dumps(r) for r in claims), encoding="utf-8")
    corpus_path.write_text("\n".join(json.dumps(r) for r in corpus), encoding="utf-8")
    manifest.update(claims_sha256=file_hash(claims_path), corpus_sha256=file_hash(corpus_path))
    if failure == "hash": manifest["corpus_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError): load_inputs(claims_path, corpus_path, manifest, synthetic=True)


def test_verdict_and_cross_split_leakage_flags():
    claims = [dict(sample_id="a", claim="same claim", split="train"), dict(sample_id="b", claim="SAME CLAIM", split="dev")]
    result = leakage_scan(claims, [doc("Verdict: the claim is false. Same claim.")])
    assert result["verdict_pattern_documents"] == ["a"]
    assert result["cross_split_exact_claim_groups"] == [["a", "b"]]
    assert len(result["claim_text_in_documents"]) == 1


def test_all_four_configs_schema_alignment_cache_reuse_and_seal(config, workspace):
    encoder = SyntheticEncoder()
    destination = run(config, workspace / "run", synthetic_backends=(encoder, SyntheticReranker()), index_cache=workspace / "index")
    manifest = verify_run(destination)
    assert manifest["config"]["synthetic"] is True
    assert manifest["source_labels_used"] is False
    assert encoder.document_rows_encoded == 64
    by_method = {}
    for method in METHODS:
        rows = [json.loads(x) for x in (destination / f"{method}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [r["sample_id"] for r in rows] == ["train:0", "dev:0", "dev:1"]
        assert all(len(r["evidence"]) == 8 and r["shortfall"] == 0 for r in rows)
        assert all(r["peak_gpu_memory_bytes"] is None for r in rows)
        assert all(len(r["ranked_candidates"]) == (50 if method in {"R1", "R2"} else 40) for r in rows)
        by_method[method] = rows
    assert {r["passage_id"] for r in by_method["R3"][0]["ranked_candidates"]} == {r["passage_id"] for r in by_method["R4"][0]["ranked_candidates"]}
    evaluated = evaluate(destination, workspace / "eval")
    comparison = json.loads((evaluated / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["methods"]["R1"]["overall"]["means"]["recall@5"] is None
    assert set(comparison["methods"]["R1"]["by_split"]) == {"train", "dev"}
    second_encoder = SyntheticEncoder()
    run(config, workspace / "rerun", max_claims=1, synthetic_backends=(second_encoder, SyntheticReranker()), index_cache=workspace / "index")
    assert second_encoder.document_rows_encoded == 0
    with pytest.raises(FileExistsError): run(config, destination, synthetic_backends=(SyntheticEncoder(), SyntheticReranker()), index_cache=workspace / "index")
    (destination / "R1.jsonl").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"): verify_run(destination)


def test_qrels_only_in_separate_evaluation(config, workspace):
    destination = run(config, workspace / "run", methods=["R1"], synthetic_backends=(SyntheticEncoder(), SyntheticReranker()))
    rows = [json.loads(x) for x in (destination / "R1.jsonl").read_text(encoding="utf-8").splitlines()]
    qrels = workspace / "qrels.jsonl"
    qrels.write_text(json.dumps({"sample_id": rows[0]["sample_id"], "passage_id": rows[0]["evidence"][0]["passage_id"], "relevance": 1}), encoding="utf-8")
    metadata = dict(validated=True, no_inferred_labels=True, annotation_provenance="Synthetic hand-written unit-test judgment",
        claims_sha256=file_hash(destination / "claims.jsonl"), passages_sha256=file_hash(destination / "passages.jsonl"),
        qrels_sha256=file_hash(qrels), relevance_unit="passage", synthetic=True, unjudged_policy="unjudged_as_nonrelevant")
    path = workspace / "qrels_manifest.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    evaluated = evaluate(destination, workspace / "eval", qrels, path)
    metrics = json.loads((evaluated / "comparison.json").read_text(encoding="utf-8"))
    assert metrics["judged_claim_count"] == 1
    assert metrics["methods"]["R1"]["overall"]["means"]["recall@5"] == 1
    assert metrics["methods"]["R1"]["overall"]["measured_counts"]["recall@5"] == 1
    metadata["no_inferred_labels"] = False
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError): evaluate(destination, workspace / "bad_eval", qrels, path)


def test_shortfall_and_provenance_schema(config, workspace):
    run(config, workspace / "run", methods=["R1"], max_claims=1, synthetic_backends=(SyntheticEncoder(), SyntheticReranker()))
    row = json.loads((workspace / "run" / "R1.jsonl").read_text(encoding="utf-8").splitlines()[0])
    p = {item["passage_id"]: Passage(**item) for item in map(json.loads, (workspace / "run" / "passages.jsonl").read_text(encoding="utf-8").splitlines())}
    validate_result(row, p)
    row["evidence"][0]["source_url"] = "https://wrong.example.org"
    with pytest.raises(ValueError): validate_result(row, p)
    with pytest.raises(ValueError): output_directory(workspace.parent.parent / "outside")


def test_plan_is_model_free_and_synthetic_guard(config):
    report = plan(config, max_claims=1, synthetic=True)
    assert report["reranker_pairs_upper_bound"] == 40
    assert report["embedding_storage_bytes_float32"] == 64 * 1024 * 4
    with pytest.raises(ValueError): plan(config)  # Synthetic corpus cannot masquerade as AVeriTeC.
    with pytest.raises(ValueError): plan(config, max_claims=0, synthetic=True)


def test_bounded_smoke_corpus_is_shared_and_marked(config, workspace):
    report = plan(config, max_claims=1, synthetic=True, max_passages=4)
    assert report["passages"] == 4
    assert report["scope"] == "bounded_smoke_not_quality_benchmark"
    destination = run(config, workspace / "bounded", max_claims=1, max_passages=4,
                      synthetic_backends=(SyntheticEncoder(), SyntheticReranker()), index_cache=workspace / "index")
    for method in METHODS:
        row = json.loads((destination / f"{method}.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert len(row["evidence"]) == 4
        assert row["shortfall"] == 4


def test_near_duplicate_claim_and_source_flags():
    claims = [dict(sample_id="a", claim="city population 100 in 2020", split="train"),
              dict(sample_id="b", claim="CITY POPULATION 100 IN 2020", split="dev")]
    d = doc("city population 100 in 2020")
    d["source_url"] = "https://www.factcheck.example.org/article"
    scan = leakage_scan(claims, [d, dict(d, document_id="b")])
    assert scan["near_duplicate_pair_count"] == 1
    assert scan["near_duplicate_claim_pairs"][0]["cross_split"] is True
    assert scan["exact_document_duplicate_groups"] == [["a", "b"]]
    assert scan["fact_check_source_url_flags"] == ["a", "b"]


def test_revision_snapshot_hash_and_dependency_guard(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
    (tmp_path / "weights.safetensors").write_bytes(b"FAKE TEST FILE NOT A MODEL")
    fingerprint = snapshot_fingerprint(tmp_path)
    spec = dict(model_id="Qwen/Qwen3-Embedding-0.6B", local_path=str(tmp_path), revision="a" * 40, snapshot_sha256=fingerprint["snapshot_sha256"])
    validate_spec(spec, spec["model_id"], verify_bytes=True)
    with pytest.raises(ValueError): validate_spec(dict(spec, revision="main"), spec["model_id"])
    (tmp_path / "weights.safetensors").write_bytes(b"CHANGED")
    with pytest.raises(ValueError): validate_spec(spec, spec["model_id"], verify_bytes=True)
    monkeypatch.setattr("tffv.retrieval.frozen.importlib.metadata.version", lambda _: "4.45.2")
    with pytest.raises(RuntimeError, match="NEW environment"): runtime_check()
