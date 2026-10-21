import json
from pathlib import Path
import shutil
import uuid

import pytest
from tffv.data import read_jsonl
from tffv.retrieval.benchmark import validate_result
from tffv.retrieval.data import Passage
from tffv.retrieval.prepare_store_candidates import POLICY, prepare_candidates
from tffv.retrieval.store_retrieval import METHODS, plan, run


@pytest.fixture
def candidate_fixture(tmp_path):
    root = Path(__file__).resolve().parents[1] / "outputs" / ("pytest-storeret-" + uuid.uuid4().hex)
    root.mkdir(parents=True)
    source = tmp_path / "train.json"
    source.write_text(json.dumps([dict(claim="Hunter Biden text zero"), dict(claim="Trump tax cut text one")]), encoding="utf-8")
    from tffv.provenance import file_hash
    from tffv.retrieval.prepare_claims import prepare_source_claims
    prepare_source_claims(source, root / "claims", file_hash(source), 2)

    import binascii
    stores = []
    for ordinal in range(2):
        rows = [
            dict(claim_id=str(ordinal), url="https://source.example/a", url2text=[f"Document zero for claim {ordinal}", "Second paragraph"]),
            dict(claim_id=str(ordinal), url="https://source.example/b", url2text=[f"Document one for claim {ordinal}", "More facts"]),
            dict(claim_id=str(ordinal), url="https://bbc.com/news/a", url2text=["Verdict: false"]),
            dict(claim_id=str(ordinal), url="https://source.example/empty", url2text=[]),
        ]
        path = tmp_path / f"train-{ordinal}.json"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        stores.append(dict(
            source_ordinal=ordinal,
            file=path.name,
            full_member_crc_verified=True,
            raw_bytes=path.stat().st_size,
            raw_sha256=file_hash(path),
            member_crc32=f"{binascii.crc32(path.read_bytes()) & 0xffffffff:08x}",
        ))

    subset_manifest = tmp_path / "subset_manifest.json"
    subset_manifest.write_text(json.dumps(dict(status="complete_raw_subset_not_admitted", labels_used=False, stores=stores)), encoding="utf-8")
    from tffv.retrieval.audit_subset import audit
    summary = audit(subset_manifest, root / "audit", 64)
    cand_manifest = prepare_candidates(summary, root / "claims" / "claims_manifest.json", root / "candidates", POLICY)

    try:
        yield cand_manifest, root
    finally:
        shutil.rmtree(root)


def test_preflight_blockers_and_synthetic_readiness(candidate_fixture):
    cand_manifest, _ = candidate_fixture

    # Real models without checkpoints should report blockers in preflight
    p_real = plan(cand_manifest, methods=["R1", "R2", "R3", "R4"], synthetic=False)
    assert p_real["ready_to_execute"] is False
    assert len(p_real["blockers"]) > 0
    assert p_real["corpus_admitted"] is False
    assert p_real["n_claims"] == 2
    assert p_real["total_documents"] == 4  # 2 retained per store * 2 stores

    # Synthetic preflight should be ready to execute
    p_syn = plan(cand_manifest, methods=["R1", "R2", "R3", "R4"], synthetic=True)
    assert p_syn["ready_to_execute"] is True
    assert p_syn["blockers"] == []
    assert p_syn["reranker_pairs_upper_bound"] > 0


def test_run_claim_scoped_isolation_all_four_methods(candidate_fixture):
    cand_manifest, root = candidate_fixture
    out_dir = root / "run_all_methods"

    # Execute all four methods in synthetic mode
    result_path = run(cand_manifest, out_dir, methods=["R1", "R2", "R3", "R4"], synthetic=True)
    assert result_path == out_dir

    # Verify all four method files exist
    for m in METHODS:
        path = out_dir / f"{m}.jsonl"
        assert path.is_file()
        records = read_jsonl(path)
        assert len(records) == 2
        for i, rec in enumerate(records):
            store_passages = {r["passage_id"]: Passage(**r) for r in read_jsonl(out_dir / f"train-{i}" / "passages.jsonl")}
            validate_result(rec, store_passages)
            assert rec["sample_id"] == f"train:{i}"
            assert rec["method"] == m
            assert len(rec["evidence"]) <= 8
            assert rec["shortfall"] == 8 - len(rec["evidence"])
            # Ensure every retrieved passage belongs strictly to this store
            for ev in rec["evidence"]:
                assert ev["passage_id"] in store_passages
                assert f"claim {i}" in ev["text"]

    # Verify R3 candidates feed R4
    r3_records = read_jsonl(out_dir / "R3.jsonl")
    r4_records = read_jsonl(out_dir / "R4.jsonl")
    for r3_rec, r4_rec in zip(r3_records, r4_records):
        r3_ids = {c["passage_id"] for c in r3_rec["ranked_candidates"]}
        r4_ids = {c["passage_id"] for c in r4_rec["ranked_candidates"]}
        assert r3_ids == r4_ids

    # Verify summary and manifest
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["corpus_admitted"] is False
    assert set(summary["methods"]) == set(METHODS)
    for m in METHODS:
        assert all(v is None for v in summary["methods"][m]["relevance_metrics"].values())

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete_operational_smoke_not_admitted"
    assert manifest["corpus_admitted"] is False
    assert manifest["gold_labels_used"] is False
    assert "outputs_sha256" in manifest


def test_run_r1_only_cpu_mode(candidate_fixture):
    cand_manifest, root = candidate_fixture
    out_dir = root / "run_r1_only"

    result_path = run(cand_manifest, out_dir, methods=["R1"], synthetic=False)
    assert result_path == out_dir
    assert (out_dir / "R1.jsonl").is_file()
    assert not (out_dir / "R2.jsonl").exists()
    assert not (out_dir / "R3.jsonl").exists()
    assert not (out_dir / "R4.jsonl").exists()


def test_refuse_execution_when_real_model_blocked(candidate_fixture):
    cand_manifest, root = candidate_fixture
    out_dir = root / "blocked_run"
    with pytest.raises(RuntimeError, match="Cannot execute claim-scoped retrieval due to blockers"):
        run(cand_manifest, out_dir, methods=["R2"], synthetic=False)
