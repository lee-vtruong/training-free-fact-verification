import binascii
import json
from pathlib import Path
import shutil
import uuid

import pytest
from tffv.data import read_jsonl
from tffv.provenance import file_hash
from tffv.retrieval.audit_subset import audit
from tffv.retrieval.prepare_claims import prepare_source_claims
from tffv.retrieval.prepare_store_candidates import POLICY, prepare_candidates


@pytest.fixture
def prepared(tmp_path):
    root = Path(__file__).resolve().parents[1] / "outputs" / ("pytest-candidates-" + uuid.uuid4().hex)
    root.mkdir(parents=True)
    source = tmp_path / "train.json"
    source.write_text(json.dumps([dict(claim="Claim zero", label="SECRET"), dict(claim="Claim one")]), encoding="utf-8")
    prepare_source_claims(source, root / "claims", file_hash(source), 2)
    stores = []
    for ordinal in range(2):
        rows = [dict(claim_id=str(ordinal), url="https://source.example/a", url2text=[f"Text {ordinal}", "More text"]),
                dict(claim_id=str(ordinal), url="https://bbc.com/news/a", url2text=["Verdict: false"]),
                dict(claim_id=str(ordinal), url="https://politifact.com/a", url2text=["No explicit rating"]),
                dict(claim_id=str(ordinal), url="https://source.example/empty", url2text=[])]
        path = tmp_path / f"train-{ordinal}.json"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        stores.append(dict(source_ordinal=ordinal, file=path.name, full_member_crc_verified=True,
            raw_bytes=path.stat().st_size, raw_sha256=file_hash(path),
            member_crc32=f"{binascii.crc32(path.read_bytes()) & 0xffffffff:08x}"))
    manifest = tmp_path / "subset_manifest.json"
    manifest.write_text(json.dumps(dict(status="complete_raw_subset_not_admitted", labels_used=False, stores=stores)), encoding="utf-8")
    summary = audit(manifest, root / "audit", 64)
    try:
        yield summary, root / "claims" / "claims_manifest.json", root / "candidates", tmp_path
    finally:
        shutil.rmtree(root)


def test_candidate_export_preserves_claim_scope_and_accounts_for_every_record(prepared):
    summary, claims, output, raw_root = prepared
    before = [file_hash(raw_root / f"train-{i}.json") for i in range(2)]
    result = json.loads(prepare_candidates(summary, claims, output, POLICY).read_text())
    assert result["corpus_admitted"] is False
    assert result["inference_executed"] is False and result["network_accessed"] is False
    for i, store in enumerate(result["stores"]):
        assert store["sample_id"] == f"train:{i}"
        assert store["counts"] == dict(retained=1, quarantined_review_flag=2, empty_text=1)
        directory = output / store["directory"]
        corpus = read_jsonl(directory / "corpus.jsonl")
        assert len(corpus) == 1 and corpus[0]["text"] == f"Text {i}\nMore text"
        assert corpus[0]["retrieved_at"] == "unknown"
        ledger = read_jsonl(directory / "record_ledger.jsonl")
        assert len(ledger) == 4 and {r["sample_id"] for r in ledger} == {f"train:{i}"}
        assert read_jsonl(directory / "claims.jsonl")[0]["sample_id"] == f"train:{i}"
        assert store["corpus_sha256"] == file_hash(directory / "corpus.jsonl")
    assert before == [file_hash(raw_root / f"train-{i}.json") for i in range(2)]
    assert not (output / "data_manifest.json").exists()
    assert "SECRET" not in "".join(p.read_text(encoding="utf-8") for p in output.rglob("*.jsonl"))


def test_mutated_raw_store_refused_before_export(prepared):
    summary, claims, output, raw_root = prepared
    with (raw_root / "train-0.json").open("a") as stream: stream.write("{}\n")
    with pytest.raises(ValueError, match="integrity mismatch"):
        prepare_candidates(summary, claims, output, POLICY)
    assert not output.exists()


def test_mutated_claims_and_implicit_policy_refused(prepared):
    summary, claims, output, _ = prepared
    with pytest.raises(ValueError, match="Explicit supported"):
        prepare_candidates(summary, claims, output, None)
    (claims.parent / "claims.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Claim export hash"):
        prepare_candidates(summary, claims, output, POLICY)
    assert not output.exists()
