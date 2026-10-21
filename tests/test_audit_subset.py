import binascii
import json
import shutil
import uuid
from pathlib import Path
import pytest
from tffv.provenance import file_hash
from tffv.retrieval.audit_subset import audit, audit_store


@pytest.fixture
def acquired(tmp_path):
    rows = [dict(claim_id="0", url="https://source.example/a", url2text=["Independent passage."]),
            dict(claim_id="0", url="https://fullfact.org/example", url2text=["Verdict: suspected material for review."]),
            dict(claim_id="0", url="https://source.example/empty", url2text=[])]
    path = tmp_path / "train-0.json"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    metadata = dict(status="complete_raw_subset_not_admitted", labels_used=False, stores=[
        dict(source_ordinal=0, file=path.name, full_member_crc_verified=True, raw_bytes=path.stat().st_size,
             raw_sha256=file_hash(path), member_crc32=f"{binascii.crc32(path.read_bytes()) & 0xffffffff:08x}")])
    manifest = tmp_path / "subset_manifest.json"
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    output = Path(__file__).resolve().parents[1] / "outputs" / ("pytest-offline-audit-" + uuid.uuid4().hex)
    try: yield path, manifest, output
    finally:
        if output.exists(): shutil.rmtree(output)


def test_offline_audit_seals_hashes_flags_and_bounded_review(acquired):
    path, manifest, output = acquired
    before = file_hash(path)
    result = audit(manifest, output, 64)
    summary = json.loads(result.read_text(encoding="utf-8"))
    store = summary["stores"][0]
    assert store["records"] == 3
    assert store["empty_text_rows"] == 1
    assert store["verdict_pattern_rows"] == 1
    assert store["fact_check_source_url_rows"] == 1
    assert summary["network_accessed"] is False
    assert summary["corpus_admitted"] is False
    assert file_hash(path) == before
    queue = json.loads((output / "review_queue.json").read_text(encoding="utf-8"))
    assert len(queue) == 1
    assert queue[0]["line_number"] == 2
    assert "not confirmed" in queue[0]["assessment"]


def test_oversized_record_never_silently_truncates_or_skips(tmp_path):
    path = tmp_path / "large.json"
    path.write_text(json.dumps(dict(claim_id="0", url="https://source.example", url2text=["word " * 1000])), encoding="utf-8")
    with pytest.raises(ValueError, match="no truncation or row skipping"):
        audit_store(path, 0, 100)
    profile, _ = audit_store(path, 0, 10000)
    assert profile["records"] == 1
    assert profile["largest_records"][0]["words"] == 1000
    assert profile["estimated_passages_180_words_overlap30_before_filtering"] == 7


def test_modified_source_rejected_before_audit(acquired):
    path, manifest, output = acquired
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"): audit(manifest, output, 64)
    assert not output.exists()


def test_record_cap_is_explicit_and_review_queue_is_bounded(acquired):
    path, manifest, output = acquired
    with pytest.raises(ValueError): audit(manifest, output, 65)
    profile, review = audit_store(path, 0, 10000, review_limit=0)
    assert not review
    assert profile["review_queue_truncated"] is True
