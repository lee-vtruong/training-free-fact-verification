import json
import shutil
import uuid
from pathlib import Path
import pytest
from tffv.data import read_jsonl
from tffv.provenance import file_hash
from tffv.retrieval.prepare_claims import prepare_source_claims


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "verified_train.json"
    rows = [dict(claim="First claim", label="SECRET_VERDICT", questions=[dict(answer="SECRET_ANSWER")],
                 justification="SECRET_JUSTIFICATION", fact_checking_article="https://SECRET.example.org"),
            dict(claim="First claim", label="OTHER_SECRET_VERDICT")]
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


@pytest.fixture
def output():
    path = Path(__file__).resolve().parents[1] / "outputs" / ("pytest-claims-" + uuid.uuid4().hex)
    try: yield path
    finally:
        if path.exists(): shutil.rmtree(path)


def test_claim_export_preserves_all_rows_and_strips_annotations(source, output):
    original = file_hash(source)
    manifest = prepare_source_claims(source, output, original, 2)
    rows = read_jsonl(output / "claims.jsonl")
    assert [r["sample_id"] for r in rows] == ["train:0", "train:1"]
    assert rows[0]["claim"] == rows[1]["claim"] == "First claim"  # Do not silently deduplicate official claims.
    assert set(rows[0]) == {"sample_id", "dataset", "split", "claim"}
    assert "SECRET" not in "".join(p.read_text(encoding="utf-8") for p in output.iterdir())
    assert manifest["corpus_provided"] is False
    assert manifest["annotation_values_used"] is False
    assert file_hash(source) == original
    with pytest.raises(FileExistsError): prepare_source_claims(source, output, original, 2)


def test_hash_count_and_output_boundary_fail_closed(source, output, tmp_path):
    with pytest.raises(ValueError, match="hash mismatch"): prepare_source_claims(source, output, "0" * 64, 2)
    with pytest.raises(ValueError, match="count mismatch"): prepare_source_claims(source, output, file_hash(source), 3)
    with pytest.raises(ValueError, match="NEW project"): prepare_source_claims(source, tmp_path / "outside", file_hash(source), 2)
    assert not output.exists()
