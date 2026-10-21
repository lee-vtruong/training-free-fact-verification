import io
import json
import re
import shutil
import uuid
import zipfile
from pathlib import Path
from unittest.mock import patch
import pytest
from tffv.provenance import file_hash
from tffv.retrieval.acquire_subset import acquire, store_profile


class Response(io.BytesIO):
    def __init__(self, payload, headers):
        super().__init__(payload)
        self.status, self.headers = 206, headers


@pytest.fixture
def fixture(tmp_path):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for i in range(3):
            rows = [dict(claim_id=str(i), query="DO_NOT_USE_QUERY_FOR_SELECTION", type="web", url="https://source.example.org", url2text=["Independent text."]),
                    dict(claim_id=str(i), query="same", type="web", url="https://source.example.org/empty", url2text=[])]
            zipped.writestr(f"{i}.json", "\n".join(json.dumps(row) for row in rows) + "\n")
    raw = payload.getvalue()
    revision = "a" * 40
    names = ["train_0_999.zip", "train_1000_1999.zip", "train_2000_3067.zip"]
    archives = [dict(path="data_store/knowledge_store/train/" + name,
        url=f"https://huggingface.co/chenxwh/AVeriTeC/resolve/{revision}/data_store/knowledge_store/train/{name}",
        compressed_bytes=len(raw), sha256="b" * 64) for name in names]
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(dict(repo_id="chenxwh/AVeriTeC", split="train", resolved_revision=revision, archives=archives)), encoding="utf-8")
    destination = Path(__file__).resolve().parents[1] / "outputs" / ("pytest-acquire-" + uuid.uuid4().hex)
    def fetch(request, timeout):
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
        return Response(raw[start:end + 1], {"Content-Range": f"bytes {start}-{end}/{len(raw)}", "Content-Length": str(end - start + 1)})
    try: yield plan, destination, fetch
    finally:
        if destination.exists(): shutil.rmtree(destination)


def test_preflight_does_not_save_store_bodies(fixture):
    plan, output, fetch = fixture
    with patch("tffv.retrieval.remote_zip.urlopen", side_effect=fetch):
        result = acquire(plan, output, [0, 1, 2])
    assert result.name == "subset_plan.json"
    assert not list(output.glob("train-*.json"))


def test_full_members_crc_hash_and_mapping(fixture):
    plan, output, fetch = fixture
    with patch("tffv.retrieval.remote_zip.urlopen", side_effect=fetch):
        result = acquire(plan, output, [0, 1, 2], execute=True)
    report = json.loads(result.read_text(encoding="utf-8"))
    assert report["source_claim_ordinals"] == [0, 1, 2]
    assert report["corpus_admitted"] is False
    for row in report["stores"]:
        assert row["full_member_crc_verified"] is True
        assert row["whole_archive_sha256_verified"] is False
        assert row["raw_sha256"] == file_hash(output / row["file"])
        assert row["schema_audit"]["records"] == 2
        assert row["schema_audit"]["empty_text_rows"] == 1
        assert row["schema_audit"]["claim_id_mismatch_rows"] == 0
    assert "DO_NOT_USE_QUERY_FOR_SELECTION" not in result.read_text(encoding="utf-8")


def test_store_caps_fail_before_member_transfer(fixture):
    plan, output, fetch = fixture
    with patch("tffv.retrieval.remote_zip.urlopen", side_effect=fetch), patch("tffv.retrieval.acquire_subset.RAW_CAP", 1):
        with pytest.raises(ValueError, match="caps"): acquire(plan, output, [0], execute=True)
    assert not list(output.glob("train-*.json"))
    assert (output / "failure.json").is_file()


def test_bad_member_crc_never_seals_or_publishes_complete_store(fixture):
    plan, output, fetch = fixture
    def corrupted(request, timeout):
        response = fetch(request, timeout)
        payload = bytearray(response.getvalue())
        central = payload.find(b"PK\x01\x02")
        if central >= 0 and central + 20 <= len(payload):
            payload[central + 16] ^= 1  # Corrupt expected CRC in central-directory metadata.
        return Response(bytes(payload), response.headers)
    with patch("tffv.retrieval.remote_zip.urlopen", side_effect=corrupted):
        with pytest.raises(zipfile.BadZipFile): acquire(plan, output, [0], execute=True)
    assert (output / "failure.json").is_file()
    assert not (output / "subset_manifest.json").exists()
    assert not (output / "train-0.json").exists()


def test_profile_flags_identity_url_and_verdict_without_printing_content(tmp_path):
    path = tmp_path / "store.json"
    row = dict(claim_id="wrong", url="bad url", url2text=["Verdict: SECRET_CONTENT"])
    path.write_text(json.dumps(row), encoding="utf-8")
    report = store_profile(path, 0)
    assert report["claim_id_mismatch_rows"] == 1
    assert report["invalid_source_url_rows"] == 1
    assert report["verdict_pattern_rows"] == 1
    assert "SECRET_CONTENT" not in json.dumps(report)
