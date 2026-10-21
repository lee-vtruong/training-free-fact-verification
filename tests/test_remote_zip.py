import io
import json
import re
import zipfile
from unittest.mock import patch
import pytest
from tffv.retrieval.remote_zip import HTTPRangeReader, RangeBudgetError, inspect_archive, validate_plan


class Response(io.BytesIO):
    def __init__(self, payload, status=206, headers=None):
        super().__init__(payload)
        self.status, self.headers = status, headers or {}
        self.read_called = False

    def read(self, *args):
        self.read_called = True
        return super().read(*args)


def archive_bytes(zip64=False):
    target = io.BytesIO()
    with patch("zipfile.ZIP64_LIMIT", 0 if zip64 else zipfile.ZIP64_LIMIT):
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as stream:
            stream.writestr("output_train/0.jsonl", json.dumps({"url": "SECRET_URL_VALUE", "text": "PRIVATE_CONTENT"}) + "\n")
            stream.writestr("output_train/1.jsonl", "{}\n")
    return target.getvalue()


def opener(payload):
    def fetch(request, timeout):
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
        return Response(payload[start:end + 1], headers={"Content-Range": f"bytes {start}-{end}/{len(payload)}", "Content-Length": str(end - start + 1)})
    return fetch


@pytest.mark.parametrize("zip64", [False, True])
def test_bounded_zip_directory_and_schema_without_content_exports(zip64):
    payload = archive_bytes(zip64)
    archive = {"url": "https://example.org/pinned.zip", "compressed_bytes": len(payload), "sha256": "a" * 64, "path": "train.zip"}
    with patch("tffv.retrieval.remote_zip.urlopen", side_effect=opener(payload)):
        result = inspect_archive(archive, sample=True)
    assert result["file_count"] == 2
    assert result["schema_sample"]["record_shape"] == {"text": "str", "url": "str"}
    assert "SECRET_URL_VALUE" not in json.dumps(result)
    assert "PRIVATE_CONTENT" not in json.dumps(result)
    assert result["whole_archive_sha256_verified"] is False
    assert result["range_payload_bytes"] <= 4 * 1024 * 1024
    assert result["uncompressed_bytes_declared"] > 0


def test_ignored_range_is_rejected_before_reading_file():
    response = Response(b"full archive", status=200)
    with patch("tffv.retrieval.remote_zip.urlopen", return_value=response):
        with pytest.raises(ValueError, match="no full-download fallback"):
            HTTPRangeReader("https://example.org/file", 100).read(10)
    assert response.read_called is False


def test_range_identity_budget_and_cache():
    data = b"0123456789"
    reader = HTTPRangeReader("https://example.org/file", len(data), budget=4)
    with patch("tffv.retrieval.remote_zip.urlopen", side_effect=opener(data)) as fetched:
        assert reader.read(4) == b"0123"
        reader.seek(0)
        assert reader.read(4) == b"0123"
        assert fetched.call_count == 1
        with pytest.raises(RangeBudgetError): reader.read(1)
    with pytest.raises(ValueError): reader.seek(-1)
    response = Response(b"1234", headers={"Content-Range": "bytes 0-3/11"})
    with patch("tffv.retrieval.remote_zip.urlopen", return_value=response):
        with pytest.raises(ValueError, match="Content-Range"):
            HTTPRangeReader("https://example.org/file", 10).read(4)
    assert response.read_called is False


def test_plan_rejects_unpinned_or_nonofficial_urls():
    revision = "a" * 40
    names = ["train_0_999.zip", "train_1000_1999.zip", "train_2000_3067.zip"]
    archives = [{"path": "data_store/knowledge_store/train/" + name, "url": f"https://huggingface.co/chenxwh/AVeriTeC/resolve/{revision}/data_store/knowledge_store/train/{name}", "sha256": "b" * 64, "compressed_bytes": 10} for name in names]
    plan = {"repo_id": "chenxwh/AVeriTeC", "split": "train", "resolved_revision": revision, "archives": archives}
    assert validate_plan(plan) == archives
    archives[0]["url"] = archives[0]["url"].replace(revision, "main")
    with pytest.raises(ValueError): validate_plan(plan)
