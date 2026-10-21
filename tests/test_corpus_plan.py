import copy
import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import patch
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/m11_corpus_plan.py"
    spec = importlib.util.spec_from_file_location("m11_corpus_plan", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def metadata(probe):
    return {"id": probe.REPOSITORY, "sha": "a" * 40, "siblings": [
        {"rfilename": name, "size": 100 + i, "lfs": {"sha256": str(i) * 64, "size": 100 + i}}
        for i, name in enumerate(probe.ARCHIVES)] + [{"rfilename": "data/train.json", "size": 123}]}


def test_plan_uses_only_train_archives_and_pinned_urls():
    probe = module()
    plan = probe.build_plan(metadata(probe), 1000)
    assert plan["total_compressed_bytes"] == 303
    assert plan["compressed_only_fits"] is True
    assert plan["ready_for_bulk_download"] is False
    assert plan["download_executed"] is False
    assert all("/resolve/" + "a" * 40 + "/" in item["url"] for item in plan["archives"])
    assert all(item["uncompressed_bytes"] is None for item in plan["archives"])
    assert probe.build_plan(metadata(probe), 1)["compressed_only_fits"] is False


@pytest.mark.parametrize("failure", ["revision", "repo", "hash", "size", "missing", "duplicate", "requested_revision"])
def test_invalid_metadata_fails_closed(failure):
    probe = module()
    data = copy.deepcopy(metadata(probe))
    requested = "main"
    if failure == "revision": data["sha"] = "main"
    elif failure == "repo": data["id"] = "other/repository"
    elif failure == "hash": data["siblings"][0]["lfs"]["sha256"] = "a" * 40
    elif failure == "size": data["siblings"][0]["size"] = 999
    elif failure == "missing": data["siblings"].pop(0)
    elif failure == "duplicate": data["siblings"].append(data["siblings"][0])
    elif failure == "requested_revision": requested = "b" * 40
    with pytest.raises(ValueError): probe.build_plan(data, 1000, requested)


def test_fetch_requests_bounded_metadata_not_archive():
    probe = module()
    response = io.BytesIO(json.dumps(metadata(probe)).encode())
    response.headers = {"Content-Type": "application/json; charset=utf-8"}
    with patch.object(probe, "urlopen", return_value=response) as opened:
        data, raw, url = probe.fetch_metadata()
    assert "/api/models/chenxwh/AVeriTeC/revision/main?blobs=true" in url
    assert "/resolve/" not in opened.call_args.args[0].full_url
    assert data["sha"] == "a" * 40
    assert len(raw) < probe.MAX_METADATA_BYTES


def test_fetch_rejects_file_payload_and_oversize_metadata():
    probe = module()
    response = io.BytesIO(b"PK not metadata")
    response.headers = {"Content-Type": "application/zip"}
    with patch.object(probe, "urlopen", return_value=response):
        with pytest.raises(ValueError, match="archive/file"): probe.fetch_metadata()
    response = io.BytesIO(b" " * (probe.MAX_METADATA_BYTES + 1))
    response.headers = {"Content-Type": "application/json"}
    with patch.object(probe, "urlopen", return_value=response):
        with pytest.raises(ValueError, match="4 MiB"): probe.fetch_metadata()
