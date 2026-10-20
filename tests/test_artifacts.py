import uuid
import shutil
from pathlib import Path
import pytest
from tffv.inference import PredictionCache
from tffv.memory.artifacts import knowledge_manifest


def test_fold_local_knowledge_manifest():
    a = dict(sample_id="a", group_id="a", claim="claim a", gold_label="S")
    b = dict(sample_id="b", group_id="b", claim="claim b", gold_label="R")
    manifest = knowledge_manifest([a], [b], "source hash", {"frozen_model": "fixture"})
    assert manifest["allowed_sample_ids"] == ["a"]
    assert manifest["excluded_sample_ids"] == ["b"]
    assert manifest["evaluation_gold_used"] is False
    with pytest.raises(ValueError): knowledge_manifest([a], [a], "hash", {})


def test_cache_is_immutable_and_keys_safe():
    destination = Path(__file__).resolve().parents[1] / "outputs" / ("pytest-cache-" + uuid.uuid4().hex)
    try:
        cache = PredictionCache(destination)
        key = "a" * 64
        assert cache.get(key) is None
        cache.put(key, {"prediction": "S"})
        assert cache.get(key) == {"prediction": "S"}
        with pytest.raises(FileExistsError): cache.put(key, {"prediction": "R"})
        with pytest.raises(ValueError): cache.get("../unsafe")
        with pytest.raises(ValueError): cache.put("b" * 64, {"gold_label": "S"})
    finally:
        if destination.exists(): shutil.rmtree(destination)
