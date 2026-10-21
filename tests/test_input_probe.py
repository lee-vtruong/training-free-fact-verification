import importlib.util
import json
import sys
from pathlib import Path


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/m11_probe_inputs.py"
    spec = importlib.util.spec_from_file_location("m11_input_probe", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_schema_counts_source_rows_without_exporting_annotation_values(tmp_path):
    probe = module()
    row = {"claim": "PRIVATE_CLAIM_VALUE", "label": "SECRET_GOLD_VALUE", "questions": [{"answer": "SECRET_ANSWER_VALUE"}]}
    path = tmp_path / "source.json"
    path.write_text(json.dumps([row, row]), encoding="utf-8")
    result = probe.source_schema(path)
    assert result["records"] == 2
    assert result["record_shapes"][0]["label"] == "str"
    assert "SECRET" not in json.dumps(result)
    assert "PRIVATE_CLAIM_VALUE" not in json.dumps(result)
    path = tmp_path / "source.jsonl"
    path.write_text("\n".join(json.dumps(row) for _ in range(3)), encoding="utf-8")
    assert probe.source_schema(path)["records"] == 3


def test_probe_limits_size_and_searches_non_json_corpus_paths(tmp_path):
    probe = module()
    (tmp_path / "averitec_knowledge_store.zip").write_bytes(b"FAKE TEST ARCHIVE")
    (tmp_path / "__MACOSX").mkdir()
    (tmp_path / "__MACOSX" / "corpus.jsonl").write_text("ignored", encoding="utf-8")
    report = probe.candidate_paths(tmp_path)
    assert len(report["candidates"]) == 1
    assert report["candidates"][0]["path"].endswith("averitec_knowledge_store.zip")
    assert "inspection" in probe.source_schema(tmp_path / "averitec_knowledge_store.zip", max_bytes=1)


def test_environment_probe_reads_metadata_without_torch_import():
    report = module().environment_metadata(sys.executable)
    assert report["returncode"] == 0
    assert "torch" in report["metadata"]["packages"]
