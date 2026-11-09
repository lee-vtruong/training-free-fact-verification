"""Unit tests for M1.8c Full Retrieval Pipeline Runner."""
import json
import pytest
from pathlib import Path

from scripts.run_m18c_full_retrieval import (
    INVALID_SAMPLE_IDS,
    load_canonical_r4_cache,
    run_full_retrieval,
)


def test_invalid_sample_id_invariants():
    assert "train:1948" in INVALID_SAMPLE_IDS


def test_canonical_cache_scan_with_mock_root(tmp_path):
    mock_dir = tmp_path / "mock_cache"
    mock_dir.mkdir()
    r4_file = mock_dir / "R4.jsonl"
    r4_record = {
        "sample_id": "train:9999",
        "evidence": [{"passage_id": f"p_{i}", "text": f"Text {i}"} for i in range(8)],
    }
    r4_file.write_text(json.dumps(r4_record) + "\n", encoding="utf-8")

    cache = load_canonical_r4_cache(roots=[mock_dir])
    assert len(cache) == 1
    assert "train:9999" in cache
    assert len(cache["train:9999"]["record"]["evidence"]) == 8


def test_synthetic_full_retrieval_runner_execution(tmp_path, monkeypatch):
    out_dir = tmp_path / "m18c_test"

    # Mock prepare_claim_input_and_candidates to provide synthetic documents
    def mock_prepare(sid, raw_train, acq_plan_path, claim_work_dir):
        claim_meta = {
            "sample_id": sid,
            "dataset": "AVeriTeC",
            "split": "train",
            "claim": f"Synthetic claim content for {sid}",
        }
        documents = [
            {
                "document_id": f"doc_{i}",
                "source_url": f"https://example.org/{sid}/doc_{i}",
                "source_kind": "web_document",
                "retrieved_at": "2024-01-01T00:00:00Z",
                "acquisition_id": "acq_mock",
                "text": f"Document text {i} for {sid} with enough facts to satisfy retrieval.",
            }
            for i in range(12)
        ]
        return claim_meta, documents

    monkeypatch.setattr(
        "scripts.run_m18c_full_retrieval.prepare_claim_input_and_candidates",
        mock_prepare,
    )

    summary = run_full_retrieval(
        start_claim=0,
        end_claim=3,
        config_path="configs/m161_source_batch25.json",
        acq_plan_path="configs/m161_acquisition_plan.json",
        output_dir=str(out_dir),
        batch_size=16,
        reuse_canonical=False,
        synthetic=True,
        progress_interval=1,
    )

    assert summary["schema_version"] == "m18c-full-retrieval-summary-v1"
    assert summary["total_target_claims"] == 3
    assert summary["total_completed"] == 3
    assert summary["retrieved_fresh_count"] == 3

    # Check state file
    state_file = out_dir / "state.json"
    assert state_file.is_file()
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert set(state["completed"]) == {"train:0", "train:1", "train:2"}

    # Check master JSONL
    master_file = out_dir / "all_train_r4.jsonl"
    assert master_file.is_file()
    lines = [json.loads(l) for l in master_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 3
    for rec in lines:
        assert rec["sample_id"].startswith("train:")
        assert len(rec["evidence"]) == 8

    # Test Checkpointing Resumption: running again on range 0..4 should reuse 0..2
    summary2 = run_full_retrieval(
        start_claim=0,
        end_claim=4,
        config_path="configs/m161_source_batch25.json",
        acq_plan_path="configs/m161_acquisition_plan.json",
        output_dir=str(out_dir),
        batch_size=16,
        reuse_canonical=False,
        synthetic=True,
        progress_interval=1,
    )
    assert summary2["total_completed"] == 4
    assert summary2["reused_previous_run_count"] == 3
    assert summary2["retrieved_fresh_count"] == 1
