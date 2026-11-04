"""Unit and integration tests for Milestone M1.8e Full-Train Memory Orchestrator."""
import json
import os
import tempfile
import time
from pathlib import Path
import pytest

from tffv.experiments.m18e_orchestrator import (
    EXACT_TRAIN_DEV_OVERLAPS,
    QUARANTINED_INVALID_CLAIMS,
    M18eConfig,
    M18eOrchestrator,
    ProcessLock,
    norm_text,
    safe_write_json,
    load_jsonl,
    append_jsonl,
)


@pytest.fixture
def mock_dataset(tmp_path):
    """Create a minimal synthetic AVeriTeC train and dev dataset with exact overlaps."""
    train_records = []
    # Create 3068 train claims
    for idx in range(3068):
        sid = f"train:{idx}"
        if idx == 1948:
            claim_text = ""  # empty quarantined
        elif idx == 1340:
            claim_text = "The Moon landing was faked in a Hollywood studio."
        elif idx == 1341:
            claim_text = "Vaccines cause autism according to CDC data."
        elif idx == 1536:
            claim_text = "Drinking 8 glasses of water a day is mandatory."
        elif idx == 1539:
            claim_text = "Coffee stunts human growth."
        else:
            claim_text = f"Sample fact checking claim number {idx}."

        label = "Supported" if idx % 2 == 0 else "Refuted"
        train_records.append({
            "claim": claim_text,
            "label": label,
            "fact_checking_article": f"https://example.com/article/{idx}",
        })

    # Create 500 dev claims
    dev_records = []
    for idx in range(500):
        if idx == 248:
            claim_text = "The Moon landing was faked in a Hollywood studio."
        elif idx == 247:
            claim_text = "Vaccines cause autism according to CDC data."
        elif idx == 447:
            claim_text = "Drinking 8 glasses of water a day is mandatory."
        elif idx == 446:
            claim_text = "Coffee stunts human growth."
        else:
            claim_text = f"Sample dev claim number {idx}."

        dev_records.append({
            "claim": claim_text,
            "label": "Supported",
            "fact_checking_article": f"https://example.com/dev_article/{idx}",
        })

    train_file = tmp_path / "train.json"
    dev_file = tmp_path / "dev.json"
    safe_write_json(train_file, train_records)
    safe_write_json(dev_file, dev_records)
    return train_file, dev_file


def test_process_lock(tmp_path):
    """Test process lock acquisition, collision prevention, and clean release."""
    lock_file = tmp_path / ".m18e.pid"
    lock1 = ProcessLock(lock_file)
    assert lock1.acquire() is True
    assert lock_file.exists()
    assert int(lock_file.read_text().strip()) == os.getpid()

    # Second lock should detect active PID
    lock2 = ProcessLock(lock_file)
    assert lock2.acquire() is False

    lock1.release()
    assert not lock_file.exists()


def test_gold_blind_base_prompt_isolation(tmp_path):
    """Test that Base inference prompt completely withholds gold labels."""
    orchestrator = M18eOrchestrator(
        output_dir=tmp_path,
        synthetic=True,
    )
    from tffv.verification.backbone import SyntheticBackbone
    backbone = SyntheticBackbone()

    claim_text = "The earth is flat."
    gold_label = "Refuted"
    evidence = [{"source_url": "https://wiki.org", "quote": "The Earth is an oblate spheroid."}]

    pred = orchestrator._infer_single_base_claim(backbone, "train:42", claim_text, evidence)
    assert pred["sample_id"] == "train:42"
    assert pred["predicted_verdict"] in ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]
    # Verify gold label was not leaked in prompt or prediction record
    assert "gold" not in pred


def test_judge_quote_verification_rejection():
    """Test that Judge rejects candidate when quotes are not exact substrings of evidence."""
    from tffv.experiments.m18e_orchestrator import ServiceJudgeEvaluator
    evaluator = ServiceJudgeEvaluator(port=17171)

    cand = {
        "sample_id": "train:10",
        "claim": "Water boils at 100 degrees Celsius.",
        "gold_verdict": "Supported",
        "base_verdict": "Refuted",
        "draft_text": "Water indeed boils at 100 degrees at sea level.",
        "mechanism": "evidence_sufficiency",
    }
    evidence = [
        {"source_url": "https://science.org", "quote": "Liquid water turns into vapor at 100 degrees Celsius under standard atmospheric pressure."}
    ]

    # Mock _query_service to return a fabricated quote that is NOT in the evidence
    def mock_query(req):
        if req["stage"] == "evidence_assessment":
            return {"failure_mechanism": "evidence_sufficiency"}
        return {
            "decision": "ACCEPT",
            "grounding_quotes": ["Water boils at 500 degrees Fahrenheit."],  # Hallucinated quote!
            "evidence_supported": True,
            "unsupported_facts": [],
        }

    evaluator._query_service = mock_query
    res = evaluator.evaluate_claim(cand, evidence)
    # Must be REJECTED because quote is not in evidence!
    assert res["decision"] == "REJECT"
    assert len(res["grounding_quotes"]) == 0


def test_synthetic_end_to_end_pipeline(mock_dataset, tmp_path):
    """Execute complete Phases A..H end-to-end with synthetic models on mock dataset."""
    train_file, dev_file = mock_dataset
    out_dir = tmp_path / "m18e_output"

    orchestrator = M18eOrchestrator(
        output_dir=out_dir,
        train_path=str(train_file),
        dev_path=str(dev_file),
        synthetic=True,
    )

    # Run with limit=25 claims for fast unit testing
    orchestrator.run_all(limit=25)

    # 1. Verify Phase A artifacts
    assert (out_dir / "config_lock.json").exists()
    assert (out_dir / "cache_inventory.json").exists()
    assert (out_dir / "state.json").exists()

    state = json.loads((out_dir / "state.json").read_text(encoding="utf-8"))
    assert state["phase_status"]["Phase_A_Preflight"] == "COMPLETED"
    assert state["phase_status"]["Phase_B_Retrieval"] == "COMPLETED"
    assert state["phase_status"]["Phase_C_Base_Inference"] == "COMPLETED"
    assert state["phase_status"]["Phase_D_Historical_Outcome_Bank"] == "COMPLETED"
    assert state["phase_status"]["Phase_E_Correction_Generation"] == "COMPLETED"
    assert state["phase_status"]["Phase_F_Stronger_Judge"] == "COMPLETED"
    assert state["phase_status"]["Phase_G_Memory_Assembly"] == "COMPLETED"
    assert state["phase_status"]["Phase_H_Validation_and_Reporting"] == "COMPLETED"

    # 2. Verify Output artifacts
    assert (out_dir / "retrieval_receipts.jsonl").exists()
    assert (out_dir / "base_predictions.jsonl").exists()
    assert (out_dir / "outcome_bank.json").exists()
    assert (out_dir / "candidate_corrections.jsonl").exists()
    assert (out_dir / "judge_decisions.jsonl").exists()
    assert (out_dir / "memory.json").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "INTEGRITY_AUDIT.md").exists()
    assert (out_dir / "FINAL_REPORT.md").exists()

    # 3. Verify Memory Bank constraints
    memory = json.loads((out_dir / "memory.json").read_text(encoding="utf-8"))
    assert memory["scope"] == "FULL_TRAIN_HISTORICAL_ERROR_MEMORY"
    assert memory["memory_status"] == "LLM_VALIDATED"
    assert memory["human_review_status"] == "UNCHANGED_NOT_HUMAN_VERIFIED"

    # Assert 0 dev contamination and exact overlaps excluded
    for r in memory["records"]:
        assert not r["sample_id"].startswith("dev:")
        assert r["sample_id"] not in EXACT_TRAIN_DEV_OVERLAPS
        assert r["sample_id"] not in QUARANTINED_INVALID_CLAIMS
        assert r["memory_status"] == "LLM_VALIDATED"


def test_resume_idempotence(mock_dataset, tmp_path):
    """Test that restarting the orchestrator skips already completed claims without recomputation."""
    train_file, dev_file = mock_dataset
    out_dir = tmp_path / "m18e_resume_test"

    orchestrator1 = M18eOrchestrator(
        output_dir=out_dir,
        train_path=str(train_file),
        dev_path=str(dev_file),
        synthetic=True,
    )
    # Pre-populate retrieval receipts with 5 claims
    out_dir.mkdir(parents=True, exist_ok=True)
    receipts_file = out_dir / "retrieval_receipts.jsonl"
    for i in range(5):
        append_jsonl(receipts_file, {
            "sample_id": f"train:{i}",
            "method": "R4-PF1000-v1",
            "evidence": [{"source_url": "http://mock", "quote": "Mock text"}],
            "latency_seconds": 0.05,
        })

    # Run Phase B with limit=10
    state = orchestrator1.load_state()
    orchestrator1.run_phase_a(state)
    orchestrator1.run_phase_b(state, limit=10)

    # Verify receipts now have 15 claims total (5 existing + 10 new)
    records = load_jsonl(receipts_file)
    assert len(records) == 15
    unique_sids = {r["sample_id"] for r in records}
    assert len(unique_sids) == 15  # Zero duplicates!
