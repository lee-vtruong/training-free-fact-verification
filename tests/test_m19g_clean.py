"""Comprehensive unit & integration tests for Milestone M1.9g."""

import json
from pathlib import Path

import pytest

from tffv.experiments.m19g_evaluator import M19gEvaluator, parse_verdict_hardened
from tffv.experiments.m19g_guardrails import (
    EvidenceRenderBlankError,
    KnowledgeStoreUnverifiedError,
    RetrievalReceiptsMissingError,
    verify_knowledge_store,
    verify_retrieval_receipts,
)
from tffv.experiments.m19g_loader import M19gDevLoader
from tffv.experiments.m19g_orchestrator import M19gOrchestrator


def test_m19g_evaluator_metrics_and_battery():
    evaluator = M19gEvaluator(bootstrap_samples=100, seed=42)

    # 4 dummy predictions
    preds_h2 = [
        {"sample_id": "dev:0", "gold_verdict": "Supported", "predicted_verdict": "Supported"},
        {"sample_id": "dev:1", "gold_verdict": "Refuted", "predicted_verdict": "Refuted"},
        {"sample_id": "dev:2", "gold_verdict": "Not Enough Evidence", "predicted_verdict": "Refuted"},
        {"sample_id": "dev:3", "gold_verdict": "Conflicting Evidence/Cherrypicking", "predicted_verdict": "Conflicting Evidence/Cherrypicking"},
    ]
    preds_b1 = [
        {"sample_id": "dev:0", "gold_verdict": "Supported", "predicted_verdict": "Supported"},
        {"sample_id": "dev:1", "gold_verdict": "Refuted", "predicted_verdict": "Supported"},
        {"sample_id": "dev:2", "gold_verdict": "Not Enough Evidence", "predicted_verdict": "Supported"},
        {"sample_id": "dev:3", "gold_verdict": "Conflicting Evidence/Cherrypicking", "predicted_verdict": "Supported"},
    ]

    m_h2 = evaluator.compute_arm_metrics(preds_h2)
    assert m_h2.accuracy == 0.75
    assert m_h2.total_claims == 4
    assert m_h2.valid_verdict_rate == 1.0

    m_b1 = evaluator.compute_arm_metrics(preds_b1)
    assert m_b1.accuracy == 0.25

    # Hypothesis battery
    arm_preds = {
        "H2": preds_h2,
        "B1": preds_b1,
        "B0": preds_b1,
        "H1": preds_h2,
        "H3": preds_h2,
        "B2": preds_b1,
    }
    battery = evaluator.run_hypothesis_battery(arm_preds)
    assert "hypotheses_evaluated" in battery
    assert len(battery["hypotheses_evaluated"]) == 5


def test_m19g_orchestrator_mock_smoke(tmp_path: Path):
    # Setup mock data directory
    ks_dir = tmp_path / "ks"
    ks_dir.mkdir()
    for i in range(5):
        (ks_dir / f"{i}.json").write_text(
            json.dumps({"document_id": f"doc_{i}", "text": f"Document text for claim {i}"}) + "\n",
            encoding="utf-8",
        )

    receipts_file = tmp_path / "dev_retrieval_receipts.jsonl"
    with open(receipts_file, "w", encoding="utf-8") as f:
        for i in range(5):
            f.write(json.dumps({
                "sample_id": f"dev:{i}",
                "evidence": [{"passage_id": f"p_{i}", "text": f"Retrieved passage snippet {i}"}],
            }) + "\n")

    # Mock historical store
    m18e_dir = tmp_path / "m18e"
    m18e_dir.mkdir()
    (m18e_dir / "retrieval_receipts.jsonl").write_text(
        json.dumps({"sample_id": "train:0", "evidence": [{"passage_id": "p0", "text": "Train passage"}]}) + "\n",
        encoding="utf-8",
    )
    (m18e_dir / "base_predictions.jsonl").write_text(
        json.dumps({"sample_id": "train:0", "prediction": "Supported", "is_correct": False, "is_parseable": True}) + "\n",
        encoding="utf-8",
    )
    (m18e_dir / "stage2_validation_receipts.jsonl").write_text("", encoding="utf-8")

    out_dir = tmp_path / "output_m19g"

    # Canonical dev path
    dev_path = Path("data/raw/averitec/dev.json")
    if not dev_path.exists():
        pytest.skip("Canonical dev.json not available locally.")

    train_path = Path("data/raw/averitec/train.json")
    if not train_path.exists():
        pytest.skip("Canonical train.json not available locally.")

    orchestrator = M19gOrchestrator(
        m18e_dir=m18e_dir,
        output_dir=out_dir,
        dev_path=dev_path,
        dev_store_dir=ks_dir,
        dev_receipts_path=receipts_file,
        train_path=train_path,
        k=1,
    )

    res = orchestrator.run_all(run_gpu=False, limit_claims=5)
    assert res["status"] == "COMPLETED"
    assert (out_dir / "DECISION_REPORT.md").exists()
    assert (out_dir / "predictions.jsonl").exists()
    assert (out_dir / "statistical_tests.json").exists()
