"""Comprehensive Unit and Integration Tests for Milestone M1.8g Recovery Orchestrator."""
import json
import os
import tempfile
from pathlib import Path
import pytest

from tffv.experiments.m18g_recovery import (
    ALL_OFFICIAL_TRAIN_COUNT,
    AVERITEC_CLASSES,
    DYNAMIC_MECHANISMS,
    EXACT_TRAIN_DEV_OVERLAPS,
    OFFICIAL_TRAIN_LABEL_COUNTS,
    OFFICIAL_TRAIN_SHA256,
    QUARANTINED_CLAIM_ID,
    VALID_TRAIN_COUNT,
    VALID_TRAIN_LABEL_COUNTS,
    M18gRecoveryOrchestrator,
    Phase0Reconciler,
    build_generator_prompt,
    parse_generator_output,
    safe_write_json,
)


@pytest.fixture
def mock_phase0_dataset(tmp_path):
    """Create a mock dataset matching canonical Official Train distribution and SHA-256."""
    train_records = []
    # 848 valid Supported + 1 empty Supported (1948) = 849
    # 1742 Refuted
    # 282 Not Enough Evidence
    # 195 Conflicting
    for i in range(849):
        txt = "" if i == 0 else f"Supported claim {i}"
        train_records.append({"claim": txt, "label": "Supported"})
    for i in range(1742):
        train_records.append({"claim": f"Refuted claim {i}", "label": "Refuted"})
    for i in range(282):
        train_records.append({"claim": f"NEI claim {i}", "label": "Not Enough Evidence"})
    for i in range(195):
        train_records.append({"claim": f"Conflicting claim {i}", "label": "Conflicting Evidence/Cherrypicking"})

    train_file = tmp_path / "train.json"
    safe_write_json(train_file, train_records)
    return train_file


def test_generator_prompt_and_schema_validation():
    """Test substantive generator prompt generation and JSON schema output parsing."""
    claim = "Drinking hot lemon water cures cancer."
    evidence = [
        {"rendered_text": "There is no clinical evidence that lemon water cures or prevents any form of cancer."}
    ]

    messages = build_generator_prompt(claim, evidence, "Supported", "Refuted")
    assert len(messages) == 2
    assert "Official Gold Verdict" in messages[1]["content"]
    assert "decisive_quotes" in messages[1]["content"]

    # 1. Valid substantive JSON
    raw_json = json.dumps({
        "status": "GENERATED",
        "failure_mechanism": "negation_or_polarity",
        "decisive_quotes": ["There is no clinical evidence that lemon water cures or prevents any form of cancer."],
        "correction_rationale": "Medical trials show no curative effect for lemon water on cancer cells.",
        "reasoning_summary": "Direct clinical contradiction.",
    })
    parsed = parse_generator_output(raw_json, evidence)
    assert parsed["status"] == "GENERATED"
    assert parsed["failure_mechanism"] == "negation_or_polarity"
    assert len(parsed["decisive_quotes"]) == 1
    assert parsed["quotes_grounded"] is True

    # 2. Insufficient evidence output
    raw_insufficient = json.dumps({
        "status": "INSUFFICIENT_EVIDENCE",
        "failure_mechanism": "missing_or_insufficient_evidence",
        "decisive_quotes": [],
        "correction_rationale": "Retrieved passages contain no relevant facts.",
        "reasoning_summary": "Empty pool.",
    })
    parsed_inf = parse_generator_output(raw_insufficient, evidence)
    assert parsed_inf["status"] == "INSUFFICIENT_EVIDENCE"
    assert parsed_inf["quotes_grounded"] is True

    # 3. Hallucinated quote output
    raw_hallucinated = json.dumps({
        "status": "GENERATED",
        "failure_mechanism": "causal_inference",
        "decisive_quotes": ["Lemon water causes severe kidney damage."],  # Not in evidence!
        "correction_rationale": "Fabricated rationale.",
    })
    parsed_hal = parse_generator_output(raw_hallucinated, evidence)
    assert len(parsed_hal["decisive_quotes"]) == 0  # Filtered out!


def test_phase0_reconciler_on_actual_canonical_train():
    """Verify Phase 0 reconciler against real data/raw/averitec/train.json."""
    raw_train = Path("data/raw/averitec/train.json")
    raw_dev = Path("data/raw/averitec/dev.json")
    if not raw_dev.exists():
        raw_dev = Path("/home/stackops/whale/ERRORFlow/data/raw/averitec/dev.json")
    if not raw_train.exists():
        pytest.skip("Local raw train.json not available")

    reconciler = Phase0Reconciler(raw_train, raw_dev)
    res = reconciler.reconcile()

    assert res["reconciliation_passed"] is True
    assert res["actual_sha256"] == OFFICIAL_TRAIN_SHA256
    assert res["total_claims"] == 3068
    assert res["valid_claims_count"] == 3067
    assert res["valid_label_counts"] == VALID_TRAIN_LABEL_COUNTS
    assert res["quarantined_record_ok"] is True
    assert res["overlap_check_ok"] is True


def test_synthetic_recovery_pipeline_end_to_end(tmp_path):
    """Test full M1.8g recovery pipeline end-to-end with synthetic models."""
    m18e_mock_dir = tmp_path / "m18e_inputs"
    m18e_mock_dir.mkdir(parents=True, exist_ok=True)

    # Base predictions mock: 10 claims (5 correct, 5 wrong)
    preds = []
    receipts = []
    for i in range(10):
        sid = f"train:{i}"
        is_corr = i < 5
        preds.append({
            "sample_id": sid,
            "claim": f"Claim text for {sid}",
            "gold_verdict": "Refuted" if i % 2 == 0 else "Supported",
            "predicted_verdict": "Refuted" if is_corr else "Supported",
            "is_correct": is_corr,
        })
        receipts.append({
            "sample_id": sid,
            "evidence": [{"rendered_text": f"Verifiable evidence excerpt for {sid} proving truth."}],
        })

    with open(m18e_mock_dir / "base_predictions.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")

    with open(m18e_mock_dir / "retrieval_receipts.jsonl", "w", encoding="utf-8") as f:
        for r in receipts:
            f.write(json.dumps(r) + "\n")

    # M1.8e memory with 1 existing accepted record
    safe_write_json(m18e_mock_dir / "memory.json", {
        "records": [
            {
                "sample_id": "train:5",
                "claim": "Claim text for train:5",
                "gold_verdict": "Supported",
                "base_verdict": "Refuted",
                "correction": "Existing correction for train:5",
                "mechanism": "evidence_sufficiency",
                "decision": "ACCEPT",
            }
        ]
    })

    out_dir = tmp_path / "m18g_outputs"

    orchestrator = M18gRecoveryOrchestrator(
        m18e_input_dir=m18e_mock_dir,
        output_dir=out_dir,
        synthetic=True,
    )

    # Bypass Phase 0 check on mock data for unit test
    state = orchestrator.load_state()
    state["completed_phases"].append("Phase_0_Reconciliation")
    orchestrator.save_state(state)

    orchestrator.run_all()

    # 1. Verify Manifest
    assert (out_dir / "recovery_manifest.json").exists()
    manifest = json.loads((out_dir / "recovery_manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_historical_errors"] == 5

    # 2. Verify Generation Results
    assert (out_dir / "generation_results.jsonl").exists()
    gens = [json.loads(l) for l in (out_dir / "generation_results.jsonl").read_text(encoding="utf-8").strip().split("\n")]
    assert len(gens) == 5
    # Verify no generic template used
    for g in gens:
        assert "contradicting the initial assessment of" not in g["correction_rationale"]
        assert g["failure_mechanism"] in DYNAMIC_MECHANISMS

    # 3. Verify Judge Decisions & Memory
    assert (out_dir / "judge_decisions.jsonl").exists()
    assert (out_dir / "memory.json").exists()
    mem = json.loads((out_dir / "memory.json").read_text(encoding="utf-8"))
    assert mem["total_records"] > 0
    for r in mem["records"]:
        assert r["memory_status"] == "LLM_VALIDATED"
        assert r["sample_id"] not in EXACT_TRAIN_DEV_OVERLAPS

    # 4. Verify Reports
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "class_coverage.csv").exists()
    assert (out_dir / "mechanism_coverage.csv").exists()
    assert (out_dir / "FINAL_REPORT.md").exists()
