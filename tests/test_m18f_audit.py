"""Comprehensive Unit and Integration Tests for Milestone M1.8f Audit Engine."""
import csv
import json
import os
import tempfile
from pathlib import Path
import pytest

from tffv.experiments.m18f_audit import (
    ALL_OFFICIAL_TRAIN_COUNT,
    AVERITEC_CLASSES,
    E1_INSUFFICIENT,
    E3_SUFFICIENT_TOP8,
    EXACT_TRAIN_DEV_OVERLAPS,
    NORMALIZED_MECHANISMS,
    QUARANTINED_CLAIM_ID,
    TIER_A_DETERMINISTIC_BUG,
    TIER_B_REQUIRES_RERUN,
    TIER_D_LEGIT_REJECTED,
    M18fAuditor,
    normalize_mechanism,
    safe_write_json,
)


@pytest.fixture
def mock_m18e_artifacts(tmp_path):
    """Create a self-consistent synthetic set of M1.8e input artifacts."""
    input_dir = tmp_path / "m18e_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summary
    summary_data = {
        "version": "m18e-full-history-pf1000-v1",
        "total_official_train_claims": 10,
        "valid_claims_processed": 9,
        "wrong_claims_count": 5,
        "correct_claims_count": 4,
        "unparseable_claims_count": 0,
        "candidate_corrections_count": 5,
        "judge_decisions_count": 5,
        "expanded_memory_size": 2,
    }
    safe_write_json(input_dir / "summary.json", summary_data)

    # 2. Outcome bank
    outcome_data = {
        "total_claims": 10,
        "accuracy": 0.444,
        "macro_f1": 0.40,
        "correct_claims": ["train:0", "train:1", "train:2", "train:3"],
        "wrong_claims": ["train:4", "train:5", "train:6", "train:7", "train:8"],
        "unparseable_claims": [],
    }
    safe_write_json(input_dir / "outcome_bank.json", outcome_data)

    # 3. Base predictions
    preds = []
    for i in range(10):
        sid = f"train:{i}"
        is_corr = i < 4
        preds.append({
            "sample_id": sid,
            "claim": f"Claim {i}",
            "gold_verdict": "Supported" if i % 2 == 0 else "Refuted",
            "predicted_verdict": "Supported" if is_corr else "Not Enough Evidence",
            "is_correct": is_corr,
        })
    with open(input_dir / "base_predictions.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")

    # 4. Candidates
    cands = []
    for i in range(4, 9):
        sid = f"train:{i}"
        cands.append({
            "sample_id": sid,
            "claim": f"Claim {i}",
            "gold_verdict": "Supported" if i % 2 == 0 else "Refuted",
            "base_verdict": "Not Enough Evidence",
            "draft_text": f"The displayed evidence establishes that the claim is Supported, contradicting the initial assessment of Refuted.",
            "mechanism": "evidence_sufficiency" if i != 6 else "numeric_calculation",
        })
    with open(input_dir / "candidate_corrections.jsonl", "w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps(c) + "\n")

    # 5. Judge decisions
    # train:4 ACCEPT, train:5 REJECT (no excerpts), train:6 REVISE, train:7 REJECT (error), train:8 ACCEPT
    judge_decisions = [
        {
            "sample_id": "train:4",
            "decision": "ACCEPT",
            "mechanism": "evidence_sufficiency",
            "grounding_quotes": ["Exact passage quote 4"],
            "reason": "Passed validation",
        },
        {
            "sample_id": "train:5",
            "decision": "REJECT",
            "mechanism": "evidence_sufficiency",
            "grounding_quotes": [],
            "reason": "The candidate corrective rationale claims the evidence supports the claim, but no evidence excerpts were provided to substantiate this.",
        },
        {
            "sample_id": "train:6",
            "decision": "REVISE",
            "mechanism": "numeric_calculation",
            "grounding_quotes": [],
            "reason": "Needs revision of calculations",
        },
        {
            "sample_id": "train:7",
            "decision": "REJECT",
            "mechanism": "evidence_sufficiency",
            "grounding_quotes": [],
            "reason": "Judge service unreachable on port 17171: connection refused",
        },
        {
            "sample_id": "train:8",
            "decision": "ACCEPT",
            "mechanism": "evidence_sufficiency",
            "grounding_quotes": ["Exact passage quote 8"],
            "reason": "Passed validation",
        },
    ]
    with open(input_dir / "judge_decisions.jsonl", "w", encoding="utf-8") as f:
        for j in judge_decisions:
            f.write(json.dumps(j) + "\n")

    # 6. Memory
    memory_data = {
        "records": [
            {
                "sample_id": "train:4",
                "claim": "Claim 4",
                "gold_verdict": "Supported",
                "base_verdict": "Not Enough Evidence",
                "correction": "Correction for 4",
                "mechanism": "evidence_sufficiency",
                "decision": "ACCEPT",
                "memory_status": "LLM_VALIDATED",
            },
            {
                "sample_id": "train:8",
                "claim": "Claim 8",
                "gold_verdict": "Supported",
                "base_verdict": "Not Enough Evidence",
                "correction": "Correction for 8",
                "mechanism": "evidence_sufficiency",
                "decision": "ACCEPT",
                "memory_status": "LLM_VALIDATED",
            },
        ]
    }
    safe_write_json(input_dir / "memory.json", memory_data)

    # 7. Receipts
    receipts = []
    for i in range(10):
        receipts.append({
            "sample_id": f"train:{i}",
            "evidence": [{"rendered_text": f"Exact passage quote {i}"}],
        })
    with open(input_dir / "retrieval_receipts.jsonl", "w", encoding="utf-8") as f:
        for r in receipts:
            f.write(json.dumps(r) + "\n")

    # 8. Raw datasets mock
    train_records = [{"claim": "" if i == 9 else f"Claim {i}", "label": "Supported" if i % 2 == 0 else "Refuted"} for i in range(10)]
    dev_records = [{"claim": f"Dev Claim {i}", "label": "Supported"} for i in range(5)]
    safe_write_json(input_dir / "train.json", train_records)
    safe_write_json(input_dir / "dev.json", dev_records)

    return input_dir


def test_mechanism_normalization():
    """Test mechanism normalization mapping."""
    assert normalize_mechanism("numeric_quantifier_error") == "numerical_or_quantitative"
    assert normalize_mechanism("temporal_date_shift") == "temporal_reasoning"
    assert normalize_mechanism("entity_swap") == "entity_disambiguation"
    assert normalize_mechanism("negation_flip") == "negation_or_polarity"
    assert normalize_mechanism("evidence_sufficiency") == "evidence_sufficiency"
    assert normalize_mechanism("completely_unknown_tag") == "other"
    assert normalize_mechanism(None) == "unknown"


def test_funnel_reconstruction_conservation(mock_m18e_artifacts, tmp_path):
    """Test that every train claim is accounted for in the funnel without double-counting."""
    out_dir = tmp_path / "audit_out"
    auditor = M18fAuditor(
        input_dir=mock_m18e_artifacts,
        output_dir=out_dir,
        train_path=mock_m18e_artifacts / "train.json",
        dev_path=mock_m18e_artifacts / "dev.json",
    )
    auditor.load_artifacts()
    ledger, funnel = auditor.reconstruct_funnel()

    # Total claims in ledger must equal total claims in train
    assert len(ledger) == 10

    # Terminal statuses sum to 10
    statuses = [r["terminal_status"] for r in ledger]
    assert len(statuses) == 10
    assert statuses.count("QUARANTINED_INVALID") == 1  # train:9 had empty text
    assert statuses.count("BASE_CORRECT") == 4         # train:0..3
    assert statuses.count("LLM_VALIDATED_SELECTABLE") == 2  # train:4, train:8
    assert statuses.count("JUDGE_REJECT") == 2         # train:5, train:7
    assert statuses.count("JUDGE_REVISE") == 1         # train:6


def test_judge_decision_taxonomy_and_recovery_tiers(mock_m18e_artifacts, tmp_path):
    """Test separation of technical errors from legitimate judge rejections and recovery tiering."""
    out_dir = tmp_path / "audit_out"
    auditor = M18fAuditor(
        input_dir=mock_m18e_artifacts,
        output_dir=out_dir,
        train_path=mock_m18e_artifacts / "train.json",
        dev_path=mock_m18e_artifacts / "dev.json",
    )
    auditor.load_artifacts()
    ledger, funnel = auditor.reconstruct_funnel()

    ledger_map = {r["sample_id"]: r for r in ledger}

    # train:7 was a connection refused error -> Tier A
    assert ledger_map["train:7"]["recoverability"] == TIER_A_DETERMINISTIC_BUG

    # train:5 was rejected due to generic template lacking excerpts -> Tier B
    assert ledger_map["train:5"]["recoverability"] == TIER_B_REQUIRES_RERUN

    # train:0 was BASE_CORRECT -> Tier D (not an error, legitimately rejected from correction bank)
    assert ledger_map["train:0"]["recoverability"] == TIER_D_LEGIT_REJECTED


def test_leakage_and_overlap_audit(mock_m18e_artifacts, tmp_path):
    """Test that exact train/dev overlaps are strictly caught if present."""
    out_dir = tmp_path / "audit_out"
    auditor = M18fAuditor(
        input_dir=mock_m18e_artifacts,
        output_dir=out_dir,
        train_path=mock_m18e_artifacts / "train.json",
        dev_path=mock_m18e_artifacts / "dev.json",
    )
    auditor.load_artifacts()
    leakage = auditor.audit_leakage()
    # No forbidden overlaps were in memory
    assert leakage["certified_safe"] is True
    assert len(leakage["forbidden_overlaps_in_memory"]) == 0


def test_full_audit_execution_and_deliverables_generation(mock_m18e_artifacts, tmp_path):
    """Test complete execution of M18fAuditor and generation of all 18 required deliverables."""
    out_dir = tmp_path / "audit_deliverables"
    auditor = M18fAuditor(
        input_dir=mock_m18e_artifacts,
        output_dir=out_dir,
        train_path=mock_m18e_artifacts / "train.json",
        dev_path=mock_m18e_artifacts / "dev.json",
    )
    summary = auditor.run_audit()

    assert summary["verdict"] == "MODIFY"
    assert (out_dir / "FUNNEL_REPORT.md").exists()
    assert (out_dir / "JUDGE_REJECTION_REPORT.md").exists()
    assert (out_dir / "EVIDENCE_FAILURE_REPORT.md").exists()
    assert (out_dir / "CLASS_COVERAGE_REPORT.md").exists()
    assert (out_dir / "MECHANISM_COVERAGE_REPORT.md").exists()
    assert (out_dir / "LEAKAGE_AUDIT.md").exists()
    assert (out_dir / "ROOT_CAUSE_REPORT.md").exists()
    assert (out_dir / "RECOVERY_POTENTIAL.md").exists()
    assert (out_dir / "MEMORY_INTEGRITY_REPORT.md").exists()
    assert (out_dir / "DECISION_REPORT.md").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "funnel_by_claim.csv").exists()
    assert (out_dir / "judge_decisions.csv").exists()
    assert (out_dir / "class_coverage.csv").exists()
    assert (out_dir / "mechanism_coverage.csv").exists()
    assert (out_dir / "recovery_candidates.csv").exists()
    assert (out_dir / "anomalies.jsonl").exists()
    assert (out_dir / "run_receipt.json").exists()
