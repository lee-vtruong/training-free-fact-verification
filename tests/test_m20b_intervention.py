"""Unit Tests: Milestone M2.0b — Baseline Audit & Instance-Grounded Error Intervention."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tffv.experiments.m20b_baseline_audit import BaselineAuditRunner
from tffv.experiments.m20b_development_split import (
    QUARANTINED_TRAIN_IDS,
    DevelopmentSplitManager,
)
from tffv.experiments.m20b_evaluator import (
    M20bEvaluator,
    bootstrap_metric_diff,
    compute_4class_metrics,
    mcnemar_exact,
)
from tffv.experiments.m20b_orchestrator import M20bOrchestrator
from tffv.experiments.m20b_prompts import CANONICAL_SYSTEM_INSTRUCTION, M20bPromptFormatter
from tffv.experiments.m20b_schemas import (
    EXPERIMENTAL_ARMS_M20B,
    InterventionDecision,
    M20bArm,
    M20bPredictionRecord,
    M20bTransitionStats,
)
from tffv.experiments.m20b_selector import (
    HistoricalErrorCase,
    InstanceGroundedSelector,
    tokenize,
)
from tffv.experiments.m20b_track_a_analysis import TrackAErrorAnalyzer


def test_baseline_audit(tmp_path: Path):
    """Test Phase A Baseline Audit runs and generates diff CSV and report."""
    m19g_dir = tmp_path / "m19g"
    m20a_dir = tmp_path / "m20a"
    out_dir = tmp_path / "m20b_out"
    m19g_dir.mkdir()
    m20a_dir.mkdir()

    # Create synthetic M1.9g and M2.0a predictions
    m19g_preds = [
        {"sample_id": "dev:0", "arm": "B0", "claim": "Claim 0", "gold_verdict": "Supported", "predicted_verdict": "Supported", "is_correct": True, "raw_completion": "Verdict: Supported"},
        {"sample_id": "dev:1", "arm": "B0", "claim": "Claim 1", "gold_verdict": "Refuted", "predicted_verdict": "Refuted", "is_correct": True, "raw_completion": "Verdict: Refuted"},
    ]
    m20a_preds = [
        {"sample_id": "dev:0", "arm": "B0", "claim": "Claim 0", "gold_verdict": "Supported", "predicted_verdict": "Conflicting Evidence/Cherrypicking", "is_correct": False, "raw_completion": "Verdict: Conflicting Evidence/Cherrypicking"},
        {"sample_id": "dev:1", "arm": "B0", "claim": "Claim 1", "gold_verdict": "Refuted", "predicted_verdict": "Refuted", "is_correct": True, "raw_completion": "Verdict: Refuted"},
    ]

    with open(m19g_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in m19g_preds:
            f.write(json.dumps(r) + "\n")
    with open(m20a_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in m20a_preds:
            f.write(json.dumps(r) + "\n")

    runner = BaselineAuditRunner(
        m19g_preds_path=m19g_dir / "predictions.jsonl",
        m20a_preds_path=m20a_dir / "predictions.jsonl",
        output_dir=out_dir,
    )
    res = runner.run_audit()

    assert res["status"] == "AUDIT_COMPLETE"
    assert res["canonical_contract_locked"] is True
    assert (out_dir / "baseline_per_claim_diff.csv").exists()
    assert (out_dir / "BASELINE_AUDIT.md").exists()

    report_text = (out_dir / "BASELINE_AUDIT.md").read_text(encoding="utf-8")
    assert "M1.9g B0" in report_text
    assert "M2.0a B0" in report_text
    assert "Prompt Phrasing Discrepancy" in report_text


def test_track_a_error_analyzer(tmp_path: Path):
    """Test Phase B Track A Error Analyzer identifies corrected and harmed claims."""
    m20a_dir = tmp_path / "m20a"
    out_dir = tmp_path / "m20b_out"
    m20a_dir.mkdir()

    preds = [
        {"sample_id": "dev:0", "arm": "B0", "claim": "C0", "gold_verdict": "Refuted", "predicted_verdict": "Supported", "is_correct": False, "raw_completion": "V: Supported"},
        {"sample_id": "dev:0", "arm": "E1", "claim": "C0", "gold_verdict": "Refuted", "predicted_verdict": "Refuted", "is_correct": True, "raw_completion": "V: Refuted"},
        {"sample_id": "dev:1", "arm": "B0", "claim": "C1", "gold_verdict": "Supported", "predicted_verdict": "Supported", "is_correct": True, "raw_completion": "V: Supported"},
        {"sample_id": "dev:1", "arm": "E1", "claim": "C1", "gold_verdict": "Supported", "predicted_verdict": "Conflicting Evidence/Cherrypicking", "is_correct": False, "raw_completion": "V: Conflicting"},
    ]
    with open(m20a_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in preds:
            f.write(json.dumps(r) + "\n")

    analyzer = TrackAErrorAnalyzer(
        m20a_preds_path=m20a_dir / "predictions.jsonl",
        output_dir=out_dir,
    )
    res = analyzer.analyze()

    assert res["corrected_count"] == 1
    assert res["harmed_count"] == 1
    assert res["net_benefit"] == 0
    assert (out_dir / "TRACK_A_ERROR_ANALYSIS.md").exists()
    assert (out_dir / "track_a_transitions.json").exists()


def test_development_split_isolation(tmp_path: Path):
    """Test Phase E Development Split Manager enforces zero dev exposure and quarantines."""
    out_dir = tmp_path / "splits"
    mgr = DevelopmentSplitManager(
        train_path=tmp_path / "dummy_train.json",
        output_dir=out_dir,
        dev_ratio=0.25,
        seed=42,
    )

    dummy_data = [
        {"id": "0", "claim": "Clean train claim 0", "label": "Supported"},
        {"id": "1", "claim": "Clean train claim 1", "label": "Refuted"},
        {"id": "1340", "claim": "Overlapping train claim", "label": "Refuted"},  # in QUARANTINED_TRAIN_IDS
        {"id": "dev:10", "claim": "Accidental dev claim", "label": "Supported"},  # dev prefix
    ]

    hist_source, dev_pilot, quarantined = mgr.partition_splits(dummy_data)

    assert len(quarantined) == 2
    quarantined_reasons = {q["reason"] for q in quarantined}
    assert "EXACT_TRAIN_DEV_OVERLAP" in quarantined_reasons
    assert "DEV_LABEL_LEAKAGE_GUARD" in quarantined_reasons

    assert len(hist_source) + len(dev_pilot) == 2

    # Save manifests and check
    dev_path, mem_path = mgr.save_manifests(hist_source, dev_pilot, quarantined)
    assert dev_path.exists()
    assert mem_path.exists()


def test_instance_grounded_selector():
    """Test selector relevance, grounding threshold filter, and fallback to NO_INTERVENTION."""
    cases = [
        HistoricalErrorCase(
            sample_id="train:100",
            claim="Apollo 11 landed on the Moon in July 1969.",
            gold_verdict="Supported",
            base_verdict="Refuted",
            evidence_text="NASA records confirm Apollo 11 lunar module touched down in July 1969.",
            error_lesson="Carefully check date and mission name.",
            generic_guidance="Standard verification of mission dates.",
        ),
        HistoricalErrorCase(
            sample_id="train:200",
            claim="The Eiffel Tower was built for the 1889 World Fair.",
            gold_verdict="Supported",
            base_verdict="Refuted",
            evidence_text="Gustave Eiffel designed the Eiffel Tower for the 1889 Exposition Universelle in Paris.",
            error_lesson="Verify architectural dates.",
            generic_guidance="Standard architectural verification.",
        ),
    ]

    selector = InstanceGroundedSelector(cases, min_grounding_threshold=0.10, min_shared_terms=2)

    # 1. Matching target: should INTERVENE
    dec, case, trace, audit = selector.select_intervention(
        sample_id="test:1",
        arm="B1",
        target_claim="Did Apollo 11 land on the moon in 1969?",
        target_evidence_text="Apollo 11 astronauts landed on the moon July 1969.",
    )
    assert dec == InterventionDecision.INTERVENE
    assert case is not None
    assert case.sample_id == "train:100"
    assert trace.intervention_decision == "INTERVENE"
    assert audit.evidence_grounding_verified is True

    # 2. Unrelated target: should NO_INTERVENTION
    dec_unrel, case_unrel, trace_unrel, audit_unrel = selector.select_intervention(
        sample_id="test:2",
        arm="B1",
        target_claim="Photosynthesis produces glucose and oxygen from sunlight.",
        target_evidence_text="Chloroplasts absorb sunlight to convert carbon dioxide into carbohydrates.",
    )
    assert dec_unrel == InterventionDecision.NO_INTERVENTION
    assert case_unrel is None
    assert trace_unrel.intervention_decision == "NO_INTERVENTION"
    assert audit_unrel.evidence_grounding_verified is False

    # 3. Arm B2 should select generic guidance
    dec_b2, case_b2, trace_b2, _ = selector.select_intervention(
        sample_id="test:1",
        arm="B2",
        target_claim="Did Apollo 11 land on the moon in 1969?",
        target_evidence_text="Apollo 11 astronauts landed on the moon July 1969.",
    )
    assert dec_b2 == InterventionDecision.INTERVENE
    assert trace_b2.selected_rationale_type == "GENERIC_CASE"


def test_prompt_formatter_invariance():
    """Test prompt formatter ensures canonical system instruction and evidence invariance."""
    formatter = M20bPromptFormatter()
    passages = [
        {"text": "First passage verifying that water freezes at zero degrees Celsius."},
        {"text": "Second passage on ice temperature states 32 Fahrenheit."},
    ]

    # Format across B0, B1 (with and without intervention), G2, E1
    b0_sys, b0_user, _ = formatter.format_prompt(
        arm="B0",
        claim_text="Water freezes at 0 C.",
        evidence_passages=passages,
    )
    assert CANONICAL_SYSTEM_INSTRUCTION in b0_sys
    assert "[1] First passage verifying" in b0_user

    # Case for intervention
    case = HistoricalErrorCase(
        sample_id="train:1",
        claim="Water boiling point",
        gold_verdict="Supported",
        base_verdict="Refuted",
        evidence_text="Evidence text",
        error_lesson="Check thermometer units.",
        generic_guidance="Verify temperature scale.",
    )

    b1_sys, b1_user, _ = formatter.format_prompt(
        arm="B1",
        claim_text="Water freezes at 0 C.",
        evidence_passages=passages,
        decision=InterventionDecision.INTERVENE,
        selected_case=case,
    )
    assert "[Verified Historical Error Diagnostic Lesson]" in b1_user
    assert "Check thermometer units." in b1_user

    # When NO_INTERVENTION, B1 prompt user block should NOT have the lesson block
    b1_no_sys, b1_no_user, _ = formatter.format_prompt(
        arm="B1",
        claim_text="Water freezes at 0 C.",
        evidence_passages=passages,
        decision=InterventionDecision.NO_INTERVENTION,
        selected_case=None,
    )
    assert "[Verified Historical Error Diagnostic Lesson]" not in b1_no_user
    assert b1_no_user == b0_user


def test_evaluator_conservation_and_mcnemar(tmp_path: Path):
    """Test evaluator conservation laws and statistical tests."""
    records = []
    # 4 claims, 2 arms (B1 and B2)
    # Claim 0: B1 correct, B2 wrong (Corrected by B1)
    records.append(M20bPredictionRecord("c:0", "B1", "Claim 0", "Supported", "Supported", "", True, True, 1.0, "", "INTERVENE"))
    records.append(M20bPredictionRecord("c:0", "B2", "Claim 0", "Supported", "Refuted", "", True, False, 1.0, "", "INTERVENE"))

    # Claim 1: B1 correct, B2 correct (Both correct)
    records.append(M20bPredictionRecord("c:1", "B1", "Claim 1", "Refuted", "Refuted", "", True, True, 1.0, "", "INTERVENE"))
    records.append(M20bPredictionRecord("c:1", "B2", "Claim 1", "Refuted", "Refuted", "", True, True, 1.0, "", "INTERVENE"))

    # Claim 2: B1 wrong, B2 correct (Harmed by B1)
    records.append(M20bPredictionRecord("c:2", "B1", "Claim 2", "Supported", "Refuted", "", True, False, 1.0, "", "NO_INTERVENTION"))
    records.append(M20bPredictionRecord("c:2", "B2", "Claim 2", "Supported", "Supported", "", True, True, 1.0, "", "NO_INTERVENTION"))

    # Claim 3: B1 wrong, B2 wrong (Both wrong)
    records.append(M20bPredictionRecord("c:3", "B1", "Claim 3", "Supported", "Refuted", "", True, False, 1.0, "", "INTERVENE"))
    records.append(M20bPredictionRecord("c:3", "B2", "Claim 3", "Supported", "Refuted", "", True, False, 1.0, "", "INTERVENE"))

    evaluator = M20bEvaluator(records, output_dir=tmp_path / "eval", n_bootstraps=100)
    res = evaluator.evaluate_all()

    assert "B1_vs_B2" in res["statistical_tests"]
    st = res["statistical_tests"]["B1_vs_B2"]["transitions"]
    assert st["both_correct"] == 1
    assert st["both_wrong"] == 1
    assert st["corrected"] == 1
    assert st["harmed"] == 1
    assert st["net_benefit"] == 0
    assert st["total_claims"] == 4


def test_orchestrator_smoke_pipeline(tmp_path: Path):
    """Test full orchestrator end-to-end execution in mock smoke mode."""
    out_dir = tmp_path / "m20b_orchestrator_test"

    orchestrator = M20bOrchestrator(
        output_dir=out_dir,
        m19g_dir=tmp_path / "dummy_m19g",
        m20a_dir=tmp_path / "dummy_m20a",
        train_path=tmp_path / "dummy_train.json",
    )

    dummy_target_claims = [
        {"sample_id": "test:0", "claim": "Target claim test zero.", "label": "Supported"},
        {"sample_id": "test:1", "claim": "Target claim test one.", "label": "Refuted"},
    ]
    dummy_hist_claims = [
        {"sample_id": "train:0", "claim": "Target claim test zero similar history.", "label": "Supported"},
        {"sample_id": "train:1", "claim": "Different historical claim.", "label": "Refuted"},
    ]
    dummy_receipts = {
        "test:0": [{"text": "Passage evidence verifying target claim test zero."}],
        "test:1": [{"text": "Passage evidence refuting target claim test one."}],
    }

    eval_res = orchestrator.run_inference_and_eval(
        target_claims=dummy_target_claims,
        retrieval_receipts=dummy_receipts,
        historical_claims=dummy_hist_claims,
        run_gpu=False,
        smoke_only=True,
    )

    assert (out_dir / "predictions.jsonl").exists()
    assert (out_dir / "selection_traces.jsonl").exists()
    assert (out_dir / "grounding_audit.json").exists()
    assert (out_dir / "run_receipt.json").exists()
    assert (out_dir / "DECISION_REPORT.md").exists()
    assert (out_dir / "metrics_by_arm.csv").exists()
