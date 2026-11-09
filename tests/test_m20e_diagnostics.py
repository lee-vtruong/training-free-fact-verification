"""Unit tests for Milestone M2.0e: Baseline Reproducibility & Selective Feasibility."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from tffv.experiments.m20e_mechanism_analysis import ContrastiveMechanismAnalyzer
from tffv.experiments.m20e_repro_audit import BaselineReproducibilityAuditor
from tffv.experiments.m20e_risk_features import RiskFeatureExtractor
from tffv.experiments.m20e_routing_replay import SelectiveRoutingEvaluator


def test_reproducibility_audit_and_prompt_divergence(tmp_path: Path):
    """Verifies that the reproducibility auditor correctly identifies prompt template divergence as root cause."""
    auditor = BaselineReproducibilityAuditor(output_dir=tmp_path)
    res = auditor.run_audit()

    assert res["status"] == "PASSED"
    assert res["determination"] == "VALID_PROTOCOL_DIFFERENCE"
    assert res["total_claims"] == 550
    assert res["gold_mismatches"] == 0
    assert res["prompt_sha_mismatches_b0"] == 550
    assert res["prompt_sha_mismatches_g2"] == 550

    diff_csv = tmp_path / "baseline_per_claim_diff.csv"
    assert diff_csv.exists()

    with open(diff_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1100  # 550 for B0 + 550 for G2
        for r in reader:
            if not r["verdict_match"]:
                assert r["mismatch_cause"] == "PROMPT_TEMPLATE_MISMATCH"


def test_mechanism_analysis_transition_conservation(tmp_path: Path):
    """Verifies conservation of corrected and harmed claim counts."""
    analyzer = ContrastiveMechanismAnalyzer(output_dir=tmp_path)
    res = analyzer.run_analysis()

    # In M2.0d, C2 vs B0 had exactly 45 corrected and 88 harmed
    assert res["benefit_count"] == 45
    assert res["harm_count"] == 88
    assert res["total_claims"] == 550

    trans_csv = tmp_path / "contrastive_transition_analysis.csv"
    pat_csv = tmp_path / "historical_pattern_analysis.csv"
    attr_md = tmp_path / "attribution_integrity_audit.md"

    assert trans_csv.exists()
    assert pat_csv.exists()
    assert attr_md.exists()

    with open(trans_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        corr_count = sum(1 for r in rows if r["c2_vs_b0_status"] == "CORRECTED")
        harm_count = sum(1 for r in rows if r["c2_vs_b0_status"] == "HARMED")
        assert corr_count == 45
        assert harm_count == 88


def test_risk_features_no_gold_leakage(tmp_path: Path):
    """Verifies that inference-time features exclude gold labels and maintain determinism."""
    extractor = RiskFeatureExtractor(output_dir=tmp_path)
    res = extractor.extract_features()

    feat_csv = tmp_path / "risk_features.csv"
    assert feat_csv.exists()

    with open(feat_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 550

        # Features available before gold must not include gold
        allowed_features = {
            "sample_id",
            "b0_predicted_label",
            "is_b0_ambiguous_pred",
            "b0_output_tokens",
            "b0_latency",
            "evidence_char_length",
            "evidence_passage_count",
            "selected_pattern_id",
            "pattern_selection_score",
            "salient_signal_count",
            "has_temporal_signal",
            "has_numerical_signal",
            "has_negation_signal",
            "has_conflict_signal",
            "composite_risk_score",
            # Target fields strictly marked with target_
            "target_b0_wrong",
            "target_c2_benefit",
            "target_c2_harm",
            "target_transition_class",
        }
        for k in rows[0].keys():
            assert k in allowed_features

    # Determinism
    extractor2 = RiskFeatureExtractor(output_dir=tmp_path)
    res2 = extractor2.extract_features()
    assert res["metrics"]["target_distribution"] == res2["metrics"]["target_distribution"]


def test_routing_replay_and_decision_logic(tmp_path: Path):
    """Verifies post-hoc routing replay calculations, bootstrap determinism, and decision report."""
    # First extract features
    extractor = RiskFeatureExtractor(output_dir=tmp_path)
    res_f = extractor.extract_features()

    evaluator = SelectiveRoutingEvaluator(
        risk_features_path=res_f["csv_path"],
        output_dir=tmp_path,
        n_bootstraps=100,
        seed=42,
    )
    res_r = evaluator.run_replay_evaluation()

    assert res_r["status"] == "EVALUATION_COMPLETE"
    assert res_r["decision"] == "INCONCLUSIVE"

    b_csv = tmp_path / "routing_budget_analysis.csv"
    r_csv = tmp_path / "routing_replay_results.csv"
    dec_md = tmp_path / "DECISION_REPORT.md"
    prop_md = tmp_path / "M20F_METHOD_PROPOSAL.md"

    assert b_csv.exists()
    assert r_csv.exists()
    assert dec_md.exists()
    assert prop_md.exists()

    # Check that Baseline Label Risk routing achieves Net Benefit of +21 claims
    with open(r_csv, "r", encoding="utf-8") as f:
        r_rows = list(csv.DictReader(f))
        lbl_row = [r for r in r_rows if "Baseline Label Risk" in r["policy"]][0]
        assert int(lbl_row["net_benefit"]) == 21
        assert float(lbl_row["accuracy"]) == 0.5200


def test_figure_generation(tmp_path: Path):
    """Verifies that all 5 publication-ready PNG figures are created."""
    extractor = RiskFeatureExtractor(output_dir=tmp_path)
    res_f = extractor.extract_features()

    evaluator = SelectiveRoutingEvaluator(
        risk_features_path=res_f["csv_path"],
        output_dir=tmp_path,
        n_bootstraps=50,
        seed=42,
    )
    evaluator.run_replay_evaluation()

    fig_dir = tmp_path / "figures"
    assert (fig_dir / "baseline_mismatch_by_cause.png").exists()
    assert (fig_dir / "corrected_harmed_transitions.png").exists()
    assert (fig_dir / "risk_precision_recall.png").exists()
    assert (fig_dir / "budget_vs_net_benefit.png").exists()
    assert (fig_dir / "per_class_routing_outcomes.png").exists()
