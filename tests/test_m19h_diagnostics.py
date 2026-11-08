"""Comprehensive Unit Test Suite for Milestone M1.9h Diagnostics Pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest

from tffv.experiments.m19h_orchestrator import M19hOrchestrator
from tffv.experiments.m19h_preflight import (
    PreflightIntegrityError,
    compute_metrics,
    run_preflight_audit,
)
from tffv.experiments.m19h_risk import run_risk_study
from tffv.experiments.m19h_routing import simulate_selective_routing
from tffv.experiments.m19h_schemas import (
    OFFICIAL_ARMS,
    OFFICIAL_LABELS,
    ClaimRecord,
)
from tffv.experiments.m19h_selector import analyze_selector
from tffv.experiments.m19h_transitions import analyze_transitions, exact_mcnemar


def test_exact_mcnemar_properties():
    # Symmetric case
    assert exact_mcnemar(10, 10) == 1.0
    # Zero total
    assert exact_mcnemar(0, 0) == 1.0
    # Strong asymmetry
    p = exact_mcnemar(0, 15)
    assert p < 0.001
    # Primacy of McNemar in M1.9g (20 vs 41)
    p_m19g = exact_mcnemar(41, 20)
    assert 0.009 < p_m19g < 0.010


def test_compute_metrics_correctness():
    y_true = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]
    y_pred = ["Supported", "Refuted", "Supported", "Conflicting Evidence/Cherrypicking"]
    # tp: Supported (1), Refuted (1), NEI (0), Conflicting (1)
    # accuracy = 3 / 4 = 0.75
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.75
    assert m["per_class"]["Supported"]["precision"] == 0.5  # 2 predicted, 1 true
    assert m["per_class"]["Supported"]["recall"] == 1.0  # 1 true, 1 predicted
    assert m["per_class"]["Not Enough Evidence"]["recall"] == 0.0
    assert 0.0 < m["macro_f1"] < 1.0


def test_preflight_missing_predictions_fails_closed(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    out_dir = tmp_path / "out"
    with pytest.raises(PreflightIntegrityError, match="Missing predictions file"):
        run_preflight_audit(empty_dir, out_dir)


def test_preflight_duplicate_prediction_fails_closed(tmp_path):
    in_dir = tmp_path / "dup_input"
    in_dir.mkdir()
    out_dir = tmp_path / "out"

    preds_file = in_dir / "predictions.jsonl"
    rows = []
    # Create duplicate (dev:0, B0)
    for _ in range(2):
        rows.append({
            "sample_id": "dev:0",
            "arm": "B0",
            "gold_verdict": "Supported",
            "predicted_verdict": "Supported",
        })
    preds_file.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    with pytest.raises(PreflightIntegrityError, match="Expected 3000 predictions"):
        run_preflight_audit(in_dir, out_dir)


def test_preflight_leakage_detection(tmp_path):
    in_dir = tmp_path / "leak_input"
    in_dir.mkdir()
    out_dir = tmp_path / "out"

    preds_file = in_dir / "predictions.jsonl"
    rows = []
    for i in range(500):
        for arm in OFFICIAL_ARMS:
            rows.append({
                "sample_id": f"dev:{i}",
                "arm": arm,
                "gold_verdict": "Supported",
                "predicted_verdict": "Supported",
                "exemplar_ids": ["dev:10"] if arm == "H2" else [],  # Illegal dev:10
            })
    preds_file.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    with pytest.raises(PreflightIntegrityError, match="Leakage contamination detected"):
        run_preflight_audit(in_dir, out_dir, skip_metric_reproduction=True)


def test_preflight_matching_invariance_violation(tmp_path):
    in_dir = tmp_path / "match_input"
    in_dir.mkdir()
    out_dir = tmp_path / "out"

    preds_file = in_dir / "predictions.jsonl"
    rows = []
    for i in range(500):
        for arm in OFFICIAL_ARMS:
            ex = ["train:100", "train:101"] if arm == "H2" else ["train:200", "train:201"] if arm == "B1" else []
            rows.append({
                "sample_id": f"dev:{i}",
                "arm": arm,
                "gold_verdict": "Supported",
                "predicted_verdict": "Supported",
                "exemplar_ids": ex,
            })
    preds_file.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    with pytest.raises(PreflightIntegrityError, match="Exemplar matching invariance failed"):
        run_preflight_audit(in_dir, out_dir, skip_metric_reproduction=True)


def create_synthetic_500_claims_map():
    """Generates synthetic 500-claim ClaimRecord dataset matching M1.9g metrics."""
    rng = np.random.RandomState(42)
    claims_map = {}
    labels_pool = ["Supported"] * 122 + ["Refuted"] * 305 + ["Not Enough Evidence"] * 35 + ["Conflicting Evidence/Cherrypicking"] * 38

    for i in range(500):
        sid = f"dev:{i}"
        gold = labels_pool[i]
        rec = ClaimRecord(
            claim_id=sid,
            claim_text=f"Claim {i}",
            gold_label=gold,
            evidence_count=8,
            evidence_char_len=int(rng.randint(2000, 6000)),
        )
        for arm in OFFICIAL_ARMS:
            # Synthetic predictions with varying error rates
            if arm == "B0":
                is_corr = rng.uniform() < 0.60
            elif arm == "H2":
                is_corr = rng.uniform() < 0.55
            elif arm == "B2":
                is_corr = rng.uniform() < 0.62
            else:
                is_corr = rng.uniform() < 0.59

            rec.is_correct[arm] = is_corr
            rec.predictions[arm] = gold if is_corr else rng.choice([l for l in OFFICIAL_LABELS if l != gold])

            # Synthetic similarities
            rec.chosen_exemplars[arm] = [f"train:{rng.randint(1, 1000)}", f"train:{rng.randint(1, 1000)}"]
            rec.similarity_scores[arm] = [float(rng.uniform(0.08, 0.25)), float(rng.uniform(0.05, 0.15))]

        claims_map[sid] = rec
    return claims_map


def test_transition_analysis_conservation(tmp_path):
    claims_map = create_synthetic_500_claims_map()
    out_dir = tmp_path / "trans_out"
    t_stats, t_mat, c_rows = analyze_transitions(claims_map, out_dir)

    assert (out_dir / "error_transitions.csv").exists()
    assert (out_dir / "transition_matrices.json").exists()
    assert (out_dir / "class_error_analysis.csv").exists()

    for pair_key, stats in t_stats.items():
        # Conservation: both_correct + both_wrong + harmed + corrected == total_claims
        total = stats.both_correct + stats.both_wrong + stats.harmed + stats.corrected
        assert total == 500
        assert stats.net_benefit == stats.corrected - stats.harmed
        # McNemar b and c matching
        assert stats.mcnemar_b == stats.corrected
        assert stats.mcnemar_c == stats.harmed


def test_h1_vs_b0_net_benefit_regression(tmp_path):
    """Specific regression test for H1 vs B0 transition direction and net benefit."""
    claims_map = {}
    for i in range(500):
        sid = f"dev:{i}"
        # Construct exact M1.9g counts:
        # B0 correct: 300 claims (0..299)
        # H1 correct: 304 claims
        # Both correct: 265 claims (0..264)
        # Harmed (B0=1, H1=0): 35 claims (265..299) -> 300 - 265 = 35
        # Corrected (B0=0, H1=1): 39 claims (300..338) -> 304 - 265 = 39
        # Both wrong: 161 claims (339..499) -> 500 - 265 - 35 - 39 = 161
        b0_corr = (i < 300)
        h1_corr = (i < 265) or (300 <= i <= 338)

        rec = ClaimRecord(claim_id=sid, claim_text="c", gold_label="Supported")
        rec.is_correct["B0"] = b0_corr
        rec.is_correct["H1"] = h1_corr
        rec.predictions["B0"] = "Supported" if b0_corr else "Refuted"
        rec.predictions["H1"] = "Supported" if h1_corr else "Refuted"
        for arm in ["B1", "H2", "H3", "B2"]:
            rec.is_correct[arm] = False
            rec.predictions[arm] = "Refuted"
        claims_map[sid] = rec

    out_dir = tmp_path / "h1_reg_out"
    t_stats, _, _ = analyze_transitions(claims_map, out_dir)
    h1_v_b0 = t_stats["H1_vs_B0"]

    assert h1_v_b0.both_correct == 265
    assert h1_v_b0.harmed == 35  # B0 correct -> H1 wrong
    assert h1_v_b0.corrected == 39  # B0 wrong -> H1 correct
    assert h1_v_b0.both_wrong == 161
    assert h1_v_b0.net_benefit == +4  # MUST BE +4, NOT -4!
    assert h1_v_b0.mcnemar_b == 39
    assert h1_v_b0.mcnemar_c == 35
    assert 0.72 < h1_v_b0.mcnemar_p_value < 0.73


def test_risk_study_no_gold_leakage(tmp_path):
    claims_map = create_synthetic_500_claims_map()
    out_dir = tmp_path / "risk_out"
    metrics_summary, feat_rows, curve_rows = run_risk_study(claims_map, out_dir, n_bootstraps=100, seed=42)

    assert (out_dir / "risk_features.csv").exists()
    assert (out_dir / "risk_metrics.json").exists()
    assert (out_dir / "risk_curves.csv").exists()

    # Verify that target gold is NOT in the feature fields
    for row in feat_rows:
        assert "gold_label" not in row
        assert "gold_verdict" not in row
        assert "is_correct" not in row

    # Check AUROC bounds
    for pred_name, p_data in metrics_summary["predictors"].items():
        assert 0.0 <= p_data["auroc"] <= 1.0
        assert 0.0 <= p_data["auprc"] <= 1.0


def test_selective_routing_bounds(tmp_path):
    claims_map = create_synthetic_500_claims_map()
    out_dir = tmp_path / "rout_out"
    _, feat_rows, _ = run_risk_study(claims_map, out_dir, n_bootstraps=50, seed=42)
    replay_rows = simulate_selective_routing(claims_map, feat_rows, out_dir)

    assert (out_dir / "routing_replay.csv").exists()
    for row in replay_rows:
        # Oracle bound must never be less than B0 base accuracy
        assert row["oracle_upper_bound_accuracy"] >= row["b0_base_accuracy"] - 1e-6
        # Precision must be in [0, 1]
        assert 0.0 <= row["error_detection_precision"] <= 1.0
        assert 0.0 <= row["error_detection_recall"] <= 1.0


def test_end_to_end_orchestrator_synthetic(tmp_path):
    # Set up synthetic input directory with mock M1.9g artifacts
    in_dir = tmp_path / "synth_m19g"
    in_dir.mkdir()
    out_dir = tmp_path / "synth_m19h"

    claims_map = create_synthetic_500_claims_map()

    # Write predictions.jsonl
    preds_file = in_dir / "predictions.jsonl"
    pred_rows = []
    sel_rows = []
    for sid, rec in claims_map.items():
        for arm in OFFICIAL_ARMS:
            # Ensure B1 == H2 and B2 == H3 matching
            if arm == "B1":
                rec.chosen_exemplars["B1"] = list(rec.chosen_exemplars["H2"])
            elif arm == "B2":
                rec.chosen_exemplars["B2"] = list(rec.chosen_exemplars["H3"])

            pred_rows.append({
                "sample_id": sid,
                "arm": arm,
                "claim": rec.claim_text,
                "gold_verdict": rec.gold_label,
                "predicted_verdict": rec.predictions[arm],
                "is_correct": rec.is_correct[arm],
                "is_parseable": True,
                "exemplar_ids": rec.chosen_exemplars[arm],
            })
            sel_rows.append({
                "sample_id": sid,
                "arm": arm,
                "chosen_exemplar_ids": rec.chosen_exemplars[arm],
                "similarity_scores": rec.similarity_scores[arm],
                "exemplar_outcomes": ["WRONG", "WRONG"] if arm in ["H2", "B1"] else ["CORRECT", "CORRECT"],
            })

    preds_file.write_text("\n".join(json.dumps(r) for r in pred_rows), encoding="utf-8")
    (in_dir / "selection_trace.jsonl").write_text("\n".join(json.dumps(r) for r in sel_rows), encoding="utf-8")

    orchestrator = M19hOrchestrator(
        input_dir=in_dir,
        output_dir=out_dir,
        n_bootstraps=50,
        skip_metric_reproduction=True,  # Synthetic data won't match reported M1.9g numbers
    )
    res = orchestrator.run_all()
    assert res["status"] == "GO"

    # Verify all deliverables exist
    required_files = [
        "PREFLIGHT_REPORT.md",
        "error_transitions.csv",
        "transition_matrices.json",
        "class_error_analysis.csv",
        "selector_diagnostics.csv",
        "risk_features.csv",
        "risk_metrics.json",
        "risk_curves.csv",
        "routing_replay.csv",
        "statistical_analysis.json",
        "ROOT_CAUSE_ANALYSIS.md",
        "M20_METHOD_PROPOSAL.md",
        "DECISION_REPORT.md",
        "run_receipt.json",
        "figures/fig1_error_transitions.png",
        "figures/fig2_per_class_f1_shift.png",
        "figures/fig3_similarity_vs_error_rate.png",
        "figures/fig4_risk_precision_recall_curves.png",
        "figures/fig5_selective_routing_tradeoffs.png",
    ]
    for rf in required_files:
        assert (out_dir / rf).exists(), f"Missing required file: {rf}"
