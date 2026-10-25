"""Unit tests for M1.4.1 Mechanism Audit and Confirmatory Setup."""
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest

from tffv.verification.evaluate import compute_metrics, load_gold_labels
from tffv.verification.leakage import quarantine_near_duplicates
from tffv.verification.m13_audit import find_train_json
from tffv.verification.m141_confirmatory import (
    fast_macro_f1_from_indices,
    paired_bootstrap_macro_f1_ci,
    run_paired_comparison,
)
from tffv.verification.ontology import VERDICTS
from tffv.verification.stabilization import exact_mcnemar_test_twosided


def test_fast_macro_f1_matches_standard():
    """Verify vectorized macro-f1 matches standard per-class calculation."""
    classes = list(VERDICTS)
    gold = ["Supported", "Refuted", "Supported", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]
    preds = ["Supported", "Refuted", "Refuted", "Not Enough Evidence", "Supported"]

    class_to_idx = {c: i for i, c in enumerate(classes)}
    gold_idx = np.array([class_to_idx[g] for g in gold])
    pred_idx = np.array([class_to_idx[p] for p in preds])

    eval_rows = [
        {"sample_id": f"s:{i}", "gold_label": g, "predicted_verdict": p, "is_correct": bool(g == p)}
        for i, (g, p) in enumerate(zip(gold, preds))
    ]
    std_metrics = compute_metrics(eval_rows)
    fast_f1 = fast_macro_f1_from_indices(gold_idx, pred_idx, num_classes=4)

    assert abs(std_metrics["macro_f1"] - fast_f1) < 1e-6


def test_paired_bootstrap_macro_f1_ci():
    """Verify paired bootstrap CI produces finite bounds."""
    classes = list(VERDICTS)
    gold = ["Supported"] * 20 + ["Refuted"] * 20
    preds_b = ["Supported"] * 15 + ["Refuted"] * 10 + ["Supported"] * 15
    preds_t = ["Supported"] * 18 + ["Refuted"] * 15 + ["Supported"] * 7

    point_diff, low, high = paired_bootstrap_macro_f1_ci(
        gold_labels=gold,
        preds_base=preds_b,
        preds_target=preds_t,
        classes=classes,
        n_boot=500,
        seed=42,
    )
    assert low <= point_diff <= high
    assert high > low


def test_fold_c_quarantine():
    """Verify near-duplicate quarantine between Fold A and Fold C."""
    root = Path(__file__).resolve().parent.parent
    train_json_p = find_train_json(project_root=root)
    if not train_json_p or not train_json_p.is_file():
        pytest.skip(f"train.json not found (searched project root and server paths)")

    gold_map = load_gold_labels(train_json_p)
    source_claims = [gold_map[f"train:{i}"] for i in range(50)]
    cand_claims = [gold_map[f"train:{i}"] for i in range(100, 150)]

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        admitted, quarantined, manifest_p = quarantine_near_duplicates(
            source_claims=source_claims,
            candidate_claims=cand_claims,
            output_dir=tmp_dir,
            jaccard_threshold=0.60,
            ngram_threshold=0.50,
        )
        assert len(admitted) == 50
        assert len(quarantined) == 0
        assert manifest_p.is_file()
    finally:
        shutil.rmtree(tmp_dir)


def test_exact_mcnemar_properties():
    """Verify exact McNemar test symmetry and edge cases."""
    # When discordants are equal (e.g. 2 vs 2), p-value must be 1.0
    assert exact_mcnemar_test_twosided(2, 2) == 1.0
    # When no discordants, p-value is 1.0
    assert exact_mcnemar_test_twosided(0, 0) == 1.0
    # Highly asymmetric
    p_asym = exact_mcnemar_test_twosided(0, 10)
    assert p_asym < 0.01
