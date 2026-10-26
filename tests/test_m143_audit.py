"""Unit tests for M1.4.3 Cross-Fold Failure Audit and Data Consistency."""
import json
from pathlib import Path
import pytest

from tffv.verification.evaluate import compute_metrics, load_gold_labels
from tffv.verification.ontology import VERDICTS


def test_fold_d_evaluation_report_consistency():
    """Verify Fold D metrics and paired comparisons match canonical report values."""
    root = Path(__file__).resolve().parent.parent
    report_p = root / "outputs" / "m142-confirmatory-20261025-233407" / "metrics_summary.json"
    paired_p = root / "outputs" / "m142-confirmatory-20261025-233407" / "paired_comparisons.json"

    if not report_p.is_file() or not paired_p.is_file():
        pytest.skip("Fold D confirmatory artifacts not present locally.")

    metrics = json.loads(report_p.read_text(encoding="utf-8"))
    paired = json.loads(paired_p.read_text(encoding="utf-8"))

    # Verify D1_matched beat D2
    acc_d1 = metrics["D1_matched"]["accuracy"]
    acc_d2 = metrics["D2"]["accuracy"]
    assert acc_d1 > acc_d2, f"Expected D1_matched ({acc_d1}) > D2 ({acc_d2})"
    assert metrics["D1_matched"]["correct_count"] == 58
    assert metrics["D2"]["correct_count"] == 54

    # Verify primary contrast D2 vs D1_matched
    prim = paired["D2_vs_D1_matched"]
    assert prim["net_gain"] == -4
    cats = prim["categories"]
    assert cats["unchanged_correct"] == 54
    assert cats["correct_to_wrong"] == 4
    assert cats["wrong_to_correct"] == 0
    assert cats["unchanged_wrong"] == 33
    assert prim["total_claims"] == 91


def test_quarantined_claims_integrity():
    """Verify quarantine manifest correctly accounts for 9 quarantined and 91 admitted claims."""
    root = Path(__file__).resolve().parent.parent
    q_man_p = root / "outputs" / "m142-confirmatory-20261025-233407" / "quarantine_manifest.json"

    if not q_man_p.is_file():
        pytest.skip("Quarantine manifest not present locally.")

    q_data = json.loads(q_man_p.read_text(encoding="utf-8"))
    assert q_data["admitted_claims_count"] == 91
    assert q_data["quarantined_claims_count"] == 9
    assert len(q_data["flagged_leakage_pairs"]) == 9
    assert len(q_data["admitted_sample_ids"]) == 91
