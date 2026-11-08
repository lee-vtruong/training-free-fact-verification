"""Unit and integration tests for Milestone M1.9e Integrity Audit."""
import json
from pathlib import Path
import pytest

from tffv.experiments.m19e_audit import M19eAuditor, exact_mcnemar


def test_exact_mcnemar():
    assert exact_mcnemar(0, 0) == 1.0
    assert exact_mcnemar(5, 5) == 1.0
    p_asym = exact_mcnemar(10, 0)
    assert p_asym < 0.01


def test_m19e_auditor_end_to_end(tmp_path):
    # Setup minimal synthetic M1.9 directory
    m19_mock = tmp_path / "m19_mock"
    m19_mock.mkdir()
    out_dir = tmp_path / "m19e_out"

    # Evaluation manifest (5 synthetic claims)
    eval_claims = [
        {"sample_id": f"train:{i}", "gold_verdict": "Supported", "evidence": [{"rendered_text": ""}]}
        for i in range(5)
    ]
    (m19_mock / "official_dev_manifest.json").write_text(json.dumps(eval_claims), encoding="utf-8")

    # Predictions
    preds = []
    for arm in ["B0", "B1", "H1", "H2", "H3", "B2"]:
        for i in range(5):
            preds.append({
                "arm": arm,
                "sample_id": f"train:{i}",
                "gold_verdict": "Supported",
                "predicted_verdict": "Not Enough Evidence",
                "is_correct": False,
                "raw_completion": "Explanation: The retrieved evidence passages are empty... Verdict: Not Enough Evidence",
                "exemplar_ids": ["train:10", "train:11"] if arm != "B0" else [],
            })
    with open(m19_mock / "predictions.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")

    auditor = M19eAuditor(
        m19_dir=m19_mock,
        output_dir=out_dir,
    )
    res = auditor.run_full_audit()

    assert res["verdict"] == "PROMPT_OR_EVIDENCE_REGRESSION"
    assert (out_dir / "DATASET_PROVENANCE.md").exists()
    assert (out_dir / "LABEL_MAPPING_AUDIT.md").exists()
    assert (out_dir / "BASELINE_REGRESSION.md").exists()
    assert (out_dir / "RECOMPUTED_METRICS.csv").exists()
    assert (out_dir / "CONFUSION_MATRICES.csv").exists()
    assert (out_dir / "STATISTICAL_RECHECK.json").exists()
    assert (out_dir / "ROOT_CAUSE_REPORT.md").exists()
    assert (out_dir / "DECISION_REPORT.md").exists()
    assert (out_dir / "run_receipt.json").exists()

    # Check evidence blanking was detected as 100%
    assert res["regression"]["evidence_blanking_bug_confirmed"] is True
