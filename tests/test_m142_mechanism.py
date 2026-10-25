"""Unit tests for M1.4.2 Mechanism Audit, Prompt Controls, and Fold D Confirmatory Runner."""
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest

from tffv.verification.evaluate import compute_metrics, load_gold_labels
from tffv.verification.leakage import quarantine_near_duplicates
from tffv.verification.m13_audit import find_train_json
from tffv.verification.m14_prompting import (
    ALL_CONDITIONS_M142,
    CONCISE_MATCHED_GENERIC_RULES,
    CONCISE_SOURCE_ERROR_RULES,
    CONCISE_ERROR_RULES_MINUS_R1,
    CONCISE_ERROR_RULES_MINUS_R2,
    CONCISE_SOURCE_ERROR_RULES_PARA1,
    CONCISE_SOURCE_ERROR_RULES_PARA2,
    build_prompt_for_condition_m14,
)
from tffv.verification.m142_runner import (
    fast_macro_f1_from_indices,
    holm_bonferroni,
    paired_bootstrap_macro_f1_ci,
    run_paired_comparison,
    main as m142_main,
)
from tffv.verification.ontology import VERDICTS
from tffv.retrieval.prepare_fold_evidence import verify_fold_evidence_contract


def test_d1_matched_length_and_structure_parity():
    """Verify D1_matched matches D2 in character budget and directive structure."""
    # Length parity within 5 characters
    len_d2 = len(CONCISE_SOURCE_ERROR_RULES)
    len_d1_m = len(CONCISE_MATCHED_GENERIC_RULES)
    assert abs(len_d2 - len_d1_m) <= 5, f"Length divergence: D2={len_d2}, D1_matched={len_d1_m}"

    # Both must contain exactly 2 numbered directives
    assert "1." in CONCISE_SOURCE_ERROR_RULES and "2." in CONCISE_SOURCE_ERROR_RULES
    assert "1." in CONCISE_MATCHED_GENERIC_RULES and "2." in CONCISE_MATCHED_GENERIC_RULES

    # Both must contain identical 1-sentence constraint
    sentence_constraint = "State a concise 1-sentence factual reason, then output Verdict: <Label>."
    assert sentence_constraint in CONCISE_SOURCE_ERROR_RULES
    assert sentence_constraint in CONCISE_MATCHED_GENERIC_RULES


def test_prompt_builders_all_m142_conditions():
    """Verify prompt builder dispatches correctly across all M1.4.2 conditions."""
    claim = "The Moon orbits the Earth."
    evidence = [{"passage_id": "p:1", "text": "The Moon is Earth's only natural satellite.", "source_url": "nasa.gov"}]

    for cond in ALL_CONDITIONS_M142:
        if cond in ("D3", "D4", "D5", "D6"):
            continue  # Requires memory bank, tested separately
        p_obj = build_prompt_for_condition_m14(condition=cond, claim=claim, evidence=evidence)
        assert p_obj["condition"] == cond
        assert claim in p_obj["user"]
        assert "p:1" in p_obj["user"] or "nasa.gov" in p_obj["user"]
        assert len(p_obj["messages"]) == 2
        assert len(p_obj["prompt_sha256"]) == 64


def test_holm_bonferroni_adjustment():
    """Verify Holm-Bonferroni correction preserves monotonicity and bounds."""
    p_vals = {
        "test_a": 0.01,
        "test_b": 0.04,
        "test_c": 0.05,
    }
    adj = holm_bonferroni(p_vals)
    # test_a: 0.01 * 3 = 0.03
    # test_b: max(0.03, 0.04 * 2) = 0.08
    # test_c: max(0.08, 0.05 * 1) = 0.08
    assert abs(adj["test_a"] - 0.03) < 1e-6
    assert abs(adj["test_b"] - 0.08) < 1e-6
    assert abs(adj["test_c"] - 0.08) < 1e-6

    # Verify capped at 1.0
    p_large = {"x": 0.8, "y": 0.9}
    adj_large = holm_bonferroni(p_large)
    assert adj_large["x"] == 1.0
    assert adj_large["y"] == 1.0


def test_fold_d_quarantine_integrity():
    """Verify Fold D (train:150..249) claims quarantine correctly against Fold A (train:0..49)."""
    root = Path(__file__).resolve().parent.parent
    train_json_p = find_train_json(project_root=root)
    if not train_json_p or not train_json_p.is_file():
        pytest.skip("train.json not found")

    gold_map = load_gold_labels(train_json_p)
    source_claims = [gold_map[f"train:{i}"] for i in range(50)]
    fold_d_claims = [gold_map[f"train:{i}"] for i in range(150, 250)]

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        admitted, quarantined, manifest_p = quarantine_near_duplicates(
            source_claims=source_claims,
            candidate_claims=fold_d_claims,
            output_dir=tmp_dir,
            jaccard_threshold=0.60,
            ngram_threshold=0.50,
        )
        assert len(admitted) == 91
        assert len(quarantined) == 9
        assert manifest_p.is_file()
    finally:
        shutil.rmtree(tmp_dir)


def test_m142_cli_runner_synthetic(tmp_path: Path, monkeypatch):
    """End-to-end synthetic CLI runner test for m142_runner.main."""
    root = Path(__file__).resolve().parent.parent
    train_json_p = find_train_json(project_root=root)
    if not train_json_p or not train_json_p.is_file():
        pytest.skip("train.json not found")

    gold_map = load_gold_labels(train_json_p)

    # 1. Prepare minimal claims and evidence for a 3-claim test slice
    test_sids = ["train:150", "train:151", "train:152"]
    claims_rows = [
        {"sample_id": sid, "claim": gold_map[sid]["claim"]}
        for sid in test_sids
    ]
    r4_rows = [
        {
            "sample_id": sid,
            "evidence": [
                {
                    "passage_id": f"p:{sid}:1",
                    "text": "Evidence text for verification testing.",
                    "source_url": "https://example.com/test",
                }
            ],
        }
        for sid in test_sids
    ]

    claims_p = tmp_path / "claims.jsonl"
    r4_p = tmp_path / "R4.jsonl"
    claims_p.write_text("\n".join(json.dumps(r) for r in claims_rows) + "\n", encoding="utf-8")
    r4_p.write_text("\n".join(json.dumps(r) for r in r4_rows) + "\n", encoding="utf-8")

    # 2. Source Fold A minimal B1 predictions and R4 evidence
    fold_a_sids = [f"train:{i}" for i in range(50)]
    b1_rows = [
        {"sample_id": sid, "claim": gold_map[sid]["claim"], "predicted_verdict": "Refuted"}
        for sid in fold_a_sids
    ]
    r4_source_rows = [
        {
            "sample_id": sid,
            "evidence": [{"passage_id": f"p:{sid}", "text": "Source evidence text.", "source_url": "https://source.com"}],
        }
        for sid in fold_a_sids
    ]

    b1_p = tmp_path / "b1_preds.jsonl"
    r4_source_p = tmp_path / "r4_source.jsonl"
    b1_p.write_text("\n".join(json.dumps(r) for r in b1_rows) + "\n", encoding="utf-8")
    r4_source_p.write_text("\n".join(json.dumps(r) for r in r4_source_rows) + "\n", encoding="utf-8")

    out_dir = tmp_path / "m142_eval_out"

    test_args = [
        "m142_runner",
        "--r4-evidence", str(r4_p),
        "--claims-jsonl", str(claims_p),
        "--train-json", str(train_json_p),
        "--b1-predictions", str(b1_p),
        "--r4-source-evidence", str(r4_source_p),
        "--output-dir", str(out_dir),
        "--conditions", "D0", "D1_matched", "D2",
        "--eval-start", "150",
        "--eval-end", "153",
        "--use-synthetic",
    ]

    monkeypatch.setattr("sys.argv", test_args)
    m142_main()

    assert (out_dir / "metrics_summary.json").is_file()
    assert (out_dir / "paired_comparisons.json").is_file()
    assert (out_dir / "M1_4_2_EVALUATION_REPORT.md").is_file()
    assert (out_dir / "D0" / "predictions.jsonl").is_file()
    assert (out_dir / "D1_matched" / "predictions.jsonl").is_file()
    assert (out_dir / "D2" / "predictions.jsonl").is_file()

    # Verify primary paired comparison output
    pairs_data = json.loads((out_dir / "paired_comparisons.json").read_text(encoding="utf-8"))
    assert "D2_vs_D1_matched" in pairs_data
    assert "D2_vs_D0" in pairs_data
