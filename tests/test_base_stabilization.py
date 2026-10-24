"""Unit tests for M1.2.1 Frozen Base Stabilization (B0, B1, B2)."""
import json
import pytest
from pathlib import Path

from tffv.verification.ontology import normalize_verdict
from tffv.verification.backbone import SyntheticBackbone
from tffv.verification.predict import run_prediction
from tffv.verification.evaluate import run_evaluation
from tffv.verification.stabilization import (
    analyze_paired_transitions,
    build_comparison_report,
    run_stabilization_comparison,
)


def test_post_think_verdict_extraction():
    # Case 1: Post-think text has clean verdict
    raw1 = "<think>\nLet's evaluate. The claim could be supported or refuted.\n</think>\n\nVerdict: Refuted."
    assert normalize_verdict(raw1) == "Refuted"

    # Case 2: Post-think text says 'the claim is supported'
    raw2 = "<think>\nThinking about evidence.\n</think>\n\nTherefore, the claim is Supported."
    assert normalize_verdict(raw2) == "Supported"

    # Case 3: Truncated inside thinking (unclosed think tag)
    raw3 = "<think>\nPassage 1 says something. Passage 2 says another thing. But neither is enough to prove"
    # Fallback to whole text: does it have clear verdict? No -> None
    assert normalize_verdict(raw3) is None

    # Case 4: Non-thinking output (direct text without <think>)
    raw4 = "Based on the evidence passages, the claim is contradicted.\n\nVerdict: Refuted"
    assert normalize_verdict(raw4) == "Refuted"


def test_synthetic_backbone_thinking_toggle():
    backbone = SyntheticBackbone()
    
    # Non-thinking
    res_b1 = backbone.generate([{"role": "user", "content": "claim 1"}], enable_thinking=False)
    assert "<think>" not in res_b1["generated_text"]
    assert "Verdict:" in res_b1["generated_text"]
    assert res_b1["enable_thinking"] is False

    # Thinking
    res_b2 = backbone.generate([{"role": "user", "content": "claim 1"}], max_new_tokens=1024, enable_thinking=True)
    assert "<think>" in res_b2["generated_text"]
    assert "</think>" in res_b2["generated_text"]
    assert res_b2["max_new_tokens"] == 1024
    assert res_b2["enable_thinking"] is True


def test_paired_transitions_logic():
    b0_eval = [
        {"sample_id": "1", "gold_label": "Supported", "predicted_verdict": "", "is_correct": False},
        {"sample_id": "2", "gold_label": "Refuted", "predicted_verdict": "Refuted", "is_correct": True},
        {"sample_id": "3", "gold_label": "Not Enough Evidence", "predicted_verdict": "Refuted", "is_correct": False},
        {"sample_id": "4", "gold_label": "Supported", "predicted_verdict": "Supported", "is_correct": True},
    ]

    target_eval = [
        {"sample_id": "1", "gold_label": "Supported", "predicted_verdict": "Supported", "is_correct": True},   # improved (unparsed -> correct)
        {"sample_id": "2", "gold_label": "Refuted", "predicted_verdict": "Supported", "is_correct": False},  # regressed (correct -> wrong)
        {"sample_id": "3", "gold_label": "Not Enough Evidence", "predicted_verdict": "Not Enough Evidence", "is_correct": True},  # improved (wrong -> correct)
        {"sample_id": "4", "gold_label": "Supported", "predicted_verdict": "Supported", "is_correct": True},   # unchanged correct
    ]

    res = analyze_paired_transitions(b0_eval, target_eval, "TestTarget")
    assert res["evaluated_claims"] == 4
    assert res["improved_count"] == 2
    assert "1" in res["improved_claims"]
    assert "3" in res["improved_claims"]
    assert res["regressed_count"] == 1
    assert "2" in res["regressed_claims"]
    assert res["net_gain"] == 1  # 2 - 1 = +1
    assert res["transition_counts"]["unparsed_to_correct"] == 1
    assert res["transition_counts"]["wrong_to_correct"] == 1
    assert res["transition_counts"]["correct_to_wrong"] == 1
    assert res["transition_counts"]["unchanged_correct"] == 1


def test_end_to_end_stabilization_synthetic_pipeline(tmp_path):
    # Setup test files
    claims_file = tmp_path / "claims.jsonl"
    claims_file.write_text(
        json.dumps({"sample_id": "train:0", "claim": "Water boils at 100C."}) + "\n" +
        json.dumps({"sample_id": "train:1", "claim": "The moon is green cheese."}) + "\n",
        encoding="utf-8"
    )

    r4_file = tmp_path / "R4.jsonl"
    r4_file.write_text(
        json.dumps({"sample_id": "train:0", "evidence": [{"passage_id": "p0", "source_url": "url0", "text": "Water boils at 100C."}]}) + "\n" +
        json.dumps({"sample_id": "train:1", "evidence": [{"passage_id": "p1", "source_url": "url1", "text": "Moon rocks are basalt."}]}) + "\n",
        encoding="utf-8"
    )

    train_json = tmp_path / "train.json"
    train_json.write_text(
        json.dumps([
            {"claim": "Water boils at 100C.", "label": "Supported"},
            {"claim": "The moon is green cheese.", "label": "Refuted"},
        ]),
        encoding="utf-8"
    )

    # 1. Simulate B0
    b0_pred = tmp_path / "b0_preds"
    b0_eval = tmp_path / "b0_eval"
    run_prediction(str(r4_file), str(claims_file), str(b0_pred), synthetic=True, max_new_tokens=512, enable_thinking=True, experiment_id="B0")
    run_evaluation(str(b0_pred), str(train_json), str(b0_eval))

    # 2. Simulate B1 (non-thinking)
    b1_pred = tmp_path / "b1_preds"
    b1_eval = tmp_path / "b1_eval"
    run_prediction(str(r4_file), str(claims_file), str(b1_pred), synthetic=True, max_new_tokens=512, enable_thinking=False, experiment_id="B1")
    run_evaluation(str(b1_pred), str(train_json), str(b1_eval))

    # 3. Simulate B2 (thinking 1024)
    b2_pred = tmp_path / "b2_preds"
    b2_eval = tmp_path / "b2_eval"
    run_prediction(str(r4_file), str(claims_file), str(b2_pred), synthetic=True, max_new_tokens=1024, enable_thinking=True, experiment_id="B2")
    run_evaluation(str(b2_pred), str(train_json), str(b2_eval))

    # 4. Run stabilization comparison
    comp_dir = tmp_path / "comparison"
    run_stabilization_comparison(
        b0_eval_dir=str(b0_eval),
        b1_eval_dir=str(b1_eval),
        b2_eval_dir=str(b2_eval),
        output_dir=str(comp_dir),
    )

    assert (comp_dir / "comparison_report.md").is_file()
    assert (comp_dir / "comparison_summary.json").is_file()
    
    summary = json.loads((comp_dir / "comparison_summary.json").read_text(encoding="utf-8"))
    assert "B0" in summary["summaries"]
    assert "B1" in summary["summaries"]
    assert "B2" in summary["summaries"]
    assert "B1" in summary["transitions"]
    assert "B2" in summary["transitions"]
