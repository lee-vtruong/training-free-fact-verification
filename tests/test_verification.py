"""Unit tests for M1.2 verification pipeline, evaluation, and error profiling."""
import json
import pytest
from pathlib import Path

from tffv.verification.ontology import VERDICTS, normalize_verdict
from tffv.verification.prompting import build_verification_prompt, format_evidence_block
from tffv.verification.backbone import SyntheticBackbone
from tffv.verification.predict import run_prediction
from tffv.verification.evaluate import compute_metrics, run_evaluation
from tffv.verification.error_profiling import run_error_profiling
from tffv.verification.group_safety import verify_fold_isolation, create_pilot_group_safety_plan


def test_ontology_verdict_parsing():
    assert normalize_verdict("Verdict: Supported") == "Supported"
    assert normalize_verdict("Verdict: Refuted.") == "Refuted"
    assert normalize_verdict("Final Verdict: **Not Enough Evidence**") == "Not Enough Evidence"
    assert normalize_verdict("The claim is Conflicting Evidence/Cherrypicking") == "Conflicting Evidence/Cherrypicking"
    assert normalize_verdict("Analysis: blah\n\nVerdict: refuted\n") == "Refuted"
    assert normalize_verdict("Random gibberish that matches nothing") is None
    assert normalize_verdict("") is None


def test_prompting_structure_and_neutrality():
    claim = "The Eiffel Tower is in Paris."
    evidence = [
        {"text": "The Eiffel Tower is a landmark in Paris, France.", "source_url": "https://example.com/paris"},
        {"text": "It was completed in 1889.", "source_url": "https://example.com/history"},
    ]
    prompt = build_verification_prompt(claim, evidence)
    assert "The Eiffel Tower is in Paris." in prompt["user"]
    assert "https://example.com/paris" in prompt["user"]
    assert "Supported" in prompt["user"]
    assert "Refuted" in prompt["user"]
    assert "Not Enough Evidence" in prompt["user"]
    assert "Conflicting Evidence/Cherrypicking" in prompt["user"]
    assert len(prompt["messages"]) == 2
    assert prompt["prompt_sha256"] is not None


def test_synthetic_backbone_deterministic():
    backbone = SyntheticBackbone()
    res = backbone.generate([{"role": "user", "content": "Test claim"}])
    assert "generated_text" in res
    assert "Verdict:" in res["generated_text"]
    assert backbone.forward_passes == 1


def test_compute_metrics_accuracy_and_macro_f1():
    eval_rows = [
        {"sample_id": "1", "gold_label": "Supported", "predicted_verdict": "Supported", "is_correct": True},
        {"sample_id": "2", "gold_label": "Refuted", "predicted_verdict": "Refuted", "is_correct": True},
        {"sample_id": "3", "gold_label": "Not Enough Evidence", "predicted_verdict": "Refuted", "is_correct": False},
        {"sample_id": "4", "gold_label": "Conflicting Evidence/Cherrypicking", "predicted_verdict": "Supported", "is_correct": False},
    ]
    metrics = compute_metrics(eval_rows)
    assert metrics["total_claims"] == 4
    assert metrics["correct_claims"] == 2
    assert metrics["accuracy"] == 0.5
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    cm = metrics["confusion_matrix"]
    assert cm["Supported"]["Supported"] == 1
    assert cm["Refuted"]["Refuted"] == 1
    assert cm["Not Enough Evidence"]["Refuted"] == 1
    assert cm["Conflicting Evidence/Cherrypicking"]["Supported"] == 1


def test_group_safety_and_leakage_detection(tmp_path):
    eval_ids = ["train:0", "train:1", "train:2"]
    safe_mem_ids = ["train:100", "train:101"]
    res = verify_fold_isolation(eval_ids, safe_mem_ids)
    assert res["isolation_verified"] is True

    leaky_mem_ids = ["train:2", "train:100"]
    with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
        verify_fold_isolation(eval_ids, leaky_mem_ids)

    plan_path = tmp_path / "fold_safety.json"
    plan = create_pilot_group_safety_plan(eval_ids, plan_path)
    assert plan["pilot_sample_count"] == 3
    assert plan["blind_test_permitted"] is False
    assert plan_path.is_file()


def test_end_to_end_synthetic_pipeline(tmp_path):
    # 1. Create dummy claims.jsonl
    claims_file = tmp_path / "claims.jsonl"
    claims_file.write_text(
        json.dumps({"sample_id": "train:0", "claim": "Water boils at 100C.", "split": "train"}) + "\n" +
        json.dumps({"sample_id": "train:1", "claim": "The moon is made of green cheese.", "split": "train"}) + "\n",
        encoding="utf-8"
    )

    # 2. Create dummy R4.jsonl
    r4_file = tmp_path / "R4.jsonl"
    r4_file.write_text(
        json.dumps({"sample_id": "train:0", "evidence": [{"passage_id": "p0", "source_url": "https://science.org", "text": "Water boils at 100C."}]}) + "\n" +
        json.dumps({"sample_id": "train:1", "evidence": [{"passage_id": "p1", "source_url": "https://nasa.gov", "text": "Moon rocks are basaltic."}]}) + "\n",
        encoding="utf-8"
    )

    # 3. Create dummy train.json
    train_json = tmp_path / "train.json"
    train_json.write_text(
        json.dumps([
            {"claim": "Water boils at 100C.", "label": "Supported"},
            {"claim": "The moon is made of green cheese.", "label": "Refuted"},
        ]),
        encoding="utf-8"
    )

    # 4. Run predict (synthetic)
    pred_dir = tmp_path / "preds"
    run_prediction(str(r4_file), str(claims_file), str(pred_dir), synthetic=True)
    assert (pred_dir / "predictions.jsonl").is_file()
    manifest = json.loads((pred_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["gold_labels_used"] is False

    # 5. Run evaluate
    eval_dir = tmp_path / "eval"
    run_evaluation(str(pred_dir), str(train_json), str(eval_dir))
    assert (eval_dir / "evaluation_summary.json").is_file()
    assert (eval_dir / "evaluation_report.md").is_file()
    assert (eval_dir / "per_claim_evaluation.csv").is_file()

    # 6. Run error profiling
    err_dir = tmp_path / "errors"
    run_error_profiling(str(eval_dir), str(pred_dir / "predictions.jsonl"), output_dir=str(err_dir))
    assert (err_dir / "error_analysis_summary.json").is_file()
    assert (err_dir / "error_profiling_report.md").is_file()
