"""Unit tests for M1.4 Controlled Fact Verification Pilot (D0–D6)."""
import json
import pytest
from pathlib import Path
from typing import Any, Dict, List

from tffv.verification.error_memory import ErrorMemoryBank, build_error_memory_bank
from tffv.verification.m14_prompting import (
    ALL_CONDITIONS_M14,
    build_prompt_for_condition_m14,
    compute_claim_similarity,
    retrieve_top_k_similar_exemplars,
)
from tffv.verification.m14_runner import (
    evaluate_condition_m14,
    generate_m14_markdown_report,
    run_condition_inference_m14,
    run_paired_comparison_m14,
)
from tffv.verification.backbone import SyntheticBackbone
from tffv.verification.predict import dump_jsonl


def test_compute_claim_similarity():
    # Exact match
    assert compute_claim_similarity("The moon is made of cheese", "The moon is made of cheese") == 1.0

    # Completely disjoint
    assert compute_claim_similarity("Alpha beta gamma", "Xylophone zebra kayak") == 0.0

    # Partial overlap
    sim = compute_claim_similarity("The government raised taxes", "The government raised income taxes")
    assert 0.0 < sim < 1.0

    # Empty string edge cases
    assert compute_claim_similarity("", "Some text") == 0.0
    assert compute_claim_similarity("", "") == 0.0


def test_retrieve_top_k_similar_exemplars():
    candidates = [
        {"sample_id": "train:1", "claim": "Vaccines cause autism according to study"},
        {"sample_id": "train:2", "claim": "The economic inflation rate fell sharply in 2023"},
        {"sample_id": "train:3", "claim": "Childhood vaccines are linked to neurological disorders"},
    ]
    target = "Vaccines are dangerous and linked to autism"
    top2 = retrieve_top_k_similar_exemplars(target, candidates, k=2)

    assert len(top2) == 2
    # The two vaccine claims should be selected over the inflation claim
    sids = [r["sample_id"] for r in top2]
    assert "train:1" in sids
    assert "train:3" in sids
    assert "train:2" not in sids


@pytest.fixture
def mock_memory_bank(tmp_path: Path) -> ErrorMemoryBank:
    records = {
        "train:0": {
            "sample_id": "train:0",
            "claim": "Claim zero is completely true.",
            "gold_label": "Supported",
            "base_predicted_verdict": "Supported",
            "is_base_correct": True,
            "confusion_pair": "None",
            "compact_evidence": [{"text": "Passage confirms claim zero.", "source_url": "https://example.com/0"}],
        },
        "train:1": {
            "sample_id": "train:1",
            "claim": "Claim one is thoroughly refuted.",
            "gold_label": "Refuted",
            "base_predicted_verdict": "Not Enough Evidence",
            "is_base_correct": False,
            "confusion_pair": "Refuted->Not Enough Evidence",
            "compact_evidence": [{"text": "Passage refutes claim one.", "source_url": "https://example.com/1"}],
        },
        "train:2": {
            "sample_id": "train:2",
            "claim": "Claim two shows cherrypicked data.",
            "gold_label": "Conflicting Evidence/Cherrypicking",
            "base_predicted_verdict": "Supported",
            "is_base_correct": False,
            "confusion_pair": "Conflicting Evidence/Cherrypicking->Supported",
            "compact_evidence": [{"text": "Passage shows contradiction.", "source_url": "https://example.com/2"}],
        },
    }
    return ErrorMemoryBank(
        memory_data={
            "records": records,
            "by_class": {"Supported": ["train:0"], "Refuted": ["train:1"], "Conflicting Evidence/Cherrypicking": ["train:2"]},
            "by_confusion": {"None": ["train:0"], "Refuted->Not Enough Evidence": ["train:1"], "Conflicting Evidence/Cherrypicking->Supported": ["train:2"]},
            "correct_sids": ["train:0"],
            "wrong_sids": ["train:1", "train:2"],
            "source_split": "train:0..2",
            "manifest": {"total_source_records": 3},
        }
    )


def test_m14_prompt_builders_all_conditions(mock_memory_bank: ErrorMemoryBank):
    claim = "Target claim for testing M1.4 verification."
    evidence = [{"text": "Evidence for target claim.", "source_url": "https://example.com/target"}]

    for cond in ALL_CONDITIONS_M14:
        prompt_data = build_prompt_for_condition_m14(
            condition=cond,
            claim=claim,
            evidence=evidence,
            memory_bank=mock_memory_bank,
            seed=42,
        )
        assert "messages" in prompt_data
        assert "prompt_sha256" in prompt_data
        assert len(prompt_data["messages"]) == 2

        user_content = prompt_data["user"]
        assert claim in user_content
        assert "https://example.com/target" in user_content

        if cond in ["D3", "D4", "D5", "D6"]:
            assert len(prompt_data["exemplar_sample_ids"]) > 0
            # Ensure none of the exemplars match the target
            assert "target" not in prompt_data["exemplar_sample_ids"]


def test_m14_synthetic_end_to_end_runner(tmp_path: Path, mock_memory_bank: ErrorMemoryBank):
    # 1. Setup mock evaluation data
    eval_sids = ["train:50", "train:51", "train:52", "train:53"]
    claims_map = {
        sid: {"sample_id": sid, "claim": f"Evaluation claim text {sid}"}
        for sid in eval_sids
    }
    evidence_map = {
        sid: [{"text": f"Evidence text for {sid}", "source_url": f"https://example.com/{sid}"}]
        for sid in eval_sids
    }

    train_json_data = [{"claim_id": i, "label": "Supported", "claim": f"Evaluation claim text train:{i}"} for i in range(55)]
    train_json_data[50]["label"] = "Supported"
    train_json_data[51]["label"] = "Refuted"
    train_json_data[52]["label"] = "Not Enough Evidence"
    train_json_data[53]["label"] = "Conflicting Evidence/Cherrypicking"
    train_json_p = tmp_path / "train.json"
    train_json_p.write_text(json.dumps(train_json_data), encoding="utf-8")

    backbone = SyntheticBackbone(seed=42)

    # 2. Run inference across D0 and D1
    out_dir = tmp_path / "m14_run"
    summaries = {}
    for cond in ["D0", "D1", "D4"]:
        preds_p = run_condition_inference_m14(
            condition=cond,
            target_sids=eval_sids,
            evidence_map=evidence_map,
            claims_map=claims_map,
            output_dir=out_dir,
            backbone=backbone,
            memory_bank=mock_memory_bank,
        )
        assert preds_p.is_file()

        eval_summary = evaluate_condition_m14(
            condition=cond,
            predictions_dir=out_dir,
            train_json_path=train_json_p,
            output_dir=out_dir,
        )
        summaries[cond] = {"metrics": eval_summary, "timing": {"mean_latency_seconds": 0.1, "mean_prompt_tokens": 500, "mean_generation_tokens": 50}}

    # 3. Paired comparison
    paired = {}
    for cond in ["D1", "D4"]:
        paired[cond] = run_paired_comparison_m14(
            baseline_condition="D0",
            target_condition=cond,
            predictions_dir=out_dir,
            train_json_path=train_json_p,
        )
        assert paired[cond]["total_claims"] == 4
        assert sum(paired[cond]["categories"].values()) == 4
        assert paired[cond]["net_gain"] == paired[cond]["delta_correct"]

    # 4. Report generation
    report_p = generate_m14_markdown_report(
        condition_summaries=summaries,
        paired_comparisons=paired,
        output_dir=out_dir,
        partition_manifest={"total_source_claims": 3, "admitted_claims_count": 4, "quarantined_ids": []},
    )
    assert report_p.is_file()
    content = report_p.read_text(encoding="utf-8")
    assert "M1.4 — Controlled Fact Verification Pilot Report" in content
    assert "D0" in content
    assert "D1" in content


def test_m14_cli_runner_synthetic(monkeypatch, tmp_path: Path):
    from tffv.verification.m14_runner import main

    claims_rows = [
        {"sample_id": f"train:{i}", "claim": f"Claim statement {i}"}
        for i in range(55)
    ]
    claims_p = tmp_path / "claims.jsonl"
    dump_jsonl(claims_p, claims_rows)

    evidence_rows = [
        {"sample_id": f"train:{i}", "evidence": [{"text": f"Evidence {i}", "source_url": f"https://example.com/{i}"}]}
        for i in range(55)
    ]
    evidence_p = tmp_path / "R4.jsonl"
    dump_jsonl(evidence_p, evidence_rows)

    train_json_data = [
        {"claim_id": i, "label": "Supported", "claim": f"Claim statement {i}"}
        for i in range(55)
    ]
    train_json_p = tmp_path / "train.json"
    train_json_p.write_text(json.dumps(train_json_data), encoding="utf-8")

    b1_preds_rows = [
        {"sample_id": f"train:{i}", "predicted_verdict": "Supported" if i % 2 == 0 else "Refuted", "raw_generation": "Verdict: Supported"}
        for i in range(50)
    ]
    b1_preds_p = tmp_path / "b1_preds.jsonl"
    dump_jsonl(b1_preds_p, b1_preds_rows)

    out_dir = tmp_path / "cli_out"
    test_args = [
        "m14_runner.py",
        "--claims", str(claims_p),
        "--evidence", str(evidence_p),
        "--train-json", str(train_json_p),
        "--b1-predictions", str(b1_preds_p),
        "--output", str(out_dir),
        "--synthetic",
        "--conditions", "D0", "D1", "D4",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    main()

    assert (out_dir / "partition_manifest.json").is_file()
    assert (out_dir / "D0" / "predictions.jsonl").is_file()
    assert (out_dir / "D1" / "predictions.jsonl").is_file()
    assert (out_dir / "M1_4_PILOT_REPORT.md").is_file()

