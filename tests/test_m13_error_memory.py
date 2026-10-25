"""Unit and integration tests for M1.3 Error-Informed Fact Verification."""
import json
from pathlib import Path
import pytest

from tffv.verification.ontology import VERDICTS
from tffv.verification.leakage import (
    assert_fold_disjointness,
    compute_jaccard_similarity,
    compute_ngram_containment,
    find_near_duplicates,
    quarantine_near_duplicates,
)
from tffv.verification.error_memory import (
    ErrorMemoryBank,
    build_error_memory_bank,
)
from tffv.verification.m13_prompting import (
    build_c0_prompt,
    build_c1_prompt,
    build_c2_prompt,
    build_c3_prompt,
    build_c4_prompt,
    build_c5_prompt,
    build_c6_prompt,
    build_prompt_for_condition,
)
from tffv.verification.m13_runner import (
    compare_conditions_against_baseline,
    evaluate_condition,
    run_condition_inference,
)
from tffv.verification.backbone import SyntheticBackbone


@pytest.fixture
def dummy_train_json(tmp_path: Path) -> Path:
    p = tmp_path / "train.json"
    data = [
        {
            "claim": f"Claim {i} about scientific topic",
            "label": VERDICTS[i % len(VERDICTS)],
            "justification": f"Factual justification for claim {i}.",
            "speaker": "Scientist",
            "claim_date": "2024-01-01",
        }
        for i in range(16)
    ]
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


@pytest.fixture
def dummy_r4_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "R4.jsonl"
    rows = [
        {
            "sample_id": f"train:{i}",
            "evidence": [
                {
                    "passage_id": f"p_{i}_1",
                    "source_url": f"https://example.com/article_{i}",
                    "text": f"This is official verified evidence text confirming or denying claim {i}.",
                }
            ],
        }
        for i in range(16)
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


@pytest.fixture
def dummy_b1_predictions(tmp_path: Path) -> Path:
    p = tmp_path / "b1_predictions.jsonl"
    rows = []
    for i in range(16):
        gold = VERDICTS[i % len(VERDICTS)]
        # First 8 are correct (2 per class), next 8 are wrong (2 per class)
        pred = gold if i < 8 else VERDICTS[(i + 1) % len(VERDICTS)]
        rows.append({
            "sample_id": f"train:{i}",
            "predicted_verdict": pred,
            "raw_generation": f"Analysis... Verdict: {pred}",
        })
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_leakage_similarity_and_quarantine(tmp_path: Path):
    text1 = "Warm water cures severe migraines completely."
    text2 = "Warm water is effective in curing severe migraines completely."
    text3 = "Unrelated claim about economic policy and inflation rates."

    jacc = compute_jaccard_similarity(text1, text2)
    assert jacc >= 0.5
    assert compute_jaccard_similarity(text1, text3) < 0.1

    ngram = compute_ngram_containment(text1, text2, n=3)
    assert ngram >= 0.2

    source_claims = [{"sample_id": "train:1", "claim": text1}]
    candidate_claims = [
        {"sample_id": "train:101", "claim": text2},
        {"sample_id": "train:102", "claim": text3},
    ]

    admitted, quarantined, manifest_path = quarantine_near_duplicates(
        source_claims, candidate_claims, output_dir=tmp_path, jaccard_threshold=0.5
    )
    assert len(admitted) == 1
    assert admitted[0]["sample_id"] == "train:102"
    assert len(quarantined) == 1
    assert quarantined[0]["sample_id"] == "train:101"
    assert manifest_path.is_file()


def test_assert_fold_disjointness():
    with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
        assert_fold_disjointness({"train:1", "train:2"}, {"train:2", "train:3"})

    # Should not raise when disjoint
    assert_fold_disjointness({"train:1", "train:2"}, {"train:3", "train:4"})


def test_error_memory_bank_construction_and_sampling(
    dummy_train_json: Path,
    dummy_b1_predictions: Path,
    dummy_r4_jsonl: Path,
    tmp_path: Path,
):
    mem_path = tmp_path / "memory_bank.json"
    bank = build_error_memory_bank(
        train_json_path=dummy_train_json,
        b1_predictions_path=dummy_b1_predictions,
        r4_evidence_path=dummy_r4_jsonl,
        output_path=mem_path,
    )
    assert mem_path.is_file()
    assert len(bank.records) == 16
    assert len(bank.correct_sids) == 8
    assert len(bank.wrong_sids) == 8

    # Test sampling
    c2_ex = bank.sample_random_source_exemplars(k=4, seed=42)
    assert len(c2_ex) == 4
    labels = [r["gold_label"] for r in c2_ex]
    assert len(set(labels)) == 4

    c3_ex = bank.sample_correct_only_exemplars(k=4, seed=42)
    assert len(c3_ex) == 4
    for r in c3_ex:
        assert r["is_correct"] is True

    c4_ex = bank.sample_error_only_exemplars(k=4, seed=42)
    assert len(c4_ex) == 4
    for r in c4_ex:
        assert r["is_correct"] is False
        assert r["confusion_pair"] is not None

    rules = bank.get_source_derived_rules()
    assert "Verification Checklist" in rules
    assert "Media Debate" in rules


def test_prompt_builders_c0_to_c6(
    dummy_train_json: Path,
    dummy_b1_predictions: Path,
    dummy_r4_jsonl: Path,
):
    bank = build_error_memory_bank(
        train_json_path=dummy_train_json,
        b1_predictions_path=dummy_b1_predictions,
        r4_evidence_path=dummy_r4_jsonl,
    )
    claim = "Target test claim to verify"
    evidence = [{"source_url": "https://gov.test", "text": "Passage text"}]

    for cond in ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]:
        prompt_data = build_prompt_for_condition(
            condition=cond,
            claim=claim,
            evidence=evidence,
            memory_bank=bank,
        )
        assert "messages" in prompt_data
        assert "prompt_sha256" in prompt_data
        assert len(prompt_data["messages"]) == 2
        user_content = prompt_data["messages"][1]["content"]
        assert claim in user_content
        assert "Verdict:" in user_content


def test_end_to_end_synthetic_runner(
    dummy_train_json: Path,
    dummy_b1_predictions: Path,
    dummy_r4_jsonl: Path,
    tmp_path: Path,
):
    source_sids = {f"train:{i}" for i in range(8)}
    target_sids = [f"train:{i}" for i in range(8, 16)]

    bank = build_error_memory_bank(
        train_json_path=dummy_train_json,
        b1_predictions_path=dummy_b1_predictions,
        r4_evidence_path=dummy_r4_jsonl,
        allowed_sample_ids=source_sids,
    )
    backbone = SyntheticBackbone()

    claims_map = {
        f"train:{i}": {"sample_id": f"train:{i}", "claim": f"Claim {i}"}
        for i in range(16)
    }
    evidence_map = {
        f"train:{i}": [{"source_url": "url", "text": f"Evidence for {i}"}]
        for i in range(16)
    }

    # Run C0
    c0_pred = run_condition_inference(
        condition="C0",
        target_sids=target_sids,
        evidence_map=evidence_map,
        claims_map=claims_map,
        output_dir=tmp_path,
        backbone=backbone,
        memory_bank=bank,
    )
    assert c0_pred.is_file()

    c0_eval = evaluate_condition(
        condition="C0",
        predictions_path=c0_pred,
        train_json_path=dummy_train_json,
        output_dir=tmp_path,
    )
    assert "metrics" in c0_eval

    # Run C4
    c4_pred = run_condition_inference(
        condition="C4",
        target_sids=target_sids,
        evidence_map=evidence_map,
        claims_map=claims_map,
        output_dir=tmp_path,
        backbone=backbone,
        memory_bank=bank,
    )
    assert c4_pred.is_file()

    c4_eval = evaluate_condition(
        condition="C4",
        predictions_path=c4_pred,
        train_json_path=dummy_train_json,
        output_dir=tmp_path,
    )
    assert "metrics" in c4_eval

    # Paired comparison
    comp = compare_conditions_against_baseline(
        base_eval_csv=tmp_path / "C0" / "eval" / "per_claim_evaluation.csv",
        target_eval_csv=tmp_path / "C4" / "eval" / "per_claim_evaluation.csv",
        base_id="C0",
        target_id="C4",
    )
    assert comp["total_claims"] == 8
    assert "net_gain" in comp
    assert "mcnemar_p_value" in comp
    assert "bootstrap_95_ci_lower" in comp


def test_generate_m13_markdown_report_unwraps_metrics(tmp_path: Path):
    from tffv.verification.m13_runner import generate_m13_markdown_report

    cond_summaries = {
        "C0": {
            "metrics": {
                "accuracy": 0.6087,
                "macro_f1": 0.4883,
                "macro_precision": 0.5028,
                "macro_recall": 0.6453,
                "total_claims": 46,
                "correct_claims": 28,
                "confusion_matrix": {
                    "Supported": {"Supported": 9, "Unparseable": 0},
                },
                "per_class": {
                    "Supported": {"f1": 0.6923},
                    "Refuted": {"f1": 0.7442},
                    "Not Enough Evidence": {"f1": 0.2500},
                    "Conflicting Evidence/Cherrypicking": {"f1": 0.2667},
                },
            },
            "timing": {
                "mean_latency_seconds": 4.47,
                "mean_prompt_tokens": 2483,
                "mean_generation_tokens": 192.3,
            },
        }
    }
    paired = {}
    part_manifest = {"total_source_claims": 50, "admitted_claims_count": 46}

    report_p = generate_m13_markdown_report(
        condition_summaries=cond_summaries,
        paired_comparisons=paired,
        output_dir=tmp_path,
        partition_manifest=part_manifest,
    )
    content = report_p.read_text(encoding="utf-8")
    assert "60.9%" in content
    assert "0.4883" in content
    assert "46/46" in content
    assert "0.6923" in content
    assert "0/0" not in content


def test_canonical_m13_audit_integrity():
    from tffv.verification.m13_audit import run_comprehensive_audit_and_repair
    summary = run_comprehensive_audit_and_repair()
    
    assert summary["total_evaluation_claims"] == 46
    metrics = summary["metrics_by_condition"]
    paired = summary["paired_comparisons"]

    assert len(metrics) == 7
    for cond in ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]:
        assert metrics[cond]["total_claims"] == 46
        assert metrics[cond]["valid_verdicts"] + metrics[cond]["unparsed_claims"] == 46
        assert metrics[cond]["mean_output_tokens"] > 100.0  # Token accounting verified nonzero!

    # Paired transition assertions
    for cond in ["C1", "C2", "C3", "C4", "C5", "C6"]:
        p = paired[cond]
        assert p["net_gain"] == p["delta_correct"]
        assert p["improved_total"] - p["regressed_total"] == p["net_gain"]
        cats = p["categories"]
        assert sum(cats.values()) == 46  # All 9 categories sum to 46

