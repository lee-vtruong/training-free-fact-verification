"""
Unit and integration tests for Milestone M1.9 Historical Outcome Memory Suite.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tffv.experiments.m19_evaluator import M19Evaluator, parse_verdict
from tffv.experiments.m19_orchestrator import EXPERIMENTAL_ARMS, M19Orchestrator
from tffv.experiments.m19_outcome_store import (
    ALL_OFFICIAL_TRAIN_COUNT,
    AVERITEC_CLASSES,
    EXACT_TRAIN_DEV_OVERLAPS,
    HistoricalOutcomeStoreBuilder,
    HistoricalRecord,
    norm_claim_text,
)
from tffv.experiments.m19_selector import (
    ClaimSimilarityIndex,
    HistoricalMemorySelector,
)


@pytest.fixture
def mock_m19_env(tmp_path: Path):
    """Sets up a mock environment with synthetic historical predictions and train data."""
    m18e_dir = tmp_path / "m18e"
    m18g_dir = tmp_path / "m18g"
    out_dir = tmp_path / "m19_out"
    m18e_dir.mkdir()
    m18g_dir.mkdir()

    # Generate synthetic records (10 claims)
    records = []
    base_preds = []
    receipts = []

    for i in range(10):
        sid = f"train:{i}"
        claim = f"Claim about economy and inflation number {i}"
        gold = "Supported" if i % 2 == 0 else "Refuted"
        pred = gold if i < 6 else ("Refuted" if gold == "Supported" else "Supported")
        correctness = "CORRECT" if pred == gold else "WRONG"

        ev = [{"source_url": "http://test.org", "rendered_text": f"Evidence passage text for claim {i}"}]

        r = HistoricalRecord(
            sample_id=sid,
            claim=claim,
            gold_verdict=gold,
            base_verdict=pred,
            correctness=correctness,
            evidence_count=1,
            evidence_hash="abc",
            evidence_references=ev,
            retrieval_fingerprint="R4-PF1000-v1",
            verifier_fingerprint="Qwen3-8B-BF16-greedy",
            leakage_group=f"grp_{i}",
            is_selectable_correct=(correctness == "CORRECT"),
            is_selectable_wrong=(correctness == "WRONG"),
            exclusion_reason="NONE",
        )
        records.append(r)

        base_preds.append({
            "sample_id": sid,
            "claim": claim,
            "predicted_verdict": pred,
            "gold_verdict": gold,
            "is_correct": (pred == gold),
        })
        receipts.append({
            "sample_id": sid,
            "evidence": ev,
        })

    with open(m18e_dir / "base_predictions.jsonl", "w", encoding="utf-8") as f:
        for bp in base_preds:
            f.write(json.dumps(bp) + "\n")

    with open(m18e_dir / "retrieval_receipts.jsonl", "w", encoding="utf-8") as f:
        for rc in receipts:
            f.write(json.dumps(rc) + "\n")

    # Mock dev.json (4 claims)
    dev_claims = [
        {"claim": f"Dev claim testing inflation {i}", "label": "Supported", "questions": []}
        for i in range(4)
    ]
    dev_path = tmp_path / "dev.json"
    dev_path.write_text(json.dumps(dev_claims), encoding="utf-8")

    train_path = tmp_path / "train.json"
    train_claims = [
        {"claim": f"Claim about economy and inflation number {i}", "label": "Supported" if i % 2 == 0 else "Refuted"}
        for i in range(10)
    ]
    train_path.write_text(json.dumps(train_claims), encoding="utf-8")

    return {
        "m18e_dir": m18e_dir,
        "m18g_dir": m18g_dir,
        "out_dir": out_dir,
        "records": records,
        "train_path": train_path,
        "dev_path": dev_path,
    }


def test_verdict_parser():
    """Test robust extraction of AVeriTeC verdicts from completions."""
    res1, ok1 = parse_verdict("Explanation: The claim is true.\nVerdict: Supported")
    assert res1 == "Supported" and ok1

    res2, ok2 = parse_verdict("Verdict: Conflicting Evidence/Cherrypicking")
    assert res2 == "Conflicting Evidence/Cherrypicking" and ok2

    res3, ok3 = parse_verdict("Verdict: Refuted.")
    assert res3 == "Refuted" and ok3

    res4, ok4 = parse_verdict("There is Not Enough Evidence to support this claim.")
    assert res4 == "Not Enough Evidence" and ok4


def test_claim_similarity_index(mock_m19_env):
    """Test deterministic TF-IDF claim similarity index."""
    records = mock_m19_env["records"]
    index = ClaimSimilarityIndex(records)

    cands = [r.sample_id for r in records]
    ranked1 = index.query_similarity("economy and inflation", cands, top_k=3)
    ranked2 = index.query_similarity("economy and inflation", cands, top_k=3)

    assert len(ranked1) == 3
    # 100% deterministic
    assert ranked1 == ranked2
    assert all(score >= 0.0 for _, score in ranked1)


def test_historical_memory_selector_pools_and_leakage(mock_m19_env):
    """Test pool restrictions, self-exclusion, and matched controls."""
    records = mock_m19_env["records"]
    selector = HistoricalMemorySelector(records, k=2)

    # 1. H1 only selects from correct pool
    sel_h1 = selector.select_exemplars_for_arm(
        arm="H1",
        target_sample_id="dev:1",
        target_claim="economy test",
        target_evidence=[],
    )
    for eid in sel_h1.chosen_exemplar_ids:
        assert selector.record_by_id[eid].correctness == "CORRECT"

    # 2. H2 only selects from wrong pool
    sel_h2 = selector.select_exemplars_for_arm(
        arm="H2",
        target_sample_id="dev:1",
        target_claim="economy test",
        target_evidence=[],
    )
    for eid in sel_h2.chosen_exemplar_ids:
        assert selector.record_by_id[eid].correctness == "WRONG"

    # 3. B1 matched control uses EXACT same exemplars as H2
    sel_b1 = selector.select_exemplars_for_arm(
        arm="B1",
        target_sample_id="dev:1",
        target_claim="economy test",
        target_evidence=[],
        matched_exemplars=sel_h2.chosen_exemplar_ids,
    )
    assert sel_b1.chosen_exemplar_ids == sel_h2.chosen_exemplar_ids

    # 4. Target self-exclusion
    target_sid = "train:0"
    target_c = records[0].claim
    sel_self = selector.select_exemplars_for_arm(
        arm="H1",
        target_sample_id=target_sid,
        target_claim=target_c,
        target_evidence=[],
    )
    assert target_sid not in sel_self.chosen_exemplar_ids

    # 5. Prompt content check: B1 hides previous prediction, H2 shows it
    assert "Previous Model Prediction" not in sel_b1.formatted_prompt
    assert "Verdict:" in sel_b1.formatted_prompt
    assert "Previous Model Prediction" in sel_h2.formatted_prompt
    assert "Do NOT repeat this error" in sel_h2.formatted_prompt


def test_m19_evaluator_and_mcnemar():
    """Test 4-class Macro-F1 computation and exact McNemar test."""
    evaluator = M19Evaluator(bootstrap_resamples=100, seed=42)

    preds_a = [
        {"sample_id": f"s:{i}", "gold_verdict": "Supported", "predicted_verdict": "Supported", "is_parseable": True}
        for i in range(8)
    ] + [
        {"sample_id": f"s:{i}", "gold_verdict": "Supported", "predicted_verdict": "Refuted", "is_parseable": True}
        for i in range(8, 10)
    ]

    preds_b = [
        {"sample_id": f"s:{i}", "gold_verdict": "Supported", "predicted_verdict": "Supported", "is_parseable": True}
        for i in range(5)
    ] + [
        {"sample_id": f"s:{i}", "gold_verdict": "Supported", "predicted_verdict": "Refuted", "is_parseable": True}
        for i in range(5, 10)
    ]

    m_a = evaluator.compute_arm_metrics("ArmA", preds_a)
    m_b = evaluator.compute_arm_metrics("ArmB", preds_b)

    assert m_a.accuracy == 0.8
    assert m_b.accuracy == 0.5
    assert m_a.macro_f1 > m_b.macro_f1

    # Test paired bootstrap
    boot = evaluator.paired_bootstrap_test(preds_a, preds_b, "ArmA", "ArmB")
    assert boot["observed_delta_accuracy"] == 0.3
    assert boot["contingency_table"]["arm_a_correct_arm_b_wrong"] == 3
    assert boot["contingency_table"]["arm_a_wrong_arm_b_correct"] == 0
    assert boot["mcnemar_exact_p_value"] < 0.30


def test_full_m19_orchestrator_synthetic(mock_m19_env):
    """Test full synthetic end-to-end execution of M19Orchestrator."""
    env = mock_m19_env

    # Subclass to bypass raw 3068 count check on small test fixture
    class TestOrchestrator(M19Orchestrator):
        def _load_eval_claims(self, limit_claims=None, pilot_mode=False, pilot_claims=None, records=None, **kwargs):
            return [
                {
                    "sample_id": f"test_dev:{i}",
                    "claim": f"Test dev claim {i}",
                    "gold_verdict": "Supported" if i % 2 == 0 else "Refuted",
                    "evidence": [{"rendered_text": "Sample test evidence"}],
                }
                for i in range(6)
            ]

    orch = TestOrchestrator(
        m18e_dir=env["m18e_dir"],
        output_dir=env["out_dir"],
        train_path=env["train_path"],
        dev_path=env["dev_path"],
        k=2,
        seed=42,
    )

    res = orch.run_all(run_gpu=False, limit_claims=6)
    assert res is not None

    # Verify all 6 arms exist in metrics
    for arm in EXPERIMENTAL_ARMS:
        assert arm in res["metrics"]
        assert (env["out_dir"] / "metrics_by_arm.csv").exists()

    # Verify required deliverables exist
    required_artifacts = [
        "dataset_integrity_report.md",
        "historical_store.jsonl",
        "historical_store_summary.json",
        "memory_index_manifest.json",
        "pilot_manifest.json",
        "official_dev_manifest.json",
        "selection_trace.jsonl",
        "predictions.jsonl",
        "per_claim_metrics.csv",
        "metrics_by_arm.csv",
        "statistical_tests.json",
        "confusion_matrices.json",
        "per_class_metrics.csv",
        "latency_breakdown.csv",
        "FINAL_REPORT.md",
        "DECISION_REPORT.md",
        "run_receipt.json",
        "config_lock.json",
        "failures.jsonl",
    ]
    for art in required_artifacts:
        assert (env["out_dir"] / art).exists(), f"Missing deliverable: {art}"
