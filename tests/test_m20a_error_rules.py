"""Unit & Regression Tests for Milestone M2.0-A: Error-Derived Verification Rules.

Validates:
1. Pattern extraction strictly distinguishes observed errors from hypothesized causes.
2. Grouped-fold isolation guards reject Dev/Test contamination with ValueError.
3. Provenance and cryptographic hash determinism for all rule sets.
4. Length matching property between G2/E2 and E1 (+/- 15%).
5. Target task invariant: identical target evidence rendering across all 5 arms.
6. Evaluator transition conservation: Delta Accuracy == (Corrected - Harmed) / N.
7. End-to-end synthetic orchestrator execution on CPU.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from tffv.experiments.m20a_evaluator import M20aEvaluator
from tffv.experiments.m20a_orchestrator import M20aOrchestrator, MockVerifierRunner
from tffv.experiments.m20a_pattern_extractor import ErrorPatternExtractor
from tffv.experiments.m20a_prompts import M20aPromptFormatter
from tffv.experiments.m20a_rule_generator import M20aRuleGenerator
from tffv.experiments.m20a_schemas import (
    EXPERIMENTAL_ARMS_M20A,
    ErrorTaxonomyCategory,
    M20aArm,
    M20aPredictionRecord,
    OFFICIAL_CLASSES,
)


def test_pattern_extraction_observed_vs_hypothesized():
    extractor = ErrorPatternExtractor(source_fold_name="train_fold_1")
    
    # Claim with temporal mismatch
    pat = extractor.extract_pattern(
        sample_id="train:100",
        claim_text="The bridge opened in 2020.",
        gold_verdict="Refuted",
        base_verdict="Supported",
        evidence_passages=[{"text": "State records confirm the bridge completed construction and opened in 2017."}],
        group_id="grp-123",
    )
    assert pat is not None
    assert pat.pattern_id.startswith("pat_train:100_")
    assert pat.source_fold == "train_fold_1"
    assert pat.group_id == "grp-123"

    # 1. Observed Error: factual fields
    obs = pat.observed
    assert obs.sample_id == "train:100"
    assert obs.gold_verdict == "Refuted"
    assert obs.base_verdict == "Supported"
    assert obs.transition_type == "Predicted Supported when Gold is Refuted"
    assert "temporal_mismatch_detected" in obs.discrepancy_signals
    assert len(obs.evidence_snippets) == 1

    # 2. Hypothesized Cause: explicitly marked hypothesis
    hyp = pat.hypothesized
    assert hyp.is_hypothesized is True
    assert hyp.category == ErrorTaxonomyCategory.TEMPORAL_CONTRADICTION
    assert "Model overlooks date/year differences" in hyp.hypothesized_mechanism
    assert "Enforce an explicit rule" in hyp.remedy_hypothesis


def test_isolation_guard_rejects_dev_and_disallowed_splits():
    extractor = ErrorPatternExtractor(
        source_fold_name="train_fold_1",
        allowed_sample_ids={"train:1", "train:2"},
        disallowed_sample_ids={"train:999"},
    )

    # 1. Reject Official Dev claims
    with pytest.raises(ValueError, match=r"\[ISOLATION VIOLATION\]"):
        extractor.extract_pattern(
            sample_id="dev:42",
            claim_text="Sample claim",
            gold_verdict="Refuted",
            base_verdict="Supported",
            evidence_passages=["Evidence"],
        )

    # 2. Reject Official Test claims
    with pytest.raises(ValueError, match=r"\[ISOLATION VIOLATION\]"):
        extractor.extract_pattern(
            sample_id="test:10",
            claim_text="Sample claim",
            gold_verdict="Refuted",
            base_verdict="Supported",
            evidence_passages=["Evidence"],
        )

    # 3. Reject explicitly disallowed claims
    with pytest.raises(ValueError, match=r"\[ISOLATION VIOLATION\]"):
        extractor.extract_pattern(
            sample_id="train:999",
            claim_text="Sample claim",
            gold_verdict="Refuted",
            base_verdict="Supported",
            evidence_passages=["Evidence"],
        )

    # 4. Reject claims not in allowed partition
    with pytest.raises(ValueError, match=r"\[ISOLATION VIOLATION\]"):
        extractor.extract_pattern(
            sample_id="train:3",
            claim_text="Sample claim",
            gold_verdict="Refuted",
            base_verdict="Supported",
            evidence_passages=["Evidence"],
        )


def test_rule_generator_seals_provenance_and_hashes():
    gen = M20aRuleGenerator(source_fold="train_fold_1")
    rule_sets = gen.build_all_rule_sets(patterns=None)

    assert set(rule_sets.keys()) == set(EXPERIMENTAL_ARMS_M20A)

    # Arm B0 has empty rules
    assert len(rule_sets["B0"].rules) == 0
    assert rule_sets["B0"].token_count_estimate == 0

    # Arm E1 has 4 error-derived rules
    assert len(rule_sets["E1"].rules) == 4
    assert rule_sets["E1"].provenance.group_isolation_verified is True
    assert rule_sets["E1"].provenance.source_fold == "train_fold_1"
    assert len(rule_sets["E1"].provenance.rule_set_hash) == 64

    # Hash determinism
    h1 = rule_sets["E1"].compute_hash()
    h2 = rule_sets["E1"].compute_hash()
    assert h1 == h2 == rule_sets["E1"].provenance.rule_set_hash


def test_length_matched_property_g2_e2_vs_e1():
    gen = M20aRuleGenerator(source_fold="train_fold_1")
    rule_sets = gen.build_all_rule_sets(patterns=None)

    e1_tokens = rule_sets["E1"].token_count_estimate
    g2_tokens = rule_sets["G2"].token_count_estimate
    e2_tokens = rule_sets["E2"].token_count_estimate

    assert e1_tokens > 0
    # G2 and E2 token length must match E1 within +/- 15%
    diff_g2 = abs(g2_tokens - e1_tokens) / e1_tokens
    diff_e2 = abs(e2_tokens - e1_tokens) / e1_tokens

    assert diff_g2 <= 0.15, f"G2 ({g2_tokens}) differs from E1 ({e1_tokens}) by {diff_g2:.1%}"
    assert diff_e2 <= 0.15, f"E2 ({e2_tokens}) differs from E1 ({e1_tokens}) by {diff_e2:.1%}"


def test_prompt_formatter_identical_target_task():
    gen = M20aRuleGenerator(source_fold="train_fold_1")
    rule_sets = gen.build_all_rule_sets(patterns=None)
    formatter = M20aPromptFormatter(rule_sets)

    claim = "The economic stimulus was approved in March 2021."
    evidence = [
        {"passage_idx": 1, "text": "Congress passed the American Rescue Plan Act in March 2021."},
        {"passage_idx": 2, "text": "The bill provided $1.9 trillion in federal funding."},
    ]

    prompts = {}
    target_sections = {}

    for arm in EXPERIMENTAL_ARMS_M20A:
        prompt_text, sha, tok = formatter.format_prompt(
            arm=arm, claim_text=claim, evidence_passages=evidence, context_id="test:1"
        )
        prompts[arm] = prompt_text
        assert len(sha) == 64
        assert tok > 50

        # Extract target task block
        target_part = prompt_text.split("### Target Verification Task:")[1]
        target_sections[arm] = target_part

    # Target task section must be 100% byte-for-byte identical across all arms
    base_target = target_sections["B0"]
    for arm in ["G1", "G2", "E1", "E2"]:
        assert target_sections[arm] == base_target, f"Target block mismatch in arm {arm}"

    # Rules injection checks
    assert "### Error-Derived Verification Rules" in prompts["E1"]
    assert "### Standard Verification Guidelines" in prompts["G1"]
    assert "### Detailed Methodological Verification Guidelines" in prompts["G2"]
    assert "### Linguistic & Textual Scrutiny Protocol" in prompts["E2"]
    assert "### " not in prompts["B0"].split("### Target Verification Task:")[0]


def test_prompt_formatter_fails_on_empty_evidence():
    gen = M20aRuleGenerator(source_fold="train_fold_1")
    rule_sets = gen.build_all_rule_sets(patterns=None)
    formatter = M20aPromptFormatter(rule_sets)

    with pytest.raises(ValueError, match=r"non-empty list of passages"):
        formatter.format_prompt(
            arm="B0", claim_text="Some claim", evidence_passages=[], context_id="fail:1"
        )


def test_evaluator_transition_conservation_and_mcnemar():
    # Build synthetic paired predictions for 5 claims
    golds = ["Supported", "Refuted", "Supported", "Not Enough Evidence", "Refuted"]

    # Candidate E1: 4 correct (claims 0, 1, 2, 4)
    # Baseline G1:  3 correct (claims 0, 1, 3)
    # Harmed: claim 3 (G1 correct, E1 wrong) -> n_01 = 1
    # Corrected: claim 2, 4 (G1 wrong, E1 correct) -> n_10 = 2
    # Net benefit = 2 - 1 = +1
    # Cand acc = 4/5 = 80%, Base acc = 3/5 = 60%, diff = +20% (1/5)
    e1_preds = ["Supported", "Refuted", "Supported", "Refuted", "Refuted"]
    g1_preds = ["Supported", "Refuted", "Refuted", "Not Enough Evidence", "Supported"]

    e1_records = [
        M20aPredictionRecord(
            sample_id=f"c:{i}", arm="E1", claim=f"Claim {i}", gold_verdict=golds[i],
            predicted_verdict=e1_preds[i], raw_completion=f"Verdict: {e1_preds[i]}",
            is_parseable=True, is_correct=(e1_preds[i] == golds[i]), latency_seconds=0.1,
            prompt_sha256="abc", rule_set_id="rules_e1_v1",
        )
        for i in range(5)
    ]
    g1_records = [
        M20aPredictionRecord(
            sample_id=f"c:{i}", arm="G1", claim=f"Claim {i}", gold_verdict=golds[i],
            predicted_verdict=g1_preds[i], raw_completion=f"Verdict: {g1_preds[i]}",
            is_parseable=True, is_correct=(g1_preds[i] == golds[i]), latency_seconds=0.1,
            prompt_sha256="xyz", rule_set_id="rules_g1_v1",
        )
        for i in range(5)
    ]

    evaluator = M20aEvaluator({"E1": e1_records, "G1": g1_records}, n_bootstraps=200)
    results = evaluator.evaluate_all()

    trans = results["transitions"]["E1_vs_G1"]
    assert trans["total_claims"] == 5
    assert trans["both_correct"] == 2
    assert trans["both_wrong"] == 0
    assert trans["harmed"] == 1
    assert trans["corrected"] == 2
    assert trans["net_benefit"] == 1
    assert trans["harm_to_correction_ratio"] == 0.5
    assert trans["mcnemar_b"] == 2
    assert trans["mcnemar_c"] == 1
    assert trans["mcnemar_p_value"] > 0.0
    # Conservation holds
    assert trans["delta_accuracy"] == 0.20


def test_synthetic_orchestrator_dry_run_generates_all_artifacts(tmp_path: Path):
    out_dir = tmp_path / "m20a_dry_run"

    claims = [
        {"sample_id": f"dev:{i}", "claim": f"Test claim statement {i}", "gold_verdict": "Supported"}
        for i in range(6)
    ]
    receipts = {
        f"dev:{i}": [
            {"passage_idx": 1, "text": f"Direct corroboration statement for test claim statement {i}."}
        ]
        for i in range(6)
    }

    orchestrator = M20aOrchestrator(output_dir=out_dir, source_fold="train_fold_1", target_split_name="dev")
    results = orchestrator.run(
        target_claims=claims,
        retrieval_receipts=receipts,
        source_train_records=None,
        run_gpu=False,
        smoke_only=False,
    )

    # Check that all expected deliverables exist and are non-empty
    expected_files = [
        "config_lock.json",
        "rules_manifest.json",
        "prompt_receipts.jsonl",
        "predictions.jsonl",
        "run_receipt.json",
        "metrics_by_arm.csv",
        "per_class_metrics.csv",
        "statistical_tests.json",
        "confusion_matrices.json",
        "FINAL_REPORT.md",
        "DECISION_REPORT.md",
    ]

    for fname in expected_files:
        p = out_dir / fname
        assert p.exists(), f"Missing deliverable: {fname}"
        assert p.stat().st_size > 0, f"Empty deliverable: {fname}"

    # Verify predictions record count: 6 claims * 5 arms = 30 records
    with open(out_dir / "predictions.jsonl", "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 30

    # Verify rules manifest contains all 5 arms
    with open(out_dir / "rules_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert set(manifest.keys()) == set(EXPERIMENTAL_ARMS_M20A)
