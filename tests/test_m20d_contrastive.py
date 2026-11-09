"""Unit tests for Milestone M2.0d: Training-Free Contrastive Evidence Verification (Track C)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tffv.experiments.m20d_contrastive import (
    ContrastivePromptBuilder,
    ContrastiveVerdictParser,
)
from tffv.experiments.m20d_dataset_audit import (
    QUARANTINED_EXACT_OVERLAPS,
    QUARANTINED_NEAR_DEV_DUPLICATES,
    DatasetExposureAuditor,
)
from tffv.experiments.m20d_error_patterns import (
    HISTORICAL_PATTERNS,
    HistoricalPatternSelector,
    PatternCategory,
)
from tffv.experiments.m20d_evaluator import M20dEvaluator
from tffv.experiments.m20d_runner import M20dPreflightAuditor
from tffv.experiments.m20d_schemas import (
    EvidenceAttribution,
    HistoricalErrorPattern,
    M20dArm,
    M20dPredictionRecord,
    PatternSelectionDecision,
)


def test_exposure_audit_and_memory_isolation(tmp_path: Path):
    """Verifies that the dataset exposure audit isolates memory and excludes all dev/eval IDs."""
    auditor = DatasetExposureAuditor(output_dir=tmp_path)
    res = auditor.run_audit()

    assert res["status"] == "AUDIT_COMPLETE"
    assert res["exposure_determination"] == "NO_UNTOUCHED_EVALUATION_DATASET_EXISTS"

    split_path = tmp_path / "track_c_split_manifest.json"
    mem_path = tmp_path / "memory_eligibility_manifest.json"
    assert split_path.exists()
    assert mem_path.exists()

    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    with open(mem_path, "r", encoding="utf-8") as f:
        mem_data = json.load(f)

    mem_ids = set(mem_data["eligible_sample_ids"])
    dev_ids = set(split_data["development_ids"])
    bench_ids = set(split_data["benchmark_evaluation_ids"])

    # Invariants: Zero leakage
    assert len(mem_ids.intersection(dev_ids)) == 0
    assert len(mem_ids.intersection(bench_ids)) == 0
    assert not any(sid.startswith("dev:") for sid in mem_ids)
    for q_id in QUARANTINED_EXACT_OVERLAPS.union(QUARANTINED_NEAR_DEV_DUPLICATES):
        assert q_id not in mem_ids


def test_pattern_selector_signals_and_determinism():
    """Verifies pattern selection based on surface signals and deterministic behavior."""
    selector = HistoricalPatternSelector()

    # 1. Temporal claim
    claim_temp = "In 2021, the company announced its quarterly earnings earlier than in 2019."
    ev_temp = "The report published in 2020 mentioned the event."
    dec_temp, pat_temp, trace_temp = selector.select_pattern("test:temp", claim_temp, ev_temp)
    assert dec_temp == PatternSelectionDecision.PATTERN_SELECTED
    assert pat_temp is not None
    assert pat_temp.category == PatternCategory.TEMPORAL_CONTRADICTION

    # 2. Numerical claim
    claim_num = "The inflation rate reached 15.4% while unemployment doubled to 8%."
    ev_num = "Statistics show numbers increased."
    dec_num, pat_num, trace_num = selector.select_pattern("test:num", claim_num, ev_num)
    assert dec_num == PatternSelectionDecision.PATTERN_SELECTED
    assert pat_num is not None
    assert pat_num.category == PatternCategory.NUMERICAL_PRECISION

    # 3. Negation claim
    claim_neg = "The minister denied that any funds were missing and rejected the fake allegations."
    ev_neg = "Press release statement."
    dec_neg, pat_neg, trace_neg = selector.select_pattern("test:neg", claim_neg, ev_neg)
    assert dec_neg == PatternSelectionDecision.PATTERN_SELECTED
    assert pat_neg is not None
    assert pat_neg.category == PatternCategory.PREDICATE_NEGATION_OVERLOOKED

    # 4. Determinism
    dec2, pat2, _ = selector.select_pattern("test:temp", claim_temp, ev_temp)
    assert dec_temp == dec2
    assert pat_temp.pattern_id == pat2.pattern_id


def test_contrastive_prompt_rendering_and_matched_controls():
    """Verifies prompt rendering, SHA-256 fingerprinting, and matched control contracts."""
    evidence = [
        {"passage_id": 1, "text": "First passage confirming event.", "url": "http://a.com"},
        {"passage_id": 2, "text": "Second passage discussing context.", "url": "http://b.com"},
    ]
    claim = "The event occurred in 2022."
    selector = HistoricalPatternSelector()
    dec, pat, _ = selector.select_pattern("test:1", claim, "2022 evidence")
    placebo = selector.select_placebo_pattern("test:1")

    # Render prompts across 6 arms
    prompts = {}
    for arm in [M20dArm.B0, M20dArm.G2, M20dArm.C1, M20dArm.C2, M20dArm.C3, M20dArm.C4]:
        sys_i, usr, sha = ContrastivePromptBuilder.build_prompt(
            arm=arm,
            claim_text=claim,
            evidence_passages=evidence,
            pattern_decision=dec,
            selected_pattern=pat,
            placebo_pattern=placebo,
        )
        prompts[arm] = (sys_i, usr, sha)
        assert "[1] First passage confirming event." in usr
        assert "[2] Second passage discussing context." in usr
        assert len(sha) == 64

    # Matched Control Contract: C2 and C3 must have identical user prompts and near-identical sys length
    c2_sys, c2_usr, _ = prompts[M20dArm.C2]
    c3_sys, c3_usr, _ = prompts[M20dArm.C3]
    assert c2_usr == c3_usr
    assert abs(len(c2_sys.split()) - len(c3_sys.split())) <= 10


def test_contrastive_verdict_parser_success_and_failures():
    """Verifies parsing of structured evidence attribution and verdicts."""
    parser = ContrastiveVerdictParser()

    # 1. Valid full structured output
    completion_valid = (
        "[EVIDENCE ATTRIBUTION]\n"
        "- Supporting Passage IDs: [1, 2]\n"
        "- Refuting Passage IDs: None\n"
        "- Scope & Relevance Assessment: Addresses exact 2022 timeframe.\n"
        "- Sufficiency & Conflict Assessment: Passage 1 verifies predicate directly.\n\n"
        "[VERDICT]\n"
        "Supported"
    )
    verdict, status, attr = parser.parse_completion(completion_valid, num_passages=8)
    assert verdict == "Supported"
    assert status == "SUCCESS"
    assert attr.is_valid_attribution is True
    assert attr.supporting_passage_ids == [1, 2]
    assert attr.refuting_passage_ids == []

    # 2. Refuted with conflicting evidence
    completion_refuted = (
        "[EVIDENCE ATTRIBUTION]\n"
        "- Supporting Passage IDs: []\n"
        "- Refuting Passage IDs: [3, 4]\n"
        "- Scope & Relevance Assessment: Official dates contradict claim.\n"
        "- Sufficiency & Conflict Assessment: Passages 3 and 4 directly debunk statement.\n\n"
        "[VERDICT]\n"
        "Refuted"
    )
    verdict, status, attr = parser.parse_completion(completion_refuted, num_passages=8)
    assert verdict == "Refuted"
    assert status == "SUCCESS"
    assert attr.refuting_passage_ids == [3, 4]

    # 3. Unparseable output
    completion_bad = "I cannot determine whether this claim is true because I am an AI."
    verdict, status, attr = parser.parse_completion(completion_bad, num_passages=8)
    assert verdict is None
    assert status == "UNPARSEABLE"


def test_preflight_auditor(tmp_path: Path):
    """Verifies the preflight integrity checker."""
    split_man = {
        "development_ids": ["train:10", "train:11"],
        "benchmark_evaluation_ids": ["train:20", "train:21"],
        "quarantined_ids": ["train:1340"],
    }
    mem_man = {
        "eligible_sample_ids": ["train:30", "train:31"],
    }
    receipts = {
        "train:10": [{"text": "ev"}],
        "train:11": [{"text": "ev"}],
        "train:20": [{"text": "ev"}],
        "train:21": [{"text": "ev"}],
    }
    auditor = M20dPreflightAuditor(
        output_dir=tmp_path,
        split_manifest=split_man,
        memory_manifest=mem_man,
        retrieval_receipts=receipts,
    )
    passed, msg = auditor.run_preflight()
    assert passed is True
    assert (tmp_path / "PREFLIGHT_REPORT.md").exists()


def test_evaluator_decision_battery(tmp_path: Path):
    """Verifies that M20dEvaluator computes metrics, paired transitions, and pre-registered decisions."""
    # Synthetic records for C2 and C3
    records = []
    for i in range(20):
        sid = f"train:{i}"
        gold = "Supported" if i < 10 else "Refuted"
        # C3 correct on 10, C2 correct on 9 -> Net benefit: -1 -> NO-GO
        c3_pred = gold if i < 10 else ("Supported" if i < 15 else "Refuted")
        c2_pred = gold if i < 9 else ("Supported" if i < 15 else "Refuted")

        rec_c3 = M20dPredictionRecord(
            sample_id=sid,
            arm="C3",
            raw_claim="Claim",
            gold_label=gold,
            model_name="mock",
            model_revision="rev",
            retrieval_hash="hash",
            prompt_sha256="sha",
            raw_completion="[VERDICT] " + c3_pred,
            parsed_verdict=c3_pred,
            parse_status="SUCCESS",
            attribution=EvidenceAttribution(is_valid_attribution=True),
            pattern_decision="PATTERN_SELECTED",
            selected_pattern_id="PAT-TEMP-01",
            prompt_tokens=100,
            output_tokens=30,
            latency_seconds=0.1,
        )
        rec_c2 = M20dPredictionRecord(
            sample_id=sid,
            arm="C2",
            raw_claim="Claim",
            gold_label=gold,
            model_name="mock",
            model_revision="rev",
            retrieval_hash="hash",
            prompt_sha256="sha",
            raw_completion="[VERDICT] " + c2_pred,
            parsed_verdict=c2_pred,
            parse_status="SUCCESS",
            attribution=EvidenceAttribution(is_valid_attribution=True),
            pattern_decision="PATTERN_SELECTED",
            selected_pattern_id="PAT-TEMP-01",
            prompt_tokens=100,
            output_tokens=30,
            latency_seconds=0.1,
        )
        records.extend([rec_c3, rec_c2])

    evaluator = M20dEvaluator(records=records, output_dir=tmp_path, n_bootstraps=200)
    res = evaluator.evaluate_all()

    assert res["decision"] == "NO-GO"
    assert (tmp_path / "DECISION_REPORT.md").exists()
    assert (tmp_path / "metrics_by_arm.csv").exists()
    assert (tmp_path / "paired_transitions.csv").exists()
