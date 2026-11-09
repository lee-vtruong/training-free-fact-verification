"""Unit tests for Milestone M2.0g: Historical Error Knowledge Robustness & Attribution.

Tests:
1. Data Contracts and Schemas.
2. Certified Exemplar Bank and Multi-Seed Balancer.
3. Knowledge Component Prompt Formatting (K1, K2, K3, K4).
4. Preflight Isolation and Group-Safety Verification.
5. Statistical Evaluator Math (E1 seed mean/std, E2 component ablation, E3 generalization).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from tffv.experiments.m20d_contrastive import ContrastiveVerdictParser
from tffv.experiments.m20g_evaluator import M20gEvaluator
from tffv.experiments.m20g_exemplars import (
    CERTIFIED_EXEMPLAR_BANK,
    ExemplarItem,
    ExemplarManager,
)
from tffv.experiments.m20g_prompts import M20gPromptBuilder
from tffv.experiments.m20g_runner import (
    M20gPreflightAuditor,
    MockM20gVerifier,
)
from tffv.experiments.m20g_schemas import (
    ExperimentPhase,
    KnowledgeComponent,
    M20gPredictionRecord,
)


class TestM20gSchemas:
    """Verifies schema contracts and data models."""

    def test_enums_completeness(self):
        expected_phases = {"E1_ROBUSTNESS", "E2_ABLATION", "E3_GENERALIZATION"}
        assert {p.value for p in ExperimentPhase} == expected_phases

        expected_components = {
            "K1_GOLD_ONLY",
            "K2_GOLD_NEUTRAL_EXPL",
            "K3_WRONG_PRED_GOLD_CORR",
            "K4_FULL_GROUNDED_ERROR",
        }
        assert {c.value for c in KnowledgeComponent} == expected_components

    def test_prediction_record_serialization(self):
        rec = M20gPredictionRecord(
            sample_id="train:1001",
            experiment_phase="E1_ROBUSTNESS",
            arm_name="E1_k2_s1_error",
            seed=1,
            k_shots=2,
            knowledge_component="K4_FULL_GROUNDED_ERROR",
            raw_claim="Test claim",
            gold_label="Supported",
            model_name="Qwen3-8B",
            model_revision="b968826d",
            prompt_sha256="sha123",
            raw_completion="Verdict: Supported",
            parsed_verdict="Supported",
            parse_status="SUCCESS",
        )
        d = rec.to_dict()
        assert d["sample_id"] == "train:1001"
        assert d["seed"] == 1
        assert d["k_shots"] == 2
        assert d["knowledge_component"] == "K4_FULL_GROUNDED_ERROR"


class TestM20gExemplars:
    """Verifies exemplar bank integrity and multi-seed balancer."""

    def test_exemplar_bank_coverage(self):
        assert len(CERTIFIED_EXEMPLAR_BANK) >= 8
        labels = {ex.gold_verdict for ex in CERTIFIED_EXEMPLAR_BANK}
        assert labels == {
            "Supported",
            "Refuted",
            "Not Enough Evidence",
            "Conflicting Evidence/Cherrypicking",
        }

    def test_seed_selection_reproducibility(self):
        for k in [2, 4, 6]:
            ex_s1_a = ExemplarManager.get_seed_exemplars(seed=1, k=k)
            ex_s1_b = ExemplarManager.get_seed_exemplars(seed=1, k=k)
            assert [e.sample_id for e in ex_s1_a] == [e.sample_id for e in ex_s1_b]

            ex_s2 = ExemplarManager.get_seed_exemplars(seed=2, k=k)
            assert len(ex_s1_a) == k
            assert len(ex_s2) == k

    def test_knowledge_component_formatting(self):
        exs = ExemplarManager.get_seed_exemplars(seed=42, k=2)
        for comp in KnowledgeComponent:
            text = ExemplarManager.format_demonstrations(exs, comp)
            assert len(text) > 0
            assert "[EXAMPLE 1]" in text
            assert "[EXAMPLE 2]" in text
            if comp == KnowledgeComponent.K1_GOLD_ONLY:
                assert "[VERIFICATION REASONING]" not in text
                assert "[HISTORICAL ERROR CONTRAST]" not in text
            elif comp == KnowledgeComponent.K2_GOLD_NEUTRAL_EXPL:
                assert "[VERIFICATION REASONING]" in text
                assert "Factual Basis:" in text
            elif comp == KnowledgeComponent.K3_WRONG_PRED_GOLD_CORR:
                assert "Prior Baseline Verdict:" in text
            elif comp == KnowledgeComponent.K4_FULL_GROUNDED_ERROR:
                assert "[HISTORICAL ERROR CONTRAST]" in text
                assert "Observed Mistake:" in text


class TestM20gPromptBuilder:
    """Verifies prompt generation across components."""

    @pytest.fixture
    def sample_inputs(self):
        claim = "Test claim text."
        passages = [{"text": "Sample passage content verifying claim."}]
        return claim, passages

    def test_prompt_rendering_determinism(self, sample_inputs):
        claim, passages = sample_inputs
        exs = ExemplarManager.get_seed_exemplars(seed=1, k=2)

        for comp in KnowledgeComponent:
            sys_1, usr_1, sha_1 = M20gPromptBuilder.build_few_shot_prompt(claim, passages, exs, comp)
            sys_2, usr_2, sha_2 = M20gPromptBuilder.build_few_shot_prompt(claim, passages, exs, comp)
            assert sha_1 == sha_2
            assert sys_1 == sys_2
            assert usr_1 == usr_2
            assert "Sample passage content verifying claim." in usr_1


class TestM20gPreflightAuditor:
    """Verifies group-isolation and safety checks."""

    def test_preflight_passes_on_valid_setup(self, tmp_path):
        bench_ids = ["train:1006", "train:1010"]
        outer_ids = ["train:2001", "train:2002"]
        receipts = {
            "train:1006": [{"text": "p"}],
            "train:1010": [{"text": "p"}],
            "train:2001": [{"text": "p"}],
            "train:2002": [{"text": "p"}],
        }
        claim_to_group = {
            "train:1006": "grp-bench-1",
            "train:1010": "grp-bench-2",
            "train:2001": "grp-outer-1",
            "train:2002": "grp-outer-2",
        }
        auditor = M20gPreflightAuditor(tmp_path, bench_ids, outer_ids, receipts, claim_to_group)
        passed, summary = auditor.run_preflight()
        assert passed is True
        assert "PASSED" in summary

    def test_preflight_fails_on_group_leakage(self, tmp_path):
        bench_ids = ["train:1006"]
        outer_ids = ["train:2001"]
        receipts = {"train:1006": [{"text": "p"}], "train:2001": [{"text": "p"}]}
        # Artificially assign bench group to match an exemplar group
        claim_to_group = {
            "train:1006": CERTIFIED_EXEMPLAR_BANK[0].group_id,
            "train:2001": "grp-outer-1",
        }
        auditor = M20gPreflightAuditor(tmp_path, bench_ids, outer_ids, receipts, claim_to_group)
        passed, summary = auditor.run_preflight()
        assert passed is False
        assert "BENCHMARK_GROUP_ISOLATION" in summary
        assert "FAILED" in summary


class TestM20gEvaluatorMath:
    """Verifies multi-seed statistics and component ablation math."""

    @pytest.fixture
    def synthetic_records(self):
        records = []
        samples = [f"sample:{i}" for i in range(20)]
        gold_labels = ["Supported"] * 10 + ["Refuted"] * 10

        # E1: k=2, seeds 1..5
        for seed in [1, 2, 3, 4, 5]:
            err_arm = f"E1_k2_s{seed}_error"
            gold_arm = f"E1_k2_s{seed}_gold"
            for sid, g in zip(samples, gold_labels):
                records.append(M20gPredictionRecord(
                    sample_id=sid, experiment_phase="E1_ROBUSTNESS", arm_name=err_arm,
                    seed=seed, k_shots=2, knowledge_component="K4_FULL_GROUNDED_ERROR",
                    raw_claim="c", gold_label=g, model_name="m", model_revision="r",
                    prompt_sha256="s", raw_completion="", parsed_verdict=g, parse_status="SUCCESS"
                ))
                records.append(M20gPredictionRecord(
                    sample_id=sid, experiment_phase="E1_ROBUSTNESS", arm_name=gold_arm,
                    seed=seed, k_shots=2, knowledge_component="K2_GOLD_NEUTRAL_EXPL",
                    raw_claim="c", gold_label=g, model_name="m", model_revision="r",
                    prompt_sha256="s", raw_completion="", parsed_verdict="Supported", parse_status="SUCCESS"
                ))

        # E2: K1, K2, K3, K4
        for arm, v in [
            ("E2_K1_gold_only", "Supported"),
            ("E2_K2_neutral_expl", "Supported"),
            ("E2_K3_wrong_pred", "Supported"),
            ("E2_K4_grounded_error", "Supported"),
        ]:
            for sid, g in zip(samples, gold_labels):
                pred = g if arm == "E2_K4_grounded_error" else v
                records.append(M20gPredictionRecord(
                    sample_id=sid, experiment_phase="E2_ABLATION", arm_name=arm,
                    seed=42, k_shots=2, knowledge_component="K4",
                    raw_claim="c", gold_label=g, model_name="m", model_revision="r",
                    prompt_sha256="s", raw_completion="", parsed_verdict=pred, parse_status="SUCCESS"
                ))

        # E3: Outer Fold
        for arm, is_perfect in [("E3_B0_baseline", False), ("E3_A1_gold", False), ("E3_A3_error", True)]:
            for sid, g in zip(samples, gold_labels):
                pred = g if is_perfect else "Supported"
                records.append(M20gPredictionRecord(
                    sample_id=sid, experiment_phase="E3_GENERALIZATION", arm_name=arm,
                    seed=42, k_shots=2, knowledge_component="K4",
                    raw_claim="c", gold_label=g, model_name="m", model_revision="r",
                    prompt_sha256="s", raw_completion="", parsed_verdict=pred, parse_status="SUCCESS"
                ))

        return records

    def test_evaluator_e1_e2_e3(self, tmp_path, synthetic_records):
        evaluator = M20gEvaluator(synthetic_records, output_dir=tmp_path, n_bootstraps=100)
        res = evaluator.evaluate_all()

        e1 = res["e1_results"]
        # Error accuracy: 100% across all seeds
        assert e1["k_2"]["error_mean"] == 1.0
        assert e1["k_2"]["error_std"] == 0.0
        # Gold accuracy: 50% across all seeds (10/20 correct)
        assert e1["k_2"]["gold_mean"] == 0.5
        assert e1["k_2"]["mean_delta"] == 0.5

        e2 = res["e2_results"]
        assert e2["delta_K4_vs_K1"] == 0.5
        assert e2["delta_K4_vs_K2"] == 0.5

        e3 = res["e3_results"]
        assert e3["delta_A3_vs_B0"] == 0.5

        assert "HISTORICAL_ERROR_ROBUSTNESS_CONFIRMED" in res["decision"]


class TestMockVerifier:
    """Verifies mock verifier completion structure."""

    def test_mock_verifier_returns_parsable_completion(self):
        verifier = MockM20gVerifier()
        for comp in KnowledgeComponent:
            exs = ExemplarManager.get_seed_exemplars(seed=1, k=2)
            sys_pr, usr_pr, _ = M20gPromptBuilder.build_few_shot_prompt("Claim", [{"text": "P"}], exs, comp)
            raw = verifier.verify(sys_pr, usr_pr)
            verdict, status, _ = ContrastiveVerdictParser.parse_completion(raw)
            assert status == "SUCCESS"
            assert verdict == "Supported"
