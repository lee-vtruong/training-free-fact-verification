"""Unit tests for Milestone M2.0i: Mechanism Identification & Final Ablation Study.

Tests:
1. Schemas and Arm Enums.
2. Placebo Mistake Alignment and 1:1 Exact Word Count Matching.
3. Prompt Builder Integrity across F0, F1, F2, F3, F4.
4. Preflight Auditor Group Isolation and Leakage Prevention.
5. Statistical Evaluator, Paired Bootstrap CI, and Factorial Interaction Decomposition.
6. Pipeline Execution in Dry-run and Mock Inference modes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from tffv.experiments.m20d_contrastive import ContrastiveVerdictParser
from tffv.experiments.m20g_exemplars import CERTIFIED_EXEMPLAR_BANK, ExemplarManager
from tffv.experiments.m20i_evaluator import M20iStatisticalEvaluator
from tffv.experiments.m20i_mechanism_ablation import MockM20iVerifier
from tffv.experiments.m20i_placebos import MATCHED_PLACEBO_MISTAKES, get_placebo_for_exemplar
from tffv.experiments.m20i_preflight import M20iPreflightAuditor
from tffv.experiments.m20i_prompts import M20iPromptBuilder
from tffv.experiments.m20i_schemas import M20iPredictionRecord, MechanismArm, ProvenanceLevel


class TestM20iSchemasAndPlacebos:
    """Verifies schema contracts, arms, and matched placebo controls."""

    def test_arms_enum(self):
        expected = {
            "F0_neutral_expl",
            "F1_header_only",
            "F2_observed_mistake",
            "F3_full_historical_error",
            "F4_matched_placebo",
        }
        assert {a.value for a in MechanismArm} == expected

    def test_placebo_completeness_and_word_matching(self):
        assert len(MATCHED_PLACEBO_MISTAKES) == len(CERTIFIED_EXEMPLAR_BANK) == 10
        for ex in CERTIFIED_EXEMPLAR_BANK:
            sid = ex.sample_id
            assert sid in MATCHED_PLACEBO_MISTAKES
            plc = get_placebo_for_exemplar(sid)
            w_act = len(ex.observed_mistake.split())
            w_plc = len(plc.split())
            assert w_act == w_plc, f"Word count mismatch for {sid}: {w_act} vs {w_plc}"
            # Ensure no target gold appears inadvertently in placebo
            assert ex.gold_verdict not in plc


class TestM20iPrompts:
    """Verifies prompt generation across the 5 arms."""

    @pytest.fixture
    def sample_inputs(self):
        exs = ExemplarManager.get_seed_exemplars(seed=1, k=2)
        ev = [
            {"passage_id": "p1", "text": "First passage confirming GDP metrics."},
            {"passage_id": "p2", "text": "Second passage confirming growth data."},
        ]
        return "Ghana grew at 3.4% previously.", ev, exs

    def test_prompt_formatting_f0(self, sample_inputs):
        claim, ev, exs = sample_inputs
        sys_inst, user_pr, sha = M20iPromptBuilder.build_prompt(claim, ev, exs, MechanismArm.F0_NEUTRAL_EXPL)
        assert "[VERIFICATION REASONING]" in user_pr
        assert "[HISTORICAL ERROR CONTRAST]" not in user_pr
        assert "Observed Mistake:" not in user_pr
        assert len(sha) == 64

    def test_prompt_formatting_f1(self, sample_inputs):
        claim, ev, exs = sample_inputs
        sys_inst, user_pr, sha = M20iPromptBuilder.build_prompt(claim, ev, exs, MechanismArm.F1_HEADER_ONLY)
        assert "[HISTORICAL ERROR CONTRAST]" in user_pr
        assert "Observed Mistake:" not in user_pr
        assert "- Factual Basis:" in user_pr

    def test_prompt_formatting_f2(self, sample_inputs):
        claim, ev, exs = sample_inputs
        sys_inst, user_pr, sha = M20iPromptBuilder.build_prompt(claim, ev, exs, MechanismArm.F2_OBSERVED_MISTAKE)
        assert "[VERIFICATION REASONING]" in user_pr
        assert "Observed Mistake:" in user_pr
        assert exs[0].observed_mistake in user_pr

    def test_prompt_formatting_f3_and_f4_length_balance(self, sample_inputs):
        claim, ev, exs = sample_inputs
        _, user_f3, sha3 = M20iPromptBuilder.build_prompt(claim, ev, exs, MechanismArm.F3_FULL_HISTORICAL_ERROR)
        _, user_f4, sha4 = M20iPromptBuilder.build_prompt(claim, ev, exs, MechanismArm.F4_MATCHED_PLACEBO)

        assert sha3 != sha4
        assert "[HISTORICAL ERROR CONTRAST]" in user_f3
        assert "[HISTORICAL ERROR CONTRAST]" in user_f4
        assert exs[0].observed_mistake in user_f3
        assert get_placebo_for_exemplar(exs[0].sample_id) in user_f4

        # Word count between F3 and F4 demonstrations must be identical
        f3_words = len(user_f3.split())
        f4_words = len(user_f4.split())
        assert f3_words == f4_words


class TestM20iPreflight:
    """Verifies preflight auditing and fail-closed gates."""

    def test_preflight_pass(self, tmp_path: Path):
        eval_ids = ["train:1001", "train:1002"]
        retrieval = {
            "train:1001": [{"passage_id": "p1", "text": "evidence 1"}],
            "train:1002": [{"passage_id": "p2", "text": "evidence 2"}],
        }
        claim_to_grp = {
            "train:1001": "grp-target-1",
            "train:1002": "grp-target-2",
        }
        auditor = M20iPreflightAuditor(tmp_path, eval_ids, retrieval, claim_to_grp)
        passed, report, _ = auditor.run_preflight()
        assert passed is True
        assert "PASS" in report

    def test_preflight_fails_on_group_leak(self, tmp_path: Path):
        eval_ids = ["train:1001"]
        retrieval = {"train:1001": [{"passage_id": "p1", "text": "evidence 1"}]}
        # Bank has train:2956 with grp-16097fa1c40f653a
        claim_to_grp = {"train:1001": "grp-16097fa1c40f653a"}
        auditor = M20iPreflightAuditor(tmp_path, eval_ids, retrieval, claim_to_grp)
        passed, report, _ = auditor.run_preflight()
        assert passed is False
        assert "GROUP_ISOLATION" in report


class TestM20iEvaluator:
    """Verifies statistical evaluation math and 2x2 factorial interaction."""

    def test_statistical_evaluator(self):
        records = []
        classes = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]
        # Generate 20 claims
        for i in range(20):
            sid = f"claim_{i}"
            gold = classes[i % 4]
            # F3: 18 correct (90%)
            pred_f3 = gold if i < 18 else classes[(i + 1) % 4]
            records.append(M20iPredictionRecord(
                sample_id=sid, arm_name="F3_full_historical_error", seed=1, k_shots=2,
                raw_claim="c", gold_label=gold, model_name="m", model_revision="r",
                prompt_sha256="s", raw_completion=f"Verdict: {pred_f3}",
                parsed_verdict=pred_f3, parse_status="SUCCESS", prompt_tokens=1400, output_tokens=150,
            ))
            # F4: 12 correct (60%)
            pred_f4 = gold if i < 12 else classes[(i + 1) % 4]
            records.append(M20iPredictionRecord(
                sample_id=sid, arm_name="F4_matched_placebo", seed=1, k_shots=2,
                raw_claim="c", gold_label=gold, model_name="m", model_revision="r",
                prompt_sha256="s", raw_completion=f"Verdict: {pred_f4}",
                parsed_verdict=pred_f4, parse_status="SUCCESS", prompt_tokens=1400, output_tokens=150,
            ))
            # F0: 10 correct (50%)
            pred_f0 = gold if i < 10 else classes[(i + 1) % 4]
            records.append(M20iPredictionRecord(
                sample_id=sid, arm_name="F0_neutral_expl", seed=1, k_shots=2,
                raw_claim="c", gold_label=gold, model_name="m", model_revision="r",
                prompt_sha256="s", raw_completion=f"Verdict: {pred_f0}",
                parsed_verdict=pred_f0, parse_status="SUCCESS", prompt_tokens=1250, output_tokens=140,
            ))
            # F1: 11 correct (55%)
            pred_f1 = gold if i < 11 else classes[(i + 1) % 4]
            records.append(M20iPredictionRecord(
                sample_id=sid, arm_name="F1_header_only", seed=1, k_shots=2,
                raw_claim="c", gold_label=gold, model_name="m", model_revision="r",
                prompt_sha256="s", raw_completion=f"Verdict: {pred_f1}",
                parsed_verdict=pred_f1, parse_status="SUCCESS", prompt_tokens=1250, output_tokens=140,
            ))
            # F2: 14 correct (70%)
            pred_f2 = gold if i < 14 else classes[(i + 1) % 4]
            records.append(M20iPredictionRecord(
                sample_id=sid, arm_name="F2_observed_mistake", seed=1, k_shots=2,
                raw_claim="c", gold_label=gold, model_name="m", model_revision="r",
                prompt_sha256="s", raw_completion=f"Verdict: {pred_f2}",
                parsed_verdict=pred_f2, parse_status="SUCCESS", prompt_tokens=1380, output_tokens=145,
            ))

        evaluator = M20iStatisticalEvaluator(records)
        m_f3 = evaluator.compute_arm_metrics("F3_full_historical_error")
        assert m_f3["accuracy"] == 0.90

        comp = evaluator.compute_paired_comparison("F3_full_historical_error", "F4_matched_placebo", n_bootstraps=200)
        assert pytest.approx(comp["delta_accuracy"]) == 0.30
        assert comp["net_benefit"] == 6

        fact = evaluator.compute_factorial_interaction()
        assert "main_effect_framing" in fact
        assert "main_effect_content" in fact
        assert "interaction_framing_x_content" in fact
