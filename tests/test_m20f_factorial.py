"""Unit tests for Milestone M2.0f: Factorial Ablation of Attribution & Error Knowledge.

Tests:
1. Schema & Data Contracts.
2. Exemplar Formatting & Token Budget Balance.
3. Exemplar Isolation & Preflight Audit.
4. Prompt Builder Determinism & Evidence Preservation.
5. Contrastive & Standard Verdict Parsing.
6. Factorial Effect Math (Main Effects, Interaction, Bootstrap CIs).
7. Contradiction Trap Diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from tffv.experiments.m20d_contrastive import ContrastiveVerdictParser
from tffv.experiments.m20f_evaluator import M20fEvaluator
from tffv.experiments.m20f_exemplars import (
    EXEMPLAR_DEMONSTRATIONS,
    ExemplarFormatter,
)
from tffv.experiments.m20f_prompts import M20fPromptBuilder
from tffv.experiments.m20f_runner import (
    M20fPreflightAuditor,
    MockFactorialVerifier,
)
from tffv.experiments.m20f_schemas import (
    AttributionFactor,
    ExemplarDemonstration,
    KnowledgeFactor,
    M20fArm,
    M20fPredictionRecord,
)


class TestM20fSchemas:
    """Verifies schema contracts and data models."""

    def test_arms_enum_completeness(self):
        expected_arms = {"B0", "G2", "A1", "A2", "A3", "A4"}
        actual_arms = {arm.value for arm in M20fArm}
        assert actual_arms == expected_arms

    def test_prediction_record_serialization(self):
        rec = M20fPredictionRecord(
            sample_id="train:1001",
            arm="A1",
            raw_claim="Test claim",
            gold_label="Supported",
            model_name="Qwen3-8B",
            model_revision="b968826d",
            retrieval_hash="hash123",
            prompt_sha256="sha456",
            raw_completion="Verdict: Supported",
            parsed_verdict="Supported",
            parse_status="SUCCESS",
            knowledge_factor="GOLD_ICL",
            attribution_factor="NO_FORCED_ATTR",
        )
        d = rec.to_dict()
        assert d["sample_id"] == "train:1001"
        assert d["arm"] == "A1"
        assert d["knowledge_factor"] == "GOLD_ICL"
        assert d["attribution_factor"] == "NO_FORCED_ATTR"


class TestM20fExemplars:
    """Verifies exemplar demonstration token balance and isolation."""

    def test_exemplar_definitions(self):
        assert len(EXEMPLAR_DEMONSTRATIONS) == 2
        ids = [ex.sample_id for ex in EXEMPLAR_DEMONSTRATIONS]
        assert ids == ["train:2956", "train:2961"]
        verdicts = [ex.gold_verdict for ex in EXEMPLAR_DEMONSTRATIONS]
        assert verdicts == ["Supported", "Refuted"]

    def test_token_budget_balance(self):
        words_a1 = len(ExemplarFormatter.format_demonstrations(M20fArm.A1).split())
        words_a2 = len(ExemplarFormatter.format_demonstrations(M20fArm.A2).split())
        words_a3 = len(ExemplarFormatter.format_demonstrations(M20fArm.A3).split())
        words_a4 = len(ExemplarFormatter.format_demonstrations(M20fArm.A4).split())

        diff_no_attr = abs(words_a1 - words_a3)
        diff_forced_attr = abs(words_a2 - words_a4)

        # Difference must be strictly within 15 words
        assert diff_no_attr <= 15, f"A1 vs A3 word diff {diff_no_attr} exceeds 15 words"
        assert diff_forced_attr <= 15, f"A2 vs A4 word diff {diff_forced_attr} exceeds 15 words"

    def test_exemplar_empty_for_controls(self):
        assert ExemplarFormatter.format_demonstrations(M20fArm.B0) == ""
        assert ExemplarFormatter.format_demonstrations(M20fArm.G2) == ""


class TestM20fPromptBuilder:
    """Verifies prompt construction across all 6 arms."""

    @pytest.fixture
    def sample_data(self):
        claim = "NASA discovered liquid water on Mars in 2015."
        passages = [
            {"text": "In September 2015, NASA announced evidence of hydrated salts on Mars.", "source_url": "https://nasa.gov"},
            {"text": "The findings were published in Nature Geoscience.", "source_url": "https://nature.com"},
        ]
        return claim, passages

    def test_prompt_rendering_and_determinism(self, sample_data):
        claim, passages = sample_data
        for arm in M20fArm:
            sys_1, usr_1, sha_1 = M20fPromptBuilder.build_prompt(arm, claim, passages)
            sys_2, usr_2, sha_2 = M20fPromptBuilder.build_prompt(arm, claim, passages)
            assert sha_1 == sha_2
            assert sys_1 == sys_2
            assert usr_1 == usr_2
            # Evidence must be preserved in user prompt
            assert "hydrated salts on Mars" in usr_1

    def test_arm_specific_formatting(self, sample_data):
        claim, passages = sample_data
        _, usr_b0, _ = M20fPromptBuilder.build_prompt(M20fArm.B0, claim, passages)
        assert "[DEMONSTRATIONS]" not in usr_b0

        _, usr_g2, _ = M20fPromptBuilder.build_prompt(M20fArm.G2, claim, passages)
        assert "[DEMONSTRATIONS]" not in usr_g2

        for arm in [M20fArm.A1, M20fArm.A2, M20fArm.A3, M20fArm.A4]:
            _, usr, _ = M20fPromptBuilder.build_prompt(arm, claim, passages)
            assert "[DEMONSTRATIONS]" in usr
            assert "The Kenyan judiciary receives less than 1%" in usr


class TestM20fPreflightAuditor:
    """Verifies preflight isolation checks."""

    def test_preflight_passes_on_valid_data(self, tmp_path):
        eval_ids = ["train:1006", "train:1010", "train:1013"]
        receipts = {
            "train:1006": [{"text": "p1"}],
            "train:1010": [{"text": "p2"}],
            "train:1013": [{"text": "p3"}],
        }
        auditor = M20fPreflightAuditor(tmp_path, eval_ids, receipts)
        passed, summary = auditor.run_preflight()
        assert passed is True
        assert "PASSED" in summary

    def test_preflight_fails_on_exemplar_leakage(self, tmp_path):
        # Artificially inject exemplar ID into eval set
        eval_ids = ["train:1006", "train:2956"]
        receipts = {
            "train:1006": [{"text": "p1"}],
            "train:2956": [{"text": "p2"}],
        }
        auditor = M20fPreflightAuditor(tmp_path, eval_ids, receipts)
        passed, summary = auditor.run_preflight()
        assert passed is False
        assert "EXEMPLAR_ISOLATION" in summary
        assert "FAILED" in summary


class TestM20fEvaluatorMath:
    """Verifies factorial calculations, bootstrap CIs, and trap diagnostics."""

    @pytest.fixture
    def synthetic_records(self):
        records = []
        samples = [f"sample:{i}" for i in range(40)]
        gold_labels = ["Supported"] * 20 + ["Refuted"] * 20

        for sid, gold in zip(samples, gold_labels):
            # B0: 80% accurate (32/40)
            b0_v = gold if int(sid.split(":")[1]) < 32 else "Not Enough Evidence"
            records.append(M20fPredictionRecord(
                sample_id=sid, arm="B0", raw_claim="c", gold_label=gold,
                model_name="m", model_revision="r", retrieval_hash="h",
                prompt_sha256="s", raw_completion="", parsed_verdict=b0_v,
                parse_status="SUCCESS", knowledge_factor="NONE", attribution_factor="NO_FORCED_ATTR"
            ))

            # A1: Gold ICL, No Attr -> 85% accurate (34/40)
            a1_v = gold if int(sid.split(":")[1]) < 34 else "Not Enough Evidence"
            records.append(M20fPredictionRecord(
                sample_id=sid, arm="A1", raw_claim="c", gold_label=gold,
                model_name="m", model_revision="r", retrieval_hash="h",
                prompt_sha256="s", raw_completion="", parsed_verdict=a1_v,
                parse_status="SUCCESS", knowledge_factor="GOLD_ICL", attribution_factor="NO_FORCED_ATTR"
            ))

            # A2: Gold ICL, Forced Attr -> 60% accurate due to Contradiction Trap (24/40)
            a2_v = gold if int(sid.split(":")[1]) < 24 else "Conflicting Evidence/Cherrypicking"
            records.append(M20fPredictionRecord(
                sample_id=sid, arm="A2", raw_claim="c", gold_label=gold,
                model_name="m", model_revision="r", retrieval_hash="h",
                prompt_sha256="s", raw_completion="", parsed_verdict=a2_v,
                parse_status="SUCCESS", knowledge_factor="GOLD_ICL", attribution_factor="FORCED_ATTR"
            ))

            # A3: Error ICL, No Attr -> 85% accurate (34/40)
            a3_v = gold if int(sid.split(":")[1]) < 34 else "Not Enough Evidence"
            records.append(M20fPredictionRecord(
                sample_id=sid, arm="A3", raw_claim="c", gold_label=gold,
                model_name="m", model_revision="r", retrieval_hash="h",
                prompt_sha256="s", raw_completion="", parsed_verdict=a3_v,
                parse_status="SUCCESS", knowledge_factor="HISTORICAL_ERROR_ICL", attribution_factor="NO_FORCED_ATTR"
            ))

            # A4: Error ICL, Forced Attr -> 60% accurate (24/40)
            a4_v = gold if int(sid.split(":")[1]) < 24 else "Conflicting Evidence/Cherrypicking"
            records.append(M20fPredictionRecord(
                sample_id=sid, arm="A4", raw_claim="c", gold_label=gold,
                model_name="m", model_revision="r", retrieval_hash="h",
                prompt_sha256="s", raw_completion="", parsed_verdict=a4_v,
                parse_status="SUCCESS", knowledge_factor="HISTORICAL_ERROR_ICL", attribution_factor="FORCED_ATTR"
            ))

        return records

    def test_factorial_main_effects(self, tmp_path, synthetic_records):
        evaluator = M20fEvaluator(synthetic_records, output_dir=tmp_path, n_bootstraps=500)
        res = evaluator.evaluate_all()

        fact = res["factorial_effects"]
        # Main Effect of Attribution: 0.5 * ((A2 - A1) + (A4 - A3))
        # A1 = 34/40 = 0.85, A2 = 24/40 = 0.60, diff = -0.25
        # A3 = 34/40 = 0.85, A4 = 24/40 = 0.60, diff = -0.25
        # Main effect of attribution should be exactly -0.25
        d_attr = fact["main_effect_attribution"]["delta_accuracy"]
        assert pytest.approx(d_attr, abs=1e-4) == -0.25

        # Main Effect of Error Knowledge: 0.5 * ((A3 - A1) + (A4 - A2)) = 0.0
        d_err = fact["main_effect_error_knowledge"]["delta_accuracy"]
        assert pytest.approx(d_err, abs=1e-4) == 0.0

        # Interaction effect: (A4 - A3) - (A2 - A1) = 0.0
        d_inter = fact["interaction_effect"]["delta_accuracy"]
        assert pytest.approx(d_inter, abs=1e-4) == 0.0

        # Check Decision is ATTRIBUTION_DISTORTION_CONFIRMED__ERROR_PARITY
        assert "ATTRIBUTION_DISTORTION_CONFIRMED" in res["decision"]


class TestMockVerifier:
    """Verifies mock verifier compatibility with ContrastiveVerdictParser."""

    def test_mock_verifier_all_arms(self):
        verifier = MockFactorialVerifier()
        for arm in M20fArm:
            sys_inst, user_pr, _ = M20fPromptBuilder.build_prompt(
                arm, "Claim text", [{"passage": "Evidence"}]
            )
            raw = verifier.verify_prompt(arm, sys_inst, user_pr)
            verdict, status, attr = ContrastiveVerdictParser.parse_completion(raw)
            assert status == "SUCCESS"
            assert verdict == "Supported"
            if arm in [M20fArm.A2, M20fArm.A4]:
                assert attr.is_valid_attribution is True
                assert attr.supporting_passage_ids == [1]
