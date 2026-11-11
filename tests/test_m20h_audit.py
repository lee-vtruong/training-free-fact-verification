"""Unit tests for Milestone M2.0h Scientific Validation & Audit Pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from tffv.experiments.m20g_exemplars import CERTIFIED_EXEMPLAR_BANK, ExemplarManager
from tffv.experiments.m20g_schemas import (
    ExperimentPhase,
    KnowledgeComponent,
    M20gPredictionRecord,
)
from tffv.experiments.m20h_scientific_audit import M20hScientificAuditor


@pytest.fixture
def mock_audit_env(tmp_path: Path) -> Dict[str, Path]:
    m20g_dir = tmp_path / "m20g"
    m20g_dir.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "m20h"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create mock train.json with 10 claims
    train_file = tmp_path / "train.json"
    train_records = []
    classes = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]
    for i in range(1, 11):
        train_records.append({
            "claim_id": i,
            "claim": f"Mock claim {i}",
            "label": classes[i % 4],
            "group_id": f"grp-mock-{i % 3}",
        })
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_records, f)

    # 2. Create mock predictions.jsonl with paired arms
    pred_file = m20g_dir / "predictions.jsonl"
    mock_preds: List[M20gPredictionRecord] = []

    # E1 mock records: k=2, seed=1, error vs gold for 10 claims
    for i in range(1, 11):
        sid = f"train:{i}"
        gold = classes[i % 4]
        wrong = "Refuted" if gold != "Refuted" else "Supported"
        # error arm: 80% correct (8/10)
        pred_err = gold if i <= 8 else wrong
        rec_err = M20gPredictionRecord(
            sample_id=sid,
            experiment_phase=ExperimentPhase.E1_ROBUSTNESS.value,
            arm_name="E1_k2_s1_error",
            seed=1,
            k_shots=2,
            knowledge_component=KnowledgeComponent.K4_FULL_GROUNDED_ERROR.value,
            raw_claim=f"Mock claim {i}",
            gold_label=gold,
            model_name="mock-qwen3",
            model_revision="mock",
            prompt_sha256="mock_sha",
            raw_completion=f"[VERIFICATION REASONING]\nVerdict: {pred_err}",
            parsed_verdict=pred_err,
            parse_status="SUCCESS",
            prompt_tokens=1400,
            output_tokens=150,
            latency_seconds=3.0,
        )
        mock_preds.append(rec_err)

        # gold arm: 50% correct (5/10)
        pred_gold = gold if i <= 5 else wrong
        rec_gold = M20gPredictionRecord(
            sample_id=sid,
            experiment_phase=ExperimentPhase.E1_ROBUSTNESS.value,
            arm_name="E1_k2_s1_gold",
            seed=1,
            k_shots=2,
            knowledge_component=KnowledgeComponent.K2_GOLD_NEUTRAL_EXPL.value,
            raw_claim=f"Mock claim {i}",
            gold_label=gold,
            model_name="mock-qwen3",
            model_revision="mock",
            prompt_sha256="mock_sha",
            raw_completion=f"[VERIFICATION REASONING]\nVerdict: {pred_gold}",
            parsed_verdict=pred_gold,
            parse_status="SUCCESS",
            prompt_tokens=1350,
            output_tokens=140,
            latency_seconds=2.8,
        )
        mock_preds.append(rec_gold)

    # E2 mock records: K1, K2, K3, K4
    for arm_name, comp, p_tok, corr_count in [
        ("E2_K1_gold_only", KnowledgeComponent.K1_GOLD_ONLY, 1200, 4),
        ("E2_K2_neutral_expl", KnowledgeComponent.K2_GOLD_NEUTRAL_EXPL, 1350, 5),
        ("E2_K3_wrong_pred", KnowledgeComponent.K3_WRONG_PRED_GOLD_CORR, 1370, 7),
        ("E2_K4_grounded_error", KnowledgeComponent.K4_FULL_GROUNDED_ERROR, 1410, 8),
    ]:
        for i in range(1, 11):
            sid = f"train:{i}"
            gold = classes[i % 4]
            wrong = "Refuted" if gold != "Refuted" else "Supported"
            pred = gold if i <= corr_count else wrong
            mock_preds.append(M20gPredictionRecord(
                sample_id=sid,
                experiment_phase=ExperimentPhase.E2_ABLATION.value,
                arm_name=arm_name,
                seed=42,
                k_shots=2,
                knowledge_component=comp.value,
                raw_claim=f"Mock claim {i}",
                gold_label=gold,
                model_name="mock-qwen3",
                model_revision="mock",
                prompt_sha256="mock_sha",
                raw_completion=f"[VERIFICATION REASONING]\nVerdict: {pred}",
                parsed_verdict=pred,
                parse_status="SUCCESS",
                prompt_tokens=p_tok,
                output_tokens=140,
                latency_seconds=2.5,
            ))

    # E3 mock records: B0, A1, A3
    for arm_name, comp, corr_count in [
        ("E3_B0_baseline", KnowledgeComponent.K1_GOLD_ONLY, 5),
        ("E3_A1_gold", KnowledgeComponent.K2_GOLD_NEUTRAL_EXPL, 4),
        ("E3_A3_error", KnowledgeComponent.K4_FULL_GROUNDED_ERROR, 8),
    ]:
        for i in range(1, 11):
            sid = f"train:{i}"
            gold = classes[i % 4]
            wrong = "Refuted" if gold != "Refuted" else "Supported"
            pred = gold if i <= corr_count else wrong
            mock_preds.append(M20gPredictionRecord(
                sample_id=sid,
                experiment_phase=ExperimentPhase.E3_GENERALIZATION.value,
                arm_name=arm_name,
                seed=42,
                k_shots=2,
                knowledge_component=comp.value,
                raw_claim=f"Mock claim {i}",
                gold_label=gold,
                model_name="mock-qwen3",
                model_revision="mock",
                prompt_sha256="mock_sha",
                raw_completion=f"[VERIFICATION REASONING]\nVerdict: {pred}",
                parsed_verdict=pred,
                parse_status="SUCCESS",
                prompt_tokens=1300,
                output_tokens=140,
                latency_seconds=2.2,
            ))

    with open(pred_file, "w", encoding="utf-8") as f:
        for r in mock_preds:
            f.write(json.dumps(r.to_dict()) + "\n")

    return {
        "m20g_dir": m20g_dir,
        "out_dir": out_dir,
        "train_file": train_file,
    }


def test_m20h_phase_a_integrity(mock_audit_env: Dict[str, Path]) -> None:
    auditor = M20hScientificAuditor(
        m20g_dir=mock_audit_env["m20g_dir"],
        train_raw_path=mock_audit_env["train_file"],
        output_dir=mock_audit_env["out_dir"],
    )
    auditor.load_inputs()
    pa = auditor.audit_phase_a()

    assert pa["total_records"] == 90
    assert pa["is_strictly_unique"] is True
    assert pa["parser_mismatches"] == 0
    assert pa["gold_mismatches"] == 0
    assert "E1_k2_s1_error" in pa["per_arm_recomputed_metrics"]
    assert pa["per_arm_recomputed_metrics"]["E1_k2_s1_error"]["accuracy"] == 0.8


def test_m20h_phase_b_e1_paired_bootstrap(mock_audit_env: Dict[str, Path]) -> None:
    auditor = M20hScientificAuditor(
        m20g_dir=mock_audit_env["m20g_dir"],
        train_raw_path=mock_audit_env["train_file"],
        output_dir=mock_audit_env["out_dir"],
    )
    auditor.load_inputs()
    pb = auditor.audit_phase_b()

    assert "k2_s1" in pb["configurations"]
    cfg = pb["configurations"]["k2_s1"]
    assert cfg["error_accuracy"] == 0.8
    assert cfg["gold_accuracy"] == 0.5
    assert cfg["delta_accuracy"] == pytest.approx(0.3)
    assert len(cfg["bootstrap_95_ci"]) == 2
    assert cfg["contingency"]["net_benefit"] == 3


def test_m20h_phase_c_e2_component_ablation(mock_audit_env: Dict[str, Path]) -> None:
    auditor = M20hScientificAuditor(
        m20g_dir=mock_audit_env["m20g_dir"],
        train_raw_path=mock_audit_env["train_file"],
        output_dir=mock_audit_env["out_dir"],
    )
    auditor.load_inputs()
    pc = auditor.audit_phase_c()

    assert "K4_vs_K1" in pc["stepwise_differences"]
    diff = pc["stepwise_differences"]["K4_vs_K1"]
    assert diff["delta_accuracy"] == pytest.approx(0.4)
    assert pc["prompt_length_confounding"]["status"] == "NOT_IDENTIFIED"


def test_m20h_phase_d_k4_provenance(tmp_path: Path) -> None:
    auditor = M20hScientificAuditor(output_dir=tmp_path / "m20h_prov")
    pd = auditor.audit_phase_d()

    assert pd["total_exemplars"] == 10
    assert pd["all_identical_resolution_to_neutral"] is True
    # Provenance classifications should cover P1 and P2
    assert "P1_VALIDATED_GROUNDED" in pd["classification_counts"] or "P2_SOURCE_TRACEABLE_UNVALIDATED" in pd["classification_counts"]


def test_m20h_phase_e_e3_generalization(mock_audit_env: Dict[str, Path]) -> None:
    auditor = M20hScientificAuditor(
        m20g_dir=mock_audit_env["m20g_dir"],
        train_raw_path=mock_audit_env["train_file"],
        output_dir=mock_audit_env["out_dir"],
    )
    auditor.load_inputs()
    pe = auditor.audit_phase_e()

    assert pe["total_outer_claims"] == 10
    assert pe["a3_metrics"]["accuracy"] == 0.8
    assert pe["b0_metrics"]["accuracy"] == 0.5
    assert pe["a3_vs_b0"]["delta_accuracy"] == pytest.approx(0.3)
    assert pe["a3_vs_b0"]["net_benefit"] == 3


def test_m20h_end_to_end_report_generation(mock_audit_env: Dict[str, Path]) -> None:
    auditor = M20hScientificAuditor(
        m20g_dir=mock_audit_env["m20g_dir"],
        train_raw_path=mock_audit_env["train_file"],
        output_dir=mock_audit_env["out_dir"],
    )
    decision = auditor.run_full_audit()

    assert decision.determination in ["VALIDATED_WITH_LIMITATIONS", "PARTIALLY_SUPPORTED"]
    assert (mock_audit_env["out_dir"] / "PREFLIGHT_REPORT.md").exists()
    assert (mock_audit_env["out_dir"] / "E1_ROBUSTNESS_AUDIT.md").exists()
    assert (mock_audit_env["out_dir"] / "E2_COMPONENT_AUDIT.md").exists()
    assert (mock_audit_env["out_dir"] / "K4_PROVENANCE_AUDIT.md").exists()
    assert (mock_audit_env["out_dir"] / "E3_SOURCE_FOLD_AUDIT.md").exists()
    assert (mock_audit_env["out_dir"] / "MILESTONE_RECONCILIATION.md").exists()
    assert (mock_audit_env["out_dir"] / "DECISION_REPORT.md").exists()
    assert (mock_audit_env["out_dir"] / "run_receipt.json").exists()
