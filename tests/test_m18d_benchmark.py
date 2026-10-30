"""Unit and Integration Tests for Milestone M1.8d Automated Retrieval Benchmark."""
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from tffv.retrieval.m18d_arms import AVAILABLE_ARMS, ArmRunner, ArmSkippedError
from tffv.retrieval.m18d_manifest import build_stratified_samples, generate_m18d_manifests
from tffv.retrieval.m18d_orchestrator import (
    DEV_OVERLAP_SIDS,
    INVALID_SIDS,
    BenchmarkOrchestrator,
)
from tffv.retrieval.m18d_stats import (
    holm_bonferroni_correction,
    mcnemar_test,
    paired_bootstrap_test,
)


def test_manifest_disjointness_and_exclusions():
    """Verify that pilot (N=50) and holdout (N=200) manifests are disjoint and exclude contamination."""
    pilot_p = Path("configs/m18d_manifests/pilot_manifest.json")
    holdout_p = Path("configs/m18d_manifests/holdout_manifest.json")

    assert pilot_p.is_file(), "Pilot manifest missing"
    assert holdout_p.is_file(), "Holdout manifest missing"

    pilot_data = json.loads(pilot_p.read_text(encoding="utf-8"))
    holdout_data = json.loads(holdout_p.read_text(encoding="utf-8"))

    pilot_sids = set(c["sample_id"] for c in pilot_data.get("claims", []))
    holdout_sids = set(c["sample_id"] for c in holdout_data.get("claims", []))

    assert len(pilot_sids) == 50
    assert len(holdout_sids) == 200

    # 1. Strictly non-overlapping
    overlap = pilot_sids.intersection(holdout_sids)
    assert len(overlap) == 0, f"Pilot and holdout overlap on: {overlap}"

    # 2. Exclude dev overlaps
    dev_overlap = (pilot_sids | holdout_sids).intersection(DEV_OVERLAP_SIDS)
    assert len(dev_overlap) == 0, f"Contaminated with dev overlap: {dev_overlap}"

    # 3. Exclude invalid sample train:1948
    assert "train:1948" not in pilot_sids
    assert "train:1948" not in holdout_sids

    # 4. Check all 4 labels are present
    assert len(pilot_data["label_distribution"]) == 4
    assert len(holdout_data["label_distribution"]) == 4


def test_mcnemar_exact_binomial():
    """Verify McNemar test calculation on known discordant pairs."""
    gold = {"c0": "A", "c1": "B", "c2": "C", "c3": "D"}
    ref_preds = {"c0": "A", "c1": "B", "c2": "C", "c3": "D"}       # 4/4 correct
    cand_preds = {"c0": "A", "c1": "B", "c2": "Wrong", "c3": "D"}  # 3/4 correct, 1 discordant (c=1, b=0)

    res = mcnemar_test(candidate_preds=cand_preds, reference_preds=ref_preds, gold_labels=gold)
    assert res["contingency_table"]["reference_only_correct"] == 1
    assert res["contingency_table"]["candidate_only_correct"] == 0
    assert res["discordant_verdicts_count"] == 1
    assert res["p_value"] == pytest.approx(1.0, abs=0.1)

    # Identical predictions should yield p_value = 1.0
    res_ident = mcnemar_test(candidate_preds=ref_preds, reference_preds=ref_preds, gold_labels=gold)
    assert res_ident["p_value"] == 1.0
    assert res_ident["discordant_verdicts_count"] == 0


def test_paired_bootstrap_test():
    """Verify paired bootstrap produces valid confidence interval bounds."""
    gold = {f"c{i}": ("Supported" if i < 20 else "Refuted") for i in range(40)}
    ref_preds = {f"c{i}": gold[f"c{i}"] if i % 5 != 0 else "Wrong" for i in range(40)}
    cand_preds = {f"c{i}": gold[f"c{i}"] if i % 3 != 0 else "Wrong" for i in range(40)}

    boot = paired_bootstrap_test(
        candidate_preds=cand_preds,
        reference_preds=ref_preds,
        gold_labels=gold,
        n_bootstrap=200,
        seed=42,
    )
    assert "delta_accuracy" in boot
    assert "delta_macro_f1" in boot
    assert len(boot["delta_accuracy_ci_95"]) == 2
    assert boot["delta_accuracy_ci_95"][0] <= boot["delta_accuracy_ci_95"][1]
    assert len(boot["delta_macro_f1_ci_95"]) == 2
    assert boot["delta_macro_f1_ci_95"][0] <= boot["delta_macro_f1_ci_95"][1]


def test_holm_bonferroni_correction():
    """Verify Holm-Bonferroni step-down family-wise error rate control."""
    p_dict = {"arm1": 0.01, "arm2": 0.04, "arm3": 0.03, "arm4": 0.005}
    corrected = holm_bonferroni_correction(p_dict)
    assert len(corrected) == len(p_dict)
    for arm, item in corrected.items():
        assert item["adjusted_p_value"] >= item["raw_p_value"]


def test_arm_runner_synthetic():
    """Verify ArmRunner executes under synthetic mode producing valid Top-8 hits."""
    claim = {"sample_id": "train:0", "claim": "Sample test claim text"}
    documents = [
        {
            "document_id": f"doc_{i}",
            "source_url": f"http://example.com/{i}",
            "source_kind": "web",
            "retrieved_at": "2026-10-30T00:00:00Z",
            "acquisition_id": "acq_0",
            "text": f"Passage document text {i} with factual content and multiple words for segmentation",
        }
        for i in range(15)
    ]
    cfg = {
        "candidate_depth": 10,
        "fusion_depth": 10,
        "rerank_depth": 8,
        "bm25": {"k1": 1.2, "b": 0.75},
        "segmentation": {"words": 5, "overlap": 1},
    }

    # Test R0
    runner_r0 = ArmRunner(
        arm_id="R0_CANONICAL_R4",
        retrieval_cfg=cfg,
        reranker=None,
        synthetic=True,
    )
    rec_r0, timings_r0 = runner_r0.retrieve(claim, documents)
    assert rec_r0["sample_id"] == "train:0"
    assert len(rec_r0["evidence"]) == 8
    assert "total_wall" in timings_r0
    runner_r0.close()

    # Test R5 (BM25 Prefilter)
    runner_r5 = ArmRunner(
        arm_id="R5_BM25_PF2000",
        retrieval_cfg=cfg,
        reranker=None,
        synthetic=True,
    )
    rec_r5, timings_r5 = runner_r5.retrieve(claim, documents)
    assert rec_r5["sample_id"] == "train:0"
    assert len(rec_r5["evidence"]) == 8
    runner_r5.close()

    # Test R8 skipped
    runner_r8 = ArmRunner(
        arm_id="R8_SPLADE_HYBRID",
        retrieval_cfg=cfg,
        reranker=None,
        synthetic=False,
    )
    assert not runner_r8.available
    with pytest.raises(ArmSkippedError):
        runner_r8.retrieve(claim, documents)


def test_orchestrator_synthetic_end_to_end(tmp_path):
    """Verify full synthetic end-to-end execution of BenchmarkOrchestrator."""
    # Build mini test dataset with 4 classes
    labels = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]
    raw_train = [
        {"claim": f"Test claim number {i}", "label": labels[i % len(labels)]}
        for i in range(10)
    ]
    train_file = tmp_path / "train.json"
    train_file.write_text(json.dumps(raw_train), encoding="utf-8")

    pilot_manifest = {
        "claims": [
            {"sample_id": "train:0", "claim": "Test claim 0", "gold_label": "Supported"},
            {"sample_id": "train:1", "claim": "Test claim 1", "gold_label": "Refuted"},
        ],
        "sample_ids": ["train:0", "train:1"],
        "count": 2,
        "label_distribution": {"Supported": 1, "Refuted": 1},
    }
    pilot_m_file = tmp_path / "pilot_manifest.json"
    pilot_m_file.write_text(json.dumps(pilot_manifest), encoding="utf-8")

    holdout_manifest = {
        "claims": [
            {"sample_id": "train:2", "claim": "Test claim 2", "gold_label": "Not Enough Evidence"},
            {"sample_id": "train:3", "claim": "Test claim 3", "gold_label": "Conflicting Evidence/Cherrypicking"},
        ],
        "sample_ids": ["train:2", "train:3"],
        "count": 2,
        "label_distribution": {"Not Enough Evidence": 1, "Conflicting Evidence/Cherrypicking": 1},
    }
    holdout_m_file = tmp_path / "holdout_manifest.json"
    holdout_m_file.write_text(json.dumps(holdout_manifest), encoding="utf-8")

    cfg_path = Path("configs/m161_source_batch25.json")
    acq_plan_path = Path("configs/m161_acquisition_plan.json")

    bench_out = tmp_path / "bench_out"

    orchestrator = BenchmarkOrchestrator(
        config_path=cfg_path,
        acq_plan_path=acq_plan_path,
        raw_train_path=train_file,
        pilot_manifest_path=pilot_m_file,
        holdout_manifest_path=holdout_m_file,
        output_dir=bench_out,
        synthetic=True,
        skip_downstream=False,
        resume=False,
        active_arms=["R0_CANONICAL_R4", "R1_QWEN3_DENSE_OPT", "R5_BM25_PF2000"],
    )

    # Run Stage 1: Preflight
    stage1 = orchestrator.run_stage1_preflight()
    assert stage1["status"] == "PASSED"

    # Pre-populate candidate data mock for the 4 test claims to avoid network acquisition
    dummy_docs = [
        {
            "document_id": f"doc_{i}",
            "source_url": f"http://test.org/{i}",
            "source_kind": "web",
            "retrieved_at": "2026-10-30T00:00:00Z",
            "acquisition_id": "acq_0",
            "text": f"Dummy passage text {i} for fact checking purposes with sufficient tokens.",
        }
        for i in range(15)
    ]
    for sid in ["train:0", "train:1", "train:2", "train:3"]:
        cand_dir = bench_out / "work_claims" / sid.replace(":", "_") / "candidates"
        cand_dir.mkdir(parents=True, exist_ok=True)
        manifest_payload = {
            "dataset": "AVeriTeC",
            "split": "train",
            "sample_id": sid,
            "candidates_count": len(dummy_docs),
            "documents": dummy_docs,
        }
        (cand_dir / "candidate_manifest.json").write_text(json.dumps(manifest_payload), encoding="utf-8")

    # Run Stage 2: Pilot
    pilot_res = orchestrator.run_stage2_pilot()
    assert "R0_CANONICAL_R4" in pilot_res

    # Run Stage 3: Pareto Selection
    selected = orchestrator.run_stage3_pareto_selection()
    assert "R0_CANONICAL_R4" in selected

    # Run Stage 4: Holdout
    holdout_res = orchestrator.run_stage4_holdout(selected)
    assert len(holdout_res) > 0

    # Run Stage 5: Verification
    preds = orchestrator.run_stage5_verification(selected, holdout_res)
    assert len(preds) > 0

    # Run Stage 6: Decision and Report
    decision = orchestrator.run_stage6_decision_and_report(selected, holdout_res, preds)
    assert "winner_arm" in decision
    assert (bench_out / "DECISION_REPORT.md").is_file()
    assert (bench_out / "decision_manifest.json").is_file()
