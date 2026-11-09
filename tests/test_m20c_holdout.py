"""Unit Tests: Milestone M2.0c — Independent Holdout Evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tffv.experiments.m20b_schemas import M20bPredictionRecord
from tffv.experiments.m20c_audit import M20bPilotAuditor
from tffv.experiments.m20c_evaluator import M20cEvaluator, holm_bonferroni_correction
from tffv.experiments.m20c_holdout import IndependentHoldoutBuilder, jaccard_similarity, norm_text
from tffv.experiments.m20c_preflight import M20cPreflightAuditor, PreflightCheckError


def test_pilot_auditor_execution(tmp_path: Path):
    """Test Phase A Pilot Auditor recomputes metrics and validates transition conservation."""
    m20b_dir = tmp_path / "m20b"
    out_dir = tmp_path / "m20c"
    m20b_dir.mkdir()
    out_dir.mkdir()

    # Generate synthetic 50 claims across 6 arms matching pilot
    arms = ["B0", "G2", "E1", "B1", "B2", "B3"]
    preds = []
    traces = []
    audits = []

    for idx in range(50):
        sid = f"train:{idx}"
        gold = "Supported" if idx % 2 == 0 else "Refuted"
        for arm in arms:
            is_corr = (idx < 25)  # 25 correct for baseline
            pred = gold if is_corr else ("Refuted" if gold == "Supported" else "Supported")

            # B1 and B2 transitions: make 1 corrected, 1 harmed
            if arm == "B1" and idx == 25:  # corrected by B1
                pred = gold
                is_corr = True
            elif arm == "B2" and idx == 26:  # harmed by B1, correct in B2
                pred = gold
                is_corr = True

            dec = "INTERVENE" if idx < 27 else "NO_INTERVENTION"
            prompt_sha = f"sha_common_{idx}" if dec == "NO_INTERVENTION" else f"sha_{arm}_{idx}"

            preds.append({
                "sample_id": sid,
                "arm": arm,
                "claim": f"Claim {idx}",
                "gold_verdict": gold,
                "predicted_verdict": pred,
                "is_correct": is_corr,
                "is_parseable": True,
                "latency_seconds": 1.0,
                "prompt_sha256": prompt_sha,
                "intervention_decision": dec,
            })

            if arm == "B1":
                traces.append({
                    "sample_id": sid,
                    "arm": "B1",
                    "intervention_decision": dec,
                    "relevance_score": 0.5,
                    "grounding_overlap_score": 0.2 if dec == "INTERVENE" else 0.05,
                    "token_budget": 50,
                })
                audits.append({
                    "sample_id": sid,
                    "evidence_grounding_verified": (dec == "INTERVENE"),
                    "filter_rationale": "GROUNDING_VERIFIED" if dec == "INTERVENE" else "BELOW_THRESHOLD",
                })

    with open(m20b_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in preds:
            f.write(json.dumps(r) + "\n")
    with open(m20b_dir / "selection_traces.jsonl", "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
    with open(m20b_dir / "grounding_audit.json", "w", encoding="utf-8") as f:
        json.dump(audits, f)
    with open(m20b_dir / "BASELINE_AUDIT.md", "w", encoding="utf-8") as f:
        f.write("GO (Contract Discrepancy Resolved)\nCanonical Baseline Contract Locked\n")

    auditor = M20bPilotAuditor(m20b_dir=m20b_dir, output_dir=out_dir)
    res = auditor.run_audit()

    assert res["status"] == "GO"
    assert res["total_claims"] == 50
    assert res["corrected"] == 1
    assert res["harmed"] == 1
    assert res["net_benefit"] == 0
    assert res["intervene_count"] == 27
    assert res["no_intervene_count"] == 23
    assert res["no_intervene_prompt_match"] == 23
    assert (out_dir / "PILOT_AUDIT.md").exists()


def test_independent_holdout_builder(tmp_path: Path):
    """Test Phase B Holdout Builder quarantines pilot claims and near-duplicates."""
    out_dir = tmp_path / "holdout_out"
    m20b_dir = tmp_path / "m20b_manifests"
    m20b_dir.mkdir()

    # 60 claims: first 10 pilot, remaining 50 holdout candidates
    claims = []
    pilot_ids = [f"train:{i}" for i in range(10)]
    for i in range(60):
        claims.append({
            "sample_id": f"train:{i}",
            "claim": f"Factual claim number {i} regarding science.",
            "label": "Supported",
        })

    with open(m20b_dir / "development_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"pilot_50_ids": pilot_ids, "claims": claims}, f)
    with open(m20b_dir / "memory_eligibility_manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "eligible_sample_ids": [f"train:{i}" for i in range(100, 200)],
            "quarantined_claims": [{"sample_id": "train:1948", "reason": "OVERLAP"}],
        }, f)

    builder = IndependentHoldoutBuilder(
        m20b_dev_manifest=m20b_dir / "development_manifest.json",
        m20b_mem_manifest=m20b_dir / "memory_eligibility_manifest.json",
        output_dir=out_dir,
    )
    res = builder.build_holdout()

    assert res["status"] == "GO"
    assert res["total_clean_holdout"] == 50
    assert (out_dir / "holdout_manifest.json").exists()
    assert (out_dir / "memory_eligibility_manifest.json").exists()
    assert (out_dir / "partition_audit.md").exists()

    with open(out_dir / "holdout_manifest.json") as f:
        h_data = json.load(f)
    holdout_ids = set(c["sample_id"] for c in h_data["claims"])
    assert holdout_ids.isdisjoint(set(pilot_ids))


def test_m20c_preflight_leakage_detection(tmp_path: Path):
    """Test Preflight raises error when leakage or overlap is detected."""
    out_dir = tmp_path / "preflight_out"
    out_dir.mkdir()

    # Construct conflicting manifests
    holdout_path = out_dir / "holdout.json"
    memory_path = out_dir / "memory.json"
    receipts_path = out_dir / "receipts.jsonl"

    with open(holdout_path, "w") as f:
        json.dump({"claims": [{"sample_id": "train:10", "claim": "Overlap claim"}]}, f)
    with open(memory_path, "w") as f:
        # train:10 is in both holdout and memory!
        json.dump({"eligible_sample_ids": ["train:10", "train:11"]}, f)
    receipts_path.touch()

    auditor = M20cPreflightAuditor(
        holdout_manifest_path=holdout_path,
        memory_manifest_path=memory_path,
        retrieval_receipts_path=receipts_path,
        output_dir=out_dir,
    )

    with pytest.raises(PreflightCheckError):
        auditor.run_preflight(fail_closed=True)


def test_m20c_evaluator_and_holm_bonferroni(tmp_path: Path):
    """Test M20c evaluator computes Holm-Bonferroni correction and subgroup metrics."""
    records = []
    # 4 claims, 6 arms
    arms = ["B0", "G2", "E1", "B1", "B2", "B3"]
    for i in range(10):
        gold = "Supported" if i % 2 == 0 else "Refuted"
        for arm in arms:
            pred = gold if (arm == "B1" or i < 5) else ("Refuted" if gold == "Supported" else "Supported")
            dec = "INTERVENE" if i < 6 else "NO_INTERVENTION"
            records.append(
                M20bPredictionRecord(
                    sample_id=f"c:{i}",
                    arm=arm,
                    claim=f"Claim {i}",
                    gold_verdict=gold,
                    predicted_verdict=pred,
                    raw_completion="Verdict: ...",
                    is_parseable=True,
                    is_correct=(pred == gold),
                    latency_seconds=1.0,
                    prompt_sha256="",
                    intervention_decision=dec,
                )
            )

    evaluator = M20cEvaluator(records=records, output_dir=tmp_path / "eval", n_bootstraps=100)
    res = evaluator.evaluate_all()

    assert "decision" in res
    assert "B1_vs_B2" in res["statistical_tests"]
    assert "INTERVENE" in res["subgroup_metrics"]
    assert "NO_INTERVENTION" in res["subgroup_metrics"]
    assert (tmp_path / "eval" / "DECISION_REPORT.md").exists()
    assert (tmp_path / "eval" / "FINAL_REPORT.md").exists()
    assert (tmp_path / "eval" / "paired_transitions.csv").exists()

    # Test Holm-Bonferroni function directly
    raw_p = [("test1", 0.01), ("test2", 0.04), ("test3", 0.001)]
    hb = holm_bonferroni_correction(raw_p)
    assert hb["test3"]["rank"] == 1
    assert hb["test3"]["is_significant_holm"] is True
