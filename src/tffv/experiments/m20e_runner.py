"""Milestone M2.0e: Complete Diagnostic & Feasibility Suite Runner.

Executes CPU-only forensic audit and selective contrastive feasibility analysis:
1. Phase A: Baseline Integrity & Reproducibility Audit.
2. Phase B: Contrastive Mechanism & Transition Diagnostics.
3. Phase C: Inference-Time Risk Features Construction.
4. Phase D & E: Selective Routing Replay, Bootstrap Significance & Figure Generation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure src is on sys.path
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tffv.experiments.m20e_mechanism_analysis import ContrastiveMechanismAnalyzer
from tffv.experiments.m20e_repro_audit import BaselineReproducibilityAuditor
from tffv.experiments.m20e_risk_features import RiskFeatureExtractor
from tffv.experiments.m20e_routing_replay import SelectiveRoutingEvaluator


def run_all_diagnostics(
    m20c_dir: str = "outputs/m20c-independent-holdout-v1",
    m20d_dir: str = "outputs/m20d-contrastive-verification-v1",
    output_dir: str = "outputs/m20e-selective-feasibility-v1",
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> None:
    t_start = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Milestone M2.0e: Baseline Reproducibility & Selective Contrastive Feasibility")
    print(f" Output Directory: {out_path}")
    print(" Execution Mode: CPU-Only Forensic Diagnostics")
    print("=" * 70)

    # Step 1: Phase A Audit
    print("\n[M2.0e] >>> Step 1: Executing Baseline Integrity & Reproducibility Audit...")
    repro_auditor = BaselineReproducibilityAuditor(
        m20c_dir=m20c_dir,
        m20d_dir=m20d_dir,
        output_dir=out_path,
    )
    res_a = repro_auditor.run_audit()
    print(f"[M2.0e] Step 1 Complete: Status = {res_a['status']} ({res_a['determination']}).")

    # Step 2: Phase B Diagnostics
    print("\n[M2.0e] >>> Step 2: Executing Contrastive Mechanism & Transition Diagnostics...")
    mech_analyzer = ContrastiveMechanismAnalyzer(
        m20d_dir=m20d_dir,
        output_dir=out_path,
    )
    res_b = mech_analyzer.run_analysis()
    print(f"[M2.0e] Step 2 Complete: Benefit = {res_b['benefit_count']} claims, Harm = {res_b['harm_count']} claims.")

    # Step 3: Phase C Risk Features
    print("\n[M2.0e] >>> Step 3: Extracting Inference-Time Risk Features & AUROC...")
    feat_extractor = RiskFeatureExtractor(
        m20d_dir=m20d_dir,
        output_dir=out_path,
    )
    res_c = feat_extractor.extract_features()
    print(f"[M2.0e] Step 3 Complete: Features sealed in {res_c['csv_path']}.")

    # Step 4: Phase D & E Selective Routing Replay
    print("\n[M2.0e] >>> Step 4: Evaluating Selective Routing Replay & Generating Figures...")
    routing_evaluator = SelectiveRoutingEvaluator(
        risk_features_path=res_c["csv_path"],
        m20d_dir=m20d_dir,
        output_dir=out_path,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )
    res_d = routing_evaluator.run_replay_evaluation()
    print(f"[M2.0e] Step 4 Complete: Determination = {res_d['decision']}.")

    elapsed = round(time.time() - t_start, 2)
    print("\n" + "=" * 70)
    print(f" Milestone M2.0e Diagnostics Complete in {elapsed}s!")
    print(f" Empirical Determination: {res_d['decision']}")
    print(f" Deliverables sealed in: {out_path}")
    print("=" * 70)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M2.0e: Baseline Reproducibility & Feasibility Runner.")
    parser.add_argument("--m20c-dir", default="outputs/m20c-independent-holdout-v1")
    parser.add_argument("--m20d-dir", default="outputs/m20d-contrastive-verification-v1")
    parser.add_argument("--output-dir", default="outputs/m20e-selective-feasibility-v1")
    parser.add_argument("--n-bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    run_all_diagnostics(
        m20c_dir=args.m20c_dir,
        m20d_dir=args.m20d_dir,
        output_dir=args.output_dir,
        n_bootstraps=args.n_bootstraps,
        seed=args.seed,
    )
