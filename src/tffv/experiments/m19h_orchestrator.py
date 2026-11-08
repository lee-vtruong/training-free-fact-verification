"""Milestone M1.9h: End-to-End CPU Diagnostics & Feasibility Pipeline Orchestrator."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

from tffv.experiments.m19h_figures import generate_all_figures
from tffv.experiments.m19h_preflight import run_preflight_audit
from tffv.experiments.m19h_reporting import generate_reports
from tffv.experiments.m19h_risk import run_risk_study
from tffv.experiments.m19h_routing import simulate_selective_routing
from tffv.experiments.m19h_selector import analyze_selector
from tffv.experiments.m19h_transitions import analyze_transitions


class M19hOrchestrator:
    def __init__(
        self,
        input_dir: str | Path = "outputs/m19g-evidence-complete-v1",
        output_dir: str | Path = "outputs/m19h-error-diagnostics-v1",
        tolerance: float = 1e-3,
        n_bootstraps: int = 5000,
        seed: int = 42,
        skip_metric_reproduction: bool = False,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.tolerance = tolerance
        self.n_bootstraps = n_bootstraps
        self.seed = seed
        self.skip_metric_reproduction = skip_metric_reproduction
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self) -> Dict[str, Any]:
        t0 = time.time()
        print("======================================================================")
        print(" Milestone M1.9h: Historical Error Diagnostics & Selective Feasibility")
        print(f" Input Directory:  {self.input_dir}")
        print(f" Output Directory: {self.output_dir}")
        print(" Mode: CPU-Only Deterministic Analysis")
        print("======================================================================")

        # Step 1: Mandatory Integrity Preflight
        print("\n[M1.9h] >>> Step 1: Running Mandatory Integrity Preflight...")
        claims_map, preflight_summary = run_preflight_audit(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            tolerance=self.tolerance,
            skip_metric_reproduction=self.skip_metric_reproduction,
        )
        print(f"[M1.9h] Preflight PASSED: {preflight_summary['total_claims']} claims verified across 6 arms.")

        # Step 2: Per-Claim Error Transitions & Class Diagnostics
        print("\n[M1.9h] >>> Step 2: Analyzing Error Transitions & Class Behavior...")
        transition_stats, transition_matrices, class_analysis = analyze_transitions(
            claims_map=claims_map,
            output_dir=self.output_dir,
        )
        h2_v_b1 = transition_stats.get("H2_vs_B1")
        if h2_v_b1:
            print(f"[M1.9h] H2 vs B1: {h2_v_b1.corrected} corrected, {h2_v_b1.harmed} harmed (Net: {h2_v_b1.net_benefit}, McNemar p={h2_v_b1.mcnemar_p_value:.5f}).")

        # Step 3: Historical Selector Diagnostics
        print("\n[M1.9h] >>> Step 3: Performing Selector & Similarity Diagnostics...")
        diag_rows, selector_summary = analyze_selector(
            claims_map=claims_map,
            output_dir=self.output_dir,
        )
        print(f"[M1.9h] Selector diagnostics complete: Pearson r(sim, B0_correct) = {selector_summary['pearson_sim_vs_b0_correct']:.4f}.")

        # Step 4: Historical Error Risk Estimation Feasibility
        print(f"\n[M1.9h] >>> Step 4: Historical Error Risk Estimation ({self.n_bootstraps} bootstraps)...")
        risk_metrics, risk_features, risk_curves = run_risk_study(
            claims_map=claims_map,
            output_dir=self.output_dir,
            n_bootstraps=self.n_bootstraps,
            seed=self.seed,
        )
        sim_pred = risk_metrics["predictors"].get("Max Sim to Historical Errors", {})
        print(f"[M1.9h] Error Detection AUROC (Max Sim Wrong): {sim_pred.get('auroc', 0.0):.4f} (95% CI: {sim_pred.get('auroc_95_ci', [])}).")

        # Step 5: Selective Intervention Simulation
        print("\n[M1.9h] >>> Step 5: Simulating Selective Intervention Routing Policies...")
        routing_replay = simulate_selective_routing(
            claims_map=claims_map,
            risk_features=risk_features,
            output_dir=self.output_dir,
        )
        print(f"[M1.9h] Routing simulation complete: {len(routing_replay)} budget configurations evaluated.")

        # Step 6: Generate Publication Figures
        print("\n[M1.9h] >>> Step 6: Generating Diagnostic Figures...")
        figs = generate_all_figures(
            transition_stats=transition_stats,
            class_analysis_rows=class_analysis,
            risk_metrics=risk_metrics,
            risk_curves=risk_curves,
            routing_replay=routing_replay,
            output_dir=self.output_dir,
        )
        print(f"[M1.9h] Generated {len(figs)} figures in outputs/figures/.")

        # Step 7: Generate Final Reports & Run Receipt
        print("\n[M1.9h] >>> Step 7: Compiling Final Reports & Proposal...")
        reports = generate_reports(
            claims_map=claims_map,
            transition_stats=transition_stats,
            class_analysis_rows=class_analysis,
            selector_summary=selector_summary,
            risk_metrics=risk_metrics,
            risk_curves=risk_curves,
            routing_replay=routing_replay,
            output_dir=self.output_dir,
            t0=t0,
        )
        elapsed = time.time() - t0
        print(f"[M1.9h] Reports generated: {list(reports.keys())}")
        print(f"======================================================================")
        print(f" [M1.9h SUCCESS] Diagnostics complete in {elapsed:.2f}s!")
        print(f" Deliverables written to: {self.output_dir.resolve()}")
        print(f"======================================================================")

        return {
            "status": "GO",
            "elapsed_seconds": elapsed,
            "total_claims": len(claims_map),
            "output_dir": str(self.output_dir),
            "reports": list(reports.keys()),
            "figures": figs,
        }
