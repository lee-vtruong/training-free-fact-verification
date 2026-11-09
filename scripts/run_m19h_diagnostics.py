"""CLI entrypoint for Milestone M1.9h CPU Diagnostics Pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tffv.experiments.m19h_orchestrator import M19hOrchestrator


def main():
    parser = argparse.ArgumentParser(description="Milestone M1.9h CPU Diagnostics & Feasibility Pipeline")
    parser.add_argument("--input-dir", type=str, default="outputs/m19g-evidence-complete-v1", help="M1.9g output directory")
    parser.add_argument("--output-dir", type=str, default="outputs/m19h-error-diagnostics-v1", help="M1.9h diagnostics output directory")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Metric reproduction numeric tolerance")
    parser.add_argument("--n-bootstraps", type=int, default=5000, help="Number of bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrapping")
    parser.add_argument("--skip-metric-reproduction", action="store_true", help="Skip metric reproduction (test mode)")
    args = parser.parse_args()

    orchestrator = M19hOrchestrator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tolerance=args.tolerance,
        n_bootstraps=args.n_bootstraps,
        seed=args.seed,
        skip_metric_reproduction=args.skip_metric_reproduction,
    )
    result = orchestrator.run_all()
    if result.get("status") == "GO":
        sys.exit(0)
    else:
        print("[M1.9h] Pipeline finished with non-GO status.")
        sys.exit(1)


if __name__ == "__main__":
    main()
