"""
Command-line interface for Milestone M1.8h Diagnostic & Recoverability Suite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tffv.experiments.m18h_diagnostic import M18hDiagnosticOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="M1.8h Rejection Root-Cause Analysis & Recoverability Audit")
    parser.add_argument(
        "--m18e-dir",
        type=str,
        default="outputs/m18e-full-history-pf1000-v1",
        help="Path to M1.8e input artifacts",
    )
    parser.add_argument(
        "--m18g-dir",
        type=str,
        default="outputs/m18g-correction-recovery-v1",
        help="Path to M1.8g recovery artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/m18h-rejection-diagnostic-v1",
        help="Path to M1.8h diagnostic output directory",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default="data/raw/averitec/train.json",
        help="Path to raw AVeriTeC train.json",
    )
    parser.add_argument(
        "--dev-path",
        type=str,
        default="data/raw/averitec/dev.json",
        help="Path to raw AVeriTeC dev.json",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=150,
        help="Stratified sample size from rejected pool (default: 150)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42)",
    )
    parser.add_argument(
        "--run-gpu-intervention",
        action="store_true",
        help="Flag to enable GPU intervention experiment (default: False, CPU simulation)",
    )

    args = parser.parse_args()

    orchestrator = M18hDiagnosticOrchestrator(
        m18e_dir=args.m18e_dir,
        m18g_dir=args.m18g_dir,
        output_dir=args.output_dir,
        train_path=args.train_path,
        dev_path=args.dev_path,
        sample_size=args.sample_size,
        random_seed=args.seed,
    )

    orchestrator.run_all(run_gpu_intervention=args.run_gpu_intervention)


if __name__ == "__main__":
    main()
