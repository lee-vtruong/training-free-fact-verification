"""
Command-line interface for Milestone M1.9 Historical Outcome Evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tffv.experiments.m19_orchestrator import M19Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="M1.9 Historical Outcome Evaluation")
    parser.add_argument(
        "--m18e-dir",
        type=str,
        default="outputs/m18e-full-history-pf1000-v1",
        help="Path to M1.8e input artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/m19-historical-outcome-eval-v1",
        help="Path to M1.9 output directory",
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
        "--k",
        type=int,
        default=2,
        help="Number of historical exemplars per prompt (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--run-gpu",
        action="store_true",
        help="Enable production GPU verification with frozen Qwen3-8B BF16",
    )
    parser.add_argument(
        "--pilot-only",
        action="store_true",
        help="Run only sanity pilot on 30 internal development claims",
    )
    parser.add_argument(
        "--limit-claims",
        type=int,
        default=None,
        help="Limit number of evaluation claims (for smoke testing)",
    )

    args = parser.parse_args()

    orchestrator = M19Orchestrator(
        m18e_dir=args.m18e_dir,
        output_dir=args.output_dir,
        train_path=args.train_path,
        dev_path=args.dev_path,
        k=args.k,
        seed=args.seed,
    )

    orchestrator.run_all(
        run_gpu=args.run_gpu,
        pilot_only=args.pilot_only,
        limit_claims=args.limit_claims,
    )


if __name__ == "__main__":
    main()
