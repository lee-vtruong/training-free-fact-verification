"""CLI Entrypoint for Milestone M1.8g Substantive Correction Recovery & Memory Revalidation."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tffv.experiments.m18g_recovery import M18gRecoveryOrchestrator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m18e-dir",
        default=str(ROOT / "outputs" / "m18e-full-history-pf1000-v1"),
        help="Path to M1.8e input directory containing base predictions and retrieval receipts",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "m18g-correction-recovery-v1"),
        help="Path to new versioned M1.8g output directory",
    )
    parser.add_argument(
        "--train-json",
        default=None,
        help="Path to raw AVeriTeC train.json",
    )
    parser.add_argument(
        "--dev-json",
        default=None,
        help="Path to raw AVeriTeC dev.json",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run in offline synthetic mode without GPU models",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of claims to process (bounded run / testing)",
    )

    args = parser.parse_args()

    orchestrator = M18gRecoveryOrchestrator(
        m18e_input_dir=args.m18e_dir,
        output_dir=args.output_dir,
        train_path=args.train_json,
        dev_path=args.dev_json,
        synthetic=args.synthetic,
    )

    orchestrator.run_all(limit=args.limit)


if __name__ == "__main__":
    main()
