"""CLI Entrypoint for Milestone M1.8f Correction Yield, Judge Rejection & Memory Coverage Audit."""
import argparse
import sys
from pathlib import Path

# Add src to pythonpath
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tffv.experiments.m18f_audit import M18fAuditor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=str(ROOT / "outputs" / "m18e-full-history-pf1000-v1"),
        help="Path to M1.8e input directory containing artifacts",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "m18f-correction-yield-audit-v1"),
        help="Path to directory where M1.8f audit deliverables will be written",
    )
    parser.add_argument(
        "--train-json",
        default=None,
        help="Path to raw AVeriTeC train.json (optional, auto-detected if omitted)",
    )
    parser.add_argument(
        "--dev-json",
        default=None,
        help="Path to raw AVeriTeC dev.json (optional, auto-detected if omitted)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail closed if any non-critical input artifact is missing",
    )

    args = parser.parse_args()

    auditor = M18fAuditor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_path=args.train_json,
        dev_path=args.dev_json,
        strict=args.strict,
    )

    summary = auditor.run_audit()
    print("\n" + "=" * 70)
    print(" Milestone M1.8f Audit Finished Successfully")
    print(f" Verdict: {summary['verdict']}")
    print(f" Deliverables written to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
