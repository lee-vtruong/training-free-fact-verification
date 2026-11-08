"""CLI entrypoint for Milestone M1.9e Integrity, Label-Mapping & Provenance Audit."""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tffv.experiments.m19e_audit import M19eAuditor


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone M1.9e Critical Integrity Audit")
    parser.add_argument("--m19-dir", type=str, default="outputs/m19-historical-outcome-eval-v1")
    parser.add_argument("--output-dir", type=str, default="outputs/m19e-integrity-audit-v1")
    parser.add_argument("--train-path", type=str, default="data/raw/averitec/train.json")
    parser.add_argument("--dev-manifest", type=str, default="outputs/m16-partitions-v2/development_manifest.json")
    parser.add_argument("--m18d-dir", type=str, default="outputs/m18d-retrieval-benchmark-v1")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    auditor = M19eAuditor(
        m19_dir=args.m19_dir,
        output_dir=args.output_dir,
        train_path=args.train_path,
        dev_manifest_v2=args.dev_manifest,
        m18d_dir=args.m18d_dir,
        seed=args.seed,
    )
    res = auditor.run_full_audit()
    print(f"\nFinal Audit Verdict: {res['verdict']}")


if __name__ == "__main__":
    main()
