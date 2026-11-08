"""CLI entrypoint for Milestone M1.9g Clean Evidence-Complete Official Dev Evaluation."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tffv.experiments.m19g_orchestrator import M19gOrchestrator


def main():
    parser = argparse.ArgumentParser(description="Milestone M1.9g Evaluation Orchestrator")
    parser.add_argument("--m18e-dir", type=str, default="outputs/m18e-full-history-pf1000-v1")
    parser.add_argument("--output-dir", type=str, default="outputs/m19g-evidence-complete-v1")
    parser.add_argument("--dev-path", type=str, default="data/raw/averitec/dev.json")
    parser.add_argument("--dev-store-dir", type=str, default="data_store/knowledge_store/dev/uncompressed")
    parser.add_argument("--dev-receipts-path", type=str, default=None)
    parser.add_argument("--train-path", type=str, default="data/raw/averitec/train.json")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--limit-claims", type=int, default=None)

    args = parser.parse_args()

    orchestrator = M19gOrchestrator(
        m18e_dir=args.m18e_dir,
        output_dir=args.output_dir,
        dev_path=args.dev_path,
        dev_store_dir=args.dev_store_dir,
        dev_receipts_path=args.dev_receipts_path,
        train_path=args.train_path,
        k=args.k,
        seed=args.seed,
    )

    orchestrator.run_all(
        run_gpu=args.run_gpu,
        smoke_only=args.smoke_only,
        limit_claims=args.limit_claims,
    )


if __name__ == "__main__":
    main()
