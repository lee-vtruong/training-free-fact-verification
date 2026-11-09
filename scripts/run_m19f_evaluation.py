"""CLI entrypoint for Milestone M1.9f Clean Official Dev 500 Evaluation."""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tffv.experiments.m19f_orchestrator import M19fOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone M1.9f Clean Official Dev Evaluation")
    parser.add_argument("--m18e-dir", type=str, default="outputs/m18e-full-history-pf1000-v1")
    parser.add_argument("--output-dir", type=str, default="outputs/m19f-clean-official-dev-v1")
    parser.add_argument("--dev-path", type=str, default="data/raw/averitec/dev.json")
    parser.add_argument("--dev-evidence-path", type=str, default=None)
    parser.add_argument("--train-path", type=str, default="data/raw/averitec/train.json")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-gpu", action="store_true", help="Run frozen Qwen3-8B BF16 on GPU")
    parser.add_argument("--smoke-only", action="store_true", help="Run mandatory 10-claim smoke test and stop")
    parser.add_argument("--limit-claims", type=int, default=None, help="Limit number of evaluation claims")

    args = parser.parse_args()

    orchestrator = M19fOrchestrator(
        m18e_dir=args.m18e_dir,
        output_dir=args.output_dir,
        dev_path=args.dev_path,
        dev_evidence_path=args.dev_evidence_path,
        train_path=args.train_path,
        k=args.k,
        seed=args.seed,
    )

    res = orchestrator.run_all(
        run_gpu=args.run_gpu,
        smoke_only=args.smoke_only,
        limit_claims=args.limit_claims,
    )
    print("\n[M1.9f] Orchestration completed successfully.")


if __name__ == "__main__":
    main()
