"""CLI Entrypoint for Milestone M1.8e Full-Train Historical Memory Construction."""
import argparse
import os
import sys
from pathlib import Path

# Add src to pythonpath
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tffv.experiments.m18e_orchestrator import (
    DEFAULT_OUTPUT_DIR,
    M18eConfig,
    M18eOrchestrator,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory root")
    parser.add_argument("--train-json", default=None, help="Path to raw AVeriTeC train.json")
    parser.add_argument("--dev-json", default=None, help="Path to raw AVeriTeC dev.json")
    parser.add_argument("--phase", choices=["all", "A", "B", "C", "D", "E", "F", "G", "H"], default="all",
                        help="Specific phase to run, or 'all'")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of claims for testing/bounded run")
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic test doubles (offline mode)")
    parser.add_argument("--judge-port", type=int, default=17171, help="Port for Qwen3-14B-AWQ judge service")
    parser.add_argument("--force-unlock", action="store_true", help="Remove stale lockfile before launching")

    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    if args.force_unlock:
        lock_file = out_dir / ".m18e.pid"
        if lock_file.exists():
            print(f"[M1.8e] Removing lockfile at {lock_file}")
            lock_file.unlink()

    orchestrator = M18eOrchestrator(
        output_dir=out_dir,
        config=M18eConfig(),
        train_path=args.train_json,
        dev_path=args.dev_json,
        synthetic=args.synthetic,
        judge_port=args.judge_port,
    )

    if args.phase == "all":
        orchestrator.run_all(limit=args.limit)
    else:
        state = orchestrator.load_state()
        if args.phase == "A":
            orchestrator.run_phase_a(state)
        elif args.phase == "B":
            orchestrator.run_phase_b(state, limit=args.limit)
        elif args.phase == "C":
            orchestrator.run_phase_c(state, limit=args.limit)
        elif args.phase == "D":
            orchestrator.run_phase_d(state)
        elif args.phase == "E":
            orchestrator.run_phase_e(state)
        elif args.phase == "F":
            orchestrator.run_phase_f(state, limit=args.limit)
        elif args.phase == "G":
            orchestrator.run_phase_g(state)
        elif args.phase == "H":
            orchestrator.run_phase_h(state)


if __name__ == "__main__":
    main()
