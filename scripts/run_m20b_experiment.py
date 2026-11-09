#!/usr/bin/env python3
"""CLI Runner: Milestone M2.0b — Baseline Audit & Instance-Grounded Error Intervention.

Usage:
    # 1. Phase A & B pure CPU diagnostics:
    python scripts/run_m20b_experiment.py --audit-only

    # 2. CPU Smoke test (10 claims):
    python scripts/run_m20b_experiment.py --smoke-only

    # 3. Pilot evaluation on 50 independent development claims:
    python scripts/run_m20b_experiment.py --pilot --run-gpu

    # 4. Full GPU execution across all 6 arms on server:
    python scripts/run_m20b_experiment.py --run-gpu

    # 5. Recompute evaluation only from sealed predictions:
    python scripts/run_m20b_experiment.py --eval-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tffv.experiments.m20b_evaluator import M20bEvaluator
from tffv.experiments.m20b_orchestrator import M20bOrchestrator
from tffv.experiments.m20b_schemas import M20bPredictionRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M2.0b Baseline Audit & Instance-Grounded Intervention Runner")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "m20b-instance-grounded-v1"),
        help="Directory to store outputs and evaluations.",
    )
    parser.add_argument(
        "--m19g-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "m19g-evidence-complete-v1"),
        help="Path to sealed M1.9g output directory.",
    )
    parser.add_argument(
        "--m20a-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "m20a-error-rules-v1"),
        help="Path to sealed M2.0a output directory.",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "averitec" / "train.json"),
        help="Path to official train.json.",
    )
    parser.add_argument(
        "--dev-claims-file",
        type=str,
        default="",
        help="Path to target claims JSON.",
    )
    parser.add_argument(
        "--evidence-receipts",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "m19g-evidence-complete-v1" / "dev_retrieval_receipts.jsonl"),
        help="Path to sealed retrieval receipts JSONL.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Custom path to Qwen checkpoint.",
    )
    parser.add_argument(
        "--run-gpu",
        action="store_true",
        help="Launch live GPU inference using frozen Qwen3-8B BF16.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run only Phase A (Baseline Audit) and Phase B (Track A Error Analysis).",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run on 50 independent development claims from Train Fold-2.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run lightweight smoke test on 10 claims.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Recompute evaluation and statistical tests from existing predictions.jsonl.",
    )
    return parser.parse_args()


def load_retrieval_receipts(path_str: str) -> Dict[str, List[Dict[str, Any]]]:
    path = Path(path_str)
    if not path.exists():
        print(f"[Warning] Retrieval receipts not found at {path}. Using synthetic passages for test.")
        return {}

    receipts = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec.get("claim_id") or rec.get("sample_id")
            passages = rec.get("passages", rec.get("evidence", []))
            if sid:
                receipts[sid] = passages
    return receipts


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orchestrator = M20bOrchestrator(
        output_dir=out_dir,
        m19g_dir=args.m19g_dir,
        m20a_dir=args.m20a_dir,
        train_path=args.train_path,
    )

    # Always run Phase A and Phase B
    orchestrator.run_phase_a()
    orchestrator.run_phase_b()

    if args.audit_only:
        print("\n[M2.0b] --audit-only flag set. Completed Phase A and Phase B diagnostics. Exiting.")
        sys.exit(0)

    # Phase E: Split & Leakage Isolation
    historical_source, dev_pilot = orchestrator.run_phase_e()

    if args.eval_only:
        preds_file = out_dir / "predictions.jsonl"
        if not preds_file.exists():
            print(f"[Error] Cannot recompute evaluation: {preds_file} not found!")
            sys.exit(1)
        all_preds = []
        with open(preds_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_preds.append(M20bPredictionRecord(**json.loads(line)))
        evaluator = M20bEvaluator(records=all_preds, output_dir=out_dir)
        evaluator.evaluate_all()
        print(f"[M2.0b] Re-evaluation completed from {preds_file}.")
        sys.exit(0)

    # Determine target claims
    receipts = load_retrieval_receipts(args.evidence_receipts)

    if args.pilot:
        target_claims = dev_pilot[:50]
        print(f"[M2.0b] Running in --pilot mode on {len(target_claims)} independent development claims.")
    elif args.dev_claims_file:
        with open(args.dev_claims_file, "r", encoding="utf-8") as f:
            target_claims = json.load(f)
    else:
        # Default target claims: independent dev_pilot fold to prevent Dev 500 overfitting
        target_claims = dev_pilot[:100] if len(dev_pilot) >= 100 else dev_pilot
        print(f"[M2.0b] Target split: Independent Development Subset ({len(target_claims)} claims).")

    # Run inference and evaluation
    eval_results = orchestrator.run_inference_and_eval(
        target_claims=target_claims,
        retrieval_receipts=receipts,
        historical_claims=historical_source,
        run_gpu=args.run_gpu,
        smoke_only=args.smoke_only,
        model_path=args.model_path if args.model_path else None,
    )

    decision = eval_results.get("decision", "INCONCLUSIVE")
    print(f"\n=======================================================")
    print(f" M2.0b Run Complete! Final Decision: {decision}")
    print(f" Outputs: {out_dir}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
