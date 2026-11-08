#!/usr/bin/env python3
"""CLI Runner: Milestone M2.0-A — Error-Derived Verification Rules.

Usage:
    # Offline CPU dry-run / smoke test:
    python scripts/run_m20a_experiment.py --smoke-only

    # Full GPU execution on server:
    python scripts/run_m20a_experiment.py --run-gpu

    # Evaluation only from existing predictions:
    python scripts/run_m20a_experiment.py --eval-only
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

from tffv.experiments.m19f_loader import EXPECTED_OFFICIAL_DEV_COUNT, OfficialDevClaim
from tffv.experiments.m20a_evaluator import M20aEvaluator
from tffv.experiments.m20a_orchestrator import M20aOrchestrator
from tffv.experiments.m20a_schemas import M20aPredictionRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M2.0-A Error-Derived Verification Rules Runner")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "m20a-error-rules-v1"),
        help="Directory to store outputs and evaluations.",
    )
    parser.add_argument(
        "--evidence-receipts",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "m19g-evidence-complete-v1" / "dev_retrieval_receipts.jsonl"),
        help="Path to sealed retrieval receipts JSONL.",
    )
    parser.add_argument(
        "--dev-claims-file",
        type=str,
        default="",
        help="Path to official dev.json.",
    )
    parser.add_argument(
        "--run-gpu",
        action="store_true",
        help="Launch live GPU inference using frozen Qwen3-8B BF16.",
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


def load_dev_claims(dev_path_str: str) -> List[Dict[str, Any]]:
    candidates = []
    if dev_path_str:
        candidates.append(Path(dev_path_str))
    candidates.extend([
        PROJECT_ROOT / "data" / "raw" / "averitec" / "dev.json",
        PROJECT_ROOT / "data_store" / "averitec" / "data_averitec" / "dev.json",
        PROJECT_ROOT / "outputs" / "m19g-evidence-complete-v1" / "predictions.jsonl",
    ])

    resolved_path = None
    for cand in candidates:
        if cand.exists():
            resolved_path = cand
            break

    if not resolved_path:
        raise FileNotFoundError(f"Dev claims not found in candidates: {candidates}")

    if resolved_path.name.endswith(".jsonl"):
        seen = set()
        claims = []
        with open(resolved_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r["sample_id"] not in seen:
                        seen.add(r["sample_id"])
                        claims.append({
                            "sample_id": r["sample_id"],
                            "claim": r["claim"],
                            "gold_verdict": r["gold_verdict"],
                        })
        return claims

    with open(resolved_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    claims = []
    for idx, item in enumerate(data):
        claims.append({
            "sample_id": f"dev:{idx}",
            "claim": item["claim"],
            "gold_verdict": item["label"],
        })
    return claims


def load_retrieval_receipts(receipts_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    if not receipts_path.exists():
        raise FileNotFoundError(f"Retrieval receipts not found at: {receipts_path}")

    receipts: Dict[str, List[Dict[str, Any]]] = {}
    with open(receipts_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                sid = row["sample_id"]
                # Supports evidence list, passages list, or result_record passages
                passages = row.get("evidence", row.get("passages", row.get("retrieved_passages", [])))
                receipts[sid] = passages
    return receipts


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        preds_file = out_dir / "predictions.jsonl"
        if not preds_file.exists():
            sys.exit(f"Cannot run eval-only: {preds_file} does not exist.")
        print(f"[M2.0-A] Recomputing evaluations from {preds_file}...")
        records_by_arm: Dict[str, List[M20aPredictionRecord]] = {}
        with open(preds_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rec = M20aPredictionRecord(**r)
                    records_by_arm.setdefault(rec.arm, []).append(rec)
        evaluator = M20aEvaluator(records_by_arm)
        evaluator.evaluate_all(output_dir=out_dir)
        print(f"[M2.0-A] Evaluation complete. Results saved in {out_dir}")
        return

    dev_claims = load_dev_claims(args.dev_claims_file)
    retrieval_receipts = load_retrieval_receipts(Path(args.evidence_receipts))

    orchestrator = M20aOrchestrator(
        output_dir=out_dir,
        source_fold="train_fold_1",
        target_split_name="dev",
    )

    orchestrator.run(
        target_claims=dev_claims,
        retrieval_receipts=retrieval_receipts,
        source_train_records=None,  # Uses sealed Train Fold-1 seed patterns
        run_gpu=args.run_gpu,
        smoke_only=args.smoke_only,
    )


if __name__ == "__main__":
    main()
