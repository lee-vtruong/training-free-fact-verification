#!/usr/bin/env python3
"""Milestone M1.8d — Automated Retrieval Benchmark & Model Selection CLI.

Executes end-to-end benchmark comparing retrieval arms R0..R8 on NVIDIA RTX 5090.
Outputs unified comparison tables, Pareto frontier, downstream Qwen3-8B verification,
McNemar/Bootstrap statistical hypothesis testing, and DECISION_REPORT.md.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tffv.retrieval.m18d_orchestrator import BenchmarkOrchestrator


def parse_args():
    parser = argparse.ArgumentParser(description="M1.8d Automated Retrieval Benchmark & Model Selection")
    parser.add_argument(
        "--config",
        default="configs/m161_source_batch25.json",
        help="Path to retrieval configuration JSON",
    )
    parser.add_argument(
        "--acq-plan",
        default="configs/m161_acquisition_plan.json",
        help="Path to acquisition plan JSON",
    )
    parser.add_argument(
        "--raw-train",
        default="data/raw/averitec/train.json",
        help="Path to raw AVeriTeC train JSON",
    )
    parser.add_argument(
        "--pilot-manifest",
        default="configs/m18d_manifests/pilot_manifest.json",
        help="Path to stratified pilot manifest (N=50)",
    )
    parser.add_argument(
        "--holdout-manifest",
        default="configs/m18d_manifests/holdout_manifest.json",
        help="Path to stratified holdout manifest (N=200)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/m18d-retrieval-benchmark-v1",
        help="Output directory for benchmark artifacts",
    )
    parser.add_argument(
        "--arms",
        default=None,
        help="Comma-separated list of specific retrieval arms to evaluate (e.g. 'R0_CANONICAL_R4,R1_QWEN3_DENSE_OPT,R6_BM25_PF5000')",
    )
    parser.add_argument(
        "--dense-batch-size",
        type=int,
        default=64,
        help="Dense embedding batch size (tuned for RTX 5090)",
    )
    parser.add_argument(
        "--rerank-batch-size",
        type=int,
        default=16,
        help="Cross-encoder reranker batch size",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run CPU synthetic simulation without GPU models",
    )
    parser.add_argument(
        "--skip-downstream",
        action="store_true",
        help="Skip Qwen3-8B verification stage",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from existing checkpoints, re-run from scratch",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    active_arms = [a.strip() for a in args.arms.split(",")] if args.arms else None

    orchestrator = BenchmarkOrchestrator(
        config_path=Path(args.config),
        acq_plan_path=Path(args.acq_plan),
        raw_train_path=Path(args.raw_train),
        pilot_manifest_path=Path(args.pilot_manifest),
        holdout_manifest_path=Path(args.holdout_manifest),
        output_dir=Path(args.output_dir),
        synthetic=args.synthetic,
        skip_downstream=args.skip_downstream,
        dense_batch_size=args.dense_batch_size,
        rerank_batch_size=args.rerank_batch_size,
        resume=not args.no_resume,
        active_arms=active_arms,
    )
    orchestrator.run_all()


if __name__ == "__main__":
    main()
