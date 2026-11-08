"""CLI entrypoint for Milestone M1.9g Frozen Dev Retrieval."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tffv.experiments.m19g_retrieval import M19gRetrievalPipeline


def main():
    parser = argparse.ArgumentParser(description="Milestone M1.9g Official Dev Retrieval")
    parser.add_argument("--dev-path", type=str, default="data/raw/averitec/dev.json")
    parser.add_argument("--dev-store-dir", type=str, default="data_store/knowledge_store/dev/uncompressed")
    parser.add_argument("--output-dir", type=str, default="outputs/m19g-evidence-complete-v1")
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pipeline = M19gRetrievalPipeline(
        dev_path=args.dev_path,
        dev_store_dir=args.dev_store_dir,
        output_dir=args.output_dir,
        run_gpu=args.run_gpu,
    )
    pipeline.run(limit=args.limit)


if __name__ == "__main__":
    main()
