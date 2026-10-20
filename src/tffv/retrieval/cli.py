"""M1.1 commands are local programs; the user runs server workloads explicitly."""
import argparse
import json
import os
from pathlib import Path
from tffv.provenance import write_json
from tffv.retrieval.benchmark import METHODS, output_directory, plan, run
from tffv.retrieval.evaluate import evaluate
from tffv.retrieval.frozen import snapshot_fingerprint


def expand(value):
    if isinstance(value, str): return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, dict): return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list): return [expand(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    fingerprint = subs.add_parser("fingerprint", help="Read local model bytes, no inference")
    fingerprint.add_argument("--model-dir", required=True)
    fingerprint.add_argument("--output", required=True, help="Fresh subdirectory of outputs/")
    benchmark = subs.add_parser("benchmark", help="Preflight by default; --execute explicitly starts retrieval")
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    benchmark.add_argument("--max-claims", type=int)
    benchmark.add_argument("--max-passages", type=int, help="Deterministic corpus prefix, bounded smoke only")
    benchmark.add_argument("--output")
    benchmark.add_argument("--index-cache")
    benchmark.add_argument("--execute", action="store_true")
    evaluation = subs.add_parser("evaluate")
    evaluation.add_argument("--run", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--qrels")
    evaluation.add_argument("--qrels-manifest")
    args = parser.parse_args()
    if args.command == "fingerprint":
        report = snapshot_fingerprint(args.model_dir)
        directory = output_directory(args.output)
        write_json(directory / "snapshot.json", report)
        print(json.dumps({"snapshot_sha256": report["snapshot_sha256"], "output": str(directory)}))
    elif args.command == "benchmark":
        config = expand(json.loads(Path(args.config).read_text(encoding="utf-8")))
        workload = plan(config, args.methods, args.max_claims, max_passages=args.max_passages)
        print(json.dumps(workload, indent=2))
        if args.execute:
            if not args.output: parser.error("--execute requires --output")
            print(run(config, args.output, args.methods, args.max_claims, index_cache=args.index_cache, max_passages=args.max_passages))
    else:
        print(evaluate(args.run, args.output, args.qrels, args.qrels_manifest))


if __name__ == "__main__": main()
