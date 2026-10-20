"""Read-only source-cache audit. No model inference occurs in this command."""
import argparse
import json
import os
import platform
import subprocess
from pathlib import Path
from tffv.data import load_audit_inputs
from tffv.error_analysis import summarize
from tffv.memory import grouped_folds, source_partition
from tffv.provenance import canonical, digest, file_hash, write_json


def run(data, predictions, manifest, output, folds=5, seed=2026):
    records, metadata = load_audit_inputs(data, predictions, manifest)
    assignment = grouped_folds(records, folds, seed)
    for fold in range(folds):
        source_partition(records, assignment, fold)
    profile = summarize(records, metadata["labels"])
    if profile["duplicates"]["split_group_clusters"]:
        raise ValueError("Supplied grouping splits normalized duplicates")
    config = dict(schema_version=1, operation="source_base_audit", folds=folds, seed=seed,
                  source_sha256=file_hash(data), predictions_sha256=file_hash(predictions),
                  manifest_sha256=file_hash(manifest))
    # Hash implementation files too; reruns with changed audit logic are distinct.
    package = Path(__file__).resolve().parents[1]
    code = {str(p.relative_to(package)).replace("\\", "/"): file_hash(p) for p in sorted(package.rglob("*.py"))}
    config["implementation_sha256"] = digest(code)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    destination = Path(output).resolve()
    # CLI writes only inside this independent checkout's outputs directory.
    allowed = Path(__file__).resolve().parents[3] / "outputs"
    if not destination.is_relative_to(allowed.resolve()) or destination == allowed.resolve():
        raise ValueError("Output must be a fresh subdirectory of the new project's outputs/")
    destination.mkdir(parents=True, exist_ok=False)
    profile["scope"] = "exploratory source descriptive audit; not an OOF method evaluation"
    profile["population_status"] = metadata["population_status"]
    write_json(destination / "metrics.json", profile)
    write_json(destination / "folds.json", assignment)
    with (destination / "audit_records.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in records:
            stream.write(canonical(row) + "\n")
    report = (f"# Base error audit\n\nExploratory source audit: {len(records)} records.\n\n"
              f"Accuracy: {profile['accuracy']:.8f}\n\nMacro-F1: {profile['macro_f1']:.8f}\n\n"
              "Confusion matrix, per-class metrics, distributions, and example IDs: `metrics.json`.\n\n"
              "Full claims, evidence references and predictions: `audit_records.jsonl`.\n\n"
              "Global descriptive summaries must never become OOF knowledge artifacts.\n")
    (destination / "report.md").write_text(report, encoding="utf-8")
    for path, key in ((data, "source_sha256"), (predictions, "predictions_sha256"), (manifest, "manifest_sha256")):
        if file_hash(path) != config[key]:
            raise ValueError("Input changed during audit; output remains unsealed")
    output_hashes = {p.name: file_hash(p) for p in sorted(destination.iterdir()) if p.is_file()}
    write_json(destination / "manifest.json", dict(config=config, config_sha256=digest(config),
               input_manifest=metadata, input_paths={"data": str(Path(data).resolve()),
               "predictions": str(Path(predictions).resolve()), "manifest": str(Path(manifest).resolve())},
               code_files=code, git_revision=revision, python=platform.python_version(),
               output_sha256=output_hashes,
               metric_definition="fixed ontology, zero_division=0", status="complete"))
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, env in (("data", "TFFV_SOURCE_DATA"), ("predictions", "TFFV_BASE_PREDICTIONS"), ("manifest", "TFFV_BASE_MANIFEST")):
        parser.add_argument(f"--{name}", default=os.getenv(env))
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not all((args.data, args.predictions, args.manifest)):
        parser.error("Supply data, predictions and manifest via flags or environment")
    profile = run(args.data, args.predictions, args.manifest, args.output, args.folds, args.seed)
    print(json.dumps({key: profile[key] for key in ("n", "accuracy", "macro_f1")}))


if __name__ == "__main__":
    main()
