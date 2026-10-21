"""Metadata/schema-only follow-up, run by the user; never inference or conversion."""
import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


def shape(value, depth=2):
    """Describe types and field names, never primitive values (including gold)."""
    if isinstance(value, dict) and depth:
        return {str(key): shape(item, depth - 1) for key, item in sorted(value.items())}
    if isinstance(value, list) and depth:
        return {"type": "list", "first_item_shape": shape(value[0], depth - 1) if value else None}
    return type(value).__name__


def source_schema(path, max_bytes=64 * 1024 * 1024):
    """Only call on source-train files. Evaluation annotations are not inspected."""
    path = Path(path)
    result = {"path": str(path), "exists": path.is_file()}
    if not path.is_file(): return result
    result["bytes"] = path.stat().st_size
    if result["bytes"] > max_bytes:
        return dict(result, inspection="skipped: exceeds bounded schema-inspection size")
    raw = path.read_bytes()
    result["sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        if path.suffix == ".jsonl":
            count, samples = 0, []
            for line in raw.decode("utf-8-sig").splitlines():
                if line.strip():
                    row = json.loads(line)
                    count += 1
                    if len(samples) < 3: samples.append(row)
        else:
            payload = json.loads(raw.decode("utf-8-sig"))
            if not isinstance(payload, list):
                return dict(result, top_level_shape=shape(payload), records=None)
            count, samples = len(payload), payload[:3]
        result.update(records=count, sampled_records=len(samples), record_shapes=[shape(row) for row in samples])
    except (ValueError, UnicodeError):
        result["inspection_error"] = "Invalid JSON or UTF-8; record values omitted"
    return result


def candidate_paths(root, max_files=20000, max_results=300):
    root = Path(root).expanduser().resolve()
    result = dict(root=str(root), exists=root.is_dir(), candidates=[], inspected_files=0)
    if not root.is_dir(): return result
    marker = re.compile(r"averitec|knowledge|corpus|document|passage|qwen3[-_]embedding|qwen3[-_]reranker", re.I)
    for base, dirs, names in os.walk(root, followlinks=False):
        depth = len(Path(base).relative_to(root).parts)
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in {"__MACOSX", "__pycache__", "node_modules", "outputs", "output", "tmp"}) if depth < 5 else []
        for name in dirs:
            if marker.search(name) and len(result["candidates"]) < max_results:
                result["candidates"].append({"path": str(Path(base, name)), "kind": "directory"})
        for name in sorted(names):
            result["inspected_files"] += 1
            path = Path(base, name)
            if marker.search(str(path.relative_to(root))) and len(result["candidates"]) < max_results:
                try:
                    result["candidates"].append({"path": str(path), "kind": "file", "bytes": path.stat().st_size})
                except OSError:
                    result["candidates"].append({"path": str(path), "kind": "unreadable"})
            if result["inspected_files"] >= max_files:
                return dict(result, scan_truncated=True)
    return dict(result, scan_truncated=False, candidate_limit_reached=len(result["candidates"]) >= max_results)


def environment_metadata(executable):
    # Do not resolve symlinks: resolving .venv/bin/python would lose venv identity.
    executable = Path(executable).expanduser().absolute()
    result = {"executable": str(executable), "exists": executable.is_file()}
    if not executable.is_file(): return result
    code = '''import json, sys, importlib.metadata as m
versions = {}
for name in ("torch", "transformers", "numpy", "pytest", "huggingface-hub"):
    try: versions[name] = m.version(name)
    except m.PackageNotFoundError: versions[name] = None
print(json.dumps({"python": sys.version.split()[0], "prefix": sys.prefix, "packages": versions}))
'''
    try:
        response = subprocess.run([str(executable), "-I", "-B", "-c", code], capture_output=True, text=True, timeout=20)
        result["returncode"] = response.returncode
        if response.returncode == 0:
            result["metadata"] = json.loads(response.stdout)
        else:
            result["error"] = "Environment metadata query failed; no model imported"
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        result["error_type"] = type(exc).__name__
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", required=True)
    parser.add_argument("--search-root", action="append", default=[])
    parser.add_argument("--env-python", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    old = Path(args.old_root).expanduser().resolve()
    destination = Path(args.output).expanduser().resolve()
    allowed = Path(__file__).resolve().parents[1] / "outputs"
    if destination == allowed.resolve() or not destination.is_relative_to(allowed.resolve()):
        parser.error("Use a fresh subdirectory of NEW project outputs/")
    report = dict(
        source_train_schemas=[source_schema(old / "data/raw/averitec/train.json"),
                              source_schema(old / "data/processed/v2/averitec/train_all.jsonl")],
        candidate_search=[candidate_paths(root) for root in args.search_root],
        existing_environments=[environment_metadata(path) for path in args.env_python],
        limitations=["Source-train schema only: no primitive values or evaluation labels exported.",
                     "Candidate path names do not establish independent corpus provenance.",
                     "Installed package metadata does not verify CUDA/model execution compatibility.",
                     "No package installs, downloads, neural imports, corpus conversion or old-file writes."])
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "input_probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination / "input_probe.json")


if __name__ == "__main__": main()
