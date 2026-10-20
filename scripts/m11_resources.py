"""User-run server reconnaissance. No SSH, inference, downloads or old-project writes."""
import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def inventory(root):
    root = Path(root).expanduser().resolve()
    result = {"root": str(root), "exists": root.is_dir(), "files": []}
    if not root.is_dir(): return result
    count = 0
    for base, dirs, names in os.walk(root, followlinks=False):
        depth = len(Path(base).relative_to(root).parts)
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in {"paper", "node_modules", "__pycache__", "tmp", "output", "outputs"}) if depth < 5 else []
        for name in sorted(names):
            path = Path(base, name)
            if path.suffix not in {".json", ".jsonl", ".safetensors"}:
                continue
            info = {"path": str(path), "bytes": path.stat().st_size}
            if path.name == "config.json":
                try:
                    config = json.loads(path.read_text(encoding="utf-8"))
                    info["model_type"] = config.get("model_type")
                    info["architectures"] = config.get("architectures")
                except (OSError, ValueError) as exc: info["read_error"] = str(exc)
            result["files"].append(info)
            count += 1
            if count >= 2000:
                result["truncated"] = True
                return result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", required=True)
    parser.add_argument("--model-root", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    allowed = Path(__file__).resolve().parents[1] / "outputs"
    destination = Path(args.output).expanduser().resolve()
    if destination == allowed.resolve() or not destination.is_relative_to(allowed.resolve()):
        parser.error("Output must be inside NEW project outputs/")
    versions = {}
    for name in ("torch", "transformers", "numpy", "pytest", "huggingface-hub"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    try:
        import torch
        cuda = {"available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
    except (ImportError, OSError) as exc:
        cuda = {"error": str(exc)}
    report = dict(python=platform.python_version(), dependencies=versions, cuda=cuda,
        disk_usage=shutil.disk_usage(Path.cwd())._asdict(),
        gpu=command(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"]),
        git_status=command(["git", "status", "--short", "--branch"]),
        git_head=command(["git", "rev-parse", "HEAD"]),
        git_remote=command(["git", "remote", "-v"]),
        old_repository=inventory(args.old_root),
        model_roots=[inventory(root) for root in args.model_root],
        notes=["Bounded metadata-only scan; no proof of dataset/model completeness.",
               "Old outputs and hidden directories are skipped; use an explicit model-root to inspect HF snapshots."])
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "resources.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination / "resources.json")


if __name__ == "__main__": main()
