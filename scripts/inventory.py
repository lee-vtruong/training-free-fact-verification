"""Bounded, read-only inventory; does not import or execute historical code."""
import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.getenv("ERRORFLOW_ROOT"), required=not os.getenv("ERRORFLOW_ROOT"))
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    result = dict(root=str(root), exists=root.is_dir(), python=platform.python_version(),
                  disk_usage=shutil.disk_usage(Path.cwd())._asdict(), cwd=str(Path.cwd()),
                  gpu=command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
                  environment={k: os.getenv(k) for k in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "VIRTUAL_ENV", "CONDA_PREFIX")})
    if root.is_dir():
        result["git_head"] = command(["git", "-C", str(root), "rev-parse", "HEAD"])
        result["git_status"] = command(["git", "--no-optional-locks", "-C", str(root), "status", "--porcelain"])
        result["files"] = []
        for base, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules"} and not Path(base, d).is_symlink())
            for name in sorted(names):
                path = Path(base, name)
                if path.is_symlink() or path.suffix.lower() not in {".jsonl", ".json", ".csv", ".yaml", ".toml", ".py"}:
                    continue
                item = dict(path=str(path), format=path.suffix, bytes=path.stat().st_size)
                if path.suffix == ".jsonl":
                    try:
                        count = 0
                        keys = set()
                        with path.open(encoding="utf-8") as stream:
                            for line in stream:
                                if line.strip():
                                    row = json.loads(line)
                                    count += 1
                                    if isinstance(row, dict): keys.update(row)
                        item.update(records=count, field_names=sorted(keys))
                    except (OSError, ValueError) as exc:
                        item["error"] = str(exc)
                result["files"].append(item)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
