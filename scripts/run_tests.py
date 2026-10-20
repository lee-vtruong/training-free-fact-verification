"""Run pytest with an isolated temporary directory and no shared disk cache."""
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    project = Path(__file__).resolve().parents[1]
    # Each account owns the fresh directory it creates. Never reuse or remove
    # pytest-of-USER or a cache created by a different account/sandbox.
    with tempfile.TemporaryDirectory(prefix="tffv-tests-") as temporary:
        arguments = sys.argv[1:] or ["-q"]
        return subprocess.call(
            [sys.executable, "-m", "pytest", *arguments,
             "-p", "no:cacheprovider", "--basetemp", str(Path(temporary) / "pytest")],
            cwd=project,
        )


if __name__ == "__main__":
    raise SystemExit(main())
