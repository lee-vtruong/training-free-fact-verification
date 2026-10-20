"""Ensure a clean Git source export includes required Python packages."""
import shutil
import subprocess
import sys
from pathlib import Path
import pytest


def test_clean_source_export_imports_data_and_retrieval(tmp_path):
    if not shutil.which("git"):
        pytest.skip("Git required for clean source export validation")
    project = Path(__file__).resolve().parents[1]
    # Include tracked files plus eligible untracked changes, as in a code handoff.
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "src"],
        cwd=project, capture_output=True, text=True, check=True,
    )
    for name in set(result.stdout.splitlines()):
        relative = Path(name)
        if relative.suffix != ".py":
            continue
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / relative, target)
    assert (tmp_path / "src/tffv/data/records.py").is_file(), "Required data package excluded from Git export"
    code = ("import sys; sys.path.insert(0, sys.argv[1]); "
            "from tffv.data import read_jsonl, load_audit_inputs; "
            "from tffv.retrieval.benchmark import plan; "
            "from tffv.retrieval.evaluate import evaluate")
    subprocess.run([sys.executable, "-I", "-c", code, str(tmp_path / "src")],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
