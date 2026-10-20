"""Cache keys contain inference-visible inputs, never evaluation gold labels."""
from tffv.provenance import digest
from tffv.provenance import write_json
from pathlib import Path
import json


def prediction_key(example, spec, knowledge_sha256):
    return digest({"input": vars(example), "spec": vars(spec), "knowledge_sha256": knowledge_sha256})


class PredictionCache:
    """Immutable per-example cache in the new checkout only; no model execution."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        allowed = Path(__file__).resolve().parents[3] / "outputs"
        if not self.directory.is_relative_to(allowed.resolve()) or self.directory == allowed.resolve():
            raise ValueError("Cache must live in a subdirectory of new outputs/")
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key):
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid cache key")
        return self.directory / (key + ".json")

    def put(self, key, output):
        if "gold_label" in output:
            raise ValueError("Inference cache must not contain evaluation gold")
        write_json(self._path(key), {"cache_key": key, "output": output})

    def get(self, key):
        path = self._path(key)
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["cache_key"] != key:
            raise ValueError("Cache identity mismatch")
        return record["output"]
