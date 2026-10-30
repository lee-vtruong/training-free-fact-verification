"""Local-only frozen Qwen adapters. No downloads, optimizer or generation."""
import gc
import importlib.metadata
import json
import re
from pathlib import Path
from tffv.provenance import digest, file_hash


TASK = "Given a factual claim, retrieve passages relevant to verifying or refuting the claim."
RERANK_SYSTEM = ('Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
                 'Note that the answer can only be "yes" or "no".')


def snapshot_fingerprint(directory):
    directory = Path(directory).expanduser().resolve()
    if not (directory / "config.json").is_file():
        raise ValueError("Local snapshot missing config.json")
    files = []
    for p in sorted(directory.rglob("*")):
        relative = p.relative_to(directory)
        if not p.is_file() or any(part.startswith(".") for part in relative.parts):
            continue
        if p.suffix not in {".json", ".safetensors", ".model", ".txt", ".tiktoken", ".jinja"}:
            continue
        files.append({"path": relative.as_posix(), "bytes": p.stat().st_size, "sha256": file_hash(p)})
    if not any(p["path"].endswith(".safetensors") for p in files):
        raise ValueError("Snapshot has no safetensors weights")
    return {"algorithm": "tffv_snapshot_files_v1", "snapshot_sha256": digest(files), "files": files}


def validate_spec(spec, expected_id, verify_bytes=False):
    if spec.get("model_id") != expected_id:
        raise ValueError(f"Expected {expected_id}")
    if not re.fullmatch(r"[0-9a-f]{40}", spec.get("revision", "")):
        raise ValueError("Pin a full 40-character model revision; main is not a revision")
    if not re.fullmatch(r"[0-9a-f]{64}", spec.get("snapshot_sha256", "")):
        raise ValueError("A snapshot content fingerprint is required")
    path = Path(spec["local_path"]).expanduser().resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError("Local model snapshot does not exist")
    if json.loads((path / "config.json").read_text(encoding="utf-8")).get("model_type") != "qwen3":
        raise ValueError("Expected Qwen3 snapshot")
    if re.fullmatch(r"[0-9a-f]{40}", path.name) and path.name != spec["revision"]:
        raise ValueError("HF snapshot directory revision mismatch")
    if verify_bytes and snapshot_fingerprint(path)["snapshot_sha256"] != spec["snapshot_sha256"]:
        raise ValueError("Snapshot content hash mismatch")
    return path


def runtime_check():
    version = importlib.metadata.version("transformers")
    numbers = tuple(int(n) for n in version.split(".")[:2])
    if numbers < (4, 51):
        raise RuntimeError(f"Qwen3 requires transformers>=4.51; found {version}. Use a NEW environment.")


class QwenFrozen:
    def __init__(self, spec, causal=False, device="cpu", dtype="float32", max_length=1024, batch_size=4):
        runtime_check()
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
        expected_id = "Qwen/Qwen3-Reranker-0.6B" if causal else "Qwen/Qwen3-Embedding-0.6B"
        path = validate_spec(spec, expected_id, verify_bytes=True)
        if device not in {"cpu", "cuda"} or dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Explicit cpu/cuda device and supported dtype required")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if device == "cpu" and dtype != "float32":
            raise ValueError("Use float32 for CPU runs")
        if type(batch_size) is not int or batch_size < 1 or type(max_length) is not int or not 128 <= max_length <= 32768:
            raise ValueError("Invalid batch size or context limit")
        self.torch, self.device = torch, device
        self.max_length, self.batch_size = max_length, batch_size
        self.forward_passes = 0
        self.token_count = 0
        self.truncated_inputs = 0
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False, padding_side="left")
        loader = AutoModelForCausalLM if causal else AutoModel
        self.model = loader.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                           torch_dtype=getattr(torch, dtype), attn_implementation="eager")
        self.model.to(device).eval()
        self.model.requires_grad_(False)
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("Failed to freeze parameters")
        if causal:
            self.prefix = self.tokenizer.encode(f"<|im_start|>system\n{RERANK_SYSTEM}<|im_end|>\n<|im_start|>user\n", add_special_tokens=False)
            self.suffix = self.tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)
            self.yes = self.tokenizer.encode("yes", add_special_tokens=False)
            self.no = self.tokenizer.encode("no", add_special_tokens=False)
            if len(self.yes) != 1 or len(self.no) != 1 or self.yes == self.no:
                raise ValueError("Reranker yes/no must be distinct single tokens")

    def sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def reset_peak(self):
        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()

    def peak_bytes(self):
        return self.torch.cuda.max_memory_allocated() if self.device == "cuda" else None

    def close(self):
        del self.model
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()

    def encode(self, texts, query=False):
        import numpy as np
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            if query:
                batch = [f"Instruct: {TASK}\nQuery:{s}" for s in batch]
            raw = self.tokenizer(batch, add_special_tokens=True, truncation=False)["input_ids"]
            self.truncated_inputs += sum(len(ids) > self.max_length for ids in raw)
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            self.token_count += int(inputs["attention_mask"].sum().item())
            with self.torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state[:, -1]
                vectors.append(self.torch.nn.functional.normalize(hidden.float(), p=2, dim=1).cpu().numpy())
            del inputs, hidden
            self.forward_passes += 1
            if self.device == "cuda" and self.forward_passes % 10 == 0:
                self.torch.cuda.empty_cache()
        return np.concatenate(vectors) if vectors else np.empty((0, 1024), dtype=np.float32)

    def score(self, claim, texts):
        scores = []
        budget = self.max_length - len(self.prefix) - len(self.suffix)
        if budget <= 0:
            raise ValueError("Reranker context is shorter than fixed template")
        for start in range(0, len(texts), self.batch_size):
            batch = [f"<Instruct>: {TASK}\n<Query>: {claim}\n<Document>: {s}" for s in texts[start:start + self.batch_size]]
            raw = self.tokenizer(batch, add_special_tokens=False, truncation=False)["input_ids"]
            self.truncated_inputs += sum(len(ids) > budget for ids in raw)
            ids = [self.prefix + item[:budget] + self.suffix for item in raw]
            inputs = self.tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(self.device)
            self.token_count += int(inputs["attention_mask"].sum().item())
            with self.torch.inference_mode():
                logits = self.model(**inputs).logits[:, -1, :].float()
                values = self.torch.softmax(logits[:, [self.no[0], self.yes[0]]], dim=1)[:, 1]
                scores.extend(values.cpu().tolist())
            self.forward_passes += 1
        return scores
