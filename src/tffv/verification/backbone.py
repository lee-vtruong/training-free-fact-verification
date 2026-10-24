"""Frozen Qwen3-8B verification backbone adapter and test doubles."""
import gc
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from tffv.provenance import digest, file_hash

EXPECTED_QWEN3_8B_ID = "Qwen/Qwen3-8B"
EXPECTED_QWEN3_8B_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def validate_backbone_spec(spec: Dict[str, Any], expected_id: str = EXPECTED_QWEN3_8B_ID) -> Path:
    """Validate snapshot directory and identity for frozen verification backbone."""
    if spec.get("model_id") != expected_id:
        raise ValueError(f"Expected model_id {expected_id}; got {spec.get('model_id')}")
    
    revision = spec.get("revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Pin a full 40-character commit revision for the verification model")

    local_path = Path(spec["local_path"]).expanduser().resolve()
    if not local_path.is_dir() or not (local_path / "config.json").is_file():
        raise ValueError(f"Local backbone snapshot directory does not exist: {local_path}")

    config_data = json.loads((local_path / "config.json").read_text(encoding="utf-8"))
    if config_data.get("model_type") != "qwen3":
        raise ValueError(f"Expected model_type 'qwen3'; found {config_data.get('model_type')}")

    return local_path


class Qwen3FrozenBackbone:
    """Frozen causal language model adapter for zero-shot verification inference."""

    def __init__(
        self,
        spec: Dict[str, Any],
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_context_length: int = 8192,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = validate_backbone_spec(spec)
        if device not in {"cuda", "cpu"}:
            raise ValueError("Device must be 'cuda' or 'cpu'")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system")

        self.torch = torch
        self.device = device
        self.max_context_length = max_context_length

        torch_dtype = getattr(torch, dtype) if dtype else torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa" if device == "cuda" else "eager",
        )
        self.model.to(device).eval()
        self.model.requires_grad_(False)

        # Strict safety check: certify all parameters are frozen
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("Failed to freeze verification backbone parameters; requires_grad is True!")

        self.forward_passes = 0
        self.total_generated_tokens = 0

    def reset_peak(self) -> None:
        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()

    def peak_bytes(self) -> Optional[int]:
        if self.device == "cuda":
            return self.torch.cuda.max_memory_allocated()
        return None

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
    ) -> Dict[str, Any]:
        """Execute deterministic greedy generation on structured messages."""
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_length,
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]

        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # strictly deterministic greedy decoding
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][input_len:]
        output_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        self.forward_passes += 1
        self.total_generated_tokens += len(gen_tokens)

        return {
            "prompt_text": prompt_text,
            "generated_text": output_text,
            "prompt_tokens": input_len,
            "generated_tokens": len(gen_tokens),
        }

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()


class SyntheticBackbone:
    """Deterministic synthetic test double for offline pipeline tests."""

    def __init__(self, spec: Optional[Dict[str, Any]] = None, **kwargs):
        self.device = "cpu"
        self.forward_passes = 0
        self.total_generated_tokens = 0

    def reset_peak(self) -> None:
        pass

    def peak_bytes(self) -> Optional[int]:
        return 0

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 512) -> Dict[str, Any]:
        user_text = messages[-1]["content"] if messages else ""
        h = digest(user_text)
        val = int(h[:4], 16) % 4
        verdicts = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]
        chosen = verdicts[val]
        text = f"Synthetic analysis based on passages provided.\nVerdict: {chosen}"
        self.forward_passes += 1
        self.total_generated_tokens += len(text.split())
        return {
            "prompt_text": user_text,
            "generated_text": text,
            "prompt_tokens": len(user_text.split()),
            "generated_tokens": len(text.split()),
        }

    def close(self) -> None:
        pass
