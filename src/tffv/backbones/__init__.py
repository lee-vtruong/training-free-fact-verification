"""Inference-only interfaces. No concrete model runner at milestone one."""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class VerificationInput:
    sample_id: str
    dataset: str
    claim: str
    evidence: str
    evidence_protocol: str


@dataclass(frozen=True)
class VerificationOutput:
    sample_id: str
    prediction: str
    label_scores: dict[str, float] | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    forward_passes: int | None = None
    peak_gpu_memory_bytes: int | None = None
    runtime_seconds: float | None = None


class FrozenVerifier(Protocol):
    def verify(self, example: VerificationInput, prompt: str, labels: tuple[str, ...]) -> VerificationOutput: ...
