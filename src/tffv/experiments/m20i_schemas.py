"""Milestone M2.0i: Data schemas for Factorial Mechanism Ablation.

Covers:
- Five ablation arms: F0, F1, F2, F3, F4.
- Provenance classifications: P1, P2, P3, P4.
- Prediction records and evaluation manifest data models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class MechanismArm(str, Enum):
    """The five experimental arms for the 2x2 factorial + placebo study."""

    F0_NEUTRAL_EXPL = "F0_neutral_expl"
    F1_HEADER_ONLY = "F1_header_only"
    F2_OBSERVED_MISTAKE = "F2_observed_mistake"
    F3_FULL_HISTORICAL_ERROR = "F3_full_historical_error"
    F4_MATCHED_PLACEBO = "F4_matched_placebo"


class ProvenanceLevel(str, Enum):
    """Provenance classifications defined in M2.0h audit."""

    P1_VALIDATED_GROUNDED = "P1_VALIDATED_GROUNDED"
    P2_SOURCE_TRACEABLE_UNVALIDATED = "P2_SOURCE_TRACEABLE_UNVALIDATED"
    P3_HEURISTIC_DERIVED = "P3_HEURISTIC_DERIVED"
    P4_UNSUPPORTED = "P4_UNSUPPORTED"


@dataclass
class M20iPredictionRecord:
    """Prediction record for Milestone M2.0i evaluations."""

    sample_id: str
    arm_name: str
    seed: int
    k_shots: int
    raw_claim: str
    gold_label: str
    model_name: str
    model_revision: str
    prompt_sha256: str
    raw_completion: str
    parsed_verdict: str
    parse_status: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
