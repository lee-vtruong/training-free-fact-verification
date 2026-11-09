"""Milestone M2.0b: Baseline Audit & Instance-Grounded Error Intervention Schemas.

Strictly defines:
1. Six experimental arms: B0, G2, E1, B1, B2, B3.
2. Instance-grounded intervention decision contracts (INTERVENE vs NO_INTERVENTION).
3. Selection traces, grounding audit, and memory eligibility data models.
4. Paired evaluation schemas with conservation invariants.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class M20bArm(str, enum.Enum):
    """The 6 experimental arms in M2.0b."""
    B0 = "B0"  # Locked Canonical Zero-Shot Baseline
    G2 = "G2"  # Length-Matched Generic Rules
    E1 = "E1"  # Global Error-Derived Rules
    B1 = "B1"  # Instance-Grounded Error Intervention (Primary Method)
    B2 = "B2"  # Instance-Grounded Generic Control (Matched Selection/Tokens)
    B3 = "B3"  # Irrelevant/Random Historical Knowledge Control


EXPERIMENTAL_ARMS_M20B = [
    M20bArm.B0.value,
    M20bArm.G2.value,
    M20bArm.E1.value,
    M20bArm.B1.value,
    M20bArm.B2.value,
    M20bArm.B3.value,
]

OFFICIAL_CLASSES_4 = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]


class InterventionDecision(str, enum.Enum):
    """Decision by the grounding filter on whether to intervene."""
    INTERVENE = "INTERVENE"
    NO_INTERVENTION = "NO_INTERVENTION"


@dataclass
class SelectionTrace:
    """Trace of exemplar/rule selection for a specific claim and arm."""
    sample_id: str
    arm: str
    intervention_decision: str  # INTERVENE or NO_INTERVENTION
    relevance_score: float
    grounding_overlap_score: float
    selected_exemplar_id: Optional[str]
    selected_rationale_type: str  # VALIDATED_CORRECTION, ERROR_CASE, GENERIC_CASE, IRRELEVANT, NONE
    token_budget: int
    is_validated_correction: bool = False
    source_fold: str = "train_fold_1"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GroundingAuditRecord:
    """Audit verification that historical knowledge is grounded in target evidence."""
    sample_id: str
    target_claim: str
    historical_sample_id: Optional[str]
    evidence_grounding_verified: bool
    overlapping_terms: List[str]
    filter_rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class M20bPredictionRecord:
    """Prediction output record for M2.0b."""
    sample_id: str
    arm: str
    claim: str
    gold_verdict: str
    predicted_verdict: str
    raw_completion: str
    is_parseable: bool
    is_correct: bool
    latency_seconds: float
    prompt_sha256: str
    intervention_decision: str  # INTERVENE or NO_INTERVENTION
    selected_exemplar_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class M20bTransitionStats:
    """Claim-level transition counts between two arms satisfying conservation invariants."""
    arm_a: str
    arm_b: str
    both_correct: int
    both_wrong: int
    corrected: int  # Wrong in arm_a, Correct in arm_b
    harmed: int     # Correct in arm_a, Wrong in arm_b
    net_benefit: int  # corrected - harmed
    mcnemar_pvalue: float
    odds_ratio: float
    total_claims: int

    def __post_init__(self):
        expected_total = self.both_correct + self.both_wrong + self.corrected + self.harmed
        if self.total_claims != expected_total:
            raise ValueError(
                f"Transition conservation invariant violated: total {self.total_claims} != sum {expected_total}"
            )
        if self.net_benefit != (self.corrected - self.harmed):
            raise ValueError(
                f"Net benefit invariant violated: {self.net_benefit} != {self.corrected} - {self.harmed}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class M20bBaselineDiffRecord:
    """Diff record comparing M1.9g B0 vs M2.0a B0 for a single claim."""
    sample_id: str
    claim: str
    gold_verdict: str
    m19g_predicted: str
    m19g_correct: bool
    m20a_predicted: str
    m20a_correct: bool
    is_mismatch: bool
    mismatch_type: str  # AGREED_CORRECT, AGREED_WRONG, M19G_ONLY_CORRECT, M20A_ONLY_CORRECT
    m19g_completion_snippet: str = ""
    m20a_completion_snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
