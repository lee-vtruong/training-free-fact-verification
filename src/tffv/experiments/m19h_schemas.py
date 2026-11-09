"""Milestone M1.9h: Constants, Data Models, and Common Schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

OFFICIAL_LABELS = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]

OFFICIAL_ARMS = ["B0", "B1", "H1", "H2", "H3", "B2"]

EXACT_TRAIN_DEV_OVERLAPS = ["train:1340", "train:1341", "train:1536", "train:1539"]
QUARANTINED_TRAIN_IDS = EXACT_TRAIN_DEV_OVERLAPS + ["train:1948"]

M19G_REPORTED_METRICS = {
    "B0": {"macro_f1": 0.4838, "accuracy": 0.6000},
    "B1": {"macro_f1": 0.4428, "accuracy": 0.5940},
    "H1": {"macro_f1": 0.4474, "accuracy": 0.6080},
    "H2": {"macro_f1": 0.4286, "accuracy": 0.5520},
    "H3": {"macro_f1": 0.4586, "accuracy": 0.5960},
    "B2": {"macro_f1": 0.4713, "accuracy": 0.6240},
}


class TransitionCategory(str, Enum):
    BOTH_CORRECT = "both_correct"
    BOTH_WRONG = "both_wrong"
    HARMED = "harmed"  # Baseline correct -> Candidate wrong
    CORRECTED = "corrected"  # Baseline wrong -> Candidate correct
    BOTH_WRONG_SHIFTED = "both_wrong_shifted"  # Both wrong but prediction class changed


@dataclass
class ClaimRecord:
    claim_id: str
    claim_text: str
    gold_label: str
    predictions: Dict[str, str] = field(default_factory=dict)
    is_correct: Dict[str, bool] = field(default_factory=dict)
    is_parseable: Dict[str, bool] = field(default_factory=dict)
    evidence_count: int = 0
    evidence_char_len: int = 0
    target_evidence_hash: str = ""
    prompt_tokens: Dict[str, int] = field(default_factory=dict)
    chosen_exemplars: Dict[str, List[str]] = field(default_factory=dict)
    similarity_scores: Dict[str, List[float]] = field(default_factory=dict)
    exemplar_outcomes: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class TransitionStats:
    comparison: str
    arm_candidate: str
    arm_baseline: str
    total_claims: int
    both_correct: int
    both_wrong: int
    harmed: int  # baseline correct -> cand wrong (n_01)
    corrected: int  # baseline wrong -> cand correct (n_10)
    both_wrong_shifted: int
    net_benefit: int  # corrected - harmed (n_10 - n_01)
    harm_rate: float
    correction_rate: float
    net_benefit_rate: float
    mcnemar_b: int  # arm_candidate correct, arm_baseline wrong (n_10 / corrected)
    mcnemar_c: int  # arm_candidate wrong, arm_baseline correct (n_01 / harmed)
    mcnemar_p_value: float
