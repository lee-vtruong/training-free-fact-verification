"""Milestone M2.0d: Schemas and Data Contracts for Contrastive Evidence Verification (Track C).

Defines:
- M20dArm: 6 experimental arms (B0, G2, C1, C2, C3, C4).
- PatternCategory: Inductive taxonomy of failure patterns.
- HistoricalErrorPattern: Structured pattern with provenance.
- PatternSelectionDecision: PATTERN_SELECTED vs NO_PATTERN_SELECTED.
- SelectionTrace: Diagnostic log of pattern selection.
- EvidenceAttribution: Extracted supporting/refuting passage IDs and assessments.
- M20dPredictionRecord: End-to-end telemetry and verification output.
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set


class M20dArm(str, enum.Enum):
    """Six experimental arms evaluated in Track C."""
    B0 = "B0"  # Canonical Zero-Shot Baseline
    G2 = "G2"  # Length-Matched Generic Rules
    C1 = "C1"  # Generic Contrastive Evidence Verification
    C2 = "C2"  # Historical-Error-Informed Contrastive Verification (Primary)
    C3 = "C3"  # Matched Generic Contrastive Control
    C4 = "C4"  # Contrastive Verification with Irrelevant/Shuffled Patterns (Placebo)


class PatternCategory(str, enum.Enum):
    """Taxonomy of failure patterns with provenance."""
    TEMPORAL_CONTRADICTION = "TEMPORAL_CONTRADICTION"
    NUMERICAL_PRECISION = "NUMERICAL_PRECISION"
    PREDICATE_NEGATION_OVERLOOKED = "PREDICATE_NEGATION_OVERLOOKED"
    ENTITY_ATTRIBUTION_CONFOUND = "ENTITY_ATTRIBUTION_CONFOUND"
    EVIDENCE_INSUFFICIENCY = "EVIDENCE_INSUFFICIENCY"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


class PatternSelectionDecision(str, enum.Enum):
    """Decision emitted by the pattern selector."""
    PATTERN_SELECTED = "PATTERN_SELECTED"
    NO_PATTERN_SELECTED = "NO_PATTERN_SELECTED"


@dataclass
class HistoricalErrorPattern:
    """A failure pattern description grounded in historical source errors."""
    pattern_id: str
    category: PatternCategory
    name: str
    description: str
    contrastive_directive: str
    historical_source_ids: List[str]
    provenance_status: str  # e.g. "VALIDATED_HISTORICAL"
    token_estimate: int
    matched_generic_directive: str  # For C3 matched control

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d


@dataclass
class PatternSelectionTrace:
    """Trace recording the deterministic pattern selection process."""
    sample_id: str
    target_claim: str
    decision: PatternSelectionDecision
    selected_pattern_id: Optional[str]
    category: Optional[str]
    salient_signals: List[str]
    selection_score: float
    token_cost: int
    matched_control_id: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d


@dataclass
class EvidenceAttribution:
    """Parsed evidence attribution from contrastive verification output."""
    supporting_passage_ids: List[int] = field(default_factory=list)
    refuting_passage_ids: List[int] = field(default_factory=list)
    scope_assessment: str = ""
    sufficiency_assessment: str = ""
    is_valid_attribution: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class M20dPredictionRecord:
    """Complete prediction and provenance record for M2.0d."""
    sample_id: str
    arm: str
    raw_claim: str
    gold_label: Optional[str]
    model_name: str
    model_revision: str
    retrieval_hash: str
    prompt_sha256: str
    raw_completion: str
    parsed_verdict: Optional[str]
    parse_status: str  # SUCCESS, UNPARSEABLE, AMBIGUOUS
    attribution: EvidenceAttribution
    pattern_decision: str
    selected_pattern_id: Optional[str]
    prompt_tokens: int
    output_tokens: int
    latency_seconds: float
    num_model_calls: int = 1

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["attribution"] = self.attribution.to_dict()
        return d
