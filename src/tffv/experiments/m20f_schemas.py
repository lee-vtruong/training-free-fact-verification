"""Milestone M2.0f: Schemas and Data Contracts for Factorial Ablation of Attribution and Historical Error Knowledge.

Defines:
- M20fArm: 6 experimental arms:
  * B0: Canonical Zero-Shot Baseline
  * G2: Generic Evidence-Guided Reasoning Control
  * A1: Matched Gold ICL, No Forced Attribution (Factor A: Gold, Factor B: NoAttr)
  * A2: Matched Gold ICL + Forced Evidence Attribution (Factor A: Gold, Factor B: ForcedAttr)
  * A3: Historical Correct/Wrong ICL, No Forced Attribution (Factor A: Error, Factor B: NoAttr)
  * A4: Historical Correct/Wrong ICL + Forced Evidence Attribution (Factor A: Error, Factor B: ForcedAttr)
- KnowledgeFactor: GOLD_ICL vs HISTORICAL_ERROR_ICL
- AttributionFactor: NO_FORCED_ATTR vs FORCED_ATTR
- ExemplarDemonstration: Structured few-shot exemplar record.
- M20fPredictionRecord: Full verification telemetry.
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class M20fArm(str, enum.Enum):
    """Six experimental arms evaluated in Milestone M2.0f."""
    B0 = "B0"  # Canonical Zero-Shot Baseline
    G2 = "G2"  # Generic Evidence-Guided Reasoning Control
    A1 = "A1"  # Matched Gold ICL, No Forced Attribution
    A2 = "A2"  # Matched Gold ICL + Forced Attribution
    A3 = "A3"  # Historical Error ICL, No Forced Attribution
    A4 = "A4"  # Historical Error ICL + Forced Attribution


class KnowledgeFactor(str, enum.Enum):
    """Factor A in 2x2 Factorial Design."""
    NONE = "NONE"
    GOLD_ICL = "GOLD_ICL"
    HISTORICAL_ERROR_ICL = "HISTORICAL_ERROR_ICL"


class AttributionFactor(str, enum.Enum):
    """Factor B in 2x2 Factorial Design."""
    NO_FORCED_ATTR = "NO_FORCED_ATTR"
    FORCED_ATTR = "FORCED_ATTR"


@dataclass
class ExemplarDemonstration:
    """A clean few-shot demonstration from Train Fold-1."""
    sample_id: str
    claim_text: str
    evidence_snippet: str
    gold_verdict: str
    # Gold reasoning fields (for Factor A = GOLD_ICL)
    scope_analysis: str
    factual_basis: str
    # Historical error fields (for Factor A = HISTORICAL_ERROR_ICL)
    historical_mistake: str
    corrective_resolution: str
    # Attribution fields (for Factor B = FORCED_ATTR)
    supporting_passage_ids: List[int] = field(default_factory=list)
    refuting_passage_ids: List[int] = field(default_factory=list)
    scope_assessment: str = ""
    sufficiency_assessment: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class M20fPredictionRecord:
    """Complete prediction and telemetry record for Milestone M2.0f."""
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
    parse_status: str
    knowledge_factor: str
    attribution_factor: str
    supporting_passage_ids: List[int] = field(default_factory=list)
    refuting_passage_ids: List[int] = field(default_factory=list)
    scope_assessment: str = ""
    sufficiency_assessment: str = ""
    is_valid_attribution: bool = False
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
