"""Milestone M2.0g: Schemas and Data Contracts for Historical Error Robustness & Attribution.

Defines contracts for:
- E1: Exemplar Robustness & Seed Sensitivity (k in {2, 4, 6}, seeds 1..5)
- E2: Knowledge Component Ablation (K1, K2, K3, K4)
- E3: Source-Fold Generalization (Group-Isolated Outer Fold)
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class ExperimentPhase(str, enum.Enum):
    """Three primary experimental phases of Milestone M2.0g."""
    E1_ROBUSTNESS = "E1_ROBUSTNESS"
    E2_ABLATION = "E2_ABLATION"
    E3_GENERALIZATION = "E3_GENERALIZATION"


class KnowledgeComponent(str, enum.Enum):
    """The 4 knowledge component levels in Experiment E2."""
    K1_GOLD_ONLY = "K1_GOLD_ONLY"                          # Claim + Evidence -> Gold Verdict
    K2_GOLD_NEUTRAL_EXPL = "K2_GOLD_NEUTRAL_EXPL"          # Claim + Evidence -> Neutral Explanation -> Gold Verdict
    K3_WRONG_PRED_GOLD_CORR = "K3_WRONG_PRED_GOLD_CORR"    # Claim + Evidence -> Prior Wrong + Gold Verdict
    K4_FULL_GROUNDED_ERROR = "K4_FULL_GROUNDED_ERROR"      # Claim + Evidence -> Observed Mistake + Corrective Resolution -> Gold Verdict


@dataclass
class ExemplarItem:
    """Certified exemplar demonstration from Train Fold-1."""
    sample_id: str
    group_id: str
    claim_text: str
    evidence_snippet: str
    gold_verdict: str
    historical_wrong_verdict: str
    neutral_explanation: str
    observed_mistake: str
    corrective_resolution: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class M20gPredictionRecord:
    """Unified telemetry record for Milestone M2.0g."""
    sample_id: str
    experiment_phase: str
    arm_name: str
    seed: int
    k_shots: int
    knowledge_component: str
    raw_claim: str
    gold_label: Optional[str]
    model_name: str
    model_revision: str
    prompt_sha256: str
    raw_completion: str
    parsed_verdict: Optional[str]
    parse_status: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
