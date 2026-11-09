"""Milestone M2.0-A: Error-Derived Verification Rules Schemas & Data Contracts.

Strictly defines:
1. Observed error vs. Hypothesized error cause contracts.
2. Verification rule structures with full provenance & hash verification.
3. Five experimental arms: B0, G1, G2, E1, E2.
4. Evaluation metrics and transition schemas with conservation invariants.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class M20aArm(str, enum.Enum):
    """The 5 experimental arms in M2.0-A."""
    B0 = "B0"  # Zero-Shot Base (No rules)
    G1 = "G1"  # Generic Rules (High-level standard fact-checking guidelines)
    G2 = "G2"  # Length-Matched Generic Rules (Detailed guidelines matching E1 token length)
    E1 = "E1"  # Error-Derived Verification Rules (Mined from historical failure patterns)
    E2 = "E2"  # Shuffled/Irrelevant Rules (Matched-length negative control for prompt distraction)


EXPERIMENTAL_ARMS_M20A = [
    M20aArm.B0.value,
    M20aArm.G1.value,
    M20aArm.G2.value,
    M20aArm.E1.value,
    M20aArm.E2.value,
]

OFFICIAL_CLASSES = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]


class ErrorTaxonomyCategory(str, enum.Enum):
    """Inductive taxonomy of observed model failure modes."""
    TEMPORAL_CONTRADICTION = "TEMPORAL_CONTRADICTION"
    NUMERICAL_PRECISION = "NUMERICAL_PRECISION"
    PARTIAL_SUPPORT_CHERRYPICKING = "PARTIAL_SUPPORT_CHERRYPICKING"
    PREDICATE_NEGATION_OVERLOOKED = "PREDICATE_NEGATION_OVERLOOKED"
    ENTITY_ATTRIBUTION_CONFOUND = "ENTITY_ATTRIBUTION_CONFOUND"
    ABSTENTION_THRESHOLD_BIAS = "ABSTENTION_THRESHOLD_BIAS"
    SCOPE_GENERALIZATION = "SCOPE_GENERALIZATION"


@dataclass
class ObservedError:
    """Factual, directly observed prediction failure (No speculative reasoning)."""
    sample_id: str
    claim_text: str
    gold_verdict: str
    base_verdict: str
    transition_type: str  # e.g., 'Predicted Supported when Gold is Refuted'
    evidence_count: int
    evidence_snippets: List[str]
    discrepancy_signals: List[str]  # e.g. ["date_mismatch", "negation_token_present"]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HypothesizedCause:
    """Inductive explanation of why the verifier made the observed error."""
    category: ErrorTaxonomyCategory
    hypothesized_mechanism: str
    remedy_hypothesis: str
    is_hypothesized: bool = True  # Invariant: strictly marked as hypothesis

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "hypothesized_mechanism": self.hypothesized_mechanism,
            "remedy_hypothesis": self.remedy_hypothesis,
            "is_hypothesized": self.is_hypothesized,
        }


@dataclass
class ErrorPatternRecord:
    """Unified record pairing an observed error with its hypothesized cause and provenance."""
    pattern_id: str
    observed: ObservedError
    hypothesized: HypothesizedCause
    source_fold: str
    group_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "observed": self.observed.to_dict(),
            "hypothesized": self.hypothesized.to_dict(),
            "source_fold": self.source_fold,
            "group_id": self.group_id,
        }


@dataclass
class VerificationRule:
    """Individual symbolic verification directive."""
    rule_id: str
    name: str
    condition: str
    directive: str
    target_classes: List[str]
    elaboration: str = ""
    source_category: Optional[str] = None
    exemplifying_source_ids: List[str] = field(default_factory=list)

    def render_markdown(self) -> str:
        md = f"**Rule {self.rule_id} ({self.name}):**\n"
        md += f"- *Condition:* {self.condition}\n"
        md += f"- *Directive:* {self.directive}\n"
        if self.elaboration:
            md += f"- *Guidance:* {self.elaboration}\n"
        return md


@dataclass
class RuleSetProvenance:
    """Cryptographic provenance sealing a rule set."""
    rule_set_id: str
    arm: str
    source_fold: str
    source_claim_ids: List[str]
    source_claims_hash: str
    generator_name: str
    generation_timestamp: str
    rule_set_hash: str
    group_isolation_verified: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RuleSet:
    """Collection of verification rules with provenance."""
    rule_set_id: str
    arm: str
    title: str
    description: str
    rules: List[VerificationRule]
    provenance: RuleSetProvenance
    token_count_estimate: int = 0

    def render_rules_block(self) -> str:
        if not self.rules:
            return ""
        rendered = [f"### {self.title}\n", f"{self.description}\n\n"]
        for r in self.rules:
            rendered.append(r.render_markdown())
            rendered.append("\n")
        return "".join(rendered).strip()

    def compute_hash(self) -> str:
        canonical = json.dumps([asdict(r) for r in self.rules], sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class M20aPromptReceipt:
    sample_id: str
    arm: str
    prompt_sha256: str
    token_count_estimate: int
    rule_set_id: str


@dataclass
class M20aPredictionRecord:
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
    rule_set_id: str


@dataclass
class M20aTransitionStats:
    """Pairwise error transition statistics with conservation invariant."""
    comparison: str
    cand_arm: str
    base_arm: str
    total_claims: int
    both_correct: int
    both_wrong: int
    harmed: int      # Base correct -> Cand wrong (n_01)
    corrected: int   # Base wrong -> Cand correct (n_10)
    net_benefit: int # corrected - harmed
    harm_rate: float
    correction_rate: float
    net_benefit_rate: float
    harm_to_correction_ratio: float
    mcnemar_b: int   # corrected
    mcnemar_c: int   # harmed
    mcnemar_p_value: float
    delta_macro_f1: float
    delta_macro_f1_ci: Tuple[float, float]
    delta_accuracy: float
    delta_accuracy_ci: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
