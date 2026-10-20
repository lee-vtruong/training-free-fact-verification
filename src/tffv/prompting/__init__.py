"""B0-B8 specifications only; method implementations follow source audit."""
from dataclasses import dataclass


BASELINES = {
    "B0": "Frozen Base, original prompt",
    "B1": "Generic verification instruction",
    "B2": "Generic structured evidence reasoning",
    "B3": "Random source demonstrations",
    "B4": "Correct-only source demonstrations",
    "B5": "Error-only source demonstrations with explicit gold correction",
    "B6": "Mixed correct/wrong source demonstrations with explicit corrections",
    "B7": "Source-derived error rules",
    "B8": "Error-aware contrastive demonstrations with explicit corrections",
}


@dataclass(frozen=True)
class BaselineSpec:
    baseline_id: str
    checkpoint_sha256: str
    labels: tuple[str, ...]
    scoring_protocol: str
    evidence_protocol: str
    dataset: str
    prompt_sha256: str
    seed: int = 2026
    demonstration_count: int = 0
    context_token_budget: int = 4096

    def __post_init__(self):
        if self.baseline_id not in BASELINES:
            raise ValueError("Unknown baseline")
        if self.demonstration_count < 0 or self.context_token_budget <= 0:
            raise ValueError("Invalid context budget")

    def comparison_protocol(self):
        return (self.checkpoint_sha256, self.labels, self.scoring_protocol, self.evidence_protocol, self.dataset)


def require_comparable(base, candidate):
    if base.comparison_protocol() != candidate.comparison_protocol():
        raise ValueError("Incomparable model/ontology/scoring/evidence/dataset")


def correction_target(record):
    return {"historical_base_prediction": record["prediction"], "correct_verdict": record["gold_label"],
            "historical_prediction_was_wrong": not record["correct"]}
