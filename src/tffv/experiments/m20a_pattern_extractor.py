"""Milestone M2.0-A: Error Pattern Extractor.

Mines observed prediction failures from source-train records, pairs them with
reproducible inductive failure hypotheses, and enforces strict group-fold isolation.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from tffv.experiments.m19f_evidence import format_evidence_block
from tffv.experiments.m20a_schemas import (
    ErrorPatternRecord,
    ErrorTaxonomyCategory,
    HypothesizedCause,
    ObservedError,
)

# Common regex patterns for grounded discrepancy signals
YEAR_RE = re.compile(r"\b(19\d\d|20[0-3]\d)\b")
NUMBER_RE = re.compile(r"\b\d+([.,]\d+)?\b")
NEGATION_WORDS = {"not", "never", "no", "neither", "none", "false", "untrue", "debunked", "incorrect"}
HEDGING_WORDS = {"may", "might", "could", "alleged", "reportedly", "claimed", "possibly", "suggests"}


class ErrorPatternExtractor:
    """Extracts structured error patterns from source train predictions with strict isolation."""

    def __init__(
        self,
        source_fold_name: str = "train_fold_1",
        allowed_sample_ids: Optional[Set[str]] = None,
        disallowed_sample_ids: Optional[Set[str]] = None,
    ):
        self.source_fold_name = source_fold_name
        self.allowed_sample_ids = allowed_sample_ids
        self.disallowed_sample_ids = disallowed_sample_ids or set()

    def validate_isolation(self, sample_id: str) -> None:
        """Enforces that evaluation splits (Dev/Test) never leak into error extraction."""
        if sample_id.startswith("dev:") or sample_id.startswith("test:"):
            raise ValueError(
                f"[ISOLATION VIOLATION] Sample {sample_id} belongs to evaluation split. "
                f"Knowledge derivation MUST strictly use source training folds only."
            )
        if sample_id in self.disallowed_sample_ids:
            raise ValueError(
                f"[ISOLATION VIOLATION] Sample {sample_id} is explicitly disallowed from source knowledge."
            )
        if self.allowed_sample_ids is not None and sample_id not in self.allowed_sample_ids:
            raise ValueError(
                f"[ISOLATION VIOLATION] Sample {sample_id} is not part of {self.source_fold_name} partition."
            )

    def extract_pattern(
        self,
        sample_id: str,
        claim_text: str,
        gold_verdict: str,
        base_verdict: str,
        evidence_passages: Sequence[Dict[str, Any] | str],
        base_completion: str = "",
        group_id: Optional[str] = None,
    ) -> Optional[ErrorPatternRecord]:
        """Extracts an ErrorPatternRecord if base_verdict != gold_verdict."""
        self.validate_isolation(sample_id)

        # Skip if model was already correct
        if base_verdict.strip().lower() == gold_verdict.strip().lower():
            return None

        # Build observed error record
        snippets: List[str] = []
        full_evidence_text = ""
        for p in evidence_passages:
            if isinstance(p, dict):
                text = p.get("text", "")
            else:
                text = str(p)
            if text:
                snippets.append(text[:200])
                full_evidence_text += " " + text

        transition_type = f"Predicted {base_verdict} when Gold is {gold_verdict}"

        # Detect factual discrepancy signals
        signals = self._detect_discrepancy_signals(claim_text, full_evidence_text, gold_verdict, base_verdict)

        observed = ObservedError(
            sample_id=sample_id,
            claim_text=claim_text,
            gold_verdict=gold_verdict,
            base_verdict=base_verdict,
            transition_type=transition_type,
            evidence_count=len(evidence_passages),
            evidence_snippets=snippets[:5],
            discrepancy_signals=signals,
        )

        # Hypothesize inductive failure cause
        hypothesized = self._hypothesize_cause(observed, full_evidence_text)

        pattern_id = f"pat_{sample_id}_{hashlib.md5(transition_type.encode()).hexdigest()[:8]}"

        return ErrorPatternRecord(
            pattern_id=pattern_id,
            observed=observed,
            hypothesized=hypothesized,
            source_fold=self.source_fold_name,
            group_id=group_id,
        )

    def _detect_discrepancy_signals(
        self,
        claim: str,
        evidence: str,
        gold: str,
        pred: str,
    ) -> List[str]:
        signals = []
        c_lower = claim.lower()
        e_lower = evidence.lower()

        # Date / Year mismatch
        c_years = set(YEAR_RE.findall(claim))
        e_years = set(YEAR_RE.findall(evidence))
        if c_years and e_years and not (c_years & e_years):
            signals.append("temporal_mismatch_detected")

        # Number mismatch
        c_nums = set(NUMBER_RE.findall(claim))
        e_nums = set(NUMBER_RE.findall(evidence))
        if c_nums and e_nums and not (c_nums & e_nums):
            signals.append("numerical_mismatch_detected")

        # Explicit negation token in evidence
        e_words = set(re.findall(r"\b\w+\b", e_lower))
        if any(w in e_words for w in NEGATION_WORDS):
            signals.append("explicit_negation_in_evidence")

        # Hedging in claim or evidence
        if any(w in c_lower for w in HEDGING_WORDS):
            signals.append("claim_hedging_present")

        # Label transitions
        if gold == "Refuted" and pred == "Supported":
            signals.append("polarity_inversion_to_supported")
        elif gold == "Refuted" and pred in ("Not Enough Evidence", "Conflicting Evidence/Cherrypicking"):
            signals.append("abstention_shift_from_refuted")
        elif gold == "Supported" and pred == "Refuted":
            signals.append("polarity_inversion_to_refuted")

        return signals

    def _hypothesize_cause(self, observed: ObservedError, evidence: str) -> HypothesizedCause:
        """Determines the inductive error taxonomy category and testable hypothesis."""
        gold = observed.gold_verdict
        pred = observed.base_verdict
        signals = set(observed.discrepancy_signals)

        if "temporal_mismatch_detected" in signals:
            return HypothesizedCause(
                category=ErrorTaxonomyCategory.TEMPORAL_CONTRADICTION,
                hypothesized_mechanism="Model overlooks date/year differences between claim and evidence, falsely treating an event that happened at a different time as Conflicting or Supported.",
                remedy_hypothesis="Enforce an explicit rule: if event occurred at a conflicting time/year, verdict is Refuted, not Conflicting or Supported.",
            )

        if "numerical_mismatch_detected" in signals:
            return HypothesizedCause(
                category=ErrorTaxonomyCategory.NUMERICAL_PRECISION,
                hypothesized_mechanism="Model treats approximately close or conflicting figures as supported rather than checking quantitative exactness.",
                remedy_hypothesis="Enforce explicit numerical scrutiny: statistics or quantities not matching the evidence tolerance must not be marked Supported.",
            )

        if gold == "Conflicting Evidence/Cherrypicking" and pred == "Supported":
            return HypothesizedCause(
                category=ErrorTaxonomyCategory.PARTIAL_SUPPORT_CHERRYPICKING,
                hypothesized_mechanism="Model verifies only one clause of a compound claim and marks the whole claim Supported, ignoring unverified or conflicting clauses.",
                remedy_hypothesis="Enforce partial support rule: compound claims require all conjuncts to be corroborated; otherwise verdict is Conflicting Evidence or Refuted.",
            )

        if gold == "Refuted" and pred == "Not Enough Evidence" and "explicit_negation_in_evidence" in signals:
            return HypothesizedCause(
                category=ErrorTaxonomyCategory.PREDICATE_NEGATION_OVERLOOKED,
                hypothesized_mechanism="Model defaults conservatively to Not Enough Evidence even when an explicit negation sentence directly refutes the claim core predicate.",
                remedy_hypothesis="Enforce negation rule: if any evidence sentence explicitly negates the core factual assertion, mark Refuted rather than abstaining to NEI.",
            )

        if pred in ("Not Enough Evidence", "Conflicting Evidence/Cherrypicking"):
            return HypothesizedCause(
                category=ErrorTaxonomyCategory.ABSTENTION_THRESHOLD_BIAS,
                hypothesized_mechanism="Model exhibits excessive abstention bias when evidence contains nuance, defaulting away from definitive Refuted/Supported verdicts.",
                remedy_hypothesis="Enforce decision boundary rule: weigh primary direct evidence over peripheral contextual ambiguity.",
            )

        # Default fallback categorization
        return HypothesizedCause(
            category=ErrorTaxonomyCategory.SCOPE_GENERALIZATION,
            hypothesized_mechanism="Semantic drift between broad claim scope and specific evidence context.",
            remedy_hypothesis="Verify whether the claim makes a categorical generalization unsupported by sample-scoped evidence.",
        )
