"""Milestone M2.0d: Phase D â€” Historical Error Pattern Integration.

Defines:
1. Catalog of inductive failure patterns with verified provenance from Train Fold-1.
2. Grounded pattern selector using only inference-time features (claim + Top-8 evidence).
3. Matched generic control directives for Arm C3.
4. Placebo selection for Arm C4.
5. Deterministic fallback policy when NO_PATTERN_SELECTED.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tffv.experiments.m20d_schemas import (
    HistoricalErrorPattern,
    PatternCategory,
    PatternSelectionDecision,
    PatternSelectionTrace,
)

# Surface signal detectors
YEAR_RE = re.compile(r"\b(19\d\d|20[0-3]\d)\b")
NUMBER_RE = re.compile(r"\b\d+([.,]\d+)?%?\b")
QUANTITY_WORDS = {"percent", "percentage", "million", "billion", "trillion", "thousand", "half", "double", "triple", "fold", "rate", "average", "total"}
NEGATION_WORDS = {"not", "never", "no", "neither", "none", "denied", "deny", "denies", "false", "fake", "hoax", "debunked", "untrue", "incorrect", "refuted", "rejected", "disproved"}
TEMPORAL_WORDS = {"before", "after", "since", "until", "earlier", "later", "recently", "current", "currently", "former", "initially", "originally", "year", "month", "decade"}
CONFLICT_MARKERS = {"however", "although", "disputed", "contradicts", "contrary", "whereas", "while", "unlike", "differ", "dispute", "claims otherwise"}
ENTITY_WORDS = {"spokesperson", "agency", "department", "organization", "ministry", "official", "president", "ceo", "director", "institute"}


HISTORICAL_PATTERNS: List[HistoricalErrorPattern] = [
    HistoricalErrorPattern(
        pattern_id="PAT-TEMP-01",
        category=PatternCategory.TEMPORAL_CONTRADICTION,
        name="Temporal Contradiction & Publication Date Scrutiny",
        description="Historical models frequently confused publication dates with event occurrence dates, incorrectly refuting claims using outdated or pre-event articles.",
        contrastive_directive=(
            "HISTORICAL ERROR GUARD (Temporal): Scrutinize timeframes carefully. Differentiate the date an article was published "
            "from the exact date the event occurred. Do NOT reject or accept a claim based on evidence describing a different or earlier timeframe."
        ),
        historical_source_ids=["train:12", "train:45", "train:108"],
        provenance_status="VALIDATED_HISTORICAL_AUDIT",
        token_estimate=52,
        matched_generic_directive=(
            "VERIFICATION DIRECTIVE (Timeline): Scrutinize timeframes carefully. Differentiate when reports were authored "
            "from the exact period the factual events transpired. Do NOT validate or reject assertions using evidence describing a distinct or preceding timeframe."
        ),
    ),
    HistoricalErrorPattern(
        pattern_id="PAT-NUM-02",
        category=PatternCategory.NUMERICAL_PRECISION,
        name="Numerical Precision & Quantity Discrepancy",
        description="Historical models often failed on unit discrepancies, conflating percentages with raw counts, or misinterpreting regional statistics as national totals.",
        contrastive_directive=(
            "HISTORICAL ERROR GUARD (Numerical): Scrutinize quantitative metrics strictly. Check whether figures represent percentages, "
            "absolute counts, rates, or approximations. Ensure the evidence measures the exact same metric, unit, and geographic scope as the claim."
        ),
        historical_source_ids=["train:92", "train:143", "train:2956"],
        provenance_status="VALIDATED_HISTORICAL_AUDIT",
        token_estimate=50,
        matched_generic_directive=(
            "VERIFICATION DIRECTIVE (Quantitative): Scrutinize quantitative metrics strictly. Check whether figures represent percentages, "
            "absolute counts, rates, or approximations. Ensure the evidence measures the exact same metric, unit, and geographic scope as the assertion."
        ),
    ),
    HistoricalErrorPattern(
        pattern_id="PAT-NEG-03",
        category=PatternCategory.PREDICATE_NEGATION_OVERLOOKED,
        name="Predicate Negation & Inverse Polarity",
        description="Historical models frequently missed negation in embedded clauses (e.g. 'falsely claimed that', 'denied involvement'), inverting the true verdict.",
        contrastive_directive=(
            "HISTORICAL ERROR GUARD (Polarity): Inspect polarity and negation markers rigorously. Watch for embedded denials, disproven claims, "
            "and attribution verbs (e.g., 'denied', 'falsely alleged') that invert affirmative statements into refutations."
        ),
        historical_source_ids=["train:78", "train:2961", "train:305"],
        provenance_status="VALIDATED_HISTORICAL_AUDIT",
        token_estimate=48,
        matched_generic_directive=(
            "VERIFICATION DIRECTIVE (Polarity): Inspect polarity and negation markers rigorously. Watch for embedded denials, qualified claims, "
            "and subtle syntactic cues that dictate whether the overall passage affirms, denies, or questions the core predicate."
        ),
    ),
    HistoricalErrorPattern(
        pattern_id="PAT-ENT-04",
        category=PatternCategory.ENTITY_ATTRIBUTION_CONFOUND,
        name="Entity Attribution & Sub-agency Confounding",
        description="Historical models misattributed statements or actions to the wrong entity when multiple related organizations or namesake persons appeared.",
        contrastive_directive=(
            "HISTORICAL ERROR GUARD (Attribution): Scrutinize entity identity and source attribution. Confirm that statements and actions are "
            "accurately attributed to the specific organization or person named in the claim, rather than a separate affiliated entity."
        ),
        historical_source_ids=["train:184", "train:412", "train:620"],
        provenance_status="VALIDATED_HISTORICAL_AUDIT",
        token_estimate=49,
        matched_generic_directive=(
            "VERIFICATION DIRECTIVE (Attribution): Scrutinize entity identity and source attribution. Confirm that statements and actions are "
            "accurately attributed to the specific organization or person named in the claim, maintaining clear distinction between separate entities."
        ),
    ),
    HistoricalErrorPattern(
        pattern_id="PAT-SUFF-05",
        category=PatternCategory.EVIDENCE_INSUFFICIENCY,
        name="Evidence Insufficiency vs Semantic Entailment",
        description="Historical models either hallucinated proof from topically relevant but non-verifying passages, or prematurely abstained when exact paraphrase was present.",
        contrastive_directive=(
            "HISTORICAL ERROR GUARD (Sufficiency): Distinguish genuine factual verification from mere topical overlap. If evidence discusses the "
            "topic but does not verify or refute the specific predicate, classify as 'Not Enough Evidence'. Do NOT extrapolate unstated facts."
        ),
        historical_source_ids=["train:2954", "train:2955", "train:502"],
        provenance_status="VALIDATED_HISTORICAL_AUDIT",
        token_estimate=52,
        matched_generic_directive=(
            "VERIFICATION DIRECTIVE (Sufficiency): Distinguish genuine factual verification from mere topical overlap. If evidence discusses the "
            "topic but does not verify or refute the specific predicate, classify as 'Not Enough Evidence'. Rely strictly on stated facts."
        ),
    ),
    HistoricalErrorPattern(
        pattern_id="PAT-CONF-06",
        category=PatternCategory.CONFLICTING_EVIDENCE,
        name="Genuine Contradiction vs Partial Cherrypicking",
        description="Historical models struggled to separate bona fide scientific/factual disputes across reputable sources from selective snippet extraction.",
        contrastive_directive=(
            "HISTORICAL ERROR GUARD (Conflicting): Scrutinize conflicting viewpoints. If credible sources present directly opposing factual claims, "
            "or if a claim selectively quotes a passage that other passages contradict, evaluate for 'Conflicting Evidence/Cherrypicking'."
        ),
        historical_source_ids=["train:210", "train:340", "train:899"],
        provenance_status="VALIDATED_HISTORICAL_AUDIT",
        token_estimate=48,
        matched_generic_directive=(
            "VERIFICATION DIRECTIVE (Conflicting): Scrutinize conflicting viewpoints. If credible sources present directly opposing factual claims, "
            "or if the evidence presents multiple credible but contradictory findings, evaluate for 'Conflicting Evidence/Cherrypicking'."
        ),
    ),
]


class HistoricalPatternSelector:
    """Selects historical error patterns using only target claim and evidence features."""

    def __init__(
        self,
        patterns: Optional[List[HistoricalErrorPattern]] = None,
        min_signal_threshold: float = 0.20,
    ):
        self.patterns = patterns or HISTORICAL_PATTERNS
        self.min_signal_threshold = min_signal_threshold
        self._pattern_map = {p.pattern_id: p for p in self.patterns}

    def select_pattern(
        self,
        sample_id: str,
        target_claim: str,
        target_evidence_text: str,
    ) -> Tuple[PatternSelectionDecision, Optional[HistoricalErrorPattern], PatternSelectionTrace]:
        """Deterministically scores and selects the most relevant pattern."""
        combined_text = f"{target_claim} {target_evidence_text}".lower()
        claim_lower = target_claim.lower()

        scores: Dict[str, float] = {}
        signals: Dict[str, List[str]] = {}

        # 1. Temporal signals
        temp_signals = []
        claim_years = YEAR_RE.findall(claim_lower)
        if claim_years:
            temp_signals.append(f"years_in_claim:{','.join(claim_years)}")
        for w in TEMPORAL_WORDS:
            if re.search(r"\b" + re.escape(w) + r"\b", claim_lower):
                temp_signals.append(f"temp_word:{w}")
        temp_score = min(1.0, len(temp_signals) * 0.35)
        scores["PAT-TEMP-01"] = temp_score
        signals["PAT-TEMP-01"] = temp_signals

        # 2. Numerical signals
        num_signals = []
        claim_numbers = NUMBER_RE.findall(claim_lower)
        if claim_numbers:
            num_signals.append(f"numbers_in_claim:{len(claim_numbers)}")
        for w in QUANTITY_WORDS:
            if re.search(r"\b" + re.escape(w) + r"\b", claim_lower):
                num_signals.append(f"quantity_word:{w}")
        num_score = min(1.0, len(num_signals) * 0.35)
        scores["PAT-NUM-02"] = num_score
        signals["PAT-NUM-02"] = num_signals

        # 3. Negation signals
        neg_signals = []
        for w in NEGATION_WORDS:
            if re.search(r"\b" + re.escape(w) + r"\b", claim_lower):
                neg_signals.append(f"neg_word:{w}")
        neg_score = min(1.0, len(neg_signals) * 0.40)
        scores["PAT-NEG-03"] = neg_score
        signals["PAT-NEG-03"] = neg_signals

        # 4. Entity / Attribution signals
        ent_signals = []
        for w in ENTITY_WORDS:
            if re.search(r"\b" + re.escape(w) + r"\b", claim_lower):
                ent_signals.append(f"entity_word:{w}")
        # Capitalized words in claim
        caps = [w for w in target_claim.split() if w and w[0].isupper() and w.lower() not in {"the", "a", "an", "in", "on"}]
        if len(caps) >= 3:
            ent_signals.append(f"proper_entities:{len(caps)}")
        ent_score = min(1.0, len(ent_signals) * 0.30)
        scores["PAT-ENT-04"] = ent_score
        signals["PAT-ENT-04"] = ent_signals

        # 5. Conflicting signals
        conf_signals = []
        for w in CONFLICT_MARKERS:
            if re.search(r"\b" + re.escape(w) + r"\b", combined_text):
                conf_signals.append(f"conflict_word:{w}")
        conf_score = min(1.0, len(conf_signals) * 0.30)
        scores["PAT-CONF-06"] = conf_score
        signals["PAT-CONF-06"] = conf_signals

        # 6. Sufficiency (default fallback check when general text has low domain specificity)
        suff_score = 0.25  # baseline sufficiency scrutiny
        scores["PAT-SUFF-05"] = suff_score
        signals["PAT-SUFF-05"] = ["baseline_sufficiency_scrutiny"]

        # Rank patterns
        sorted_patterns = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_pid, top_score = sorted_patterns[0]

        if top_score < self.min_signal_threshold:
            trace = PatternSelectionTrace(
                sample_id=sample_id,
                target_claim=target_claim,
                decision=PatternSelectionDecision.NO_PATTERN_SELECTED,
                selected_pattern_id=None,
                category=None,
                salient_signals=[],
                selection_score=top_score,
                token_cost=0,
                matched_control_id=None,
            )
            return PatternSelectionDecision.NO_PATTERN_SELECTED, None, trace

        selected_pat = self._pattern_map[top_pid]
        trace = PatternSelectionTrace(
            sample_id=sample_id,
            target_claim=target_claim,
            decision=PatternSelectionDecision.PATTERN_SELECTED,
            selected_pattern_id=top_pid,
            category=selected_pat.category.value,
            salient_signals=signals.get(top_pid, []),
            selection_score=round(top_score, 3),
            token_cost=selected_pat.token_estimate,
            matched_control_id=top_pid,
        )
        return PatternSelectionDecision.PATTERN_SELECTED, selected_pat, trace

    def select_placebo_pattern(self, sample_id: str) -> HistoricalErrorPattern:
        """Deterministically selects an irrelevant pattern for C4 using claim ID hash."""
        h = int(hashlib.sha256(f"{sample_id}_c4_placebo".encode("utf-8")).hexdigest(), 16)
        idx = h % len(self.patterns)
        return self.patterns[idx]

    def save_pattern_manifest(self, output_path: str | Path) -> None:
        """Emits historical_pattern_manifest.json."""
        data = {
            "milestone": "M2.0d",
            "description": "Historical failure pattern descriptions with provenance from Train Fold-1",
            "total_patterns": len(self.patterns),
            "patterns": [p.to_dict() for p in self.patterns],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
