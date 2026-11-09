"""Milestone M2.0b: Phase C — Instance-Grounded Error Selector & Evidence-Grounding Filter.

Implements:
1. Evidence-conditioned historical error retrieval strictly isolated to Train Fold-1.
2. Evidence-Grounding Filter: verifies candidate error lessons share lexical/entity
   anchors with target retrieved evidence.
3. Explicit NO_INTERVENTION fallback threshold when target evidence lacks sufficient grounding.
4. Matched controls:
   - B1: Instance-Grounded Error Intervention (Primary Method)
   - B2: Instance-Grounded Generic Control (Matched Selection/Tokens)
   - B3: Irrelevant/Random Historical Control
5. Emits:
   - selection_traces.jsonl
   - grounding_audit.json
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tffv.experiments.m20b_schemas import (
    GroundingAuditRecord,
    InterventionDecision,
    M20bArm,
    SelectionTrace,
)

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
}


def tokenize(text: str) -> List[str]:
    """Tokenize and filter stopwords and non-alphanumeric tokens."""
    tokens = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


@dataclass
class HistoricalErrorCase:
    sample_id: str
    claim: str
    gold_verdict: str
    base_verdict: str
    evidence_text: str
    error_lesson: str
    generic_guidance: str
    is_validated_correction: bool = False
    source_fold: str = "train_fold_1"


class InstanceGroundedSelector:
    """Selects and grounds historical error knowledge for target claims."""

    def __init__(
        self,
        error_cases: List[HistoricalErrorCase],
        min_grounding_threshold: float = 0.12,
        min_shared_terms: int = 2,
    ):
        self.error_cases = error_cases
        self.min_grounding_threshold = min_grounding_threshold
        self.min_shared_terms = min_shared_terms
        self._build_index()

    def _build_index(self):
        """Builds TF-IDF index over historical error cases."""
        self.doc_ids = [c.sample_id for c in self.error_cases]
        self.case_map = {c.sample_id: c for c in self.error_cases}
        self.doc_tokens = {}
        self.df = Counter()
        self.num_docs = len(self.error_cases)

        for case in self.error_cases:
            combined = f"{case.claim} {case.evidence_text}"
            tokens = tokenize(combined)
            self.doc_tokens[case.sample_id] = tokens
            for term in set(tokens):
                self.df[term] += 1

        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log((self.num_docs + 1) / (freq + 1)) + 1.0

        self.doc_vectors = {}
        self.doc_norms = {}
        for sid, tokens in self.doc_tokens.items():
            tf = Counter(tokens)
            vec = {term: count * self.idf[term] for term, count in tf.items()}
            norm = math.sqrt(sum(w * w for w in vec.values()))
            self.doc_vectors[sid] = vec
            self.doc_norms[sid] = norm if norm > 0 else 1.0

    def query(
        self,
        target_claim: str,
        target_evidence_text: str,
    ) -> List[Tuple[str, float]]:
        """Computes cosine similarity of target query against all historical cases."""
        q_tokens = tokenize(f"{target_claim} {target_evidence_text}")
        q_tf = Counter(q_tokens)
        q_vec = {t: count * self.idf.get(t, 1.0) for t, count in q_tf.items()}
        q_norm = math.sqrt(sum(w * w for w in q_vec.values()))
        if q_norm == 0:
            return [(c.sample_id, 0.0) for c in self.error_cases]

        scores: List[Tuple[str, float]] = []
        for sid, d_vec in self.doc_vectors.items():
            dot = sum(q_vec[t] * d_vec[t] for t in q_vec if t in d_vec)
            score = dot / (q_norm * self.doc_norms[sid])
            scores.append((sid, score))

        # Deterministic sort: descending by score, ascending by sample_id
        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores

    def select_intervention(
        self,
        sample_id: str,
        arm: str,
        target_claim: str,
        target_evidence_text: str,
    ) -> Tuple[InterventionDecision, Optional[HistoricalErrorCase], SelectionTrace, GroundingAuditRecord]:
        """Performs grounded selection for a specific claim and arm."""
        if not self.error_cases:
            # Empty memory fallback
            trace = SelectionTrace(
                sample_id=sample_id,
                arm=arm,
                intervention_decision=InterventionDecision.NO_INTERVENTION.value,
                relevance_score=0.0,
                grounding_overlap_score=0.0,
                selected_exemplar_id=None,
                selected_rationale_type="NONE",
                token_budget=0,
            )
            audit = GroundingAuditRecord(
                sample_id=sample_id,
                target_claim=target_claim,
                historical_sample_id=None,
                evidence_grounding_verified=False,
                overlapping_terms=[],
                filter_rationale="NO_HISTORICAL_CASES_IN_STORE",
            )
            return InterventionDecision.NO_INTERVENTION, None, trace, audit

        # Arms B0, G2, E1 do not use instance-level historical cases
        if arm in (M20bArm.B0.value, M20bArm.G2.value, M20bArm.E1.value):
            trace = SelectionTrace(
                sample_id=sample_id,
                arm=arm,
                intervention_decision=InterventionDecision.NO_INTERVENTION.value,
                relevance_score=0.0,
                grounding_overlap_score=0.0,
                selected_exemplar_id=None,
                selected_rationale_type="NONE",
                token_budget=0,
            )
            audit = GroundingAuditRecord(
                sample_id=sample_id,
                target_claim=target_claim,
                historical_sample_id=None,
                evidence_grounding_verified=False,
                overlapping_terms=[],
                filter_rationale=f"ARM_{arm}_USES_NO_INSTANCE_RETRIEVAL",
            )
            return InterventionDecision.NO_INTERVENTION, None, trace, audit

        ranked_scores = self.query(target_claim, target_evidence_text)

        if arm == M20bArm.B3.value:
            # B3: Irrelevant Control -> pick lowest similarity historical item
            lowest_sid, lowest_score = ranked_scores[-1]
            case = self.case_map[lowest_sid]
            trace = SelectionTrace(
                sample_id=sample_id,
                arm=arm,
                intervention_decision=InterventionDecision.INTERVENE.value,
                relevance_score=lowest_score,
                grounding_overlap_score=0.0,
                selected_exemplar_id=lowest_sid,
                selected_rationale_type="IRRELEVANT",
                token_budget=len(case.error_lesson.split()),
                is_validated_correction=case.is_validated_correction,
            )
            audit = GroundingAuditRecord(
                sample_id=sample_id,
                target_claim=target_claim,
                historical_sample_id=lowest_sid,
                evidence_grounding_verified=False,
                overlapping_terms=[],
                filter_rationale="B3_INTENTIONAL_IRRELEVANT_CONTROL",
            )
            return InterventionDecision.INTERVENE, case, trace, audit

        # For B1 (Error Intervention) and B2 (Generic Control):
        target_tokens = set(tokenize(f"{target_claim} {target_evidence_text}"))
        best_sid, best_score = ranked_scores[0]
        candidate_case = self.case_map[best_sid]

        # Evidence Grounding Filter
        case_tokens = set(tokenize(f"{candidate_case.claim} {candidate_case.evidence_text}"))
        shared_terms = sorted(list(target_tokens.intersection(case_tokens)))
        overlap_score = len(shared_terms) / max(1, len(case_tokens))

        passes_grounding = (
            overlap_score >= self.min_grounding_threshold and len(shared_terms) >= self.min_shared_terms
        )

        if not passes_grounding:
            # NO_INTERVENTION fallback: fail-closed to canonical zero-shot
            trace = SelectionTrace(
                sample_id=sample_id,
                arm=arm,
                intervention_decision=InterventionDecision.NO_INTERVENTION.value,
                relevance_score=best_score,
                grounding_overlap_score=overlap_score,
                selected_exemplar_id=None,
                selected_rationale_type="NONE",
                token_budget=0,
            )
            audit = GroundingAuditRecord(
                sample_id=sample_id,
                target_claim=target_claim,
                historical_sample_id=best_sid,
                evidence_grounding_verified=False,
                overlapping_terms=shared_terms,
                filter_rationale=f"BELOW_GROUNDING_THRESHOLD (score={overlap_score:.3f} < {self.min_grounding_threshold})",
            )
            return InterventionDecision.NO_INTERVENTION, None, trace, audit

        # INTERVENE: Grounding passed
        rationale_type = "VALIDATED_CORRECTION" if candidate_case.is_validated_correction else "ERROR_CASE"
        if arm == M20bArm.B2.value:
            rationale_type = "GENERIC_CASE"

        trace = SelectionTrace(
            sample_id=sample_id,
            arm=arm,
            intervention_decision=InterventionDecision.INTERVENE.value,
            relevance_score=best_score,
            grounding_overlap_score=overlap_score,
            selected_exemplar_id=best_sid,
            selected_rationale_type=rationale_type,
            token_budget=len(candidate_case.error_lesson.split()),
            is_validated_correction=candidate_case.is_validated_correction,
        )
        audit = GroundingAuditRecord(
            sample_id=sample_id,
            target_claim=target_claim,
            historical_sample_id=best_sid,
            evidence_grounding_verified=True,
            overlapping_terms=shared_terms,
            filter_rationale=f"GROUNDING_VERIFIED (score={overlap_score:.3f}, shared={len(shared_terms)})",
        )
        return InterventionDecision.INTERVENE, candidate_case, trace, audit
