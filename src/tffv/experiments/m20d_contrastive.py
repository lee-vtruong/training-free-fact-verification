"""Milestone M2.0d: Phase C & F â€” Contrastive Evidence Verification Prompting & Hardened Parsing.

Implements:
1. Prompt rendering for all 6 arms:
   - B0: Canonical Zero-Shot Baseline.
   - G2: Length-Matched Generic Rules.
   - C1: Generic Contrastive Evidence Verification.
   - C2: Historical-Error-Informed Contrastive Verification (Primary).
   - C3: Matched Generic Contrastive Control.
   - C4: Placebo Control (Irrelevant/Shuffled Patterns).
2. Hardened Verdict and Attribution Parser:
   - Extracts Supporting and Refuting passage IDs.
   - Extracts Scope and Sufficiency assessments.
   - Robustly isolates final verdict against preamble, echo, and markdown artifacts.
   - Fails closed with UNPARSEABLE when invalid.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from tffv.experiments.m19f_evidence import format_evidence_block, normalize_evidence_list
from tffv.experiments.m20d_error_patterns import HistoricalErrorPattern
from tffv.experiments.m20d_schemas import (
    EvidenceAttribution,
    M20dArm,
    PatternSelectionDecision,
)

VALID_VERDICTS = {
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
}

# Regex helpers for structured attribution
SUPPORTING_IDS_RE = re.compile(r"Supporting\s+Passage\s+IDs?:\s*(\[[^\]]*\]|None|\d+(?:,\s*\d+)*)", re.IGNORECASE)
REFUTING_IDS_RE = re.compile(r"Refuting\s+Passage\s+IDs?:\s*(\[[^\]]*\]|None|\d+(?:,\s*\d+)*)", re.IGNORECASE)
SCOPE_ASSESSMENT_RE = re.compile(r"Scope\s*(?:&|and)?\s*Relevance\s*Assessment:\s*([^\n]+)", re.IGNORECASE)
SUFFICIENCY_ASSESSMENT_RE = re.compile(r"Sufficiency\s*(?:&|and)?\s*Conflict\s*Assessment:\s*([^\n]+)", re.IGNORECASE)
VERDICT_SECTION_RE = re.compile(r"\[VERDICT\]\s*([^\n]+)", re.IGNORECASE)


GENERIC_RULES_G2 = """[VERIFICATION RULES]
1. Explicit Direct Evidence: Prioritize verbatim statements from primary sources. Do NOT extrapolate or assume facts not directly stated.
2. Temporal Alignment: Ensure the evidence timeframe strictly matches the claim date. Pre-event or outdated reports cannot verify subsequent events.
3. Quantifier & Unit Precision: Distinguish absolute figures from percentages, rates, or local estimates. Scope must match exactly.
4. Selective Quoting Caution: If retrieved passages provide conflicting perspectives or omit critical context, evaluate whether the claim is cherrypicked."""


class ContrastivePromptBuilder:
    """Renders versioned system and user prompts across the 6 Track C arms."""

    VERSION = "m20d_contrastive_v1"

    @classmethod
    def build_prompt(
        cls,
        arm: M20dArm,
        claim_text: str,
        evidence_passages: List[Dict[str, Any]],
        pattern_decision: PatternSelectionDecision = PatternSelectionDecision.NO_PATTERN_SELECTED,
        selected_pattern: Optional[HistoricalErrorPattern] = None,
        placebo_pattern: Optional[HistoricalErrorPattern] = None,
    ) -> Tuple[str, str, str]:
        """Returns (system_instruction, user_prompt, prompt_sha256)."""
        ev_block = format_evidence_block(evidence_passages)

        if arm == M20dArm.B0:
            sys_inst, user_pr = cls._build_b0(claim_text, ev_block)
        elif arm == M20dArm.G2:
            sys_inst, user_pr = cls._build_g2(claim_text, ev_block)
        elif arm == M20dArm.C1:
            sys_inst, user_pr = cls._build_c1(claim_text, ev_block)
        elif arm == M20dArm.C2:
            sys_inst, user_pr = cls._build_c2(claim_text, ev_block, pattern_decision, selected_pattern)
        elif arm == M20dArm.C3:
            sys_inst, user_pr = cls._build_c3(claim_text, ev_block, pattern_decision, selected_pattern)
        elif arm == M20dArm.C4:
            sys_inst, user_pr = cls._build_c4(claim_text, ev_block, placebo_pattern)
        else:
            raise ValueError(f"Unknown arm: {arm}")

        sha = hashlib.sha256((sys_inst + user_pr).encode("utf-8")).hexdigest()
        return sys_inst, user_pr, sha

    @classmethod
    def _build_b0(cls, claim_text: str, ev_block: str) -> Tuple[str, str]:
        sys_inst = (
            "You are a fact-checking verifier. Given a claim and retrieved evidence passages, "
            "determine whether the claim is Supported, Refuted, Not Enough Evidence, or Conflicting Evidence/Cherrypicking.\n\n"
            "Verdict definitions:\n"
            "- Supported: The evidence confirms the claim is true.\n"
            "- Refuted: The evidence contradicts the claim or confirms it is false.\n"
            "- Not Enough Evidence: The evidence is insufficient to prove or disprove the claim.\n"
            "- Conflicting Evidence/Cherrypicking: Multiple credible sources contradict each other, or the claim cherry-picks partial facts.\n\n"
            "Output your final verdict on the last line in the format:\n"
            "Verdict: <verdict>"
        )
        user_pr = f"Claim: {claim_text}\n\nEvidence:\n{ev_block}\n\nVerdict:"
        return sys_inst, user_pr

    @classmethod
    def _build_g2(cls, claim_text: str, ev_block: str) -> Tuple[str, str]:
        sys_inst = (
            "You are a rigorous fact-checking verifier. Given a claim and retrieved evidence passages, "
            "determine whether the claim is Supported, Refuted, Not Enough Evidence, or Conflicting Evidence/Cherrypicking.\n\n"
            f"{GENERIC_RULES_G2}\n\n"
            "Verdict definitions:\n"
            "- Supported: The evidence confirms the claim is true.\n"
            "- Refuted: The evidence contradicts the claim or confirms it is false.\n"
            "- Not Enough Evidence: The evidence is insufficient to prove or disprove the claim.\n"
            "- Conflicting Evidence/Cherrypicking: Multiple credible sources contradict each other, or the claim cherry-picks partial facts.\n\n"
            "Output your final verdict on the last line in the format:\n"
            "Verdict: <verdict>"
        )
        user_pr = f"Claim: {claim_text}\n\nEvidence:\n{ev_block}\n\nVerdict:"
        return sys_inst, user_pr

    @classmethod
    def _build_c1(cls, claim_text: str, ev_block: str) -> Tuple[str, str]:
        sys_inst = (
            "You are an expert fact-checker performing Contrastive Evidence Verification.\n"
            "Evaluate the claim by systematically contrasting alternative factual hypotheses against the retrieved passages:\n"
            "1. Supporting Evidence: Identify which passages directly affirm or substantiate the claim predicate.\n"
            "2. Refuting Evidence: Identify which passages directly contradict or disprove the claim predicate.\n"
            "3. Relevance & Scope: Verify that the evidence pertains to the exact entity, event, timeframe, and location.\n"
            "4. Sufficiency vs Conflict: Distinguish true factual support/refutation from partial overlap (Not Enough Evidence) "
            "or direct discordance between credible sources (Conflicting Evidence/Cherrypicking).\n\n"
            "You MUST format your response exactly as follows:\n\n"
            "[EVIDENCE ATTRIBUTION]\n"
            "- Supporting Passage IDs: [list of passage numbers, e.g. [1, 2], or None]\n"
            "- Refuting Passage IDs: [list of passage numbers, e.g. [3], or None]\n"
            "- Scope & Relevance Assessment: <1 concise sentence on entity/temporal alignment>\n"
            "- Sufficiency & Conflict Assessment: <1 concise sentence on sufficiency or conflict>\n\n"
            "[VERDICT]\n"
            "<Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking>"
        )
        user_pr = f"Claim: {claim_text}\n\nEvidence Passages:\n{ev_block}\n\nBegin Contrastive Verification:"
        return sys_inst, user_pr

    @classmethod
    def _build_c2(
        cls,
        claim_text: str,
        ev_block: str,
        decision: PatternSelectionDecision,
        pattern: Optional[HistoricalErrorPattern],
    ) -> Tuple[str, str]:
        if decision == PatternSelectionDecision.NO_PATTERN_SELECTED or pattern is None:
            return cls._build_c1(claim_text, ev_block)

        sys_inst = (
            "You are an expert fact-checker performing Contrastive Evidence Verification.\n"
            "Evaluate the claim by systematically contrasting alternative factual hypotheses against the retrieved passages:\n"
            "1. Supporting Evidence: Identify which passages directly affirm or substantiate the claim predicate.\n"
            "2. Refuting Evidence: Identify which passages directly contradict or disprove the claim predicate.\n"
            "3. Relevance & Scope: Verify that the evidence pertains to the exact entity, event, timeframe, and location.\n"
            "4. Sufficiency vs Conflict: Distinguish true factual support/refutation from partial overlap (Not Enough Evidence) "
            "or direct discordance between credible sources (Conflicting Evidence/Cherrypicking).\n\n"
            f"[FAILURE PATTERN GUARD]\n{pattern.contrastive_directive}\n\n"
            "You MUST format your response exactly as follows:\n\n"
            "[EVIDENCE ATTRIBUTION]\n"
            "- Supporting Passage IDs: [list of passage numbers, e.g. [1, 2], or None]\n"
            "- Refuting Passage IDs: [list of passage numbers, e.g. [3], or None]\n"
            "- Scope & Relevance Assessment: <1 concise sentence on entity/temporal alignment>\n"
            "- Sufficiency & Conflict Assessment: <1 concise sentence on sufficiency or conflict>\n\n"
            "[VERDICT]\n"
            "<Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking>"
        )
        user_pr = f"Claim: {claim_text}\n\nEvidence Passages:\n{ev_block}\n\nBegin Contrastive Verification:"
        return sys_inst, user_pr

    @classmethod
    def _build_c3(
        cls,
        claim_text: str,
        ev_block: str,
        decision: PatternSelectionDecision,
        pattern: Optional[HistoricalErrorPattern],
    ) -> Tuple[str, str]:
        if decision == PatternSelectionDecision.NO_PATTERN_SELECTED or pattern is None:
            return cls._build_c1(claim_text, ev_block)

        sys_inst = (
            "You are an expert fact-checker performing Contrastive Evidence Verification.\n"
            "Evaluate the claim by systematically contrasting alternative factual hypotheses against the retrieved passages:\n"
            "1. Supporting Evidence: Identify which passages directly affirm or substantiate the claim predicate.\n"
            "2. Refuting Evidence: Identify which passages directly contradict or disprove the claim predicate.\n"
            "3. Relevance & Scope: Verify that the evidence pertains to the exact entity, event, timeframe, and location.\n"
            "4. Sufficiency vs Conflict: Distinguish true factual support/refutation from partial overlap (Not Enough Evidence) "
            "or direct discordance between credible sources (Conflicting Evidence/Cherrypicking).\n\n"
            f"[VERIFICATION DIRECTIVE]\n{pattern.matched_generic_directive}\n\n"
            "You MUST format your response exactly as follows:\n\n"
            "[EVIDENCE ATTRIBUTION]\n"
            "- Supporting Passage IDs: [list of passage numbers, e.g. [1, 2], or None]\n"
            "- Refuting Passage IDs: [list of passage numbers, e.g. [3], or None]\n"
            "- Scope & Relevance Assessment: <1 concise sentence on entity/temporal alignment>\n"
            "- Sufficiency & Conflict Assessment: <1 concise sentence on sufficiency or conflict>\n\n"
            "[VERDICT]\n"
            "<Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking>"
        )
        user_pr = f"Claim: {claim_text}\n\nEvidence Passages:\n{ev_block}\n\nBegin Contrastive Verification:"
        return sys_inst, user_pr

    @classmethod
    def _build_c4(
        cls,
        claim_text: str,
        ev_block: str,
        placebo_pattern: Optional[HistoricalErrorPattern],
    ) -> Tuple[str, str]:
        if placebo_pattern is None:
            return cls._build_c1(claim_text, ev_block)

        sys_inst = (
            "You are an expert fact-checker performing Contrastive Evidence Verification.\n"
            "Evaluate the claim by systematically contrasting alternative factual hypotheses against the retrieved passages:\n"
            "1. Supporting Evidence: Identify which passages directly affirm or substantiate the claim predicate.\n"
            "2. Refuting Evidence: Identify which passages directly contradict or disprove the claim predicate.\n"
            "3. Relevance & Scope: Verify that the evidence pertains to the exact entity, event, timeframe, and location.\n"
            "4. Sufficiency vs Conflict: Distinguish true factual support/refutation from partial overlap (Not Enough Evidence) "
            "or direct discordance between credible sources (Conflicting Evidence/Cherrypicking).\n\n"
            f"[FAILURE PATTERN GUARD]\n{placebo_pattern.contrastive_directive}\n\n"
            "You MUST format your response exactly as follows:\n\n"
            "[EVIDENCE ATTRIBUTION]\n"
            "- Supporting Passage IDs: [list of passage numbers, e.g. [1, 2], or None]\n"
            "- Refuting Passage IDs: [list of passage numbers, e.g. [3], or None]\n"
            "- Scope & Relevance Assessment: <1 concise sentence on entity/temporal alignment>\n"
            "- Sufficiency & Conflict Assessment: <1 concise sentence on sufficiency or conflict>\n\n"
            "[VERDICT]\n"
            "<Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking>"
        )
        user_pr = f"Claim: {claim_text}\n\nEvidence Passages:\n{ev_block}\n\nBegin Contrastive Verification:"
        return sys_inst, user_pr


class ContrastiveVerdictParser:
    """Hardened parser for contrastive evidence attributions and verdicts."""

    @classmethod
    def parse_completion(
        cls,
        completion: str,
        num_passages: int = 8,
    ) -> Tuple[Optional[str], str, EvidenceAttribution]:
        """Parses model completion. Returns (verdict, status, attribution)."""
        if not completion or not completion.strip():
            return None, "UNPARSEABLE", EvidenceAttribution()

        text = completion.strip()

        # 1. Parse Attribution Fields
        sup_ids = cls._extract_ids(text, SUPPORTING_IDS_RE, num_passages)
        ref_ids = cls._extract_ids(text, REFUTING_IDS_RE, num_passages)

        scope_m = SCOPE_ASSESSMENT_RE.search(text)
        scope_text = scope_m.group(1).strip() if scope_m else ""

        suff_m = SUFFICIENCY_ASSESSMENT_RE.search(text)
        suff_text = suff_m.group(1).strip() if suff_m else ""

        is_valid_attr = bool(sup_ids is not None or ref_ids is not None or scope_text or suff_text)

        attribution = EvidenceAttribution(
            supporting_passage_ids=sup_ids or [],
            refuting_passage_ids=ref_ids or [],
            scope_assessment=scope_text,
            sufficiency_assessment=suff_text,
            is_valid_attribution=is_valid_attr,
        )

        # 2. Parse Verdict
        # Method A: Look for [VERDICT] section
        verdict_m = VERDICT_SECTION_RE.search(text)
        if verdict_m:
            candidate = cls._clean_verdict_str(verdict_m.group(1))
            if candidate in VALID_VERDICTS:
                return candidate, "SUCCESS", attribution

        # Method B: Look for 'Verdict: <verdict>'
        v_lines = re.findall(r"Verdict:\s*([^\n]+)", text, re.IGNORECASE)
        if v_lines:
            candidate = cls._clean_verdict_str(v_lines[-1])
            if candidate in VALID_VERDICTS:
                return candidate, "SUCCESS", attribution

        # Method C: Inspect trailing lines
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in reversed(lines[-4:]):
            cleaned = cls._clean_verdict_str(line)
            if cleaned in VALID_VERDICTS:
                return cleaned, "SUCCESS", attribution

        return None, "UNPARSEABLE", attribution

    @classmethod
    def _extract_ids(cls, text: str, regex: re.Pattern, max_id: int) -> Optional[List[int]]:
        m = regex.search(text)
        if not m:
            return None
        raw = m.group(1).strip()
        if raw.lower() == "none" or raw == "[]":
            return []
        # Find all integer numbers
        digits = re.findall(r"\b\d+\b", raw)
        parsed = []
        for d in digits:
            val = int(d)
            if 1 <= val <= max_id:
                if val not in parsed:
                    parsed.append(val)
        return parsed

    @classmethod
    def _clean_verdict_str(cls, s: str) -> str:
        s = s.strip()
        # Remove bolding or markdown
        s = re.sub(r"[\*\_\[\]`]", "", s).strip()
        # Direct exact match
        for v in VALID_VERDICTS:
            if s.lower() == v.lower():
                return v
            # If line starts with verdict
            if s.lower().startswith(v.lower()):
                return v
        return s
