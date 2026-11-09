"""Milestone M2.0b: Phase C & D — Prompt Formatter for Six Experimental Arms.

Guarantees:
1. Locked Canonical System Instruction (Concise M1.9g Contract) across all arms.
2. Identical retrieved evidence passage blocks across all 6 arms.
3. Precise injection of instance-grounded lessons (B1) and matched controls (B2, B3).
4. Full fallback to canonical B0 when grounding decision is NO_INTERVENTION.
5. Invariant hash checks preventing prompt corruption or blank evidence.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tffv.experiments.m19f_evidence import (
    NormalizedPassage,
    format_evidence_block,
    normalize_evidence_list,
)
from tffv.experiments.m19g_guardrails import validate_evidence_rendering
from tffv.experiments.m20b_schemas import InterventionDecision, M20bArm
from tffv.experiments.m20b_selector import HistoricalErrorCase

CANONICAL_SYSTEM_INSTRUCTION = """You are a rigorous, objective fact-checking assistant.
Your task is to verify whether the target claim is supported, refuted, or unresolvable based strictly on the provided evidence passages.

Output Schema:
Explanation: <Concise step-by-step reasoning citing specific passages>
Verdict: <Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking>"""

STATIC_G2_GENERIC_RULES = """Standard Verification Guidelines:
1. Cross-reference all named entities, geographic locations, and dates against the provided evidence.
2. Ensure assertions of absolute frequency ("always", "never") are corroborated by unequivocal source statements.
3. Distinguish between verified factual reports and speculative quotes from secondary observers.
4. When evidence is ambiguous or incomplete, abstain rather than extrapolate."""

STATIC_E1_ERROR_RULES = """Historical Error Prevention Rules:
1. Entity Disambiguation: Verify that the exact organization, person, or legislative act mentioned is the subject of the evidence, not a related entity.
2. Temporal Accuracy: Ensure events described in the evidence occurred within the exact timeframe asserted by the claim.
3. Over-Rejection Guard: Do not mark a claim as Refuted or Conflicting merely because evidence uses different wording or omission of minor non-core trivia.
4. Cherrypicking Criteria: Only assign Conflicting Evidence if reputable sources directly disagree on core facts."""


class M20bPromptFormatter:
    """Renders deterministic, invariant-checked prompts across all 6 arms."""

    def __init__(
        self,
        canonical_system_instruction: str = CANONICAL_SYSTEM_INSTRUCTION,
        generic_rules: str = STATIC_G2_GENERIC_RULES,
        global_error_rules: str = STATIC_E1_ERROR_RULES,
    ):
        self.canonical_system_instruction = canonical_system_instruction
        self.generic_rules = generic_rules
        self.global_error_rules = global_error_rules

    def format_prompt(
        self,
        arm: str,
        claim_text: str,
        evidence_passages: Sequence[Dict[str, Any] | NormalizedPassage],
        decision: InterventionDecision = InterventionDecision.NO_INTERVENTION,
        selected_case: Optional[HistoricalErrorCase] = None,
        context_id: str = "",
    ) -> Tuple[str, str, int]:
        """Formats (system_instruction, user_prompt, token_estimate) with invariant validation."""
        if not evidence_passages or len(evidence_passages) == 0:
            raise ValueError(f"Integrity Violation: Claim {context_id} must have a non-empty list of passages.")

        evidence_block = format_evidence_block(evidence_passages)
        validate_evidence_rendering(evidence_passages, evidence_block, context=f"M2.0b Claim {context_id}")

        system_instruction = self.canonical_system_instruction
        intervention_block = ""

        if arm == M20bArm.B0.value:
            # Canonical zero-shot baseline
            pass
        elif arm == M20bArm.G2.value:
            # Length-matched generic verification rules
            system_instruction = f"{self.canonical_system_instruction}\n\n{self.generic_rules}"
        elif arm == M20bArm.E1.value:
            # Global error-derived rules
            system_instruction = f"{self.canonical_system_instruction}\n\n{self.global_error_rules}"
        elif arm == M20bArm.B1.value:
            # Instance-grounded error intervention
            if decision == InterventionDecision.INTERVENE and selected_case is not None:
                intervention_block = (
                    f"\n\n[Verified Historical Error Diagnostic Lesson]\n"
                    f"Notice regarding similar factual patterns:\n"
                    f"{selected_case.error_lesson}\n"
                    f"Apply this caution when examining the passages below."
                )
        elif arm == M20bArm.B2.value:
            # Instance-grounded generic control (matched token budget)
            if decision == InterventionDecision.INTERVENE and selected_case is not None:
                intervention_block = (
                    f"\n\n[General Verification Guidance Note]\n"
                    f"Notice regarding factual verification:\n"
                    f"{selected_case.generic_guidance}\n"
                    f"Apply this standard guidance when examining the passages below."
                )
        elif arm == M20bArm.B3.value:
            # Irrelevant historical control
            if selected_case is not None:
                intervention_block = (
                    f"\n\n[Reference Background Note]\n"
                    f"{selected_case.error_lesson}"
                )

        # Assemble canonical user prompt
        user_prompt = (
            f"Claim: {claim_text.strip()}"
            f"{intervention_block}\n\n"
            f"{evidence_block}\n\n"
            f"Provide your Explanation and Verdict:"
        )

        token_estimate = len((system_instruction + " " + user_prompt).split())
        return system_instruction, user_prompt, token_estimate
