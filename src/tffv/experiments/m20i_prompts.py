"""Milestone M2.0i: Prompt Builder for Factorial Mechanism Ablation.

Generates prompts for the 5 arms:
- F0: Neutral Explanation (Gold label + neutral explanation, standard header)
- F1: Header-Only Framing ([HISTORICAL ERROR CONTRAST] header + neutral explanation, no mistake info)
- F2: Observed Mistake Only (Standard header + actual Observed Mistake + neutral explanation)
- F3: Full Historical Error K4 ([HISTORICAL ERROR CONTRAST] + actual Observed Mistake + corrective resolution)
- F4: Matched Placebo Mistake ([HISTORICAL ERROR CONTRAST] + length-matched placebo mistake + corrective resolution)
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Tuple

from tffv.experiments.m19f_evidence import format_evidence_block
from tffv.experiments.m20g_exemplars import ExemplarItem
from tffv.experiments.m20i_placebos import get_placebo_for_exemplar
from tffv.experiments.m20i_schemas import MechanismArm


class M20iPromptBuilder:
    """Deterministic prompt generator for the 5 factorial mechanism arms."""

    VERSION = "m20i_mechanism_ablation_v1"

    BASE_DEFINITIONS = (
        "You are an expert fact-checking verifier. Given a claim and retrieved evidence passages, "
        "determine whether the claim is Supported, Refuted, Not Enough Evidence, or Conflicting Evidence/Cherrypicking.\n\n"
        "Verdict definitions:\n"
        "- Supported: The evidence confirms the claim is true.\n"
        "- Refuted: The evidence contradicts the claim or confirms it is false.\n"
        "- Not Enough Evidence: The evidence is insufficient to prove or disprove the claim.\n"
        "- Conflicting Evidence/Cherrypicking: Multiple credible sources contradict each other, or the claim cherry-picks partial facts.\n\n"
    )

    @classmethod
    def get_system_instruction(cls, arm: MechanismArm) -> str:
        """Returns system instruction tailored to each arm's formatting contract."""
        if arm == MechanismArm.F0_NEUTRAL_EXPL:
            return (
                cls.BASE_DEFINITIONS
                + "Evaluate the claim by assessing the factual basis against the retrieved evidence passages. "
                "Provide your reasoning followed by the final verdict.\n\n"
                "Format your output as:\n"
                "[VERIFICATION REASONING]\n"
                "- Factual Basis: <reasoning explaining how the evidence addresses the claim>\n"
                "Verdict: <verdict>"
            )
        elif arm == MechanismArm.F1_HEADER_ONLY:
            return (
                cls.BASE_DEFINITIONS
                + "Evaluate the claim by learning from historical verification mistakes and applying corrective reasoning to the retrieved evidence passages. "
                "Provide your reasoning followed by the final verdict.\n\n"
                "Format your output as:\n"
                "[HISTORICAL ERROR CONTRAST]\n"
                "- Factual Basis: <reasoning explaining how the evidence addresses the claim>\n"
                "Verdict: <verdict>"
            )
        elif arm == MechanismArm.F2_OBSERVED_MISTAKE:
            return (
                cls.BASE_DEFINITIONS
                + "Evaluate the claim by assessing the factual basis against the retrieved evidence passages and noting observed mistakes to avoid. "
                "Provide your reasoning followed by the final verdict.\n\n"
                "Format your output as:\n"
                "[VERIFICATION REASONING]\n"
                "- Observed Mistake: <common reasoning trap or error to avoid>\n"
                "- Factual Basis: <reasoning explaining how the evidence addresses the claim>\n"
                "Verdict: <verdict>"
            )
        elif arm in (MechanismArm.F3_FULL_HISTORICAL_ERROR, MechanismArm.F4_MATCHED_PLACEBO):
            return (
                cls.BASE_DEFINITIONS
                + "Evaluate the claim by learning from historical verification mistakes and applying corrective reasoning to the retrieved evidence passages. "
                "Provide your reasoning followed by the final verdict.\n\n"
                "Format your output as:\n"
                "[HISTORICAL ERROR CONTRAST]\n"
                "- Observed Mistake: <common reasoning trap or error to avoid>\n"
                "- Corrective Resolution: <corrective factual reasoning applying to the evidence>\n"
                "Verdict: <verdict>"
            )
        else:
            raise ValueError(f"Unknown MechanismArm: {arm}")

    @classmethod
    def format_demonstrations(cls, exemplars: List[ExemplarItem], arm: MechanismArm) -> str:
        """Formats exemplars according to the arm's specific component configuration."""
        blocks = []
        for idx, ex in enumerate(exemplars, start=1):
            if arm == MechanismArm.F0_NEUTRAL_EXPL:
                block = (
                    f"[EXAMPLE {idx}]\n"
                    f"Claim: {ex.claim_text}\n"
                    f"Evidence:\n{ex.evidence_snippet}\n"
                    f"[VERIFICATION REASONING]\n"
                    f"- Factual Basis: {ex.neutral_explanation}\n"
                    f"Verdict: {ex.gold_verdict}"
                )
            elif arm == MechanismArm.F1_HEADER_ONLY:
                block = (
                    f"[EXAMPLE {idx}]\n"
                    f"Claim: {ex.claim_text}\n"
                    f"Evidence:\n{ex.evidence_snippet}\n"
                    f"[HISTORICAL ERROR CONTRAST]\n"
                    f"- Factual Basis: {ex.neutral_explanation}\n"
                    f"Verdict: {ex.gold_verdict}"
                )
            elif arm == MechanismArm.F2_OBSERVED_MISTAKE:
                block = (
                    f"[EXAMPLE {idx}]\n"
                    f"Claim: {ex.claim_text}\n"
                    f"Evidence:\n{ex.evidence_snippet}\n"
                    f"[VERIFICATION REASONING]\n"
                    f"- Observed Mistake: {ex.observed_mistake}\n"
                    f"- Factual Basis: {ex.neutral_explanation}\n"
                    f"Verdict: {ex.gold_verdict}"
                )
            elif arm == MechanismArm.F3_FULL_HISTORICAL_ERROR:
                block = (
                    f"[EXAMPLE {idx}]\n"
                    f"Claim: {ex.claim_text}\n"
                    f"Evidence:\n{ex.evidence_snippet}\n"
                    f"[HISTORICAL ERROR CONTRAST]\n"
                    f"- Observed Mistake: {ex.observed_mistake}\n"
                    f"- Corrective Resolution: {ex.corrective_resolution}\n"
                    f"Verdict: {ex.gold_verdict}"
                )
            elif arm == MechanismArm.F4_MATCHED_PLACEBO:
                placebo_mistake = get_placebo_for_exemplar(ex.sample_id)
                block = (
                    f"[EXAMPLE {idx}]\n"
                    f"Claim: {ex.claim_text}\n"
                    f"Evidence:\n{ex.evidence_snippet}\n"
                    f"[HISTORICAL ERROR CONTRAST]\n"
                    f"- Observed Mistake: {placebo_mistake}\n"
                    f"- Corrective Resolution: {ex.corrective_resolution}\n"
                    f"Verdict: {ex.gold_verdict}"
                )
            else:
                raise ValueError(f"Unknown MechanismArm: {arm}")
            blocks.append(block)

        return "\n\n".join(blocks)

    @classmethod
    def build_prompt(
        cls,
        claim_text: str,
        evidence_passages: List[Dict[str, Any]],
        exemplars: List[ExemplarItem],
        arm: MechanismArm,
    ) -> Tuple[str, str, str]:
        """Builds full prompt (system instruction, user prompt, SHA-256 hash)."""
        sys_inst = cls.get_system_instruction(arm)
        ev_block = format_evidence_block(evidence_passages)
        demos_str = cls.format_demonstrations(exemplars, arm)

        user_pr = (
            f"[DEMONSTRATIONS]\n{demos_str}\n\n"
            f"[TARGET CLAIM TO VERIFY]\n"
            f"Claim: {claim_text}\n\n"
            f"Evidence Passages:\n{ev_block}\n\n"
            f"Begin Verification:"
        )
        sha = hashlib.sha256((sys_inst + user_pr).encode("utf-8")).hexdigest()
        return sys_inst, user_pr, sha
