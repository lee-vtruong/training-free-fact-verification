"""Milestone M2.0f: Prompt Rendering for Factorial Ablation of Attribution and Error Knowledge.

Implements prompt generation across the 6 experimental arms:
- B0: Canonical Zero-Shot Baseline.
- G2: Generic Evidence-Guided Reasoning Control.
- A1: Matched Gold ICL, No Forced Evidence Attribution.
- A2: Matched Gold ICL + Forced Evidence Attribution.
- A3: Historical Error ICL, No Forced Attribution.
- A4: Historical Error ICL + Forced Evidence Attribution.

Guarantees:
- Byte-for-byte identical evidence block (Top-8 PF1000 passages).
- Perfectly balanced token budget across matched arms (A1 vs A3, A2 vs A4).
- Deterministic SHA-256 fingerprinting.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Tuple

from tffv.experiments.m19f_evidence import format_evidence_block
from tffv.experiments.m20d_contrastive import GENERIC_RULES_G2
from tffv.experiments.m20f_exemplars import ExemplarFormatter
from tffv.experiments.m20f_schemas import M20fArm


class M20fPromptBuilder:
    """Builds standardized system and user prompts across the 6 M2.0f arms."""

    VERSION = "m20f_factorial_v1"

    @classmethod
    def build_prompt(
        cls,
        arm: M20fArm,
        claim_text: str,
        evidence_passages: List[Dict[str, Any]],
    ) -> Tuple[str, str, str]:
        """Returns (system_instruction, user_prompt, prompt_sha256)."""
        ev_block = format_evidence_block(evidence_passages)

        if arm == M20fArm.B0:
            sys_inst, user_pr = cls._build_b0(claim_text, ev_block)
        elif arm == M20fArm.G2:
            sys_inst, user_pr = cls._build_g2(claim_text, ev_block)
        elif arm == M20fArm.A1:
            sys_inst, user_pr = cls._build_a1(claim_text, ev_block)
        elif arm == M20fArm.A2:
            sys_inst, user_pr = cls._build_a2(claim_text, ev_block)
        elif arm == M20fArm.A3:
            sys_inst, user_pr = cls._build_a3(claim_text, ev_block)
        elif arm == M20fArm.A4:
            sys_inst, user_pr = cls._build_a4(claim_text, ev_block)
        else:
            raise ValueError(f"Unknown M20fArm: {arm}")

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
    def _build_a1(cls, claim_text: str, ev_block: str) -> Tuple[str, str]:
        sys_inst = (
            "You are an expert fact-checking verifier. Given a claim and retrieved evidence passages, "
            "determine whether the claim is Supported, Refuted, Not Enough Evidence, or Conflicting Evidence/Cherrypicking.\n\n"
            "Verdict definitions:\n"
            "- Supported: The evidence confirms the claim is true.\n"
            "- Refuted: The evidence contradicts the claim or confirms it is false.\n"
            "- Not Enough Evidence: The evidence is insufficient to prove or disprove the claim.\n"
            "- Conflicting Evidence/Cherrypicking: Multiple credible sources contradict each other, or the claim cherry-picks partial facts.\n\n"
            "Evaluate the claim by assessing the factual basis against the retrieved evidence passages. "
            "Provide your reasoning followed by the final verdict.\n\n"
            "Format your output as:\n"
            "[VERIFICATION REASONING]\n"
            "- Scope Analysis: <scope analysis>\n"
            "- Factual Basis: <reasoning explaining how the evidence addresses the claim>\n"
            "Verdict: <verdict>"
        )
        demos = ExemplarFormatter.format_demonstrations(M20fArm.A1)
        user_pr = (
            f"[DEMONSTRATIONS]\n{demos}\n\n"
            f"[TARGET CLAIM TO VERIFY]\n"
            f"Claim: {claim_text}\n\n"
            f"Evidence Passages:\n{ev_block}\n\n"
            f"Begin Verification:"
        )
        return sys_inst, user_pr

    @classmethod
    def _build_a2(cls, claim_text: str, ev_block: str) -> Tuple[str, str]:
        sys_inst = (
            "You are an expert fact-checker performing Evidence Attribution and Verification. "
            "Given a claim and retrieved evidence passages, "
            "determine whether the claim is Supported, Refuted, Not Enough Evidence, or Conflicting Evidence/Cherrypicking.\n\n"
            "Verdict definitions:\n"
            "- Supported: The evidence confirms the claim is true.\n"
            "- Refuted: The evidence contradicts the claim or confirms it is false.\n"
            "- Not Enough Evidence: The evidence is insufficient to prove or disprove the claim.\n"
            "- Conflicting Evidence/Cherrypicking: Multiple credible sources contradict each other, or the claim cherry-picks partial facts.\n\n"
            "Evaluate the claim by systematically attributing evidence and contrasting factual hypotheses against the retrieved passages:\n"
            "1. Supporting Evidence: Identify which passages directly affirm or substantiate the claim predicate.\n"
            "2. Refuting Evidence: Identify which passages directly contradict or disprove the claim predicate.\n"
            "3. Relevance & Scope: Verify that the evidence pertains to the exact entity, event, timeframe, and location.\n"
            "4. Sufficiency vs Conflict: Distinguish true factual support/refutation from partial overlap (Not Enough Evidence) "
            "or direct discordance between credible sources (Conflicting Evidence/Cherrypicking).\n\n"
            "You MUST format your response exactly as follows:\n\n"
            "[VERIFICATION REASONING]\n"
            "- Scope Analysis: <scope analysis>\n"
            "- Factual Basis: <concise factual basis from the evidence>\n"
            "[EVIDENCE ATTRIBUTION]\n"
            "- Supporting Passage IDs: [list of passage numbers, e.g. [1, 2], or None]\n"
            "- Refuting Passage IDs: [list of passage numbers, e.g. [3], or None]\n"
            "- Scope & Relevance Assessment: <1 concise sentence on entity/temporal alignment>\n"
            "- Sufficiency & Conflict Assessment: <1 concise sentence on sufficiency or conflict>\n\n"
            "[VERDICT]\n"
            "<Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking>"
        )
        demos = ExemplarFormatter.format_demonstrations(M20fArm.A2)
        user_pr = (
            f"[DEMONSTRATIONS]\n{demos}\n\n"
            f"[TARGET CLAIM TO VERIFY]\n"
            f"Claim: {claim_text}\n\n"
            f"Evidence Passages:\n{ev_block}\n\n"
            f"Begin Verification:"
        )
        return sys_inst, user_pr

    @classmethod
    def _build_a3(cls, claim_text: str, ev_block: str) -> Tuple[str, str]:
        sys_inst = (
            "You are an expert fact-checking verifier. Given a claim and retrieved evidence passages, "
            "determine whether the claim is Supported, Refuted, Not Enough Evidence, or Conflicting Evidence/Cherrypicking.\n\n"
            "Verdict definitions:\n"
            "- Supported: The evidence confirms the claim is true.\n"
            "- Refuted: The evidence contradicts the claim or confirms it is false.\n"
            "- Not Enough Evidence: The evidence is insufficient to prove or disprove the claim.\n"
            "- Conflicting Evidence/Cherrypicking: Multiple credible sources contradict each other, or the claim cherry-picks partial facts.\n\n"
            "Evaluate the claim by learning from historical verification mistakes and applying corrective reasoning to the retrieved evidence passages. "
            "Provide your reasoning followed by the final verdict.\n\n"
            "Format your output as:\n"
            "[HISTORICAL ERROR CONTRAST]\n"
            "- Observed Mistake: <common reasoning trap or error to avoid>\n"
            "- Corrective Resolution: <corrective factual reasoning applying to the evidence>\n"
            "Verdict: <verdict>"
        )
        demos = ExemplarFormatter.format_demonstrations(M20fArm.A3)
        user_pr = (
            f"[DEMONSTRATIONS]\n{demos}\n\n"
            f"[TARGET CLAIM TO VERIFY]\n"
            f"Claim: {claim_text}\n\n"
            f"Evidence Passages:\n{ev_block}\n\n"
            f"Begin Verification:"
        )
        return sys_inst, user_pr

    @classmethod
    def _build_a4(cls, claim_text: str, ev_block: str) -> Tuple[str, str]:
        sys_inst = (
            "You are an expert fact-checker performing Evidence Attribution and Verification. "
            "Given a claim and retrieved evidence passages, "
            "determine whether the claim is Supported, Refuted, Not Enough Evidence, or Conflicting Evidence/Cherrypicking.\n\n"
            "Verdict definitions:\n"
            "- Supported: The evidence confirms the claim is true.\n"
            "- Refuted: The evidence contradicts the claim or confirms it is false.\n"
            "- Not Enough Evidence: The evidence is insufficient to prove or disprove the claim.\n"
            "- Conflicting Evidence/Cherrypicking: Multiple credible sources contradict each other, or the claim cherry-picks partial facts.\n\n"
            "Evaluate the claim by learning from historical verification mistakes, attributing evidence, and contrasting factual hypotheses against the retrieved passages:\n"
            "1. Supporting Evidence: Identify which passages directly affirm or substantiate the claim predicate.\n"
            "2. Refuting Evidence: Identify which passages directly contradict or disprove the claim predicate.\n"
            "3. Relevance & Scope: Verify that the evidence pertains to the exact entity, event, timeframe, and location.\n"
            "4. Sufficiency vs Conflict: Distinguish true factual support/refutation from partial overlap (Not Enough Evidence) "
            "or direct discordance between credible sources (Conflicting Evidence/Cherrypicking).\n\n"
            "You MUST format your response exactly as follows:\n\n"
            "[HISTORICAL ERROR CONTRAST]\n"
            "- Observed Mistake: <common reasoning trap or error to avoid>\n"
            "- Corrective Resolution: <corrective factual reasoning applying to the evidence>\n"
            "[EVIDENCE ATTRIBUTION]\n"
            "- Supporting Passage IDs: [list of passage numbers, e.g. [1, 2], or None]\n"
            "- Refuting Passage IDs: [list of passage numbers, e.g. [3], or None]\n"
            "- Scope & Relevance Assessment: <1 concise sentence on entity/temporal alignment>\n"
            "- Sufficiency & Conflict Assessment: <1 concise sentence on sufficiency or conflict>\n\n"
            "[VERDICT]\n"
            "<Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking>"
        )
        demos = ExemplarFormatter.format_demonstrations(M20fArm.A4)
        user_pr = (
            f"[DEMONSTRATIONS]\n{demos}\n\n"
            f"[TARGET CLAIM TO VERIFY]\n"
            f"Claim: {claim_text}\n\n"
            f"Evidence Passages:\n{ev_block}\n\n"
            f"Begin Verification:"
        )
        return sys_inst, user_pr
