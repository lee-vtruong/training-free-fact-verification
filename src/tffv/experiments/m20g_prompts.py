"""Milestone M2.0g: Prompt Builder for Robustness & Knowledge Component Ablation.

Supports:
- Canonical Baseline B0 and Generic Rules G2.
- 4 Knowledge Component levels (K1_GOLD_ONLY, K2_GOLD_NEUTRAL_EXPL, K3_WRONG_PRED_GOLD_CORR, K4_FULL_GROUNDED_ERROR).
- Demonstration counts k in {2, 4, 6}.
- Byte-for-byte evidence passage preservation and deterministic SHA-256 hashing.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

from tffv.experiments.m19f_evidence import format_evidence_block
from tffv.experiments.m20d_contrastive import GENERIC_RULES_G2
from tffv.experiments.m20g_exemplars import (
    ExemplarItem,
    ExemplarManager,
)
from tffv.experiments.m20g_schemas import KnowledgeComponent


class M20gPromptBuilder:
    """Standardized prompt builder for Milestone M2.0g."""

    VERSION = "m20g_robustness_v1"

    @classmethod
    def build_b0(cls, claim_text: str, evidence_passages: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        ev_block = format_evidence_block(evidence_passages)
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
        sha = hashlib.sha256((sys_inst + user_pr).encode("utf-8")).hexdigest()
        return sys_inst, user_pr, sha

    @classmethod
    def build_g2(cls, claim_text: str, evidence_passages: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        ev_block = format_evidence_block(evidence_passages)
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
        sha = hashlib.sha256((sys_inst + user_pr).encode("utf-8")).hexdigest()
        return sys_inst, user_pr, sha

    @classmethod
    def build_few_shot_prompt(
        cls,
        claim_text: str,
        evidence_passages: List[Dict[str, Any]],
        exemplars: List[ExemplarItem],
        component: KnowledgeComponent,
    ) -> Tuple[str, str, str]:
        """Builds prompt with specified exemplars and knowledge component level."""
        ev_block = format_evidence_block(evidence_passages)
        demos_str = ExemplarManager.format_demonstrations(exemplars, component)

        if component == KnowledgeComponent.K1_GOLD_ONLY:
            sys_inst = (
                "You are an expert fact-checking verifier. Given a claim and retrieved evidence passages, "
                "determine whether the claim is Supported, Refuted, Not Enough Evidence, or Conflicting Evidence/Cherrypicking.\n\n"
                "Verdict definitions:\n"
                "- Supported: The evidence confirms the claim is true.\n"
                "- Refuted: The evidence contradicts the claim or confirms it is false.\n"
                "- Not Enough Evidence: The evidence is insufficient to prove or disprove the claim.\n"
                "- Conflicting Evidence/Cherrypicking: Multiple credible sources contradict each other, or the claim cherry-picks partial facts.\n\n"
                "Output your final verdict on the last line in the format:\n"
                "Verdict: <verdict>"
            )
        elif component == KnowledgeComponent.K2_GOLD_NEUTRAL_EXPL:
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
                "- Factual Basis: <reasoning explaining how the evidence addresses the claim>\n"
                "Verdict: <verdict>"
            )
        elif component == KnowledgeComponent.K3_WRONG_PRED_GOLD_CORR:
            sys_inst = (
                "You are an expert fact-checking verifier. Given a claim and retrieved evidence passages, "
                "determine whether the claim is Supported, Refuted, Not Enough Evidence, or Conflicting Evidence/Cherrypicking.\n\n"
                "Verdict definitions:\n"
                "- Supported: The evidence confirms the claim is true.\n"
                "- Refuted: The evidence contradicts the claim or confirms it is false.\n"
                "- Not Enough Evidence: The evidence is insufficient to prove or disprove the claim.\n"
                "- Conflicting Evidence/Cherrypicking: Multiple credible sources contradict each other, or the claim cherry-picks partial facts.\n\n"
                "Evaluate the claim by comparing alternative verdict hypotheses against the evidence passages. "
                "Provide your assessment followed by the final verdict.\n\n"
                "Format your output as:\n"
                "[VERIFICATION REASONING]\n"
                "- Hypotheses Assessment: <concise assessment of evidence support vs contradiction>\n"
                "Verdict: <verdict>"
            )
        elif component == KnowledgeComponent.K4_FULL_GROUNDED_ERROR:
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
        else:
            raise ValueError(f"Unknown KnowledgeComponent: {component}")

        user_pr = (
            f"[DEMONSTRATIONS]\n{demos_str}\n\n"
            f"[TARGET CLAIM TO VERIFY]\n"
            f"Claim: {claim_text}\n\n"
            f"Evidence Passages:\n{ev_block}\n\n"
            f"Begin Verification:"
        )
        sha = hashlib.sha256((sys_inst + user_pr).encode("utf-8")).hexdigest()
        return sys_inst, user_pr, sha
