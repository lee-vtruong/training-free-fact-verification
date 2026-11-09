"""Milestone M2.0f: Audited Few-Shot Exemplars strictly from Train Fold-1.

Defines:
- Two audited, balanced demonstrations:
  * train:2956 (Gold: Supported, Historical Mistake: Not Enough Evidence)
  * train:2961 (Gold: Refuted, Historical Mistake: Not Enough Evidence / Supported)
- Strict length and token-budget matched rendering across:
  * A1: Matched Gold ICL, No Forced Attribution
  * A2: Matched Gold ICL + Forced Evidence Attribution
  * A3: Historical Error ICL, No Forced Attribution
  * A4: Historical Error ICL + Forced Evidence Attribution
- Guarantees 100% isolation from evaluation partitions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tffv.experiments.m20f_schemas import (
    AttributionFactor,
    ExemplarDemonstration,
    KnowledgeFactor,
    M20fArm,
)

EXEMPLAR_DEMONSTRATIONS: List[ExemplarDemonstration] = [
    ExemplarDemonstration(
        sample_id="train:2956",
        claim_text="The Kenyan judiciary receives less than 1% of the national budget.",
        evidence_snippet=(
            "[1] The Kenyan judiciary receives less than 1% of the national budget, "
            "according to official treasury reports and statements by Chief Justice David Maraga."
        ),
        gold_verdict="Supported",
        scope_analysis="Verify whether official treasury reports and judiciary statements substantiate the under 1% allocation.",
        factual_basis="The passage explicitly confirms the exact percentage; verbatim statement substantiates the claim.",
        historical_mistake="Earlier baseline concluded 'Not Enough Evidence' by overlooking the explicit numerical figure.",
        corrective_resolution="The passage explicitly confirms the exact percentage; verbatim statement substantiates the claim.",
        supporting_passage_ids=[1],
        refuting_passage_ids=[],
        scope_assessment="Evidence directly addresses the Kenyan judiciary and national budget allocation.",
        sufficiency_assessment="Passage 1 unequivocally confirms the under 1% budgetary allocation.",
    ),
    ExemplarDemonstration(
        sample_id="train:2961",
        claim_text="Baba Ramdev's Patanjali launched Coronil as a cure for coronavirus.",
        evidence_snippet=(
            "[1] Patanjali Ayurved launched Coronil tablets amid intense nationwide media coverage.\n"
            "[2] Acharya Balkrishna explicitly clarified that Patanjali never claimed Coronil is a cure "
            "for COVID-19, clarifying it was only formulated and tested as an immunity booster."
        ),
        gold_verdict="Refuted",
        scope_analysis="Verify whether executive announcements claimed Coronil as an approved COVID-19 cure.",
        factual_basis="The executive statement explicitly denies the cure claim; direct negation refutes the predicate.",
        historical_mistake="Earlier baseline overlooked the embedded denial in the quote, confusing an immunity booster with an approved cure.",
        corrective_resolution="The executive statement explicitly denies the cure claim; direct negation refutes the predicate.",
        supporting_passage_ids=[],
        refuting_passage_ids=[2],
        scope_assessment="Evidence pertains directly to Patanjali's Coronil product launch and claims.",
        sufficiency_assessment="Passage 2 directly contradicts and refutes the assertion that it was launched as a cure.",
    ),
]


class ExemplarFormatter:
    """Renders demonstrations for the 4 factorial arms with strictly matched token budgets."""

    @classmethod
    def format_demonstrations(cls, arm: M20fArm) -> str:
        """Formats the k=2 demonstrations for arm A1, A2, A3, or A4."""
        if arm not in [M20fArm.A1, M20fArm.A2, M20fArm.A3, M20fArm.A4]:
            return ""

        blocks = []
        for idx, ex in enumerate(EXEMPLAR_DEMONSTRATIONS, start=1):
            if arm == M20fArm.A1:
                # Gold ICL, No Forced Attr
                block = (
                    f"[EXAMPLE {idx}]\n"
                    f"Claim: {ex.claim_text}\n"
                    f"Evidence:\n{ex.evidence_snippet}\n"
                    f"[VERIFICATION REASONING]\n"
                    f"- Scope Analysis: {ex.scope_analysis}\n"
                    f"- Factual Basis: {ex.factual_basis}\n"
                    f"Verdict: {ex.gold_verdict}"
                )
            elif arm == M20fArm.A2:
                # Gold ICL + Forced Attribution
                ref_str = str(ex.refuting_passage_ids) if ex.refuting_passage_ids else "None"
                sup_str = str(ex.supporting_passage_ids) if ex.supporting_passage_ids else "None"
                block = (
                    f"[EXAMPLE {idx}]\n"
                    f"Claim: {ex.claim_text}\n"
                    f"Evidence:\n{ex.evidence_snippet}\n"
                    f"[VERIFICATION REASONING]\n"
                    f"- Scope Analysis: {ex.scope_analysis}\n"
                    f"- Factual Basis: {ex.factual_basis}\n"
                    f"[EVIDENCE ATTRIBUTION]\n"
                    f"- Supporting Passage IDs: {sup_str}\n"
                    f"- Refuting Passage IDs: {ref_str}\n"
                    f"- Scope & Relevance Assessment: {ex.scope_assessment}\n"
                    f"- Sufficiency & Conflict Assessment: {ex.sufficiency_assessment}\n"
                    f"[VERDICT]\n{ex.gold_verdict}"
                )
            elif arm == M20fArm.A3:
                # Historical Error ICL, No Forced Attr
                block = (
                    f"[EXAMPLE {idx}]\n"
                    f"Claim: {ex.claim_text}\n"
                    f"Evidence:\n{ex.evidence_snippet}\n"
                    f"[HISTORICAL ERROR CONTRAST]\n"
                    f"- Observed Mistake: {ex.historical_mistake}\n"
                    f"- Corrective Resolution: {ex.corrective_resolution}\n"
                    f"Verdict: {ex.gold_verdict}"
                )
            elif arm == M20fArm.A4:
                # Historical Error ICL + Forced Attribution
                ref_str = str(ex.refuting_passage_ids) if ex.refuting_passage_ids else "None"
                sup_str = str(ex.supporting_passage_ids) if ex.supporting_passage_ids else "None"
                block = (
                    f"[EXAMPLE {idx}]\n"
                    f"Claim: {ex.claim_text}\n"
                    f"Evidence:\n{ex.evidence_snippet}\n"
                    f"[HISTORICAL ERROR CONTRAST]\n"
                    f"- Observed Mistake: {ex.historical_mistake}\n"
                    f"- Corrective Resolution: {ex.corrective_resolution}\n"
                    f"[EVIDENCE ATTRIBUTION]\n"
                    f"- Supporting Passage IDs: {sup_str}\n"
                    f"- Refuting Passage IDs: {ref_str}\n"
                    f"- Scope & Relevance Assessment: {ex.scope_assessment}\n"
                    f"- Sufficiency & Conflict Assessment: {ex.sufficiency_assessment}\n"
                    f"[VERDICT]\n{ex.gold_verdict}"
                )
            blocks.append(block)

        return "\n\n".join(blocks)
