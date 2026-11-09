"""Milestone M2.0-A: Prompt Formatter for Five Experimental Arms.

Ensures:
1. Identical target claim and retrieved evidence rendering across all 5 arms.
2. Invariant verification preventing blank or corrupt evidence strings.
3. Clean, modular injection of rule sets (B0, G1, G2, E1, E2).
4. Strict AVeriTeC 4-class label schema compliance.
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
from tffv.experiments.m20a_schemas import M20aArm, RuleSet

BASE_SYSTEM_INSTRUCTION = """You are an impartial, highly rigorous fact-checking system.
Your task is to determine the truthfulness of a given claim based STRICTLY and SOLELY on the retrieved evidence passages provided.
Do NOT use external knowledge not present in the evidence.

You must assign exactly ONE of the four official verdicts:
1. Supported: The evidence provides direct, unambiguous corroboration for all factual assertions in the claim.
2. Refuted: The evidence directly contradicts or disproves at least one core factual assertion in the claim.
3. Not Enough Evidence: The evidence is insufficient, absent, or incomplete, leaving the claim unverifiable.
4. Conflicting Evidence/Cherrypicking: The evidence presents contradictory findings from credible sources, or the claim cherry-picks true facts while omitting critical context that changes the overall truth value.

Format requirements:
Explanation: <Concise, evidence-grounded justification citing specific passage numbers>
Verdict: <Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking>"""


class M20aPromptFormatter:
    """Renders consistent, invariant-guarded prompts across the 5 arms."""

    def __init__(self, rule_sets: Dict[str, RuleSet]):
        self.rule_sets = rule_sets

    def format_prompt(
        self,
        arm: str,
        claim_text: str,
        evidence_passages: Sequence[Dict[str, Any] | NormalizedPassage],
        context_id: str = "",
    ) -> Tuple[str, str, int]:
        """Renders the prompt for the specified arm.
        
        Returns:
            Tuple of (formatted_prompt, prompt_sha256, estimated_tokens)
        """
        if arm not in self.rule_sets and arm not in [a.value for a in M20aArm]:
            raise ValueError(f"Unknown M2.0-A arm: {arm}. Must be one of {[a.value for a in M20aArm]}")

        if not evidence_passages or len(evidence_passages) == 0:
            raise ValueError(f"Integrity Violation: Claim {context_id} must have a non-empty list of passages.")

        # 1. Format and validate target evidence
        ev_block = format_evidence_block(evidence_passages)
        validate_evidence_rendering(evidence_passages, ev_block, context=f"M2.0-A Claim {context_id}")

        # 2. Build prompt components
        parts: List[str] = [BASE_SYSTEM_INSTRUCTION, "\n\n"]

        # 3. Inject Rules if applicable
        rule_set = self.rule_sets.get(arm)
        if rule_set and rule_set.rules:
            rendered_rules = rule_set.render_rules_block()
            parts.append(rendered_rules)
            parts.append("\n\n---\n\n")
        else:
            parts.append("---\n\n")

        # 4. Target Task Block (Identical across all arms)
        parts.append("### Target Verification Task:\n")
        parts.append(f"Claim: {claim_text.strip()}\n\n")
        parts.append(f"Evidence:\n{ev_block}\n\n")
        parts.append("Provide your reasoning and final verdict following the schema above:\n")

        full_prompt = "".join(parts)
        prompt_sha256 = hashlib.sha256(full_prompt.encode("utf-8")).hexdigest()
        token_estimate = int(len(full_prompt.split()) * 4 / 3)

        return full_prompt, prompt_sha256, token_estimate
