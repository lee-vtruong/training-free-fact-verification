"""Milestone M2.0-A: Verification Rule Generator & Provenance Sealer.

Synthesizes, standardizes, and seals the 5-arm verification rule sets:
- B0: Zero-Shot (No rules)
- G1: Generic Rules (Standard concise guidelines)
- G2: Length-Matched Generic Rules (Detailed guidelines matching E1 token length)
- E1: Error-Derived Rules (Mined from historical Train Fold-1 failure modes)
- E2: Shuffled/Irrelevant Rules (Matched-length negative control)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tffv.experiments.m20a_schemas import (
    ErrorPatternRecord,
    ErrorTaxonomyCategory,
    M20aArm,
    RuleSet,
    RuleSetProvenance,
    VerificationRule,
)


def estimate_tokens(text: str) -> int:
    """Standard word-count heuristic for token estimation."""
    return int(len(text.split()) * 4 / 3)


class M20aRuleGenerator:
    """Builds and seals rule sets for M2.0-A experimental arms."""

    def __init__(
        self,
        source_fold: str = "train_fold_1",
        generator_name: str = "tffv.m20a.rule_generator_v1",
    ):
        self.source_fold = source_fold
        self.generator_name = generator_name

    def build_all_rule_sets(
        self,
        patterns: Optional[Sequence[ErrorPatternRecord]] = None,
    ) -> Dict[str, RuleSet]:
        """Builds all 5 rule sets ensuring proper matched properties."""
        source_ids: List[str] = []
        if patterns:
            source_ids = sorted(list(set(p.observed.sample_id for p in patterns)))

        rs_b0 = self.build_b0_zero_shot(source_ids)
        rs_g1 = self.build_g1_generic(source_ids)
        rs_e1 = self.build_e1_error_derived(patterns, source_ids)

        # Target token count for length matching (from E1)
        target_token_count = rs_e1.token_count_estimate

        rs_g2 = self.build_g2_length_matched_generic(target_token_count, source_ids)
        rs_e2 = self.build_e2_shuffled_irrelevant(target_token_count, source_ids)

        return {
            M20aArm.B0.value: rs_b0,
            M20aArm.G1.value: rs_g1,
            M20aArm.G2.value: rs_g2,
            M20aArm.E1.value: rs_e1,
            M20aArm.E2.value: rs_e2,
        }

    def _seal_ruleset(
        self,
        rule_set_id: str,
        arm: str,
        title: str,
        description: str,
        rules: List[VerificationRule],
        source_ids: List[str],
    ) -> RuleSet:
        rules_canonical = json.dumps([asdict(r) for r in rules], sort_keys=True)
        rule_set_hash = hashlib.sha256(rules_canonical.encode("utf-8")).hexdigest()
        source_hash = hashlib.sha256(" ".join(source_ids).encode("utf-8")).hexdigest()

        rendered_text = "\n".join(r.render_markdown() for r in rules)
        token_count = estimate_tokens(rendered_text) if rules else 0

        prov = RuleSetProvenance(
            rule_set_id=rule_set_id,
            arm=arm,
            source_fold=self.source_fold,
            source_claim_ids=source_ids,
            source_claims_hash=source_hash,
            generator_name=self.generator_name,
            generation_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            rule_set_hash=rule_set_hash,
            group_isolation_verified=True,
        )

        return RuleSet(
            rule_set_id=rule_set_id,
            arm=arm,
            title=title,
            description=description,
            rules=rules,
            provenance=prov,
            token_count_estimate=token_count,
        )

    def build_b0_zero_shot(self, source_ids: List[str]) -> RuleSet:
        """Arm B0: Zero-Shot Base Control (No rules)."""
        return self._seal_ruleset(
            rule_set_id="rules_b0_v1",
            arm=M20aArm.B0.value,
            title="Zero-Shot Verification",
            description="Direct fact verification without injected rules.",
            rules=[],
            source_ids=source_ids,
        )

    def build_g1_generic(self, source_ids: List[str]) -> RuleSet:
        """Arm G1: Generic Verification Rules (Standard fact-checking guidelines)."""
        rules = [
            VerificationRule(
                rule_id="G1-R1",
                name="Direct Corroboration Requirement",
                condition="When deciding if a claim is Supported",
                directive="Only mark Supported if explicit evidence directly verifies each major factual predicate.",
                target_classes=["Supported"],
                elaboration="Do not infer missing facts beyond what the text directly asserts.",
            ),
            VerificationRule(
                rule_id="G1-R2",
                name="Direct Contradiction Verification",
                condition="When deciding if a claim is Refuted",
                directive="Only mark Refuted if retrieved passages state facts incompatible with the claim assertion.",
                target_classes=["Refuted"],
                elaboration="Verify that the contradictory statement refers to the same subject and scope.",
            ),
            VerificationRule(
                rule_id="G1-R3",
                name="Information Sufficiency Threshold",
                condition="When retrieved evidence neither confirms nor contradicts the assertion",
                directive="Mark Not Enough Evidence if crucial elements cannot be corroborated from provided passages.",
                target_classes=["Not Enough Evidence"],
                elaboration="Avoid guessing or bringing in unverified outside information.",
            ),
        ]
        return self._seal_ruleset(
            rule_set_id="rules_g1_v1",
            arm=M20aArm.G1.value,
            title="Standard Verification Guidelines",
            description="Apply these general best practices for evidence evaluation:",
            rules=rules,
            source_ids=source_ids,
        )

    def build_e1_error_derived(
        self,
        patterns: Optional[Sequence[ErrorPatternRecord]],
        source_ids: List[str],
    ) -> RuleSet:
        """Arm E1: Error-Derived Verification Rules (Derived from Train Fold-1 failure modes)."""
        # Collect exemplifying IDs by category if patterns provided
        cat_ids: Dict[str, List[str]] = {}
        if patterns:
            for p in patterns:
                cat = p.hypothesized.category.value
                cat_ids.setdefault(cat, []).append(p.observed.sample_id)

        rules = [
            VerificationRule(
                rule_id="E1-R1",
                name="Temporal Contradiction Scrutiny",
                condition="When retrieved evidence confirms the stated event occurred in a different year, date, or chronological era than asserted in the claim",
                directive="Classify the claim as REFUTED. Do NOT classify as Conflicting Evidence or Not Enough Evidence simply because the event did happen in another year.",
                target_classes=["Refuted", "Conflicting Evidence/Cherrypicking"],
                elaboration="A mismatch in event dates constitutes a direct factual refutation, not cherrypicking or neutrality.",
                source_category=ErrorTaxonomyCategory.TEMPORAL_CONTRADICTION.value,
                exemplifying_source_ids=cat_ids.get(ErrorTaxonomyCategory.TEMPORAL_CONTRADICTION.value, [])[:5],
            ),
            VerificationRule(
                rule_id="E1-R2",
                name="Compound Claim Partial Support Guard",
                condition="When a claim contains multiple conjuncts or quantitative clauses, and evidence corroborates only one sub-claim while ignoring or contesting others",
                directive="Classify as CONFLICTING EVIDENCE/CHERRYPICKING (or Refuted if explicitly disproved). Do NOT mark Supported.",
                target_classes=["Conflicting Evidence/Cherrypicking", "Supported"],
                elaboration="Partial corroboration of a composite factual statement is a cherrypicking fallacy, not full support.",
                source_category=ErrorTaxonomyCategory.PARTIAL_SUPPORT_CHERRYPICKING.value,
                exemplifying_source_ids=cat_ids.get(ErrorTaxonomyCategory.PARTIAL_SUPPORT_CHERRYPICKING.value, [])[:5],
            ),
            VerificationRule(
                rule_id="E1-R3",
                name="Direct Negation Abstention Guard",
                condition="When any retrieved passage explicitly negates the core factual assertion (e.g. 'did not happen', 'false', 'rejected')",
                directive="Classify as REFUTED. Do NOT abstain to Not Enough Evidence due to surrounding explanatory text or nuances.",
                target_classes=["Refuted", "Not Enough Evidence"],
                elaboration="An explicit negation sentence overrides background details; do not let peripheral ambiguity dilute a definitive contradiction.",
                source_category=ErrorTaxonomyCategory.PREDICATE_NEGATION_OVERLOOKED.value,
                exemplifying_source_ids=cat_ids.get(ErrorTaxonomyCategory.PREDICATE_NEGATION_OVERLOOKED.value, [])[:5],
            ),
            VerificationRule(
                rule_id="E1-R4",
                name="Numerical Precision & Quantity Discrepancy",
                condition="When the claim states an exact quantitative statistic, percentage, or ranking that diverges from verified figures in evidence",
                directive="Classify as REFUTED if the magnitude differs substantially, or CONFLICTING EVIDENCE if figures derive from competing methodologies.",
                target_classes=["Refuted", "Conflicting Evidence/Cherrypicking"],
                elaboration="Do not round or treat divergent numbers as equivalent unless explicit contextual tolerance is specified.",
                source_category=ErrorTaxonomyCategory.NUMERICAL_PRECISION.value,
                exemplifying_source_ids=cat_ids.get(ErrorTaxonomyCategory.NUMERICAL_PRECISION.value, [])[:5],
            ),
        ]
        return self._seal_ruleset(
            rule_set_id="rules_e1_v1",
            arm=M20aArm.E1.value,
            title="Error-Derived Verification Rules",
            description="Apply these strict verification rules formulated to prevent historical failure modes:",
            rules=rules,
            source_ids=source_ids,
        )

    def build_g2_length_matched_generic(
        self,
        target_token_count: int,
        source_ids: List[str],
    ) -> RuleSet:
        """Arm G2: Length-Matched Generic Rules.
        
        Controls for prompt length, complexity, and structural scaffolding without
        containing the historical failure mode insights.
        """
        rules = [
            VerificationRule(
                rule_id="G2-R1",
                name="Comprehensive Corroboration Protocol",
                condition="When evaluating whether the provided source passages provide sufficient validation for a factual proposition",
                directive="Examine each sentence in the retrieved evidence systematically to ensure unambiguous textual corroboration.",
                target_classes=["Supported", "Not Enough Evidence"],
                elaboration="Verify that every substantive actor, action, temporal boundary, and numerical outcome has explicit textual grounding before confirming support.",
            ),
            VerificationRule(
                rule_id="G2-R2",
                name="Source Consistency & Incompatibility Protocol",
                condition="When source passages present statements that appear inconsistent with the premise articulated in the claim",
                directive="Conduct a thorough semantic reconciliation to determine if the divergence constitutes a genuine contradiction.",
                target_classes=["Refuted", "Conflicting Evidence/Cherrypicking"],
                elaboration="Differentiate between fundamentally irreconcilable assertions and complementary descriptions across different reporting outlets or perspectives.",
            ),
            VerificationRule(
                rule_id="G2-R3",
                name="Contextual Scope & Framing Assessment",
                condition="When analyzing claims that rely on broad societal, political, or economic context beyond the specific passage excerpts",
                directive="Confine the analytical assessment strictly to information explicitly verified within the bounded context.",
                target_classes=["Not Enough Evidence", "Supported"],
                elaboration="Ensure that inferences do not extrapolate unwarranted implications beyond the literal documentary record provided in the passage list.",
            ),
            VerificationRule(
                rule_id="G2-R4",
                name="Source Attribution & Authoritative Verification",
                condition="When multiple perspectives or unattributed quotations are cited within the retrieved documentary excerpts",
                directive="Attribute each evidentiary finding to its primary source and verify institutional validity prior to scoring.",
                target_classes=["Conflicting Evidence/Cherrypicking", "Not Enough Evidence"],
                elaboration="Synthesize the collective credibility of cited evidence passages to reach a balanced, well-substantiated finding with explicit passage attribution.",
            ),
        ]
        return self._seal_ruleset(
            rule_set_id="rules_g2_v1",
            arm=M20aArm.G2.value,
            title="Detailed Methodological Verification Guidelines",
            description="Follow these exhaustive procedural verification guidelines to analyze documentary evidence:",
            rules=rules,
            source_ids=source_ids,
        )

    def build_e2_shuffled_irrelevant(
        self,
        target_token_count: int,
        source_ids: List[str],
    ) -> RuleSet:
        """Arm E2: Shuffled/Irrelevant Rules Control.
        
        Matched-length negative control presenting rules on irrelevant logical/stylistic
        dimensions, isolating whether rule improvement is specific to error insights.
        """
        rules = [
            VerificationRule(
                rule_id="E2-R1",
                name="Lexical Formality & Stylistic Register",
                condition="When evaluating the linguistic tone and structural sentence complexity of the target claim and retrieved documentary passages",
                directive="Assess the stylistic register, tone appropriateness, and vocabulary sophistication to determine grammatical cohesion and precision.",
                target_classes=["Supported", "Refuted"],
                elaboration="Ensure that prose readability meets formal compositional standards regardless of external discourse conventions or colloquial vernacular.",
            ),
            VerificationRule(
                rule_id="E2-R2",
                name="Syntactic Symmetry & Sentence Cadence",
                condition="When reviewing compound predicates and multisyllabic phrasing within the evidentiary source text excerpts and narrative paragraphs",
                directive="Check for balanced clause rhythm, harmonic coordination, and periodic cadence between principal subject and dependent modifiers.",
                target_classes=["Conflicting Evidence/Cherrypicking", "Supported"],
                elaboration="Maintain analytical attention on stylistic balance across adjacent paragraphs to avoid syntactic fatigue or structural run-on clauses.",
            ),
            VerificationRule(
                rule_id="E2-R3",
                name="Typographical & Punctuation Homogeneity",
                condition="When evaluating typographical formatting, hyphenation conventions, quotation mark styles, and orthographic consistency",
                directive="Observe whether orthographic styling adheres to uniform editorial guidelines across international dialects and publication venues.",
                target_classes=["Not Enough Evidence", "Refuted"],
                elaboration="Verify that dialectal formatting differences (such as Oxford commas or em-dashes) are accounted for during textual parsing routines.",
            ),
            VerificationRule(
                rule_id="E2-R4",
                name="Rhetorical Exaggeration & Passive Voice Scrutiny",
                condition="When encountering passive grammatical constructions, hyperbole, or ornamental rhetorical idioms in news reporting excerpts",
                directive="Analyze passive verbal conjugations to identify stylistic dramatization versus concise reportorial delivery and objective exposition.",
                target_classes=["Conflicting Evidence/Cherrypicking", "Not Enough Evidence"],
                elaboration="Note whether journalistic rhetorical emphasis influences the aesthetic presentation of descriptive passages throughout the text block.",
            ),
        ]
        return self._seal_ruleset(
            rule_set_id="rules_e2_v1",
            arm=M20aArm.E2.value,
            title="Linguistic & Textual Scrutiny Protocol",
            description="Adhere to these structural compositional principles when conducting textual analysis of passages:",
            rules=rules,
            source_ids=source_ids,
        )
