"""Milestone M2.0g: Exemplar Bank & Multi-Seed Balancer.

Defines:
- Certified exemplar bank from Train Fold-1 with 100% group isolation from evaluation partitions.
- Formatter for the 4 knowledge component levels (K1, K2, K3, K4).
- Multi-seed deterministic exemplar selector for k in {2, 4, 6} across Seeds 1..5.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from tffv.experiments.m20g_schemas import (
    ExemplarItem,
    KnowledgeComponent,
)

CERTIFIED_EXEMPLAR_BANK: List[ExemplarItem] = [
    # Supported exemplars
    ExemplarItem(
        sample_id="train:2956",
        group_id="grp-16097fa1c40f653a",
        claim_text="The Kenyan judiciary receives less than 1% of the national budget.",
        evidence_snippet=(
            "[1] The Kenyan judiciary receives less than 1% of the national budget, "
            "according to official treasury reports and statements by Chief Justice David Maraga."
        ),
        gold_verdict="Supported",
        historical_wrong_verdict="Not Enough Evidence",
        neutral_explanation="The passage explicitly confirms the exact percentage; verbatim statement substantiates the claim.",
        observed_mistake="Earlier baseline concluded 'Not Enough Evidence' by overlooking the explicit numerical figure.",
        corrective_resolution="The passage explicitly confirms the exact percentage; verbatim statement substantiates the claim.",
    ),
    ExemplarItem(
        sample_id="train:37",
        group_id="grp-d131cd3e0c0942e2",
        claim_text="Ghana's economy previously grew at 3.4% under the former President.",
        evidence_snippet=(
            "[1] World Bank economic indicators confirm Ghana's GDP growth rate was recorded at 3.4% in 2016."
        ),
        gold_verdict="Supported",
        historical_wrong_verdict="Refuted",
        neutral_explanation="Official World Bank macroeconomic reports substantiate the recorded 3.4% GDP growth rate.",
        observed_mistake="Earlier baseline confused local growth projections with final World Bank published growth rates.",
        corrective_resolution="Official World Bank macroeconomic reports substantiate the recorded 3.4% GDP growth rate.",
    ),
    ExemplarItem(
        sample_id="train:622",
        group_id="grp-1faaa8020afd400a",
        claim_text="A German Shepherd in New York was the first confirmed dog to test positive for SARS-CoV-2 in the US.",
        evidence_snippet=(
            "[1] The USDA National Veterinary Services Laboratories confirmed that a German Shepherd in New York "
            "was the first dog in the United States to test positive for SARS-CoV-2."
        ),
        gold_verdict="Supported",
        historical_wrong_verdict="Conflicting Evidence/Cherrypicking",
        neutral_explanation="The USDA official agency report unequivocally verifies the entity, location, and first positive diagnosis.",
        observed_mistake="Earlier baseline perceived conflicting media articles rather than prioritizing the primary USDA press release.",
        corrective_resolution="The USDA official agency report unequivocally verifies the entity, location, and first positive diagnosis.",
    ),

    # Refuted exemplars
    ExemplarItem(
        sample_id="train:47",
        group_id="grp-ad2fe96a6908a131",
        claim_text="South Africans consume more alcohol per person on average than anyone else in Africa.",
        evidence_snippet=(
            "[1] The WHO Global Status Report on Alcohol shows Nigeria and Gabon rank higher than South Africa "
            "in pure alcohol consumption per drinker in Africa."
        ),
        gold_verdict="Refuted",
        historical_wrong_verdict="Supported",
        neutral_explanation="WHO official tables rank Gabon and Nigeria higher, directly contradicting the continental superlative.",
        observed_mistake="Earlier baseline assumed high regional drinking rates established the absolute continental top ranking.",
        corrective_resolution="WHO official tables rank Gabon and Nigeria higher, directly contradicting the continental superlative.",
    ),
    ExemplarItem(
        sample_id="train:48",
        group_id="grp-ad2fe96a6908a131",
        claim_text="South Africans drink more per person than anywhere else in Africa.",
        evidence_snippet=(
            "[1] Data from the World Health Organization indicates per capita consumption in Namibia and Gabon exceeds South Africa."
        ),
        gold_verdict="Refuted",
        historical_wrong_verdict="Supported",
        neutral_explanation="Official World Health Organization statistics show per capita consumption in neighboring countries exceeds South Africa.",
        observed_mistake="Earlier baseline accepted anecdotal claims without verifying cross-national per capita metrics.",
        corrective_resolution="Official World Health Organization statistics show per capita consumption in neighboring countries exceeds South Africa.",
    ),
    ExemplarItem(
        sample_id="train:1863",
        group_id="grp-6fb1c68e6783b216",
        claim_text="President Buhari approves the creation of state police in Nigeria.",
        evidence_snippet=(
            "[1] Presidential spokesman Femi Adesina explicitly clarified that President Buhari did not approve "
            "state police, explaining the recommendation was merely received for further study."
        ),
        gold_verdict="Refuted",
        historical_wrong_verdict="Conflicting Evidence/Cherrypicking",
        neutral_explanation="The official presidency statement directly refutes the claim of executive approval.",
        observed_mistake="Earlier baseline mistook receiving a committee advisory report for formal presidential executive approval.",
        corrective_resolution="The official presidency statement directly refutes the claim of executive approval.",
    ),

    # Not Enough Evidence exemplars
    ExemplarItem(
        sample_id="train:35",
        group_id="grp-d131cd3e0c0942e2",
        claim_text="The current Ghana administration has resulted in a decline in foreign direct investment.",
        evidence_snippet=(
            "[1] Ghana Investment Promotion Centre released preliminary figures for 2018 showing fluctuating quarterly flows, "
            "with full audited annual balance of payments not yet finalized."
        ),
        gold_verdict="Not Enough Evidence",
        historical_wrong_verdict="Supported",
        neutral_explanation="Preliminary quarterly flows without full audited balance of payments are insufficient to prove a sustained net decline.",
        observed_mistake="Earlier baseline declared support based on a single quarter's dip, overlooking incomplete annual data.",
        corrective_resolution="Preliminary quarterly flows without full audited balance of payments are insufficient to prove a sustained net decline.",
    ),
    ExemplarItem(
        sample_id="train:49",
        group_id="grp-ad2fe96a6908a131",
        claim_text="Only 31% of South Africans drink alcohol.",
        evidence_snippet=(
            "[1] Surveys cite varying estimates ranging from 30% to 35% among adults in selected urban provinces, "
            "lacking a unified national census figure."
        ),
        gold_verdict="Not Enough Evidence",
        historical_wrong_verdict="Supported",
        neutral_explanation="Regional survey estimates with varied sampling methodology cannot definitively substantiate the exact national 31% statistic.",
        observed_mistake="Earlier baseline equated partial provincial sample estimates with a verified nationwide demographic figure.",
        corrective_resolution="Regional survey estimates with varied sampling methodology cannot definitively substantiate the exact national 31% statistic.",
    ),

    # Conflicting Evidence/Cherrypicking exemplars
    ExemplarItem(
        sample_id="train:8",
        group_id="grp-89b80b7304ffaec0",
        claim_text="Joe Biden voted for the Iraq War and supported war in Serbia, Syria and Libya.",
        evidence_snippet=(
            "[1] Official Senate records confirm Biden voted for the 2002 Iraq War authorization.\n"
            "[2] Independent analysis notes his record on Serbia and Libya is mixed, while Syria intervention claims rely on unverified campaign speeches."
        ),
        gold_verdict="Conflicting Evidence/Cherrypicking",
        historical_wrong_verdict="Refuted",
        neutral_explanation="The Iraq vote is fully verified, but attributing the entire four-conflict interventionist record mixes factual votes with unverified campaign claims.",
        observed_mistake="Earlier baseline rejected the entire statement as Refuted because one of the listed conflicts lacked primary documentation.",
        corrective_resolution="The Iraq vote is fully verified, but attributing the entire four-conflict interventionist record mixes factual votes with unverified campaign claims.",
    ),
    ExemplarItem(
        sample_id="train:12",
        group_id="grp-89b80b7304ffaec0",
        claim_text="Donald Trump inherited a stagnant economy and then rebuilt it.",
        evidence_snippet=(
            "[1] Bureau of Economic Analysis data shows GDP growth was 1.6% in 2016 and rose to 2.9% in 2018.\n"
            "[2] Non-partisan economists note low unemployment and job creation trends began steadily under the previous administration."
        ),
        gold_verdict="Conflicting Evidence/Cherrypicking",
        historical_wrong_verdict="Refuted",
        neutral_explanation="Economic indicators confirm growth rates accelerated, but attributing economic health solely to the transition cherry-picks continuous multi-year trends.",
        observed_mistake="Earlier baseline labeled the claim outright false, overlooking documented acceleration in specific quarterly GDP metrics.",
        corrective_resolution="Economic indicators confirm growth rates accelerated, but attributing economic health solely to the transition cherry-picks continuous multi-year trends.",
    ),
]


class ExemplarManager:
    """Manages exemplar selection across seeds and formats knowledge components."""

    @classmethod
    def get_seed_exemplars(cls, seed: int, k: int) -> List[ExemplarItem]:
        """Selects balanced k exemplars for a given seed."""
        supported = [e for e in CERTIFIED_EXEMPLAR_BANK if e.gold_verdict == "Supported"]
        refuted = [e for e in CERTIFIED_EXEMPLAR_BANK if e.gold_verdict == "Refuted"]
        nei = [e for e in CERTIFIED_EXEMPLAR_BANK if e.gold_verdict == "Not Enough Evidence"]
        conf = [e for e in CERTIFIED_EXEMPLAR_BANK if e.gold_verdict == "Conflicting Evidence/Cherrypicking"]

        rng = random.Random(seed)
        shuffled_sup = list(supported)
        shuffled_ref = list(refuted)
        shuffled_nei = list(nei)
        shuffled_conf = list(conf)

        rng.shuffle(shuffled_sup)
        rng.shuffle(shuffled_ref)
        rng.shuffle(shuffled_nei)
        rng.shuffle(shuffled_conf)

        if k == 2:
            # 1 Supported + 1 Refuted
            return [shuffled_sup[0], shuffled_ref[0]]
        elif k == 4:
            # 1 Supported + 1 Refuted + 1 NEI + 1 Conflicting
            return [shuffled_sup[0], shuffled_ref[0], shuffled_nei[0], shuffled_conf[0]]
        elif k == 6:
            # 2 Supported + 2 Refuted + 1 NEI + 1 Conflicting
            return [
                shuffled_sup[0],
                shuffled_ref[0],
                shuffled_sup[1 % len(shuffled_sup)],
                shuffled_ref[1 % len(shuffled_ref)],
                shuffled_nei[0],
                shuffled_conf[0],
            ]
        else:
            raise ValueError(f"Unsupported k: {k}. Supported values: 2, 4, 6.")

    @classmethod
    def format_demonstrations(
        cls,
        exemplars: List[ExemplarItem],
        component: KnowledgeComponent,
    ) -> str:
        """Formats exemplars according to the specified knowledge component level."""
        blocks = []
        for idx, ex in enumerate(exemplars, start=1):
            if component == KnowledgeComponent.K1_GOLD_ONLY:
                block = (
                    f"[EXAMPLE {idx}]\n"
                    f"Claim: {ex.claim_text}\n"
                    f"Evidence:\n{ex.evidence_snippet}\n"
                    f"Verdict: {ex.gold_verdict}"
                )
            elif component == KnowledgeComponent.K2_GOLD_NEUTRAL_EXPL:
                block = (
                    f"[EXAMPLE {idx}]\n"
                    f"Claim: {ex.claim_text}\n"
                    f"Evidence:\n{ex.evidence_snippet}\n"
                    f"[VERIFICATION REASONING]\n"
                    f"- Factual Basis: {ex.neutral_explanation}\n"
                    f"Verdict: {ex.gold_verdict}"
                )
            elif component == KnowledgeComponent.K3_WRONG_PRED_GOLD_CORR:
                block = (
                    f"[EXAMPLE {idx}]\n"
                    f"Claim: {ex.claim_text}\n"
                    f"Evidence:\n{ex.evidence_snippet}\n"
                    f"[VERIFICATION REASONING]\n"
                    f"- Prior Baseline Verdict: {ex.historical_wrong_verdict}\n"
                    f"- Correct Verdict: {ex.gold_verdict}\n"
                    f"Verdict: {ex.gold_verdict}"
                )
            elif component == KnowledgeComponent.K4_FULL_GROUNDED_ERROR:
                block = (
                    f"[EXAMPLE {idx}]\n"
                    f"Claim: {ex.claim_text}\n"
                    f"Evidence:\n{ex.evidence_snippet}\n"
                    f"[HISTORICAL ERROR CONTRAST]\n"
                    f"- Observed Mistake: {ex.observed_mistake}\n"
                    f"- Corrective Resolution: {ex.corrective_resolution}\n"
                    f"Verdict: {ex.gold_verdict}"
                )
            else:
                raise ValueError(f"Unknown KnowledgeComponent: {component}")
            blocks.append(block)

        return "\n\n".join(blocks)
