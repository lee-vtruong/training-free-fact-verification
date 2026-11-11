"""Milestone M2.0i: Placebo Mappings and Exemplar Registry for Mechanism Ablation.

Controls:
- Exact 1:1 word-matched neutral placebo mistakes for all 10 certified exemplars.
- Deterministic, fixed seed assignment.
- Verified absence of target gold correlation and semantic leakage.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from tffv.experiments.m20g_exemplars import CERTIFIED_EXEMPLAR_BANK, ExemplarItem

# Deterministic, length-matched neutral placebo observed mistakes.
# Carefully constructed to preserve syntax, domain tone, and exact word count,
# while removing the specific historical error mechanism and containing zero target gold leaks.
MATCHED_PLACEBO_MISTAKES: Dict[str, str] = {
    # 1. train:2956 (Supported, 12 words)
    "train:2956": "Earlier baseline concluded an alternate outcome by misinterpreting general narrative contextual descriptions.",
    # 2. train:37 (Supported, 13 words)
    "train:37": "Earlier baseline confused secondary background news commentary with published macroeconomic GDP growth tables.",
    # 3. train:622 (Supported, 14 words)
    "train:622": "Earlier baseline prioritized peripheral news commentary rather than consulting official primary published agency records.",
    # 4. train:47 (Refuted, 13 words)
    "train:47": "Earlier baseline assumed regional demographic trends established absolute verified continental national statistical values.",
    # 5. train:48 (Refuted, 11 words)
    "train:48": "Earlier baseline accepted preliminary survey summaries without reviewing published comparative tables.",
    # 6. train:1863 (Refuted, 13 words)
    "train:1863": "Earlier baseline mistook preliminary committee advisory deliberations for formal executive government policy decisions.",
    # 7. train:35 (Not Enough Evidence, 14 words)
    "train:35": "Earlier baseline presumed definitive trends based on preliminary quarterly figures without formal annual audits.",
    # 8. train:49 (Not Enough Evidence, 13 words)
    "train:49": "Earlier baseline generalized localized urban sample estimates to broader population demographic summary metrics.",
    # 9. train:8 (Conflicting Evidence/Cherrypicking, 17 words)
    "train:8": "Earlier baseline rejected the entire composite statement because one of the listed sub-points lacked independent separate documentation.",
    # 10. train:12 (Conflicting Evidence/Cherrypicking, 15 words)
    "train:12": "Earlier baseline labeled the statement invalid, overlooking documented acceleration across specific intermediate economic reporting periods.",
}


def get_placebo_for_exemplar(sample_id: str) -> str:
    """Returns the verified matched placebo mistake for the given exemplar ID."""
    if sample_id not in MATCHED_PLACEBO_MISTAKES:
        raise KeyError(f"No certified placebo mistake defined for exemplar {sample_id}")
    return MATCHED_PLACEBO_MISTAKES[sample_id]
