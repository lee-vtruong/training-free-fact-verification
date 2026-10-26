"""Unit tests for M1.5 Failure Mechanism Taxonomy, Observable Signals, and Protocol Integrity."""
import json
from pathlib import Path
import pytest

from tffv.verification.taxonomy import (
    ALL_MECHANISMS,
    GUIDANCE_M0,
    GUIDANCE_M1,
    MECHANISM_ATTRIBUTION,
    MECHANISM_CONFLICT,
    MECHANISM_NEGATION,
    MECHANISM_PARTIAL_TRUTH,
    MECHANISM_SCOPE_QUANTIFIER,
    MECHANISM_SUFFICIENCY,
    TAXONOMY_PROFILES,
    build_controlled_guidance,
    detect_active_mechanisms,
    extract_attribution_signal,
    extract_evidence_sufficiency_signal,
    extract_negation_signal,
    extract_partial_truth_signal,
    extract_scope_quantifier_signal,
)


def test_taxonomy_profiles_integrity():
    """Verify taxonomy structure, schema, and sample sufficiency reporting."""
    assert len(ALL_MECHANISMS) == 6
    assert len(TAXONOMY_PROFILES) == 6

    total_fold_a_support = 0
    for name in ALL_MECHANISMS:
        profile = TAXONOMY_PROFILES[name]
        assert profile.name == name
        assert len(profile.definition) > 20
        assert len(profile.inclusion_criteria) >= 2
        assert len(profile.exclusion_criteria) >= 1
        assert len(profile.gold_error_characterization) > 20
        assert len(profile.observable_signals) >= 1
        assert len(profile.ambiguous_cases) > 10
        assert profile.fold_a_support_count > 0
        assert len(profile.fold_a_example_ids) == profile.fold_a_support_count
        total_fold_a_support += profile.fold_a_support_count

    # Fold A has exactly 25 errors across all 6 categories
    assert total_fold_a_support == 25

    # Strictly verify that sample insufficiency is reported honestly:
    # Only Scope/Quantifier has adequate support (>= 8); the other 5 have n <= 5
    assert TAXONOMY_PROFILES[MECHANISM_SCOPE_QUANTIFIER].is_adequately_supported is True
    assert TAXONOMY_PROFILES[MECHANISM_PARTIAL_TRUTH].is_adequately_supported is False
    assert TAXONOMY_PROFILES[MECHANISM_SUFFICIENCY].is_adequately_supported is False
    assert TAXONOMY_PROFILES[MECHANISM_CONFLICT].is_adequately_supported is False
    assert TAXONOMY_PROFILES[MECHANISM_ATTRIBUTION].is_adequately_supported is False
    assert TAXONOMY_PROFILES[MECHANISM_NEGATION].is_adequately_supported is False


def test_observable_signal_extractors():
    """Verify label-free observable signal extractors on canonical test sentences."""
    # 1. Attribution
    assert extract_attribution_signal("Joe Biden said that he would not build the wall.") is True
    assert extract_attribution_signal('The report states "unemployment fell to 4%".') is True
    assert extract_attribution_signal("President signed a new executive order.") is True
    assert extract_attribution_signal("Inflation rose by 3% in Germany.") is False

    # 2. Scope / Quantifier
    assert extract_scope_quantifier_signal("In Nigeria, 20% goes to local government.") is True
    assert extract_scope_quantifier_signal("South Africa consumes more alcohol per person than elsewhere.") is True
    assert extract_scope_quantifier_signal("The program costs 5 billion dollars.") is True
    assert extract_scope_quantifier_signal("The minister visited the school.") is False

    # 3. Negation
    assert extract_negation_signal("The government did not authorize the funding.") is True
    assert extract_negation_signal("The hospital has no emergency beds.") is True
    assert extract_negation_signal("He was elected without opposition.") is True
    assert extract_negation_signal("The government authorized the funding.") is False

    # 4. Partial Truth / Conjunction
    assert extract_partial_truth_signal("Joe Biden voted for the Iraq War and supported war in Serbia, Syria and Libya.") is True
    assert extract_partial_truth_signal("Short claim and done.") is False

    # 5. Evidence Sufficiency
    assert extract_evidence_sufficiency_signal("claim", evidence=[]) is True
    assert extract_evidence_sufficiency_signal("claim", evidence=None) is False
    assert extract_evidence_sufficiency_signal("claim", evidence=[{"text": "short text"}]) is True
    long_ev = [{"text": "word " * 100}]
    assert extract_evidence_sufficiency_signal("claim", evidence=long_ev) is False


def test_controlled_guidance_builder():
    """Verify M0-M4 guidance formatting, fallback logic, and budget parity."""
    claim_neutral = "The minister visited the secondary school."
    claim_quant = "In Nigeria, 20% of revenue goes to local government."
    claim_attr = 'President Biden said "we will build back better".'

    # M0: Matched generic control
    g_m0, mech_m0 = build_controlled_guidance("M0", claim_neutral)
    assert g_m0 == GUIDANCE_M0
    assert mech_m0 is None

    # M1: Generic claim-type clarification
    g_m1, mech_m1 = build_controlled_guidance("M1", claim_neutral)
    assert g_m1 == GUIDANCE_M1
    assert mech_m1 is None

    # M2: Source-derived mechanism recognition
    # Fallback to M0 when no signal fires
    g_m2_neutral, mech_m2_neutral = build_controlled_guidance("M2", claim_neutral)
    assert g_m2_neutral == GUIDANCE_M0
    assert mech_m2_neutral is None

    # Triggers Scope guidance when quantifier signal is present
    g_m2_quant, mech_m2_quant = build_controlled_guidance("M2", claim_quant)
    assert "Statistical Scope Directives" in g_m2_quant
    assert mech_m2_quant == MECHANISM_SCOPE_QUANTIFIER

    # M3: Negative control (random assignment)
    import random
    rng = random.Random(42)
    g_m3, mech_m3 = build_controlled_guidance("M3", claim_neutral, rng=rng)
    assert mech_m3 in ALL_MECHANISMS
    assert len(g_m3) > 50

    # M4: Permutation consistency uses M0 prompt
    g_m4, mech_m4 = build_controlled_guidance("M4", claim_neutral)
    assert g_m4 == GUIDANCE_M0
    assert mech_m4 is None

    # Unknown condition raises ValueError
    with pytest.raises(ValueError):
        build_controlled_guidance("UNKNOWN_CONDITION", claim_neutral)


def test_fold_a_audit_consistency():
    """Verify Fold A error memory bank records and mechanism mapping."""
    root = Path(__file__).resolve().parent.parent
    mem_bank_p = root / "outputs" / "m142-confirmatory-20261025-233407" / "error_memory_bank.json"

    if not mem_bank_p.is_file():
        pytest.skip("Fold A error memory bank artifact not present locally.")

    data = json.loads(mem_bank_p.read_text(encoding="utf-8"))
    records = data["records"]
    correct_sids = set(data["correct_sids"])
    wrong_sids = set(data["wrong_sids"])

    assert len(records) == 50
    assert len(correct_sids) == 25
    assert len(wrong_sids) == 25

    # Check Scope/Quantifier examples
    scope_profile = TAXONOMY_PROFILES[MECHANISM_SCOPE_QUANTIFIER]
    for sid in scope_profile.fold_a_example_ids:
        assert sid in wrong_sids, f"Expected {sid} in wrong_sids"
        assert sid in records

    # Check Attribution examples
    attr_profile = TAXONOMY_PROFILES[MECHANISM_ATTRIBUTION]
    assert attr_profile.fold_a_support_count == 2
    assert set(attr_profile.fold_a_example_ids) == {"train:3", "train:14"}
    for sid in attr_profile.fold_a_example_ids:
        assert sid in wrong_sids
