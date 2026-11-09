"""Unit and integration tests for Milestone M1.9f Clean Official Dev 500 pipeline."""
import json
from pathlib import Path
import pytest

from tffv.experiments.m19f_evidence import (
    EvidenceSchemaError,
    extract_passage_text,
    format_evidence_block,
    normalize_evidence_list,
    normalize_passage,
)
from tffv.experiments.m19f_loader import (
    EXPECTED_DEV_SHA256,
    EXPECTED_OFFICIAL_DEV_COUNT,
    OfficialDevLoader,
)
from tffv.experiments.m19f_selector import (
    EXPERIMENTAL_ARMS,
    HistoricalMemorySelector,
)
from tffv.experiments.m19f_store import (
    EXACT_TRAIN_DEV_OVERLAPS,
    QUARANTINED_CLAIM_ID,
    HistoricalOutcomeStoreBuilder,
    HistoricalRecord,
)


def test_evidence_schema_normalization_and_fail_closed():
    # Canonical receipt schema with 'text'
    p1 = {"text": "Valid retrieved evidence content", "source_url": "https://example.com/1", "rank": 1}
    np1 = normalize_passage(p1, rank=1, fail_closed=True)
    assert np1.text == "Valid retrieved evidence content"
    assert not np1.is_blank

    # Fallback to rendered_text
    p2 = {"rendered_text": "Secondary fallback content", "rank": 2}
    np2 = normalize_passage(p2, rank=2, fail_closed=True)
    assert np2.text == "Secondary fallback content"

    # Genuinely empty passage
    p_empty = {"rank": 3}
    np_empty = normalize_passage(p_empty, rank=3, fail_closed=False)
    assert np_empty.is_blank

    # Fail closed on blanked text when content exists in source receipt
    p_corrupt = {"text": "", "has_raw_content": True, "source_url": "https://example.com"}
    with pytest.raises(EvidenceSchemaError):
        normalize_passage(p_corrupt, rank=4, fail_closed=True)

    # Format evidence block
    block = format_evidence_block([np1, np2])
    assert "[1] Valid retrieved evidence content" in block
    assert "[2] Secondary fallback content" in block


def test_official_dev_loader_integrity():
    dev_path = Path("data/raw/averitec/dev.json")
    if not dev_path.exists():
        pytest.skip("Local dev.json not found")

    loader = OfficialDevLoader(dev_path=dev_path)
    claims, summary = loader.load_and_validate()

    assert len(claims) == EXPECTED_OFFICIAL_DEV_COUNT
    assert summary["sha256_verified"] is True
    assert summary["train_ids_present"] is False
    assert claims[0].sample_id == "dev:0"
    assert claims[-1].sample_id == f"dev:{EXPECTED_OFFICIAL_DEV_COUNT - 1}"

    # Verify no claim has empty text
    for c in claims:
        assert len(c.claim) > 0
        assert c.gold_verdict in {"Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"}


def test_historical_memory_eligibility_and_overlap_guards(tmp_path):
    # Create synthetic records
    records = []
    # 1 quarantined
    records.append(
        HistoricalRecord(
            sample_id=QUARANTINED_CLAIM_ID,
            claim="",
            gold_verdict="Supported",
            base_verdict="Supported",
            correctness="QUARANTINED",
            evidence_count=0,
            evidence_hash="",
            evidence_references=[],
            retrieval_fingerprint="R4-PF1000-v1",
            verifier_fingerprint="Qwen3-8B-BF16",
            leakage_group="q",
            is_selectable_correct=False,
            is_selectable_wrong=False,
            exclusion_reason="QUARANTINED_EMPTY_CLAIM",
        )
    )
    # 4 overlaps
    for sid in EXACT_TRAIN_DEV_OVERLAPS:
        records.append(
            HistoricalRecord(
                sample_id=sid,
                claim=f"Overlap claim {sid}",
                gold_verdict="Supported",
                base_verdict="Refuted",
                correctness="WRONG",
                evidence_count=1,
                evidence_hash="h",
                evidence_references=[{"rendered_text": "sample text"}],
                retrieval_fingerprint="R4-PF1000-v1",
                verifier_fingerprint="Qwen3-8B-BF16",
                leakage_group=f"grp_{sid}",
                is_selectable_correct=False,
                is_selectable_wrong=False,
                exclusion_reason=f"TRAIN_DEV_OVERLAP_WITH_{EXACT_TRAIN_DEV_OVERLAPS[sid]}",
            )
        )
    # 10 selectable records (5 correct, 5 wrong)
    for i in range(10):
        is_corr = (i % 2 == 0)
        records.append(
            HistoricalRecord(
                sample_id=f"train:{100 + i}",
                claim=f"Historical fact claim {i}",
                gold_verdict="Supported" if is_corr else "Refuted",
                base_verdict="Supported",
                correctness="CORRECT" if is_corr else "WRONG",
                evidence_count=1,
                evidence_hash="h",
                evidence_references=[{"rendered_text": f"Fact evidence {i}"}],
                retrieval_fingerprint="R4-PF1000-v1",
                verifier_fingerprint="Qwen3-8B-BF16",
                leakage_group=f"grp_{i}",
                is_selectable_correct=is_corr,
                is_selectable_wrong=not is_corr,
                exclusion_reason="NONE",
            )
        )

    selector = HistoricalMemorySelector(records, k=2)

    # Check that quarantined and overlap IDs are not in selectable pools
    assert QUARANTINED_CLAIM_ID not in selector.correct_pool
    assert QUARANTINED_CLAIM_ID not in selector.wrong_pool
    for sid in EXACT_TRAIN_DEV_OVERLAPS:
        assert sid not in selector.correct_pool
        assert sid not in selector.wrong_pool

    assert len(selector.correct_pool) == 5
    assert len(selector.wrong_pool) == 5

    # Check matched exemplar identity (B1 == H2 and B2 == H3)
    target_claim = "Fact check query"
    target_ev = [{"rendered_text": "Target evidence"}]

    sel_h2 = selector.select_exemplars_for_arm("H2", "dev:0", target_claim, target_ev)
    sel_b1 = selector.select_exemplars_for_arm("B1", "dev:0", target_claim, target_ev, matched_exemplar_ids=sel_h2.chosen_exemplar_ids)
    assert sel_b1.chosen_exemplar_ids == sel_h2.chosen_exemplar_ids

    # Ensure B1 prompt hides Base prediction and error label
    assert "INCORRECT" not in sel_b1.formatted_prompt
    assert "Previous Model Verdict" not in sel_b1.formatted_prompt
    # Ensure H2 prompt includes explicit negative instruction
    assert "Do NOT repeat this error" in sel_h2.formatted_prompt

    sel_h3 = selector.select_exemplars_for_arm("H3", "dev:0", target_claim, target_ev)
    sel_b2 = selector.select_exemplars_for_arm("B2", "dev:0", target_claim, target_ev, matched_exemplar_ids=sel_h3.chosen_exemplar_ids)
    assert sel_b2.chosen_exemplar_ids == sel_h3.chosen_exemplar_ids
    assert "Previous Model Verdict" not in sel_b2.formatted_prompt
