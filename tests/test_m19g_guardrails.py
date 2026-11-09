"""Unit tests for Milestone M1.9g guardrails and integrity enforcement."""

import json
import pytest
from pathlib import Path

from tffv.experiments.m19g_guardrails import (
    AVERITEC_CLASSES,
    EvidenceRenderBlankError,
    KnowledgeStoreUnverifiedError,
    RetrievalReceiptsMissingError,
    parse_verdict_hardened,
    validate_evidence_rendering,
    verify_knowledge_store,
    verify_retrieval_receipts,
)


def test_verify_knowledge_store_fails_when_dir_missing(tmp_path: Path):
    with pytest.raises(KnowledgeStoreUnverifiedError, match="does not exist"):
        verify_knowledge_store(["dev:0", "dev:1"], tmp_path / "nonexistent")


def test_verify_knowledge_store_fails_when_claim_missing(tmp_path: Path):
    store_dir = tmp_path / "ks"
    store_dir.mkdir()
    (store_dir / "0.json").write_text(json.dumps({"text": "Valid text document"}), encoding="utf-8")
    
    with pytest.raises(KnowledgeStoreUnverifiedError, match="missing"):
        verify_knowledge_store(["dev:0", "dev:1"], store_dir)


def test_verify_knowledge_store_fails_when_doc_text_empty(tmp_path: Path):
    store_dir = tmp_path / "ks"
    store_dir.mkdir()
    (store_dir / "0.json").write_text(json.dumps({"text": "   "}), encoding="utf-8")
    
    with pytest.raises(KnowledgeStoreUnverifiedError, match="0 valid text documents"):
        verify_knowledge_store(["dev:0"], store_dir)


def test_verify_knowledge_store_succeeds_valid(tmp_path: Path):
    store_dir = tmp_path / "ks"
    store_dir.mkdir()
    (store_dir / "0.json").write_text(json.dumps({"text": "Evidence for 0"}), encoding="utf-8")
    (store_dir / "1.json").write_text(json.dumps({"url2text": ["Evidence for 1"]}), encoding="utf-8")
    
    summary = verify_knowledge_store(["dev:0", "dev:1"], store_dir)
    assert summary["status"] == "VERIFIED"
    assert summary["verified_claims_count"] == 2


def test_verify_retrieval_receipts_fails_missing_claim(tmp_path: Path):
    receipts_file = tmp_path / "receipts.jsonl"
    receipts_file.write_text(
        json.dumps({"sample_id": "dev:0", "evidence": [{"text": "snippet 0"}]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RetrievalReceiptsMissingError, match="claims missing"):
        verify_retrieval_receipts(["dev:0", "dev:1"], receipts_file)


def test_verify_retrieval_receipts_fails_blank_passage(tmp_path: Path):
    receipts_file = tmp_path / "receipts.jsonl"
    receipts_file.write_text(
        json.dumps({"sample_id": "dev:0", "evidence": [{"text": ""}]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RetrievalReceiptsMissingError, match="contain blank passages"):
        verify_retrieval_receipts(["dev:0"], receipts_file)


def test_validate_evidence_rendering_rule():
    # Non-empty evidence rendered as blank -> FAILS CLOSED
    with pytest.raises(EvidenceRenderBlankError, match="rendered evidence is completely blank"):
        validate_evidence_rendering([{"text": "snippet"}], "", context="Arm H2 target prompt")

    # Non-empty evidence rendered with content -> PASSES
    validate_evidence_rendering([{"text": "snippet"}], "[1] snippet content", context="Arm H2 target prompt")

    # Empty evidence with blank rendered -> PASSES
    validate_evidence_rendering([], "", context="Empty retrieval case")


def test_parse_verdict_hardened_immune_to_template_echo():
    raw_completion_echo = """
Explanation: <Concise step-by-step reasoning citing specific passages>
Verdict: <Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking>
---

Explanation: The retrieved documents do not confirm the statement.
Verdict: Not Enough Evidence
"""
    verdict, parseable, note = parse_verdict_hardened(raw_completion_echo)
    assert parseable is True
    assert verdict == "Not Enough Evidence"
    assert note == "CANONICAL_MATCH"


def test_parse_verdict_hardened_multiple_classes():
    for expected in AVERITEC_CLASSES:
        raw = f"""
Verdict: <Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking>
Some detailed reasoning steps...
Verdict: {expected}
"""
        verdict, parseable, note = parse_verdict_hardened(raw)
        assert parseable is True
        assert verdict == "expected" or verdict == expected
