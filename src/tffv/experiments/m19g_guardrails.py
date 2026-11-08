"""Milestone M1.9g: Strict Guardrails and Integrity Verification.

Enforces mandatory pipeline rules:
1. Knowledge store must be verified before retrieval or verification runs.
2. Evidence in retrieval receipts must NEVER render as an empty string.
3. Template-echo immune verdict parser prevents schema placeholders from contaminating predictions.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


AVERITEC_CLASSES = {
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
}


class KnowledgeStoreUnverifiedError(RuntimeError):
    """Raised when target knowledge store fails preflight integrity check."""
    pass


class EvidenceRenderBlankError(RuntimeError):
    """Raised when evidence exists in receipt but renders as an empty string."""
    pass


class RetrievalReceiptsMissingError(RuntimeError):
    """Raised when retrieval receipts are incomplete, missing claims, or corrupted."""
    pass


def verify_knowledge_store(
    sample_ids: Sequence[str],
    store_dir: str | Path,
    min_docs_per_claim: int = 1,
) -> Dict[str, Any]:
    """Pre-flight verification of target knowledge store.
    
    Fails closed if any claim file is missing, empty, or lacks readable text.
    """
    store_path = Path(store_dir)
    if not store_path.exists() or not store_path.is_dir():
        raise KnowledgeStoreUnverifiedError(
            f"Knowledge store directory does not exist: {store_path}"
        )

    missing_claims = []
    empty_claims = []
    total_docs = 0

    for sid in sample_ids:
        ordinal = sid.split(":")[1] if ":" in sid else sid
        claim_file = store_path / f"{ordinal}.json"
        
        if not claim_file.exists():
            missing_claims.append(sid)
            continue

        # Verify readability and non-empty content
        docs_count = 0
        has_non_empty_text = False
        try:
            with open(claim_file, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                        docs_count += 1
                        text = ""
                        if doc.get("text"):
                            text = str(doc.get("text"))
                        elif doc.get("url2text"):
                            u2t = doc.get("url2text")
                            text = "\n".join(u2t) if isinstance(u2t, list) else str(u2t)
                        if text.strip():
                            has_non_empty_text = True
                    except Exception:
                        continue
        except Exception as e:
            raise KnowledgeStoreUnverifiedError(f"Failed reading claim store {claim_file}: {e}")

        if docs_count < min_docs_per_claim or not has_non_empty_text:
            empty_claims.append(sid)
        total_docs += docs_count

    if missing_claims:
        raise KnowledgeStoreUnverifiedError(
            f"Knowledge store verification failed: {len(missing_claims)} claim files missing! "
            f"Sample missing: {missing_claims[:5]}"
        )

    if empty_claims:
        raise KnowledgeStoreUnverifiedError(
            f"Knowledge store verification failed: {len(empty_claims)} claims have 0 valid text documents! "
            f"Sample empty: {empty_claims[:5]}"
        )

    summary = {
        "status": "VERIFIED",
        "store_dir": str(store_path),
        "verified_claims_count": len(sample_ids),
        "total_documents_indexed": total_docs,
        "avg_docs_per_claim": total_docs / len(sample_ids) if sample_ids else 0.0,
    }
    return summary


def verify_retrieval_receipts(
    sample_ids: Sequence[str],
    receipts_path: str | Path,
    require_all_claims: bool = True,
) -> Dict[str, Any]:
    """Pre-flight verification of retrieval receipts before running verifier.
    
    Fails closed if any claim is missing or receipts have blank passages.
    """
    p = Path(receipts_path)
    if not p.exists():
        raise RetrievalReceiptsMissingError(f"Retrieval receipts file not found: {p}")

    receipts_by_id: Dict[str, Dict[str, Any]] = {}
    with open(p, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                sid = row.get("sample_id")
                if sid:
                    receipts_by_id[sid] = row
            except Exception as e:
                raise RetrievalReceiptsMissingError(f"Malformed JSON on line {line_no} in {p}: {e}")

    missing_ids = [sid for sid in sample_ids if sid not in receipts_by_id]
    if require_all_claims and missing_ids:
        raise RetrievalReceiptsMissingError(
            f"Retrieval receipts incomplete: {len(missing_ids)}/{len(sample_ids)} claims missing from {p}! "
            f"Sample missing: {missing_ids[:5]}"
        )

    blank_passage_claims = []
    total_passages = 0

    for sid, rec in receipts_by_id.items():
        if sid not in sample_ids:
            continue
        ev_list = rec.get("evidence", [])
        total_passages += len(ev_list)
        for passage in ev_list:
            text = (
                passage.get("text")
                or passage.get("rendered_text")
                or passage.get("quote")
                or ""
            ).strip()
            if not text:
                blank_passage_claims.append(sid)
                break

    if blank_passage_claims:
        raise RetrievalReceiptsMissingError(
            f"Retrieval receipts integrity error: {len(blank_passage_claims)} claims contain blank passages! "
            f"Sample claims: {blank_passage_claims[:5]}"
        )

    return {
        "status": "VERIFIED",
        "receipts_path": str(p),
        "total_receipts": len(receipts_by_id),
        "evaluation_claims_present": len(sample_ids) - len(missing_ids),
        "total_passages": total_passages,
    }


def validate_evidence_rendering(
    evidence_list: Sequence[Any],
    rendered_text: str,
    context: str = "",
) -> None:
    """Enforces rule: Evidence in retrieval receipts must NEVER render as empty string."""
    if evidence_list and len(evidence_list) > 0:
        if not rendered_text or not rendered_text.strip():
            raise EvidenceRenderBlankError(
                f"Integrity Violation: {len(evidence_list)} evidence passages exist in receipt, "
                f"but rendered evidence is completely blank in {context}!"
            )


def parse_verdict_hardened(
    raw_completion: str,
) -> Tuple[Optional[str], bool, str]:
    """Extracts final prediction while strictly ignoring template schema placeholders.
    
    Protects against Qwen3-8B echoing schema headers:
    'Verdict: <Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking>'
    which previously caused naive first-match regex to mistakenly classify 100% of few-shot
    outputs as 'Supported'.
    """
    if not raw_completion or not raw_completion.strip():
        return None, False, "EMPTY_COMPLETION"

    # Pattern finding all Verdict lines
    pattern = re.compile(r"Verdict:\s*([^\n\r]+)", re.IGNORECASE)
    matches = pattern.findall(raw_completion)

    if not matches:
        # Fallback to scanning lines from bottom
        lines = [line.strip() for line in raw_completion.splitlines() if line.strip()]
        for line in reversed(lines):
            # Exclude template schema line
            if "<" in line and "|" in line:
                continue
            for cls in AVERITEC_CLASSES:
                if re.search(rf"\b{re.escape(cls)}\b", line, re.IGNORECASE):
                    return cls, True, "FALLBACK_LINE_MATCH"
        return None, False, "NO_VERDICT_MATCH"

    # Iterate matches from bottom to top to get the FINAL verdict
    for match_text in reversed(matches):
        cleaned = match_text.strip()
        
        # Check if match is a template placeholder
        if "<" in cleaned and ("|" in cleaned or ">" in cleaned):
            continue

        # Match exact canonical classes
        for cls in AVERITEC_CLASSES:
            if re.search(rf"\b{re.escape(cls)}\b", cleaned, re.IGNORECASE):
                return cls, True, "CANONICAL_MATCH"

    return None, False, "ONLY_TEMPLATE_PLACEHOLDERS_FOUND"
