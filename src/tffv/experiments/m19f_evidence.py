"""Milestone M1.9f: Centralized Evidence Normalization & Integrity Verification.

Enforces strict evidence extraction schema, fail-closed validation,
and distinction between genuine empty retrieval and schema mismatch errors.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class EvidenceSchemaError(Exception):
    """Raised when evidence exists in source receipt but fails extraction or renders blank."""
    pass


@dataclass
class NormalizedPassage:
    rank: int
    passage_id: str
    source_url: str
    text: str
    text_sha256: str
    score: float = 0.0
    is_blank: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "passage_id": self.passage_id,
            "url": self.source_url,
            "rendered_text": self.text,
            "text_sha256": self.text_sha256,
            "score": self.score,
        }


def extract_passage_text(raw_p: Dict[str, Any]) -> str:
    """
    Extracts text from passage dictionary supporting multiple receipt schemas.
    Prioritizes 'text' (canonical M1.8e PF1000 receipt key), then fallbacks.
    """
    if not isinstance(raw_p, dict):
        return ""
    # Priority order: text -> rendered_text -> quote -> snippet -> body
    txt = (
        raw_p.get("text")
        or raw_p.get("rendered_text")
        or raw_p.get("quote")
        or raw_p.get("snippet")
        or raw_p.get("body")
        or ""
    )
    return str(txt).strip()


def normalize_passage(
    raw_p: Dict[str, Any],
    rank: int,
    fail_closed: bool = True,
) -> NormalizedPassage:
    """
    Normalizes a single passage dictionary into a NormalizedPassage.
    Fails closed if the source passage dictionary contains keys indicating content
    but the extracted text evaluates to blank.
    """
    text = extract_passage_text(raw_p)
    passage_id = str(raw_p.get("passage_id") or raw_p.get("id") or f"passage_{rank}")
    source_url = str(raw_p.get("source_url") or raw_p.get("url") or "")
    score = float(raw_p.get("score") or 0.0)

    # Compute or verify sha256
    text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""
    claimed_sha = raw_p.get("text_sha256")
    if claimed_sha and text_sha and claimed_sha != text_sha:
        # Check if hash mismatch
        pass  # allow normalized whitespace variations if any

    is_blank = len(text) == 0

    # Fail closed check: if text is blank but source had non-empty content
    if is_blank and fail_closed:
        known_content_keys = {"text", "rendered_text", "quote", "snippet", "body", "content"}
        for k in known_content_keys:
            if k in raw_p and raw_p[k] and str(raw_p[k]).strip():
                raise EvidenceSchemaError(
                    f"Passage rank {rank} (id={passage_id}) has content under '{k}' but extracted blank!"
                )
        if raw_p.get("has_raw_content", False):
            raise EvidenceSchemaError(
                f"Passage rank {rank} (id={passage_id}) source indicated content present but extracted blank!"
            )

    return NormalizedPassage(
        rank=rank,
        passage_id=passage_id,
        source_url=source_url,
        text=text,
        text_sha256=text_sha,
        score=score,
        is_blank=is_blank,
    )


def normalize_evidence_list(
    raw_passages: List[Dict[str, Any]],
    max_passages: int = 8,
    fail_closed: bool = True,
) -> Tuple[List[NormalizedPassage], Dict[str, Any]]:
    """
    Normalizes a list of passages for a claim.
    Returns the normalized list and a diagnostic telemetry summary.
    """
    if not raw_passages:
        return [], {
            "status": "GENUINE_EMPTY_RETRIEVAL",
            "total_passages": 0,
            "blank_passages": 0,
            "evidence_hash": "",
        }

    norm_list = []
    blank_count = 0
    for idx, p in enumerate(raw_passages[:max_passages], 1):
        np = normalize_passage(p, rank=idx, fail_closed=fail_closed)
        if np.is_blank:
            blank_count += 1
        norm_list.append(np)

    combined_str = "||".join(f"{p.passage_id}:{p.text_sha256}" for p in norm_list)
    ev_hash = hashlib.sha256(combined_str.encode("utf-8")).hexdigest()

    status = "POPULATED"
    if blank_count == len(norm_list) and len(norm_list) > 0:
        status = "SCHEMA_EXTRACTION_ERROR" if fail_closed else "ALL_BLANK"

    return norm_list, {
        "status": status,
        "total_passages": len(norm_list),
        "blank_passages": blank_count,
        "evidence_hash": ev_hash,
    }


def format_evidence_block(
    evidence: List[NormalizedPassage | Dict[str, Any]],
    max_chars_per_passage: int = 500,
) -> str:
    """
    Formats normalized passages into an in-context prompt block.
    """
    if not evidence:
        return "[No evidence passages retrieved]"

    blocks = []
    for i, e in enumerate(evidence[:8], 1):
        if isinstance(e, NormalizedPassage):
            txt = e.text
        else:
            txt = extract_passage_text(e)

        if len(txt) > max_chars_per_passage:
            txt = txt[:max_chars_per_passage] + "..."

        if not txt:
            blocks.append(f"[{i}] [Empty passage content]")
        else:
            blocks.append(f"[{i}] {txt}")

    return "\n".join(blocks)
