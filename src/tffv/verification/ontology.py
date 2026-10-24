"""Official AVeriTeC four-class verification ontology and parsing utilities."""
import re
from typing import Optional, Tuple

VERDICTS: Tuple[str, ...] = (
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
)

CANONICAL_MAP = {
    "supported": "Supported",
    "refuted": "Refuted",
    "not enough evidence": "Not Enough Evidence",
    "not enough info": "Not Enough Evidence",
    "not enough information": "Not Enough Evidence",
    "insufficient evidence": "Not Enough Evidence",
    "conflicting evidence/cherrypicking": "Conflicting Evidence/Cherrypicking",
    "conflicting evidence": "Conflicting Evidence/Cherrypicking",
    "cherrypicking": "Conflicting Evidence/Cherrypicking",
    "cherry-picking": "Conflicting Evidence/Cherrypicking",
    "conflicting": "Conflicting Evidence/Cherrypicking",
}


def _parse_verdict_from_segment(segment: str) -> Optional[str]:
    """Parse verdict from a single text segment using high-to-low confidence heuristics."""
    if not segment or not segment.strip():
        return None

    cleaned = segment.strip()

    # 1. Look for explicit 'Verdict:' or 'Final Verdict:' pattern
    verdict_patterns = [
        r"(?:final\s+)?verdict\s*:\s*[\*\_]*([A-Za-z\s\/\-]+?)[\*\_]*(?:\.|\n|$)",
        r"(?:the\s+claim\s+is\s+|claim\s+is\s+|conclusion\s*:\s*)[\*\_]*([A-Za-z\s\/\-]+?)[\*\_]*(?:\.|\n|$)",
    ]

    for pat in verdict_patterns:
        match = re.search(pat, cleaned, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip().lower().rstrip(".").strip()
            if candidate in CANONICAL_MAP:
                return CANONICAL_MAP[candidate]

    # 2. Check line by line from bottom to top
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    for line in reversed(lines):
        l_clean = re.sub(r"[\*\_\#\:\-\.]", " ", line).strip().lower()
        words = " ".join(l_clean.split())
        for key, canonical in CANONICAL_MAP.items():
            if words == key or words.endswith(" " + key) or words.startswith(key + " "):
                return canonical

    # 3. Whole-string scan for unambiguous presence of exactly one category
    found = set()
    lowered = cleaned.lower()
    for key, canonical in CANONICAL_MAP.items():
        if re.search(rf"\b{re.escape(key)}\b", lowered):
            found.add(canonical)

    if len(found) == 1:
        return found.pop()

    return None


def normalize_verdict(raw_text: str) -> Optional[str]:
    """Extract and normalize canonical verdict from model generation text.
    
    Checks post-think text first (after </think>) to prioritize the final verdict.
    Falls back to whole-generation parsing if unparsed or no </think> tag exists.
    Returns one of the four canonical VERDICTS, or None if unparseable.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None

    # Step 1: If thinking trace is closed, parse post-think section first
    if "</think>" in raw_text:
        post_think = raw_text.split("</think>")[-1].strip()
        if post_think:
            parsed = _parse_verdict_from_segment(post_think)
            if parsed is not None:
                return parsed

    # Step 2: Parse entire raw generation (for non-thinking or unclosed think traces)
    return _parse_verdict_from_segment(raw_text)
