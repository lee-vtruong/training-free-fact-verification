"""Leakage detection and partition safety for M1.3 Error-Informed Verification."""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from tffv.provenance import canonical, digest, file_hash, write_json


def tokenize_words(text: str) -> List[str]:
    """Tokenize lowercased text into alphanumeric words."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in cleaned.split() if w]


def compute_jaccard_similarity(text1: str, text2: str) -> float:
    """Compute word-level Jaccard similarity between two texts."""
    set1 = set(tokenize_words(text1))
    set2 = set(tokenize_words(text2))
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def compute_ngram_containment(text1: str, text2: str, n: int = 3) -> float:
    """Compute n-gram containment of text1 in text2."""
    words1 = tokenize_words(text1)
    words2 = tokenize_words(text2)
    if len(words1) < n or len(words2) < n:
        return compute_jaccard_similarity(text1, text2)
    ngrams1 = set(zip(*[words1[i:] for i in range(n)]))
    ngrams2 = set(zip(*[words2[i:] for i in range(n)]))
    if not ngrams1:
        return 0.0
    return len(ngrams1 & ngrams2) / len(ngrams1)


def find_near_duplicates(
    source_claims: List[Dict[str, Any]],
    candidate_claims: List[Dict[str, Any]],
    jaccard_threshold: float = 0.60,
    ngram_threshold: float = 0.50,
) -> List[Dict[str, Any]]:
    """Scan candidate evaluation claims against source memory claims for near-duplicates.
    
    Flags pairs where word Jaccard similarity exceeds jaccard_threshold OR
    3-gram containment exceeds ngram_threshold.
    """
    flagged = []
    for cand in candidate_claims:
        cand_id = cand["sample_id"]
        cand_text = cand["claim"]
        for src in source_claims:
            src_id = src["sample_id"]
            src_text = src["claim"]
            jacc = compute_jaccard_similarity(cand_text, src_text)
            ngram = compute_ngram_containment(cand_text, src_text, n=3)
            if jacc >= jaccard_threshold or ngram >= ngram_threshold:
                flagged.append({
                    "eval_sample_id": cand_id,
                    "source_sample_id": src_id,
                    "jaccard_similarity": round(jacc, 4),
                    "ngram_containment": round(ngram, 4),
                    "eval_claim": cand_text,
                    "source_claim": src_text,
                })
                break  # Flag candidate once
    return flagged


def quarantine_near_duplicates(
    source_claims: List[Dict[str, Any]],
    candidate_claims: List[Dict[str, Any]],
    output_dir: Path,
    jaccard_threshold: float = 0.60,
    ngram_threshold: float = 0.50,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Path]:
    """Quarantine near-duplicates and persist partition manifest."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    flagged_pairs = find_near_duplicates(
        source_claims, candidate_claims, jaccard_threshold, ngram_threshold
    )
    quarantined_sids = {f["eval_sample_id"] for f in flagged_pairs}

    admitted = [c for c in candidate_claims if c["sample_id"] not in quarantined_sids]
    quarantined = [c for c in candidate_claims if c["sample_id"] in quarantined_sids]

    manifest = {
        "milestone": "M1.3",
        "phase": "near_duplicate_quarantine",
        "total_source_claims": len(source_claims),
        "total_candidate_claims": len(candidate_claims),
        "admitted_claims_count": len(admitted),
        "quarantined_claims_count": len(quarantined),
        "jaccard_threshold": jaccard_threshold,
        "ngram_threshold": ngram_threshold,
        "flagged_leakage_pairs": flagged_pairs,
        "admitted_sample_ids": [c["sample_id"] for c in admitted],
        "quarantined_sample_ids": [c["sample_id"] for c in quarantined],
    }

    manifest_path = output_dir / "quarantine_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return admitted, quarantined, manifest_path


def assert_fold_disjointness(source_sids: Set[str], eval_sids: Set[str]) -> None:
    """Enforce strict disjointness between source memory and evaluation claims."""
    overlap = set(source_sids) & set(eval_sids)
    if overlap:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: {len(overlap)} sample_ids exist in both source and evaluation sets: {sorted(overlap)}"
        )
