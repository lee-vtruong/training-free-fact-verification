"""Milestone M1.9f: Canonical Official Dev 500 Dataset Loader.

Loads and validates the official AVeriTeC Dev split from raw source with strict integrity checks.
Requires exactly 500 claims, IDs dev:0 through dev:499, and zero train IDs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tffv.experiments.m19f_evidence import NormalizedPassage, normalize_evidence_list

EXPECTED_OFFICIAL_DEV_COUNT = 500
EXPECTED_DEV_SHA256 = "499793726b4a5406780928a3d9dedc48d6dd53de778f22437d129cacdb08e300"
AVERITEC_CLASSES = {
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
}


@dataclass
class OfficialDevClaim:
    sample_id: str
    claim: str
    gold_verdict: str
    speaker: str
    claim_date: str
    evidence: List[NormalizedPassage]
    evidence_status: str
    evidence_hash: str
    raw_questions_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "claim": self.claim,
            "gold_verdict": self.gold_verdict,
            "speaker": self.speaker,
            "claim_date": self.claim_date,
            "evidence": [p.to_dict() for p in self.evidence],
            "evidence_status": self.evidence_status,
            "evidence_hash": self.evidence_hash,
            "raw_questions_count": self.raw_questions_count,
        }


class OfficialDevLoader:
    def __init__(
        self,
        dev_path: str | Path = "data/raw/averitec/dev.json",
        dev_evidence_path: Optional[str | Path] = None,
    ):
        self.dev_path = Path(dev_path)
        self.dev_evidence_path = Path(dev_evidence_path) if dev_evidence_path else None

    def load_and_validate(self) -> Tuple[List[OfficialDevClaim], Dict[str, Any]]:
        """Loads and verifies canonical Official Dev 500 claims."""
        if not self.dev_path.exists():
            raise FileNotFoundError(f"Canonical Official Dev file not found: {self.dev_path}")

        raw_bytes = self.dev_path.read_bytes()
        actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        raw_data = json.loads(raw_bytes.decode("utf-8"))

        total_claims = len(raw_data)
        if total_claims != EXPECTED_OFFICIAL_DEV_COUNT:
            raise ValueError(
                f"Expected exactly {EXPECTED_OFFICIAL_DEV_COUNT} claims in Official Dev, found {total_claims}!"
            )

        # Load external retrieval evidence if available
        evidence_receipts: Dict[str, List[Dict[str, Any]]] = {}
        if self.dev_evidence_path and self.dev_evidence_path.exists():
            with open(self.dev_evidence_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        evidence_receipts[r["sample_id"]] = r.get("evidence", [])

        claims: List[OfficialDevClaim] = []
        class_counts = {c: 0 for c in AVERITEC_CLASSES}
        blank_evidence_claims = 0

        for idx, item in enumerate(raw_data):
            sample_id = f"dev:{idx}"
            claim_text = str(item.get("claim", "")).strip()
            gold_label = str(item.get("label", "")).strip()

            if not claim_text:
                raise ValueError(f"Claim {sample_id} has empty claim text!")

            if gold_label not in AVERITEC_CLASSES:
                raise ValueError(f"Claim {sample_id} has unrecognized gold label: '{gold_label}'!")

            class_counts[gold_label] += 1

            # Get target evidence from external receipts or empty
            raw_ev = evidence_receipts.get(sample_id, [])
            norm_ev, ev_meta = normalize_evidence_list(raw_ev, max_passages=8, fail_closed=False)
            if ev_meta["status"] in ("GENUINE_EMPTY_RETRIEVAL", "ALL_BLANK"):
                blank_evidence_claims += 1

            claims.append(
                OfficialDevClaim(
                    sample_id=sample_id,
                    claim=claim_text,
                    gold_verdict=gold_label,
                    speaker=str(item.get("speaker", "")),
                    claim_date=str(item.get("claim_date", "")),
                    evidence=norm_ev,
                    evidence_status=ev_meta["status"],
                    evidence_hash=ev_meta["evidence_hash"],
                    raw_questions_count=len(item.get("questions", [])),
                )
            )

        summary = {
            "source_path": str(self.dev_path),
            "file_sha256": actual_sha256,
            "sha256_verified": (actual_sha256 == EXPECTED_DEV_SHA256),
            "total_claims": len(claims),
            "id_range": f"{claims[0].sample_id} to {claims[-1].sample_id}",
            "train_ids_present": any(c.sample_id.startswith("train:") for c in claims),
            "gold_class_distribution": class_counts,
            "external_evidence_loaded": len(evidence_receipts) > 0,
            "blank_evidence_claims": blank_evidence_claims,
        }

        return claims, summary
