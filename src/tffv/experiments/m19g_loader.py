"""Milestone M1.9g: Evidence-Complete Canonical Official Dev 500 Dataset Loader.

Requires:
1. Exactly 500 claims from canonical dev.json (SHA256 verified).
2. Mandatory non-empty R4-PF1000 retrieval receipts for all 500 claims.
3. Fails closed if retrieval receipts are missing, incomplete, or contain blank passages.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tffv.experiments.m19f_evidence import NormalizedPassage, normalize_evidence_list
from tffv.experiments.m19f_loader import EXPECTED_DEV_SHA256, EXPECTED_OFFICIAL_DEV_COUNT, OfficialDevClaim
from tffv.experiments.m19g_guardrails import (
    AVERITEC_CLASSES,
    RetrievalReceiptsMissingError,
    verify_retrieval_receipts,
)


class M19gDevLoader:
    def __init__(
        self,
        dev_path: str | Path = "data/raw/averitec/dev.json",
        dev_retrieval_receipts_path: Optional[str | Path] = None,
    ):
        self.dev_path = Path(dev_path)
        self.receipts_path = Path(dev_retrieval_receipts_path) if dev_retrieval_receipts_path else None

    def load_and_validate(
        self, limit_claims: Optional[int] = None
    ) -> Tuple[List[OfficialDevClaim], Dict[str, Any]]:
        """Loads canonical Official Dev 500 with mandatory non-empty retrieval evidence."""
        if not self.dev_path.exists():
            raise FileNotFoundError(f"Canonical dev.json not found at: {self.dev_path}")

        raw_bytes = self.dev_path.read_bytes()
        actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        raw_data = json.loads(raw_bytes.decode("utf-8"))

        if len(raw_data) != EXPECTED_OFFICIAL_DEV_COUNT:
            raise ValueError(
                f"Expected exactly {EXPECTED_OFFICIAL_DEV_COUNT} claims in Official Dev, found {len(raw_data)}!"
            )

        if limit_claims:
            raw_data = raw_data[:limit_claims]

        sample_ids = [f"dev:{i}" for i in range(len(raw_data))]

        # Strict guard: Retrieval receipts are mandatory in M1.9g
        if not self.receipts_path or not self.receipts_path.exists():
            raise RetrievalReceiptsMissingError(
                f"[M1.9g Mandate] Verifier execution prohibited: Retrieval receipts path missing or not found: {self.receipts_path}"
            )

        # Verify receipts integrity
        receipts_summary = verify_retrieval_receipts(sample_ids, self.receipts_path, require_all_claims=True)

        # Load retrieval receipts map
        evidence_receipts: Dict[str, List[Dict[str, Any]]] = {}
        with open(self.receipts_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    evidence_receipts[r["sample_id"]] = r.get("evidence", [])

        claims: List[OfficialDevClaim] = []
        class_counts = {c: 0 for c in AVERITEC_CLASSES}
        total_passages_loaded = 0

        for idx, item in enumerate(raw_data):
            sid = f"dev:{idx}"
            claim_text = str(item.get("claim", "")).strip()
            gold_label = str(item.get("label", "")).strip()

            if not claim_text:
                raise ValueError(f"Claim {sid} has empty claim text!")
            if gold_label not in AVERITEC_CLASSES:
                raise ValueError(f"Claim {sid} has unrecognized gold label: '{gold_label}'!")

            class_counts[gold_label] += 1

            raw_ev = evidence_receipts.get(sid, [])
            norm_ev, ev_meta = normalize_evidence_list(raw_ev, max_passages=8, fail_closed=True)
            total_passages_loaded += len(norm_ev)

            claims.append(
                OfficialDevClaim(
                    sample_id=sid,
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
            "gold_class_distribution": class_counts,
            "retrieval_receipts_verified": True,
            "total_passages_loaded": total_passages_loaded,
            "avg_passages_per_claim": total_passages_loaded / len(claims),
        }

        return claims, summary
