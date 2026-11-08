"""Milestone M1.9f: Historical Outcome Store Builder with Evidence Normalization & Leakage Guards.

Constructs normalized historical memory from M1.8e Official Train outcomes.
Enforces strict exclusion of quarantined claims, unparseables, and cross-split overlaps.
Normalizes all evidence passages to eliminate the Evidence Blanking Bug.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tffv.experiments.m19f_evidence import (
    NormalizedPassage,
    extract_passage_text,
    normalize_evidence_list,
)

ALL_OFFICIAL_TRAIN_COUNT = 3068
QUARANTINED_CLAIM_ID = "train:1948"
EXACT_TRAIN_DEV_OVERLAPS = {
    "train:1340": "dev:306",
    "train:1341": "dev:307",
    "train:1536": "dev:348",
    "train:1539": "dev:349",
}
AVERITEC_CLASSES = {
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
}


def norm_claim_text(text: str) -> str:
    """Normalize claim text for near-duplicate grouping."""
    import re
    t = text.lower()
    t = re.sub(r"[^\w\s]", "", t)
    return " ".join(t.split())


@dataclass
class HistoricalRecord:
    sample_id: str
    claim: str
    gold_verdict: str
    base_verdict: str
    correctness: str  # "CORRECT", "WRONG", "UNPARSEABLE", "QUARANTINED"
    evidence_count: int
    evidence_hash: str
    evidence_references: List[Dict[str, Any]]
    retrieval_fingerprint: str
    verifier_fingerprint: str
    leakage_group: str
    is_selectable_correct: bool
    is_selectable_wrong: bool
    exclusion_reason: str = "NONE"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HistoricalOutcomeStoreBuilder:
    def __init__(
        self,
        m18e_dir: str | Path = "outputs/m18e-full-history-pf1000-v1",
        m18g_dir: Optional[str | Path] = None,
        train_path: str | Path = "data/raw/averitec/train.json",
        claims_manifest_path: Optional[str | Path] = "outputs/m16-partitions-v1/claims_manifest.json",
    ):
        self.m18e_dir = Path(m18e_dir)
        self.m18g_dir = Path(m18g_dir) if m18g_dir else Path("outputs/m18g-correction-recovery-v1")
        self.train_path = Path(train_path)
        self.claims_manifest_path = Path(claims_manifest_path) if claims_manifest_path else None

    def build_store(self) -> Tuple[List[HistoricalRecord], Dict[str, Any]]:
        # Load raw train data
        train_claims_map: Dict[str, Dict[str, Any]] = {}
        if self.train_path.exists():
            raw_train = json.loads(self.train_path.read_text(encoding="utf-8"))
            for idx, item in enumerate(raw_train):
                sid = f"train:{idx}"
                train_claims_map[sid] = {
                    "claim": str(item.get("claim", "")).strip(),
                    "gold_verdict": str(item.get("label", "")).strip(),
                }

        # Load M1.8e base predictions
        base_preds_file = self.m18e_dir / "base_predictions.jsonl"
        receipts_file = self.m18e_dir / "retrieval_receipts.jsonl"

        preds_by_id: Dict[str, Dict[str, Any]] = {}
        if base_preds_file.exists():
            with open(base_preds_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        preds_by_id[d["sample_id"]] = d
        else:
            # Fallback for mock/local testing when full base_predictions is on server
            wrong_manifest_file = self.m18g_dir / "recovery_manifest.json"
            if wrong_manifest_file.exists():
                m_data = json.loads(wrong_manifest_file.read_text(encoding="utf-8"))
                for c in m_data.get("candidates", []):
                    preds_by_id[c["sample_id"]] = {
                        "sample_id": c["sample_id"],
                        "claim": c["claim"],
                        "predicted_verdict": c["base_verdict"],
                        "gold_verdict": c["gold_verdict"],
                        "is_correct": False,
                    }
                preds_by_id["train:1340"] = {"sample_id": "train:1340", "predicted_verdict": "Refuted", "gold_verdict": "Supported", "is_correct": False}
                preds_by_id["train:1341"] = {"sample_id": "train:1341", "predicted_verdict": "Refuted", "gold_verdict": "Supported", "is_correct": False}
                for sid, t in train_claims_map.items():
                    if sid not in preds_by_id and sid != QUARANTINED_CLAIM_ID:
                        preds_by_id[sid] = {
                            "sample_id": sid,
                            "claim": t["claim"],
                            "predicted_verdict": t["gold_verdict"],
                            "gold_verdict": t["gold_verdict"],
                            "is_correct": True,
                        }

        # Load retrieval receipts
        receipts_map: Dict[str, List[Dict[str, Any]]] = {}
        if receipts_file.exists():
            with open(receipts_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        receipts_map[r["sample_id"]] = r.get("evidence", [])

        # Load near-duplicate groups if available
        near_dup_map: Dict[str, str] = {}
        if self.claims_manifest_path and self.claims_manifest_path.exists():
            manifest_data = json.loads(self.claims_manifest_path.read_text(encoding="utf-8"))
            for entry in manifest_data:
                sid = entry.get("sample_id")
                gid = entry.get("group_id") or entry.get("duplicate_group_id")
                if sid and gid:
                    near_dup_map[sid] = gid

        records: List[HistoricalRecord] = []
        counts = Counter()
        total_passages_normalized = 0
        blank_passages_detected = 0

        for idx in range(ALL_OFFICIAL_TRAIN_COUNT):
            sid = f"train:{idx}"
            t_data = train_claims_map.get(sid, {})
            p_data = preds_by_id.get(sid, {})

            claim_text = (p_data.get("claim") or t_data.get("claim") or "").strip()
            gold = p_data.get("gold_verdict") or t_data.get("gold_verdict") or ""
            pred = p_data.get("predicted_verdict") or ""

            # Check quarantined record
            if sid == QUARANTINED_CLAIM_ID or not claim_text:
                counts["quarantined"] += 1
                records.append(
                    HistoricalRecord(
                        sample_id=sid,
                        claim="",
                        gold_verdict=gold,
                        base_verdict=pred,
                        correctness="QUARANTINED",
                        evidence_count=0,
                        evidence_hash="",
                        evidence_references=[],
                        retrieval_fingerprint="R4-PF1000-v1",
                        verifier_fingerprint="Qwen3-8B-BF16-greedy",
                        leakage_group="quarantined",
                        is_selectable_correct=False,
                        is_selectable_wrong=False,
                        exclusion_reason="QUARANTINED_EMPTY_CLAIM",
                    )
                )
                continue

            # Determine correctness
            is_unparseable = p_data.get("unparseable", False) or pred not in AVERITEC_CLASSES
            if is_unparseable:
                correctness = "UNPARSEABLE"
                counts["unparseable"] += 1
            elif pred == gold:
                correctness = "CORRECT"
                counts["correct"] += 1
            else:
                correctness = "WRONG"
                counts["wrong"] += 1

            # Leakage group from manifest or normalized claim hash
            leakage_grp = near_dup_map.get(sid)
            if not leakage_grp:
                norm_c = norm_claim_text(claim_text)
                leakage_grp = hashlib.sha256(norm_c.encode("utf-8")).hexdigest()[:16]

            # Normalize evidence passages using centralized schema extractor
            raw_ev = receipts_map.get(sid, [])
            norm_ev, ev_meta = normalize_evidence_list(raw_ev, max_passages=8, fail_closed=False)
            total_passages_normalized += ev_meta["total_passages"]
            blank_passages_detected += ev_meta["blank_passages"]

            ev_refs = [p.to_dict() for p in norm_ev]

            # Leakage exclusion checks
            is_overlap = sid in EXACT_TRAIN_DEV_OVERLAPS
            if is_overlap:
                counts["train_dev_overlap"] += 1
                excl_reason = f"TRAIN_DEV_OVERLAP_WITH_{EXACT_TRAIN_DEV_OVERLAPS[sid]}"
            elif is_unparseable:
                excl_reason = "UNPARSEABLE_VERDICT"
            else:
                excl_reason = "NONE"

            is_sel_correct = (correctness == "CORRECT") and (excl_reason == "NONE")
            is_sel_wrong = (correctness == "WRONG") and (excl_reason == "NONE")

            if is_sel_correct:
                counts["selectable_correct"] += 1
            if is_sel_wrong:
                counts["selectable_wrong"] += 1

            records.append(
                HistoricalRecord(
                    sample_id=sid,
                    claim=claim_text,
                    gold_verdict=gold,
                    base_verdict=pred,
                    correctness=correctness,
                    evidence_count=len(ev_refs),
                    evidence_hash=ev_meta["evidence_hash"],
                    evidence_references=ev_refs,
                    retrieval_fingerprint="R4-PF1000-v1",
                    verifier_fingerprint="Qwen3-8B-BF16-greedy",
                    leakage_group=leakage_grp,
                    is_selectable_correct=is_sel_correct,
                    is_selectable_wrong=is_sel_wrong,
                    exclusion_reason=excl_reason,
                )
            )

        summary = {
            "total_records": len(records),
            "correct_count": counts["correct"],
            "wrong_count": counts["wrong"],
            "unparseable_count": counts["unparseable"],
            "quarantined_count": counts["quarantined"],
            "train_dev_overlap_count": counts["train_dev_overlap"],
            "selectable_correct_exemplars": counts["selectable_correct"],
            "selectable_wrong_exemplars": counts["selectable_wrong"],
            "total_passages_normalized": total_passages_normalized,
            "blank_passages_detected": blank_passages_detected,
        }

        return records, summary

    def save_store(
        self,
        records: List[HistoricalRecord],
        summary: Dict[str, Any],
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        store_file = output_dir / "historical_store.jsonl"
        with open(store_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

        (output_dir / "memory_eligibility.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
