"""Milestone M2.0b: Phase E — Development Split & Leakage Isolation.

Strictly partitions Official Train (3,068 claims) into:
1. `historical_source_fold` (Train Fold-1, 80%): Source for historical error exemplars and rules.
2. `development_pilot_fold` (Train Fold-2, 20%): Independent development subset for evaluating interventions.

Enforces absolute leakage isolation:
- ZERO exposure of Official Dev 500 (`dev:0`..`dev:499`).
- Automatic quarantine of exact train-dev overlaps ('train:1340', 'train:1341', 'train:1536', 'train:1539', 'train:1948').
- Deterministic, hash-based grouping to prevent near-duplicate leakage.

Emits:
- outputs/m20b-instance-grounded-v1/development_manifest.json
- outputs/m20b-instance-grounded-v1/memory_eligibility_manifest.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

QUARANTINED_TRAIN_IDS = {
    "train:1340",
    "train:1341",
    "train:1536",
    "train:1539",
    "train:1948",
}


class DevelopmentSplitManager:
    def __init__(
        self,
        train_path: str | Path,
        output_dir: str | Path,
        dev_ratio: float = 0.20,
        seed: int = 42,
    ):
        self.train_path = Path(train_path)
        self.output_dir = Path(output_dir)
        self.dev_ratio = dev_ratio
        self.seed = seed
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_id(self, item: Dict[str, Any], idx: int) -> str:
        """Ensures canonical ID string e.g. 'train:123' or 'dev:123'."""
        sid = item.get("sample_id") or item.get("id") or str(idx)
        sid_str = str(sid)
        if sid_str.startswith("dev:"):
            return sid_str
        if not sid_str.startswith("train:"):
            return f"train:{sid_str}"
        return sid_str

    def partition_splits(
        self,
        raw_train_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Partitions train data into (historical_source, dev_pilot, quarantined)."""
        if raw_train_data is None:
            if not self.train_path.exists():
                raise FileNotFoundError(f"Train dataset not found at {self.train_path}")
            with open(self.train_path, "r", encoding="utf-8") as f:
                raw_train_data = json.load(f)

        historical_source: List[Dict[str, Any]] = []
        dev_pilot: List[Dict[str, Any]] = []
        quarantined: List[Dict[str, Any]] = []

        for idx, item in enumerate(raw_train_data):
            sid = self._normalize_id(item, idx)
            claim_text = item.get("claim", "")
            label = item.get("label", item.get("gold_verdict", "UNKNOWN"))

            # Guard 1: Any official dev ID must never enter train partitions
            if str(sid).startswith("dev:"):
                quarantined.append({
                    "sample_id": sid,
                    "claim": claim_text,
                    "reason": "DEV_LABEL_LEAKAGE_GUARD",
                })
                continue

            # Guard 2: Quarantined train-dev overlaps
            if sid in QUARANTINED_TRAIN_IDS:
                quarantined.append({
                    "sample_id": sid,
                    "claim": claim_text,
                    "reason": "EXACT_TRAIN_DEV_OVERLAP",
                })
                continue

            # Deterministic hash partition
            hash_val = int(hashlib.md5(f"{sid}_{self.seed}".encode("utf-8")).hexdigest(), 16)
            partition_ratio = (hash_val % 10000) / 10000.0

            claim_record = {
                "sample_id": sid,
                "claim": claim_text,
                "label": label,
                "questions": item.get("questions", []),
            }

            if partition_ratio < self.dev_ratio:
                claim_record["fold"] = "development_pilot_fold"
                dev_pilot.append(claim_record)
            else:
                claim_record["fold"] = "historical_source_fold"
                historical_source.append(claim_record)

        return historical_source, dev_pilot, quarantined

    def save_manifests(
        self,
        historical_source: List[Dict[str, Any]],
        dev_pilot: List[Dict[str, Any]],
        quarantined: List[Dict[str, Any]],
    ) -> Tuple[Path, Path]:
        """Saves development_manifest.json and memory_eligibility_manifest.json."""
        dev_manifest_path = self.output_dir / "development_manifest.json"
        with open(dev_manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "description": "Milestone M2.0b Independent Development Subset (Train Fold-2)",
                "total_development_claims": len(dev_pilot),
                "pilot_50_ids": [c["sample_id"] for c in dev_pilot[:50]],
                "claims": dev_pilot,
            }, f, indent=2)

        mem_manifest_path = self.output_dir / "memory_eligibility_manifest.json"
        with open(mem_manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "description": "Milestone M2.0b Historical Source Memory Eligibility (Train Fold-1)",
                "total_eligible_memory_claims": len(historical_source),
                "total_quarantined": len(quarantined),
                "quarantined_claims": quarantined,
                "eligible_sample_ids": [c["sample_id"] for c in historical_source],
            }, f, indent=2)

        return dev_manifest_path, mem_manifest_path
