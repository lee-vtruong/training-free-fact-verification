"""M1.8d Stratified Manifest Builder for Pilot (50) and Confirmatory Holdout (200).

Extracts eligible AVeriTeC train claims, strictly excludes train:1948 (invalid)
and all 4 exact text train-dev overlaps + 3 URL overlap claims.
Stratifies deterministically by gold label and passage count quartiles.
"""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(ROOT / "src"))

from tffv.provenance import canonical, digest, file_hash, write_json
from tffv.verification.ontology import VERDICTS

EXCLUDED_SAMPLE_IDS = {
    "train:1948",  # Empty claim text
    # Exact normalized text overlaps with Dev
    "train:1340",
    "train:1341",
    "train:1536",
    "train:1539",
    # Fact-checking article URL overlaps with Dev
    "train:1568",
    "train:2790",
    "train:2990",
}


def load_eligible_claims(
    raw_train_path: Path,
    knowledge_store_uncompressed: Path,
) -> List[Dict[str, Any]]:
    """Load valid train claims and compute passage/line statistics from local stores."""
    raw_train = json.loads(raw_train_path.read_text(encoding="utf-8"))
    eligible = []

    for idx, r in enumerate(raw_train):
        sid = f"train:{idx}"
        if sid in EXCLUDED_SAMPLE_IDS:
            continue
        claim_text = (r.get("claim") or "").strip()
        if not claim_text:
            continue
        label = r.get("label")
        if label not in VERDICTS:
            continue

        store_file = knowledge_store_uncompressed / f"{idx}.json"
        store_bytes = store_file.stat().st_size if store_file.is_file() else 0

        eligible.append({
            "sample_id": sid,
            "ordinal": idx,
            "claim": claim_text,
            "gold_label": label,
            "store_bytes": store_bytes,
        })

    return eligible


def assign_size_stratum(store_bytes: int) -> str:
    """Bin store sizes into 4 strata: small, medium, large, huge."""
    if store_bytes < 5 * 1024 * 1024:
        return "small"
    elif store_bytes < 20 * 1024 * 1024:
        return "medium"
    elif store_bytes < 60 * 1024 * 1024:
        return "large"
    else:
        return "huge"


def build_stratified_samples(
    eligible_claims: List[Dict[str, Any]],
    pilot_size: int = 50,
    holdout_size: int = 200,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministically sample non-overlapping Pilot (50) and Holdout (200) claims."""
    rng = random.Random(seed)

    # Annotate size stratum
    for c in eligible_claims:
        c["size_stratum"] = assign_size_stratum(c["store_bytes"])

    # Group by (gold_label, size_stratum)
    bins = defaultdict(list)
    for c in eligible_claims:
        key = (c["gold_label"], c["size_stratum"])
        bins[key].append(c)

    # Sort each bin deterministically by ordinal
    for key in bins:
        bins[key].sort(key=lambda x: x["ordinal"])
        rng.shuffle(bins[key])

    pilot = []
    holdout = []

    # Proportional allocation
    total_eligible = len(eligible_claims)
    for key, items in sorted(bins.items()):
        weight = len(items) / total_eligible
        n_pilot = max(1, round(weight * pilot_size))
        n_holdout = max(2, round(weight * holdout_size))

        pilot_sample = items[:n_pilot]
        holdout_sample = items[n_pilot:n_pilot + n_holdout]

        pilot.extend(pilot_sample)
        holdout.extend(holdout_sample)

    # Trim or fill to exact targets if rounding differed slightly
    pilot = sorted(pilot, key=lambda x: x["ordinal"])[:pilot_size]
    pilot_sids = {c["sample_id"] for c in pilot}

    holdout_filtered = [c for c in holdout if c["sample_id"] not in pilot_sids]
    if len(holdout_filtered) > holdout_size:
        holdout_filtered = holdout_filtered[:holdout_size]
    elif len(holdout_filtered) < holdout_size:
        remaining = [c for c in eligible_claims if c["sample_id"] not in pilot_sids and c["sample_id"] not in {h["sample_id"] for h in holdout_filtered}]
        remaining.sort(key=lambda x: x["ordinal"])
        rng.shuffle(remaining)
        holdout_filtered.extend(remaining[:holdout_size - len(holdout_filtered)])

    holdout = sorted(holdout_filtered, key=lambda x: x["ordinal"])

    return pilot, holdout


def generate_m18d_manifests(
    raw_train_path: Path,
    knowledge_store_uncompressed: Path,
    output_dir: Path,
    seed: int = 42,
) -> Tuple[Path, Path]:
    """Generate and write frozen pilot and holdout manifests."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    eligible = load_eligible_claims(raw_train_path, knowledge_store_uncompressed)
    pilot, holdout = build_stratified_samples(eligible, pilot_size=50, holdout_size=200, seed=seed)

    pilot_sids = {c["sample_id"] for c in pilot}
    holdout_sids = {c["sample_id"] for c in holdout}

    # Strict disjointness invariant
    assert len(pilot) == 50, f"Expected 50 pilot claims, got {len(pilot)}"
    assert len(holdout) == 200, f"Expected 200 holdout claims, got {len(holdout)}"
    assert pilot_sids.isdisjoint(holdout_sids), "Pilot and Holdout must be strictly disjoint!"
    assert pilot_sids.isdisjoint(EXCLUDED_SAMPLE_IDS), "Pilot must not contain excluded claims!"
    assert holdout_sids.isdisjoint(EXCLUDED_SAMPLE_IDS), "Holdout must not contain excluded claims!"

    pilot_manifest = {
        "schema_version": "m18d-manifest-v1",
        "split_name": "pilot_50",
        "seed": seed,
        "count": len(pilot),
        "excluded_overlaps_count": len(EXCLUDED_SAMPLE_IDS),
        "label_distribution": dict(Counter(c["gold_label"] for c in pilot)),
        "size_strata_distribution": dict(Counter(c["size_stratum"] for c in pilot)),
        "claims": pilot,
    }

    holdout_manifest = {
        "schema_version": "m18d-manifest-v1",
        "split_name": "holdout_200",
        "seed": seed,
        "count": len(holdout),
        "excluded_overlaps_count": len(EXCLUDED_SAMPLE_IDS),
        "label_distribution": dict(Counter(c["gold_label"] for c in holdout)),
        "size_strata_distribution": dict(Counter(c["size_stratum"] for c in holdout)),
        "claims": holdout,
    }

    p_file = output_dir / "pilot_manifest.json"
    h_file = output_dir / "holdout_manifest.json"

    write_json(p_file, pilot_manifest)
    write_json(h_file, holdout_manifest)

    print(f"Generated frozen pilot manifest (50 claims): {p_file}")
    print(f"Generated frozen holdout manifest (200 claims): {h_file}")
    return p_file, h_file


if __name__ == "__main__":
    generate_m18d_manifests(
        raw_train_path=ROOT / "data/raw/averitec/train.json",
        knowledge_store_uncompressed=ROOT / "data_store/knowledge_store/train/uncompressed",
        output_dir=ROOT / "configs/m18d_manifests",
    )
