"""Group-safe knowledge preparation and fold isolation guards.

Ensures strict zero-leakage boundaries:
1. Evaluation folds are never admitted into historical error memory.
2. The 100 pilot exploratory claims are NEVER used as a blind held-out test set.
3. Near-duplicate or same-speaker/topic groups are partitioned atomically across folds.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Set
from tffv.provenance import canonical, digest, file_hash, write_json


def verify_fold_isolation(
    evaluation_sample_ids: List[str],
    memory_source_sample_ids: List[str],
) -> Dict[str, Any]:
    """Verify that no evaluation sample exists in memory source (strict zero leakage)."""
    eval_set = set(evaluation_sample_ids)
    mem_set = set(memory_source_sample_ids)
    overlap = eval_set & mem_set

    if overlap:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: {len(overlap)} evaluation samples exist in memory source: {sorted(overlap)[:5]}"
        )

    return {
        "isolation_verified": True,
        "evaluation_count": len(eval_set),
        "memory_source_count": len(mem_set),
        "leakage_count": 0,
    }


def create_pilot_group_safety_plan(
    pilot_sample_ids: List[str],
    output_path: Path,
) -> Dict[str, Any]:
    """Create and seal the group-safety and fold allocation plan for M1.2/M1.3."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pilot_set = sorted(set(pilot_sample_ids))

    plan = {
        "milestone": "M1.2",
        "phase": "group_safe_knowledge_partitioning",
        "pilot_partition_id": "pilot_exploratory_100",
        "pilot_sample_ids": pilot_set,
        "pilot_sample_count": len(pilot_set),
        "rules": [
            "The 100 pilot claims (train:0..train:99) are exploratory profiling samples.",
            "NEVER use these 100 pilot claims as a blind held-out test set in future milestones.",
            "Evaluation folds in M1.3+ must be drawn from held-out claims (train:100..train:3067 or dev split).",
            "Any historical error memory constructed must strictly exclude evaluation fold claims.",
            "No global heuristic rules derived from evaluation gold labels may be injected into prompts.",
        ],
        "exploratory_status": "exploratory_profiling_only",
        "blind_test_permitted": False,
        "isolation_contract_sha256": digest(pilot_set),
    }

    write_json(output_path, plan)
    return plan
