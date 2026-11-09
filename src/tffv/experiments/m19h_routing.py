"""Milestone M1.9h: Selective Intervention Feasibility & Post-hoc Replay."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from tffv.experiments.m19h_schemas import ClaimRecord


def simulate_selective_routing(
    claims_map: Dict[str, ClaimRecord],
    risk_features: List[Dict[str, Any]],
    output_dir: str | Path,
) -> List[Dict[str, Any]]:
    """Simulates selective intervention policies and post-hoc replays across budgets."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sorted_sids = sorted(claims_map.keys(), key=lambda s: int(s.split(":")[1]) if ":" in s else s)
    claims_list = [claims_map[sid] for sid in sorted_sids]
    n_claims = len(claims_list)

    # Base B0 statistics
    b0_correct = np.array([int(c.is_correct.get("B0", False)) for c in claims_list])
    b0_accuracy = float(np.mean(b0_correct))
    total_b0_errors = int(np.sum(1 - b0_correct))

    # Candidate arm correctness
    h2_correct = np.array([int(c.is_correct.get("H2", False)) for c in claims_list])
    h3_correct = np.array([int(c.is_correct.get("H3", False)) for c in claims_list])
    b2_correct = np.array([int(c.is_correct.get("B2", False)) for c in claims_list])
    h1_correct = np.array([int(c.is_correct.get("H1", False)) for c in claims_list])

    # Extract risk scores:
    # 1. Historical error similarity (max_sim_wrong)
    max_sim_w = np.array([r["max_sim_wrong"] for r in risk_features])
    # 2. Diff similarity (wrong - correct)
    diff_sim = np.array([r["diff_sim_wrong_correct"] for r in risk_features])

    policies = {
        "Historical Error Similarity (Max Sim Wrong)": max_sim_w,
        "Diff Similarity (Wrong - Correct)": diff_sim,
    }

    budgets = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    replay_rows: List[Dict[str, Any]] = []

    for pol_name, scores in policies.items():
        sorted_indices = np.argsort(-scores)  # highest risk first

        for b in budgets:
            k_claims = max(1, int(round(b * n_claims)))
            flagged_mask = np.zeros(n_claims, dtype=bool)
            flagged_mask[sorted_indices[:k_claims]] = True

            # Risk detection performance
            captured_errors = int(np.sum((1 - b0_correct) & flagged_mask))
            false_alarms = int(np.sum(b0_correct & flagged_mask))

            prec_error = captured_errors / k_claims if k_claims > 0 else 0.0
            rec_error = captured_errors / total_b0_errors if total_b0_errors > 0 else 0.0
            false_interv_rate = false_alarms / k_claims if k_claims > 0 else 0.0

            # Oracle Upper Bound: all captured errors converted to correct, 0 correct harmed
            oracle_correct = b0_correct.copy()
            oracle_correct[flagged_mask & (b0_correct == 0)] = 1
            oracle_acc = float(np.mean(oracle_correct))

            # Post-hoc replays with candidate arms (EXPLORATORY ONLY)
            # Replay with H2
            h2_routed_correct = b0_correct.copy()
            h2_routed_correct[flagged_mask] = h2_correct[flagged_mask]
            h2_routed_acc = float(np.mean(h2_routed_correct))
            h2_delta_acc = h2_routed_acc - b0_accuracy

            # Replay with H3
            h3_routed_correct = b0_correct.copy()
            h3_routed_correct[flagged_mask] = h3_correct[flagged_mask]
            h3_routed_acc = float(np.mean(h3_routed_correct))
            h3_delta_acc = h3_routed_acc - b0_accuracy

            # Replay with B2
            b2_routed_correct = b0_correct.copy()
            b2_routed_correct[flagged_mask] = b2_correct[flagged_mask]
            b2_routed_acc = float(np.mean(b2_routed_correct))
            b2_delta_acc = b2_routed_acc - b0_accuracy

            # Replay with H1
            h1_routed_correct = b0_correct.copy()
            h1_routed_correct[flagged_mask] = h1_correct[flagged_mask]
            h1_routed_acc = float(np.mean(h1_routed_correct))
            h1_delta_acc = h1_routed_acc - b0_accuracy

            replay_rows.append({
                "routing_policy": pol_name,
                "budget_fraction": b,
                "claims_flagged": k_claims,
                "captured_b0_errors": captured_errors,
                "false_alarms_correct_harmed_risk": false_alarms,
                "error_detection_precision": round(prec_error, 4),
                "error_detection_recall": round(rec_error, 4),
                "false_intervention_rate": round(false_interv_rate, 4),
                "b0_base_accuracy": round(b0_accuracy, 4),
                "oracle_upper_bound_accuracy": round(oracle_acc, 4),
                "oracle_max_possible_gain": round(oracle_acc - b0_accuracy, 4),
                "h2_posthoc_replay_accuracy": round(h2_routed_acc, 4),
                "h2_delta_accuracy": round(h2_delta_acc, 4),
                "h3_posthoc_replay_accuracy": round(h3_routed_acc, 4),
                "h3_delta_accuracy": round(h3_delta_acc, 4),
                "b2_posthoc_replay_accuracy": round(b2_routed_acc, 4),
                "b2_delta_accuracy": round(b2_delta_acc, 4),
                "h1_posthoc_replay_accuracy": round(h1_routed_acc, 4),
                "h1_delta_accuracy": round(h1_delta_acc, 4),
                "analysis_note": "POST-HOC REPLAY — EXPLORATORY ONLY",
            })

    # Save routing_replay.csv
    with open(out_path / "routing_replay.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(replay_rows[0].keys()))
        writer.writeheader()
        writer.writerows(replay_rows)

    return replay_rows
