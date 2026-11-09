"""Milestone M1.9h: Per-Claim Error Transition Analysis & Class Diagnostics."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import binomtest

from tffv.experiments.m19h_schemas import (
    OFFICIAL_ARMS,
    OFFICIAL_LABELS,
    ClaimRecord,
    TransitionCategory,
    TransitionStats,
)


COMPARISON_PAIRS = [
    ("H2", "B1", "Primary: Error-History ICL vs Matched Gold-Only ICL"),
    ("H2", "B0", "Secondary: Error-History ICL vs Zero-Shot Base"),
    ("H3", "B2", "Secondary: Mixed-History ICL vs Matched Mixed Gold-Only ICL"),
    ("H3", "B0", "Secondary: Mixed-History ICL vs Zero-Shot Base"),
    ("H1", "B0", "Secondary: Correct-History ICL vs Zero-Shot Base"),
]


def exact_mcnemar(b: int, c: int) -> float:
    """Computes two-sided exact McNemar p-value using binomial test."""
    total = b + c
    if total == 0:
        return 1.0
    res = binomtest(min(b, c), n=total, p=0.5, alternative="two-sided")
    return float(res.pvalue)


def analyze_transitions(
    claims_map: Dict[str, ClaimRecord],
    output_dir: str | Path,
) -> Tuple[Dict[str, TransitionStats], Dict[str, Any], List[Dict[str, Any]]]:
    """Computes paired error transition tables, matrices, and per-class diagnostics."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sorted_sids = sorted(claims_map.keys(), key=lambda s: int(s.split(":")[1]) if ":" in s else s)
    claims_list = [claims_map[sid] for sid in sorted_sids]

    # 1. Build and export error_transitions.csv
    csv_rows = []
    for c in claims_list:
        h2_sims = c.similarity_scores.get("H2", [])
        h1_sims = c.similarity_scores.get("H1", [])
        h2_max = max(h2_sims) if h2_sims else 0.0
        h2_mean = float(np.mean(h2_sims)) if h2_sims else 0.0
        h1_max = max(h1_sims) if h1_sims else 0.0
        h1_mean = float(np.mean(h1_sims)) if h1_sims else 0.0

        row = {
            "claim_id": c.claim_id,
            "gold_label": c.gold_label,
            "evidence_count": c.evidence_count,
            "evidence_char_len": c.evidence_char_len,
            "B0_pred": c.predictions.get("B0", ""),
            "B1_pred": c.predictions.get("B1", ""),
            "H1_pred": c.predictions.get("H1", ""),
            "H2_pred": c.predictions.get("H2", ""),
            "H3_pred": c.predictions.get("H3", ""),
            "B2_pred": c.predictions.get("B2", ""),
            "B0_correct": int(c.is_correct.get("B0", False)),
            "B1_correct": int(c.is_correct.get("B1", False)),
            "H1_correct": int(c.is_correct.get("H1", False)),
            "H2_correct": int(c.is_correct.get("H2", False)),
            "H3_correct": int(c.is_correct.get("H3", False)),
            "B2_correct": int(c.is_correct.get("B2", False)),
            "H2_max_sim": round(h2_max, 4),
            "H2_mean_sim": round(h2_mean, 4),
            "H1_max_sim": round(h1_max, 4),
            "H1_mean_sim": round(h1_mean, 4),
            "H2_chosen_exemplars": ";".join(c.chosen_exemplars.get("H2", [])),
            "H1_chosen_exemplars": ";".join(c.chosen_exemplars.get("H1", [])),
            "H2_vs_B1_transition": (
                "both_correct" if c.is_correct.get("H2") and c.is_correct.get("B1")
                else "both_wrong" if (not c.is_correct.get("H2")) and (not c.is_correct.get("B1"))
                else "corrected" if c.is_correct.get("H2") and (not c.is_correct.get("B1"))
                else "harmed"
            ),
            "H2_vs_B0_transition": (
                "both_correct" if c.is_correct.get("H2") and c.is_correct.get("B0")
                else "both_wrong" if (not c.is_correct.get("H2")) and (not c.is_correct.get("B0"))
                else "corrected" if c.is_correct.get("H2") and (not c.is_correct.get("B0"))
                else "harmed"
            ),
        }
        csv_rows.append(row)

    csv_path = out_path / "error_transitions.csv"
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    # 2. Compute transition statistics & 4x4 transition matrices
    transition_stats_map: Dict[str, TransitionStats] = {}
    transition_matrices_json: Dict[str, Any] = {}

    for cand, base, desc in COMPARISON_PAIRS:
        pair_key = f"{cand}_vs_{base}"
        both_corr = 0
        both_wrg = 0
        harmed = 0  # base correct -> cand wrong (b)
        corrected = 0  # base wrong -> cand correct (c)
        both_wrg_shift = 0

        # 4x4 matrix: row = baseline pred, col = candidate pred
        mat = {b_lbl: {c_lbl: 0 for c_lbl in OFFICIAL_LABELS + ["UNPARSEABLE"]} for b_lbl in OFFICIAL_LABELS + ["UNPARSEABLE"]}

        for c in claims_list:
            b_corr = c.is_correct.get(base, False)
            c_corr = c.is_correct.get(cand, False)
            b_pred = c.predictions.get(base, "UNPARSEABLE")
            c_pred = c.predictions.get(cand, "UNPARSEABLE")

            if b_pred in mat and c_pred in mat[b_pred]:
                mat[b_pred][c_pred] += 1

            if b_corr and c_corr:
                both_corr += 1
            elif (not b_corr) and (not c_corr):
                both_wrg += 1
                if b_pred != c_pred:
                    both_wrg_shift += 1
            elif b_corr and (not c_corr):
                harmed += 1
            elif (not b_corr) and c_corr:
                corrected += 1

        n_claims = len(claims_list)
        p_val = exact_mcnemar(harmed, corrected)

        # Verification of transition conservation with overall accuracy
        cand_correct_total = sum(1 for c in claims_list if c.is_correct.get(cand, False))
        base_correct_total = sum(1 for c in claims_list if c.is_correct.get(base, False))
        net_diff = corrected - harmed
        expected_diff = cand_correct_total - base_correct_total
        if net_diff != expected_diff:
            raise ValueError(
                f"Transition conservation check failed for {cand} vs {base}: "
                f"Corrected ({corrected}) - Harmed ({harmed}) = {net_diff}, but "
                f"Cand correct ({cand_correct_total}) - Base correct ({base_correct_total}) = {expected_diff}!"
            )

        stats = TransitionStats(
            comparison=desc,
            arm_candidate=cand,
            arm_baseline=base,
            total_claims=n_claims,
            both_correct=both_corr,
            both_wrong=both_wrg,
            harmed=harmed,
            corrected=corrected,
            both_wrong_shifted=both_wrg_shift,
            net_benefit=net_diff,
            harm_rate=harmed / n_claims if n_claims else 0.0,
            correction_rate=corrected / n_claims if n_claims else 0.0,
            net_benefit_rate=net_diff / n_claims if n_claims else 0.0,
            mcnemar_b=corrected,  # Matching m19g convention: arm_a correct, arm_b incorrect
            mcnemar_c=harmed,     # Matching m19g convention: arm_a incorrect, arm_b correct
            mcnemar_p_value=p_val,
        )
        transition_stats_map[pair_key] = stats
        transition_matrices_json[pair_key] = {
            "description": desc,
            "counts": {
                "both_correct": both_corr,
                "both_wrong": both_wrg,
                "harmed_baseline_correct_cand_wrong": harmed,
                "corrected_baseline_wrong_cand_correct": corrected,
                "both_wrong_shifted": both_wrg_shift,
                "net_benefit": net_diff,
                "candidate_total_correct": cand_correct_total,
                "baseline_total_correct": base_correct_total,
            },
            "rates": {
                "harm_rate": round(stats.harm_rate, 4),
                "correction_rate": round(stats.correction_rate, 4),
                "net_benefit_rate": round(stats.net_benefit_rate, 4),
            },
            "mcnemar": {
                "mcnemar_b_candidate_correct_baseline_wrong": corrected,
                "mcnemar_c_candidate_wrong_baseline_correct": harmed,
                "exact_p_value": round(p_val, 6),
            },
            "prediction_matrix_baseline_row_cand_col": mat,
        }

    (out_path / "transition_matrices.json").write_text(
        json.dumps(transition_matrices_json, indent=2), encoding="utf-8"
    )

    # 3. Class Error & Shift Analysis (Section 4)
    class_analysis_rows: List[Dict[str, Any]] = []

    # Per arm per class metrics
    for arm in OFFICIAL_ARMS:
        y_true = [c.gold_label for c in claims_list]
        y_pred = [c.predictions.get(arm, "UNPARSEABLE") for c in claims_list]

        for lbl in OFFICIAL_LABELS:
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lbl and yp == lbl)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != lbl and yp == lbl)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lbl and yp != lbl)
            support = sum(1 for yt in y_true if yt == lbl)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            # Count transitions specific to this gold label in H2 vs B1 and H2 vs B0
            h2_v_b1_corr = sum(1 for c in claims_list if c.gold_label == lbl and (not c.is_correct.get("B1")) and c.is_correct.get("H2"))
            h2_v_b1_harm = sum(1 for c in claims_list if c.gold_label == lbl and c.is_correct.get("B1") and (not c.is_correct.get("H2")))

            h2_v_b0_corr = sum(1 for c in claims_list if c.gold_label == lbl and (not c.is_correct.get("B0")) and c.is_correct.get("H2"))
            h2_v_b0_harm = sum(1 for c in claims_list if c.gold_label == lbl and c.is_correct.get("B0") and (not c.is_correct.get("H2")))

            class_analysis_rows.append({
                "arm": arm,
                "class": lbl,
                "support": support,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "h2_vs_b1_corrected": h2_v_b1_corr if arm == "H2" else "",
                "h2_vs_b1_harmed": h2_v_b1_harm if arm == "H2" else "",
                "h2_vs_b1_net": (h2_v_b1_corr - h2_v_b1_harm) if arm == "H2" else "",
                "h2_vs_b0_corrected": h2_v_b0_corr if arm == "H2" else "",
                "h2_vs_b0_harmed": h2_v_b0_harm if arm == "H2" else "",
                "h2_vs_b0_net": (h2_v_b0_corr - h2_v_b0_harm) if arm == "H2" else "",
            })

    class_csv_path = out_path / "class_error_analysis.csv"
    with open(class_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(class_analysis_rows[0].keys()))
        writer.writeheader()
        writer.writerows(class_analysis_rows)

    return transition_stats_map, transition_matrices_json, class_analysis_rows
