"""Milestone M1.9h: Historical Selector Diagnostics & Similarity Analysis."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

from tffv.experiments.m19h_schemas import ClaimRecord


def compute_group_stats(values: List[float]) -> Dict[str, float]:
    """Computes robust descriptive statistics."""
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "iqr": 0.0}
    arr = np.array(values, dtype=float)
    q75, q25 = np.percentile(arr, [75, 25])
    return {
        "count": len(arr),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "iqr": round(float(q75 - q25), 4),
    }


def analyze_selector(
    claims_map: Dict[str, ClaimRecord],
    output_dir: str | Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Analyzes exemplar similarity scores and their correlation with outcomes."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    claims_list = list(claims_map.values())

    # Build per-claim diagnostics data
    diag_rows: List[Dict[str, Any]] = []

    h2_sim_all = []
    h1_sim_all = []
    gap_all = []
    b0_corr_all = []
    h2_corr_all = []

    for c in claims_list:
        h2_sims = c.similarity_scores.get("H2", [])
        h1_sims = c.similarity_scores.get("H1", [])
        
        sim1 = h2_sims[0] if len(h2_sims) > 0 else 0.0
        sim2 = h2_sims[1] if len(h2_sims) > 1 else 0.0
        sim_gap = sim1 - sim2
        h2_mean = float(np.mean(h2_sims)) if h2_sims else 0.0

        h1_sim1 = h1_sims[0] if len(h1_sims) > 0 else 0.0
        h1_mean = float(np.mean(h1_sims)) if h1_sims else 0.0

        b0_c = int(c.is_correct.get("B0", False))
        b1_c = int(c.is_correct.get("B1", False))
        h2_c = int(c.is_correct.get("H2", False))
        h3_c = int(c.is_correct.get("H3", False))
        b2_c = int(c.is_correct.get("B2", False))

        # Outcome categorizations
        h2_vs_b1 = (
            "help" if (not b1_c) and h2_c
            else "harm" if b1_c and (not h2_c)
            else "neutral"
        )
        h2_vs_b0 = (
            "help" if (not b0_c) and h2_c
            else "harm" if b0_c and (not h2_c)
            else "neutral"
        )
        h3_vs_b2 = (
            "help" if (not b2_c) and h3_c
            else "harm" if b2_c and (not h3_c)
            else "neutral"
        )

        h2_sim_all.append(sim1)
        h1_sim_all.append(h1_sim1)
        gap_all.append(sim_gap)
        b0_corr_all.append(b0_c)
        h2_corr_all.append(h2_c)

        diag_rows.append({
            "claim_id": c.claim_id,
            "gold_label": c.gold_label,
            "B0_correct": b0_c,
            "H2_correct": h2_c,
            "H2_sim1_top1": round(sim1, 4),
            "H2_sim2_top2": round(sim2, 4),
            "H2_sim_gap": round(sim_gap, 4),
            "H2_sim_mean": round(h2_mean, 4),
            "H1_sim1_top1": round(h1_sim1, 4),
            "H1_sim_mean": round(h1_mean, 4),
            "diff_sim_wrong_correct": round(sim1 - h1_sim1, 4),
            "H2_vs_B1_outcome": h2_vs_b1,
            "H2_vs_B0_outcome": h2_vs_b0,
            "H3_vs_B2_outcome": h3_vs_b2,
        })

    # Group Comparisons (Section 5)
    # A. B0 correct vs B0 incorrect
    b0_corr_sims = [r["H2_sim1_top1"] for r in diag_rows if r["B0_correct"] == 1]
    b0_wrong_sims = [r["H2_sim1_top1"] for r in diag_rows if r["B0_correct"] == 0]

    # B. H2 helps vs H2 harms (relative to B1)
    h2_help_sims = [r["H2_sim1_top1"] for r in diag_rows if r["H2_vs_B1_outcome"] == "help"]
    h2_harm_sims = [r["H2_sim1_top1"] for r in diag_rows if r["H2_vs_B1_outcome"] == "harm"]

    # C. H3 helps vs H3 harms (relative to B2)
    h3_help_sims = [r["H2_sim1_top1"] for r in diag_rows if r["H3_vs_B2_outcome"] == "help"]
    h3_harm_sims = [r["H2_sim1_top1"] for r in diag_rows if r["H3_vs_B2_outcome"] == "harm"]

    # Correlation checks
    p_corr, p_pval = pearsonr(h2_sim_all, b0_corr_all) if len(set(h2_sim_all)) > 1 else (0.0, 1.0)
    s_corr, s_pval = spearmanr(h2_sim_all, b0_corr_all) if len(set(h2_sim_all)) > 1 else (0.0, 1.0)

    summary_stats = {
        "b0_correct_group": compute_group_stats(b0_corr_sims),
        "b0_wrong_group": compute_group_stats(b0_wrong_sims),
        "h2_helps_b1_group": compute_group_stats(h2_help_sims),
        "h2_harms_b1_group": compute_group_stats(h2_harm_sims),
        "h3_helps_b2_group": compute_group_stats(h3_help_sims),
        "h3_harms_b2_group": compute_group_stats(h3_harm_sims),
        "pearson_sim_vs_b0_correct": round(float(p_corr), 4),
        "pearson_p_value": round(float(p_pval), 6),
        "spearman_sim_vs_b0_correct": round(float(s_corr), 4),
        "spearman_p_value": round(float(s_pval), 6),
    }

    # Write selector_diagnostics.csv
    csv_path = out_path / "selector_diagnostics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
        writer.writeheader()
        writer.writerows(diag_rows)

    return diag_rows, summary_stats
