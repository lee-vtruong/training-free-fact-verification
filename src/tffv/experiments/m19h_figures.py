"""Milestone M1.9h: Figure Generation for Diagnostics and Feasibility Study."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # Non-interactive CPU backend
import matplotlib.pyplot as plt
import numpy as np

from tffv.experiments.m19h_schemas import OFFICIAL_ARMS, OFFICIAL_LABELS, TransitionStats


def generate_all_figures(
    transition_stats: Dict[str, TransitionStats],
    class_analysis_rows: List[Dict[str, Any]],
    risk_metrics: Dict[str, Any],
    risk_curves: List[Dict[str, Any]],
    routing_replay: List[Dict[str, Any]],
    output_dir: str | Path,
) -> List[str]:
    """Generates 5 publication-ready diagnostic figures in outputs/figures/."""
    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    generated_paths = []

    # -------------------------------------------------------------------------
    # Figure 1: Error Transitions Across Arm Comparisons
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    pairs = list(transition_stats.keys())
    labels = [p.replace("_vs_", " vs ") for p in pairs]

    corrected = [transition_stats[p].corrected for p in pairs]
    harmed = [transition_stats[p].harmed for p in pairs]
    net = [transition_stats[p].net_benefit for p in pairs]

    x = np.arange(len(pairs))
    w = 0.25

    ax.bar(x - w, corrected, width=w, label="Corrected (Base Wrong -> Cand Correct)", color="#2ca02c", alpha=0.85)
    ax.bar(x, harmed, width=w, label="Harmed (Base Correct -> Cand Wrong)", color="#d62728", alpha=0.85)
    ax.bar(x + w, net, width=w, label="Net Benefit (Corrected - Harmed)", color="#1f77b4", alpha=0.85)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Number of Claims (N=500)")
    ax.set_title("M1.9h Figure 1: Claim-Level Error Transitions Across Arm Pairs")
    ax.legend(frameon=True, fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    f1_path = str(fig_dir / "fig1_error_transitions.png")
    plt.savefig(f1_path)
    plt.close(fig)
    generated_paths.append(f1_path)

    # -------------------------------------------------------------------------
    # Figure 2: Per-Class F1 Shift Across Experimental Arms
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    classes = OFFICIAL_LABELS
    x = np.arange(len(classes))
    total_arms = len(OFFICIAL_ARMS)
    w = 0.8 / total_arms

    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#ccb974", "#64b5cd"]

    for idx, arm in enumerate(OFFICIAL_ARMS):
        f1_vals = []
        for c_lbl in classes:
            row = next((r for r in class_analysis_rows if r["arm"] == arm and r["class"] == c_lbl), None)
            f1_vals.append(row["f1"] if row else 0.0)
        offset = (idx - total_arms / 2 + 0.5) * w
        ax.bar(x + offset, f1_vals, width=w, label=arm, color=colors[idx % len(colors)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("Conflicting Evidence/Cherrypicking", "Conflicting/Cherrypick") for c in classes], fontsize=9)
    ax.set_ylabel("Macro-F1 Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("M1.9h Figure 2: Per-Class F1 Comparison Across All 6 Arms")
    ax.legend(frameon=True, loc="upper right", ncol=3, fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    f2_path = str(fig_dir / "fig2_per_class_f1_shift.png")
    plt.savefig(f2_path)
    plt.close(fig)
    generated_paths.append(f2_path)

    # -------------------------------------------------------------------------
    # Figure 3: Historical Similarity vs Base Verifier Error Rate
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    # Extract similarity bins from curves or risk features
    sim_curves = [r for r in risk_curves if r["predictor"] == "Max Sim to Historical Errors"]
    if sim_curves:
        budgets = [r["budget_fraction"] * 100 for r in sim_curves]
        error_precs = [r["precision"] * 100 for r in sim_curves]
        base_error = risk_metrics["base_b0_error_rate"] * 100

        ax.plot(budgets, error_precs, marker="o", linewidth=2, color="#d62728", label="B0 Error Rate in Top-K% Nearest to Errors")
        ax.axhline(base_error, color="black", linestyle="--", linewidth=1.2, label=f"Overall Base Error Rate ({base_error:.1f}%)")

        ax.set_xlabel("Intervention Budget / Percentile of Nearest Claims (%)")
        ax.set_ylabel("True Error Rate in Selected Group (%)")
        ax.set_title("M1.9h Figure 3: Error Concentration vs Proximity to Historical Errors")
        ax.legend(frameon=True, fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    f3_path = str(fig_dir / "fig3_similarity_vs_error_rate.png")
    plt.savefig(f3_path)
    plt.close(fig)
    generated_paths.append(f3_path)

    # -------------------------------------------------------------------------
    # Figure 4: Risk Predictors Precision-Recall Tradeoff
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    predictors = list(set(r["predictor"] for r in risk_curves))
    pred_colors = {
        "Random Baseline": "gray",
        "Evidence Length (Inverse)": "#ff7f0e",
        "B0 Predicted Class Prior": "#2ca02c",
        "Max Sim to Historical Errors": "#d62728",
        "Diff Sim (Wrong - Correct)": "#9467bd",
        "Composite Inference Heuristic (5-Fold Out-of-Fold)": "#1f77b4",
    }

    for p in predictors:
        sub = [r for r in risk_curves if r["predictor"] == p]
        sub = sorted(sub, key=lambda x: x["recall"])
        recs = [r["recall"] for r in sub]
        precs = [r["precision"] for r in sub]
        color = pred_colors.get(p, "blue")
        style = "--" if "Baseline" in p else "-"
        ax.plot(recs, precs, marker="s", markersize=4, linestyle=style, linewidth=1.8, color=color, label=p[:35])

    ax.set_xlabel("Error Recall (Fraction of Total B0 Errors Captured)")
    ax.set_ylabel("Error Precision (Fraction of Flagged Claims that are True Errors)")
    ax.set_title("M1.9h Figure 4: Precision-Recall Curves for Base Error Detection")
    ax.legend(frameon=True, fontsize=8, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    f4_path = str(fig_dir / "fig4_risk_precision_recall_curves.png")
    plt.savefig(f4_path)
    plt.close(fig)
    generated_paths.append(f4_path)

    # -------------------------------------------------------------------------
    # Figure 5: Selective Routing Budget Tradeoff vs Post-hoc Replay
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    sim_replays = [r for r in routing_replay if "Max Sim" in r["routing_policy"]]
    if sim_replays:
        budgets_pct = [r["budget_fraction"] * 100 for r in sim_replays]
        b0_acc = sim_replays[0]["b0_base_accuracy"] * 100
        oracle_acc = [r["oracle_upper_bound_accuracy"] * 100 for r in sim_replays]
        h2_acc = [r["h2_posthoc_replay_accuracy"] * 100 for r in sim_replays]
        b2_acc = [r["b2_posthoc_replay_accuracy"] * 100 for r in sim_replays]

        ax.axhline(b0_acc, color="black", linestyle="--", linewidth=1.5, label=f"B0 Base Zero-Shot ({b0_acc:.1f}%)")
        ax.plot(budgets_pct, oracle_acc, marker="^", linewidth=2, color="#2ca02c", label="Oracle Perfect Ceiling (Correct all flagged errors)")
        ax.plot(budgets_pct, b2_acc, marker="o", linewidth=1.8, color="#1f77b4", label="Replay with B2 Mixed Gold ICL")
        ax.plot(budgets_pct, h2_acc, marker="x", linewidth=1.8, color="#d62728", label="Replay with H2 Error-History ICL (Degrades)")

        ax.set_xlabel("Selective Intervention Budget (% of Claims Flagged)")
        ax.set_ylabel("Overall Dev Accuracy (%)")
        ax.set_title("M1.9h Figure 5: Selective Routing Policy vs Replay Outcomes (Exploratory)")
        ax.legend(frameon=True, fontsize=8, loc="center right")
        ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    f5_path = str(fig_dir / "fig5_selective_routing_tradeoffs.png")
    plt.savefig(f5_path)
    plt.close(fig)
    generated_paths.append(f5_path)

    return generated_paths
