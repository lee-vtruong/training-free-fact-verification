"""Milestone M1.9h: Historical Error Risk Estimation — Feasibility Study."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from tffv.experiments.m19h_schemas import ClaimRecord


def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Any,
    n_bootstraps: int = 5000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Computes point estimate and 95% percentile bootstrap confidence interval."""
    val = float(metric_fn(y_true, y_score))
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_vals = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            boot_vals.append(float(metric_fn(y_true[idx], y_score[idx])))
        except Exception:
            continue
    if not boot_vals:
        return val, val, val
    low = float(np.percentile(boot_vals, 2.5))
    high = float(np.percentile(boot_vals, 97.5))
    return val, low, high


def run_risk_study(
    claims_map: Dict[str, ClaimRecord],
    output_dir: str | Path,
    n_bootstraps: int = 5000,
    seed: int = 42,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluates whether inference-available historical features predict Base verifier errors."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sorted_sids = sorted(claims_map.keys(), key=lambda s: int(s.split(":")[1]) if ":" in s else s)
    claims_list = [claims_map[sid] for sid in sorted_sids]

    # Target label: 1 if B0 made an error, 0 otherwise (Evaluator-only!)
    y_error = np.array([1 - int(c.is_correct.get("B0", False)) for c in claims_list], dtype=int)
    base_error_rate = float(np.mean(y_error))

    # Inference-available feature extraction
    feature_rows = []
    for c in claims_list:
        h2_sims = c.similarity_scores.get("H2", [])
        h1_sims = c.similarity_scores.get("H1", [])

        max_sim_wrong = h2_sims[0] if len(h2_sims) > 0 else 0.0
        mean_sim_wrong = float(np.mean(h2_sims)) if h2_sims else 0.0
        sim_gap_wrong = (h2_sims[0] - h2_sims[1]) if len(h2_sims) > 1 else 0.0

        max_sim_correct = h1_sims[0] if len(h1_sims) > 0 else 0.0
        mean_sim_correct = float(np.mean(h1_sims)) if h1_sims else 0.0

        diff_sim = max_sim_wrong - max_sim_correct

        b0_pred = c.predictions.get("B0", "UNPARSEABLE")

        feature_rows.append({
            "claim_id": c.claim_id,
            "b0_pred_verdict": b0_pred,
            "evidence_count": c.evidence_count,
            "evidence_char_len": c.evidence_char_len,
            "max_sim_wrong": round(max_sim_wrong, 4),
            "mean_sim_wrong": round(mean_sim_wrong, 4),
            "sim_gap_wrong": round(sim_gap_wrong, 4),
            "max_sim_correct": round(max_sim_correct, 4),
            "mean_sim_correct": round(mean_sim_correct, 4),
            "diff_sim_wrong_correct": round(diff_sim, 4),
            # Target diagnostic label
            "b0_is_error": int(1 - int(c.is_correct.get("B0", False))),
        })

    # Save risk_features.csv
    with open(out_path / "risk_features.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(feature_rows[0].keys()))
        writer.writeheader()
        writer.writerows(feature_rows)

    # 1. Feature Arrays
    max_sim_w = np.array([r["max_sim_wrong"] for r in feature_rows])
    diff_sim = np.array([r["diff_sim_wrong_correct"] for r in feature_rows])
    ev_len = np.array([r["evidence_char_len"] for r in feature_rows])
    # Shorter evidence -> higher error risk score
    ev_len_score = -ev_len if np.std(ev_len) > 0 else np.zeros_like(ev_len)

    # Label-prior risk heuristic (class error rate in Dev)
    b0_preds = [r["b0_pred_verdict"] for r in feature_rows]
    unique_preds = list(set(b0_preds))
    label_error_rates = {}
    for p_lbl in unique_preds:
        mask = [p == p_lbl for p in b0_preds]
        label_error_rates[p_lbl] = float(np.mean(y_error[mask])) if sum(mask) > 0 else base_error_rate
    label_prior_scores = np.array([label_error_rates[p] for p in b0_preds])

    # 2. Composite Cross-Fitted Inference Heuristic (5-fold out-of-fold)
    X_mat = np.column_stack([
        max_sim_w,
        diff_sim,
        ev_len,
        label_prior_scores,
    ])
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    composite_scores = np.zeros(len(claims_list))

    for train_idx, test_idx in kf.split(X_mat):
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_mat[train_idx])
        X_test_s = scaler.transform(X_mat[test_idx])
        clf = LogisticRegression(C=1.0, random_state=seed)
        clf.fit(X_train_s, y_error[train_idx])
        composite_scores[test_idx] = clf.predict_proba(X_test_s)[:, 1]

    # Random baseline
    rng = np.random.RandomState(seed)
    random_scores = rng.uniform(0.0, 1.0, size=len(y_error))

    predictors = {
        "Random Baseline": random_scores,
        "Evidence Length (Inverse)": ev_len_score,
        "B0 Predicted Class Prior": label_prior_scores,
        "Max Sim to Historical Errors": max_sim_w,
        "Diff Sim (Wrong - Correct)": diff_sim,
        "Composite Inference Heuristic (5-Fold Out-of-Fold)": composite_scores,
    }

    metrics_summary = {
        "sample_size": len(claims_list),
        "total_b0_errors": int(np.sum(y_error)),
        "base_b0_error_rate": round(base_error_rate, 4),
        "predictors": {},
    }

    for name, scores in predictors.items():
        # Handle zero variance or constant scores
        if np.std(scores) == 0:
            auroc_val, auroc_low, auroc_high = 0.5, 0.5, 0.5
            auprc_val, auprc_low, auprc_high = base_error_rate, base_error_rate, base_error_rate
        else:
            auroc_val, auroc_low, auroc_high = bootstrap_metric(y_error, scores, roc_auc_score, n_bootstraps, seed)
            auprc_val, auprc_low, auprc_high = bootstrap_metric(y_error, scores, average_precision_score, n_bootstraps, seed)

        metrics_summary["predictors"][name] = {
            "auroc": round(auroc_val, 4),
            "auroc_95_ci": [round(auroc_low, 4), round(auroc_high, 4)],
            "auprc": round(auprc_val, 4),
            "auprc_95_ci": [round(auprc_low, 4), round(auprc_high, 4)],
            "gain_over_random_auroc": round(auroc_val - 0.50, 4),
            "gain_over_base_auprc": round(auprc_val - base_error_rate, 4),
        }

    # Save risk_metrics.json
    (out_path / "risk_metrics.json").write_text(
        json.dumps(metrics_summary, indent=2), encoding="utf-8"
    )

    # 3. Risk Curves across Budgets (Section 6)
    budgets = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00]
    curve_rows: List[Dict[str, Any]] = []

    for name, scores in predictors.items():
        sorted_indices = np.argsort(-scores)  # highest risk first
        total_errors = np.sum(y_error)

        for b in budgets:
            k_claims = max(1, int(round(b * len(claims_list))))
            flagged_idx = sorted_indices[:k_claims]

            tp_errors = int(np.sum(y_error[flagged_idx]))
            fp_correct = k_claims - tp_errors

            prec = tp_errors / k_claims if k_claims > 0 else 0.0
            rec = tp_errors / total_errors if total_errors > 0 else 0.0
            fpr = fp_correct / (len(claims_list) - total_errors) if (len(claims_list) - total_errors) > 0 else 0.0

            curve_rows.append({
                "predictor": name,
                "budget_fraction": b,
                "claims_flagged": k_claims,
                "captured_b0_errors": tp_errors,
                "false_alarms": fp_correct,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "false_alarm_rate": round(fpr, 4),
            })

    # Save risk_curves.csv
    with open(out_path / "risk_curves.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        writer.writeheader()
        writer.writerows(curve_rows)

    return metrics_summary, feature_rows, curve_rows
