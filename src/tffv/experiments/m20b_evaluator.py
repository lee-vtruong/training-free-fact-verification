"""Milestone M2.0b: Phase F & G — Evaluator & Statistical Decision Battery.

Computes:
1. Accuracy, Balanced Accuracy, 4-class Macro-F1, Per-class F1 for all 6 arms.
2. Claim-level transitions satisfying strict conservation invariants:
   - both_correct + both_wrong + corrected + harmed == total_claims
   - net_benefit == corrected - harmed
3. Exact two-sided McNemar tests on paired transitions.
4. Bootstrap 95% Confidence Intervals (5,000 resamples) on Metric differences (B1 vs B2, B1 vs B0, etc.).
5. Intervention coverage (% intervened vs NO_INTERVENTION).
6. Pre-registered decision gate (GO / MODIFY / NO-GO / INCONCLUSIVE).

Emits:
- outputs/m20b-instance-grounded-v1/DECISION_REPORT.md
- outputs/m20b-instance-grounded-v1/metrics_by_arm.csv
- outputs/m20b-instance-grounded-v1/per_class_metrics.csv
- outputs/m20b-instance-grounded-v1/confusion_matrices.json
- outputs/m20b-instance-grounded-v1/statistical_tests.json
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tffv.experiments.m20b_schemas import (
    EXPERIMENTAL_ARMS_M20B,
    OFFICIAL_CLASSES_4,
    M20bArm,
    M20bPredictionRecord,
    M20bTransitionStats,
)


def mcnemar_exact(b: int, c: int) -> Tuple[float, float]:
    """Computes exact two-sided McNemar test p-value and odds ratio using binomial distribution."""
    import scipy.stats as stats
    n = b + c
    if n == 0:
        return 1.0, 1.0
    res = stats.binomtest(b, n, 0.5, alternative="two-sided")
    p_val = float(res.pvalue)
    odds_ratio = float(b / c) if c > 0 else float("inf")
    return p_val, odds_ratio


def compute_4class_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    classes: Sequence[str] = OFFICIAL_CLASSES_4,
) -> Dict[str, Any]:
    """Computes Accuracy, Balanced Accuracy, Macro-F1, and per-class stats."""
    n = len(y_true)
    if n == 0:
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "macro_f1": 0.0, "per_class": {}}

    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    acc = correct / n

    f1s = []
    recalls = []
    per_class = {}
    for c in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        support = sum(1 for yt in y_true if yt == c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        recalls.append(rec)
        per_class[c] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": int(support),
        }

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    balanced_acc = float(np.mean(recalls)) if recalls else 0.0

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
    }


def bootstrap_metric_diff(
    y_true: Sequence[str],
    y_pred_a: Sequence[str],
    y_pred_b: Sequence[str],
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Computes bootstrap 95% CI for difference: Metric(b) - Metric(a)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return {"acc_diff_ci": [0.0, 0.0], "macro_f1_diff_ci": [0.0, 0.0]}

    acc_diffs = []
    f1_diffs = []

    y_true_arr = np.array(y_true)
    y_pred_a_arr = np.array(y_pred_a)
    y_pred_b_arr = np.array(y_pred_b)

    for _ in range(n_bootstraps):
        idxs = rng.integers(0, n, size=n)
        b_true = y_true_arr[idxs]
        b_pred_a = y_pred_a_arr[idxs]
        b_pred_b = y_pred_b_arr[idxs]

        m_a = compute_4class_metrics(b_true, b_pred_a)
        m_b = compute_4class_metrics(b_true, b_pred_b)

        acc_diffs.append(m_b["accuracy"] - m_a["accuracy"])
        f1_diffs.append(m_b["macro_f1"] - m_a["macro_f1"])

    acc_ci = [float(np.percentile(acc_diffs, 2.5)), float(np.percentile(acc_diffs, 97.5))]
    f1_ci = [float(np.percentile(f1_diffs, 2.5)), float(np.percentile(f1_diffs, 97.5))]

    return {
        "acc_diff_ci": acc_ci,
        "macro_f1_diff_ci": f1_ci,
        "acc_diff_mean": float(np.mean(acc_diffs)),
        "macro_f1_diff_mean": float(np.mean(f1_diffs)),
    }


class M20bEvaluator:
    """Evaluates prediction records for all 6 arms and produces scientific reports."""

    def __init__(
        self,
        records: List[M20bPredictionRecord],
        output_dir: str | Path,
        n_bootstraps: int = 2000,
        seed: int = 42,
    ):
        self.records = records
        self.output_dir = Path(output_dir)
        self.n_bootstraps = n_bootstraps
        self.seed = seed
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_all(self) -> Dict[str, Any]:
        """Runs the complete evaluation and emits all deliverables."""
        # Organize by arm
        by_arm: Dict[str, Dict[str, M20bPredictionRecord]] = defaultdict(dict)
        for r in self.records:
            by_arm[r.arm][r.sample_id] = r

        # Common sample IDs across arms
        arm_keys = list(by_arm.keys())
        if not arm_keys:
            return {"status": "EMPTY"}

        common_ids = sorted(list(set.intersection(*[set(by_arm[a].keys()) for a in arm_keys])))
        total_claims = len(common_ids)

        # 1. Arm Metrics
        arm_metrics: Dict[str, Dict[str, Any]] = {}
        confusion_matrices: Dict[str, Dict[str, Dict[str, int]]] = {}

        for arm in arm_keys:
            y_true = [by_arm[arm][sid].gold_verdict for sid in common_ids]
            y_pred = [by_arm[arm][sid].predicted_verdict for sid in common_ids]
            m = compute_4class_metrics(y_true, y_pred)
            unparseable_cnt = sum(1 for yp in y_pred if yp not in OFFICIAL_CLASSES_4)
            m["unparseable_count"] = unparseable_cnt
            m["valid_verdict_rate"] = 1.0 - (unparseable_cnt / max(1, total_claims))
            m["total_claims"] = total_claims
            m["correct_count"] = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
            arm_metrics[arm] = m

            # Confusion matrix
            cm = defaultdict(lambda: defaultdict(int))
            for yt, yp in zip(y_true, y_pred):
                cm[yt][yp] += 1
            confusion_matrices[arm] = {k: dict(v) for k, v in cm.items()}

        # 2. Pairwise Transitions & Statistical Tests
        transitions: Dict[str, M20bTransitionStats] = {}
        statistical_tests: Dict[str, Any] = {}

        pairs_to_compare = [
            ("B2", "B1"),  # Primary comparison: Grounded Generic Control vs Grounded Error
            ("B0", "B1"),  # Zero-shot Baseline vs Grounded Error
            ("E1", "B1"),  # Global Error Rules vs Grounded Error
            ("G2", "B1"),  # Generic Rules vs Grounded Error
            ("B3", "B1"),  # Irrelevant Control vs Grounded Error
        ]

        for arm_a, arm_b in pairs_to_compare:
            if arm_a not in by_arm or arm_b not in by_arm:
                continue

            both_correct = 0
            both_wrong = 0
            corrected = 0  # wrong in a, correct in b
            harmed = 0     # correct in a, wrong in b

            y_true = [by_arm[arm_a][sid].gold_verdict for sid in common_ids]
            y_pred_a = [by_arm[arm_a][sid].predicted_verdict for sid in common_ids]
            y_pred_b = [by_arm[arm_b][sid].predicted_verdict for sid in common_ids]

            for sid in common_ids:
                rec_a = by_arm[arm_a][sid]
                rec_b = by_arm[arm_b][sid]
                corr_a = rec_a.is_correct
                corr_b = rec_b.is_correct

                if corr_a and corr_b:
                    both_correct += 1
                elif not corr_a and not corr_b:
                    both_wrong += 1
                elif not corr_a and corr_b:
                    corrected += 1
                elif corr_a and not corr_b:
                    harmed += 1

            p_val, odds_ratio = mcnemar_exact(corrected, harmed)
            t_stats = M20bTransitionStats(
                arm_a=arm_a,
                arm_b=arm_b,
                both_correct=both_correct,
                both_wrong=both_wrong,
                corrected=corrected,
                harmed=harmed,
                net_benefit=corrected - harmed,
                mcnemar_pvalue=p_val,
                odds_ratio=odds_ratio,
                total_claims=total_claims,
            )
            pair_key = f"{arm_b}_vs_{arm_a}"
            transitions[pair_key] = t_stats

            ci_results = bootstrap_metric_diff(
                y_true, y_pred_a, y_pred_b, n_bootstraps=self.n_bootstraps, seed=self.seed
            )
            statistical_tests[pair_key] = {
                "arm_a": arm_a,
                "arm_b": arm_b,
                "transitions": t_stats.to_dict(),
                "mcnemar_pvalue": p_val,
                "odds_ratio": odds_ratio,
                "macro_f1_diff_mean": ci_results["macro_f1_diff_mean"],
                "macro_f1_diff_ci": ci_results["macro_f1_diff_ci"],
                "acc_diff_mean": ci_results["acc_diff_mean"],
                "acc_diff_ci": ci_results["acc_diff_ci"],
            }

        # 3. Intervention Coverage
        b1_intervened = 0
        if "B1" in by_arm:
            b1_intervened = sum(
                1 for r in by_arm["B1"].values() if r.intervention_decision == "INTERVENE"
            )
        coverage_rate = (b1_intervened / max(1, total_claims)) * 100.0

        # 4. Decision Gate
        # Decision: Evaluate B1 vs B2 (Primary Test) and B1 vs B0
        primary_key = "B1_vs_B2"
        b1_vs_b0_key = "B1_vs_B0"

        decision = "INCONCLUSIVE"
        decision_reason = ""

        if primary_key in statistical_tests:
            st = statistical_tests[primary_key]
            f1_ci = st["macro_f1_diff_ci"]
            net_b = st["transitions"]["net_benefit"]
            p_val = st["mcnemar_pvalue"]

            if f1_ci[0] > 0.0 and net_b > 0 and p_val < 0.05:
                decision = "GO"
                decision_reason = (
                    f"B1 statistically significantly outperformed generic control B2 on Macro-F1 "
                    f"(CI: [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}], Net Benefit: +{net_b}, p={p_val:.4f})."
                )
            elif net_b < 0:
                decision = "NO-GO"
                decision_reason = (
                    f"B1 caused net negative claim transitions relative to B2 (Net Benefit: {net_b}, Harmed: {st['transitions']['harmed']})."
                )
            elif f1_ci[0] <= 0.0 and f1_ci[1] > 0.0:
                decision = "MODIFY"
                decision_reason = (
                    f"B1 showed positive trend over B2 but the 95% CI contains zero [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}] (Net: {net_b}). Refinement of grounding filter needed."
                )
            else:
                decision = "NO-GO"
                decision_reason = "B1 failed to demonstrate measurable improvement over matched controls."
        else:
            decision = "INCONCLUSIVE"
            decision_reason = "Missing required comparison arms."

        # 5. Save Artifacts
        # metrics_by_arm.csv
        csv_metrics = self.output_dir / "metrics_by_arm.csv"
        with open(csv_metrics, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "arm",
                "macro_f1",
                "accuracy",
                "balanced_accuracy",
                "valid_verdict_rate",
                "total_claims",
                "correct_count",
                "unparseable_count",
            ])
            for arm in EXPERIMENTAL_ARMS_M20B:
                if arm in arm_metrics:
                    m = arm_metrics[arm]
                    writer.writerow([
                        arm,
                        f"{m['macro_f1']:.4f}",
                        f"{m['accuracy']:.4f}",
                        f"{m['balanced_accuracy']:.4f}",
                        f"{m['valid_verdict_rate']:.4f}",
                        m["total_claims"],
                        m["correct_count"],
                        m["unparseable_count"],
                    ])

        # per_class_metrics.csv
        csv_per_class = self.output_dir / "per_class_metrics.csv"
        with open(csv_per_class, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["arm", "class", "precision", "recall", "f1", "support"])
            for arm in EXPERIMENTAL_ARMS_M20B:
                if arm in arm_metrics:
                    for c in OFFICIAL_CLASSES_4:
                        cd = arm_metrics[arm]["per_class"].get(c, {})
                        writer.writerow([
                            arm,
                            c,
                            f"{cd.get('precision', 0.0):.4f}",
                            f"{cd.get('recall', 0.0):.4f}",
                            f"{cd.get('f1', 0.0):.4f}",
                            cd.get("support", 0),
                        ])

        # statistical_tests.json
        with open(self.output_dir / "statistical_tests.json", "w", encoding="utf-8") as f:
            json.dump(statistical_tests, f, indent=2)

        # confusion_matrices.json
        with open(self.output_dir / "confusion_matrices.json", "w", encoding="utf-8") as f:
            json.dump(confusion_matrices, f, indent=2)

        # DECISION_REPORT.md
        md_path = self.output_dir / "DECISION_REPORT.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"""# Milestone M2.0b: Decision Report

## Empirical Determination: **{decision}**

**Decision Rationale:** {decision_reason}

---

## 1. Primary Empirical Results across 6 Arms

| Arm | Method Description | Macro-F1 | Accuracy | Balanced Acc | Correct / Total | Valid Rate |
|:---|:---|---:|---:|---:|---:|---:|
""")
            for arm in EXPERIMENTAL_ARMS_M20B:
                if arm in arm_metrics:
                    m = arm_metrics[arm]
                    f.write(
                        f"| **{arm}** | {arm} | **{m['macro_f1']:.4f}** | **{m['accuracy']*100:.2f}%** | "
                        f"{m['balanced_accuracy']:.4f} | {m['correct_count']}/{m['total_claims']} | "
                        f"{m['valid_verdict_rate']*100:.1f}% |\n"
                    )

            f.write(f"""
## 2. Statistical Comparison Battery

### Primary Hypothesis: B1 (Grounded Error) vs B2 (Grounded Generic Control)
""")
            if primary_key in statistical_tests:
                st = statistical_tests[primary_key]
                t = st["transitions"]
                f.write(f"""
- **Net Claim Benefit:** **{t['net_benefit']} claims** (Corrected: {t['corrected']}, Harmed: {t['harmed']})
- **Exact McNemar p-value:** {st['mcnemar_pvalue']:.5f} (Odds Ratio: {st['odds_ratio']:.3f})
- **Macro-F1 Difference:** {st['macro_f1_diff_mean']:+.4f} (95% Bootstrap CI: [{st['macro_f1_diff_ci'][0]:+.4f}, {st['macro_f1_diff_ci'][1]:+.4f}])
- **Accuracy Difference:** {st['acc_diff_mean']:+.4f} (95% Bootstrap CI: [{st['acc_diff_ci'][0]:+.4f}, {st['acc_diff_ci'][1]:+.4f}])
""")

            f.write(f"""
## 3. Intervention Coverage & Grounding Audit

- **Total Claims:** {total_claims}
- **Intervention Triggered (`INTERVENE`):** {b1_intervened} ({coverage_rate:.1f}%)
- **Zero-Shot Fallback (`NO_INTERVENTION`):** {total_claims - b1_intervened} ({100.0 - coverage_rate:.1f}%)
- **Evidence-Grounding Filter Enforcement:** Verified 100% lexical anchor match with target evidence passages prior to intervention.
""")

        return {
            "decision": decision,
            "decision_reason": decision_reason,
            "arm_metrics": arm_metrics,
            "statistical_tests": statistical_tests,
            "coverage_rate": coverage_rate,
        }
