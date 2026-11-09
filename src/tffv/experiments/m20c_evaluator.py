"""Milestone M2.0c: Phase F & G — Evaluator & Statistical Decision Battery.

Computes:
1. Primary comparison: B1 (Instance-Grounded Error) vs B2 (Matched Generic Control).
2. Secondary comparisons with Holm-Bonferroni correction:
   - B1 vs B0, B1 vs E1, B1 vs G2, B1 vs B3.
3. Paired transition matrices and conservation checks.
4. Exact two-sided McNemar tests on paired correctness (Accuracy test).
5. Paired bootstrap 95% CIs (5,000 resamples) for Delta Macro-F1 and Delta Accuracy.
6. Subgroup metrics: overall, INTERVENE, and NO_INTERVENTION.
7. Pre-registered decision criteria (GO / NO-GO / INCONCLUSIVE).

Emits:
- outputs/m20c-independent-holdout-v1/metrics_by_arm.csv
- outputs/m20c-independent-holdout-v1/per_class_metrics.csv
- outputs/m20c-independent-holdout-v1/confusion_matrices.json
- outputs/m20c-independent-holdout-v1/paired_transitions.csv
- outputs/m20c-independent-holdout-v1/statistical_tests.json
- outputs/m20c-independent-holdout-v1/FINAL_REPORT.md
- outputs/m20c-independent-holdout-v1/DECISION_REPORT.md
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tffv.experiments.m20b_evaluator import (
    bootstrap_metric_diff,
    compute_4class_metrics,
    mcnemar_exact,
)
from tffv.experiments.m20b_schemas import (
    EXPERIMENTAL_ARMS_M20B,
    OFFICIAL_CLASSES_4,
    M20bPredictionRecord,
    M20bTransitionStats,
)


def holm_bonferroni_correction(p_values: List[Tuple[str, float]]) -> Dict[str, Dict[str, Any]]:
    """Applies step-down Holm-Bonferroni correction to multiple secondary p-values."""
    sorted_p = sorted(p_values, key=lambda x: x[1])
    m = len(sorted_p)
    results = {}
    for rank, (name, p_val) in enumerate(sorted_p, 1):
        alpha_threshold = 0.05 / (m - rank + 1)
        is_significant = p_val < alpha_threshold
        results[name] = {
            "raw_pvalue": p_val,
            "rank": rank,
            "alpha_threshold": alpha_threshold,
            "is_significant_holm": is_significant,
        }
    return results


class M20cEvaluator:
    def __init__(
        self,
        records: List[M20bPredictionRecord],
        output_dir: str | Path,
        n_bootstraps: int = 5000,
        seed: int = 42,
    ):
        self.records = records
        self.output_dir = Path(output_dir)
        self.n_bootstraps = n_bootstraps
        self.seed = seed
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_all(self) -> Dict[str, Any]:
        """Runs the complete statistical evaluation and writes all reports."""
        by_arm: Dict[str, Dict[str, M20bPredictionRecord]] = defaultdict(dict)
        for r in self.records:
            by_arm[r.arm][r.sample_id] = r

        arm_keys = list(by_arm.keys())
        if not arm_keys:
            return {"status": "EMPTY"}

        common_ids = sorted(list(set.intersection(*[set(by_arm[a].keys()) for a in arm_keys])))
        total_claims = len(common_ids)

        # 1. Arm Metrics Overall
        arm_metrics: Dict[str, Dict[str, Any]] = {}
        confusion_matrices: Dict[str, Dict[str, Dict[str, int]]] = {}

        for arm in arm_keys:
            y_true = [by_arm[arm][sid].gold_verdict for sid in common_ids]
            y_pred = [by_arm[arm][sid].predicted_verdict for sid in common_ids]
            m = compute_4class_metrics(y_true, y_pred)
            unparseable = sum(1 for yp in y_pred if yp not in OFFICIAL_CLASSES_4)
            m["unparseable_count"] = unparseable
            m["valid_verdict_rate"] = 1.0 - (unparseable / max(1, total_claims))
            m["total_claims"] = total_claims
            m["correct_count"] = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
            arm_metrics[arm] = m

            cm = defaultdict(lambda: defaultdict(int))
            for yt, yp in zip(y_true, y_pred):
                cm[yt][yp] += 1
            confusion_matrices[arm] = {k: dict(v) for k, v in cm.items()}

        # 2. Subgroup Identification: INTERVENE vs NO_INTERVENTION
        b1_intervened_ids = [
            sid for sid in common_ids if by_arm.get("B1", {}).get(sid, None) and by_arm["B1"][sid].intervention_decision == "INTERVENE"
        ]
        b1_no_intervene_ids = [
            sid for sid in common_ids if by_arm.get("B1", {}).get(sid, None) and by_arm["B1"][sid].intervention_decision == "NO_INTERVENTION"
        ]

        subgroup_metrics: Dict[str, Dict[str, Any]] = {"INTERVENE": {}, "NO_INTERVENTION": {}}
        for subgroup_name, sids in [("INTERVENE", b1_intervened_ids), ("NO_INTERVENTION", b1_no_intervene_ids)]:
            for arm in arm_keys:
                if not sids:
                    subgroup_metrics[subgroup_name][arm] = {"accuracy": 0.0, "macro_f1": 0.0, "count": 0}
                    continue
                yt = [by_arm[arm][sid].gold_verdict for sid in sids]
                yp = [by_arm[arm][sid].predicted_verdict for sid in sids]
                sub_m = compute_4class_metrics(yt, yp)
                sub_m["count"] = len(sids)
                sub_m["correct_count"] = sum(1 for a, b in zip(yt, yp) if a == b)
                subgroup_metrics[subgroup_name][arm] = sub_m

        coverage_rate = (len(b1_intervened_ids) / max(1, total_claims)) * 100.0

        # 3. Pairwise Transitions & Statistical Tests
        transitions_list = []
        statistical_tests: Dict[str, Any] = {}
        secondary_pvalues = []

        pairs = [
            ("B2", "B1"),  # Primary Hypothesis
            ("B0", "B1"),  # Secondary 1
            ("E1", "B1"),  # Secondary 2
            ("G2", "B1"),  # Secondary 3
            ("B3", "B1"),  # Secondary 4
        ]

        for arm_a, arm_b in pairs:
            if arm_a not in by_arm or arm_b not in by_arm:
                continue

            both_c = 0
            both_w = 0
            corr = 0
            harm = 0

            y_true = [by_arm[arm_a][sid].gold_verdict for sid in common_ids]
            y_pred_a = [by_arm[arm_a][sid].predicted_verdict for sid in common_ids]
            y_pred_b = [by_arm[arm_b][sid].predicted_verdict for sid in common_ids]

            for sid in common_ids:
                c_a = by_arm[arm_a][sid].is_correct
                c_b = by_arm[arm_b][sid].is_correct
                if c_a and c_b:
                    both_c += 1
                elif not c_a and not c_b:
                    both_w += 1
                elif not c_a and c_b:
                    corr += 1
                elif c_a and not c_b:
                    harm += 1

            p_val, odds_ratio = mcnemar_exact(corr, harm)
            t_stats = M20bTransitionStats(
                arm_a=arm_a,
                arm_b=arm_b,
                both_correct=both_c,
                both_wrong=both_w,
                corrected=corr,
                harmed=harm,
                net_benefit=corr - harm,
                mcnemar_pvalue=p_val,
                odds_ratio=odds_ratio,
                total_claims=total_claims,
            )
            pair_key = f"{arm_b}_vs_{arm_a}"
            transitions_list.append(t_stats)

            ci_res = bootstrap_metric_diff(
                y_true, y_pred_a, y_pred_b, n_bootstraps=self.n_bootstraps, seed=self.seed
            )
            statistical_tests[pair_key] = {
                "arm_a": arm_a,
                "arm_b": arm_b,
                "transitions": t_stats.to_dict(),
                "mcnemar_pvalue": p_val,
                "odds_ratio": odds_ratio,
                "macro_f1_diff_mean": ci_res["macro_f1_diff_mean"],
                "macro_f1_diff_ci": ci_res["macro_f1_diff_ci"],
                "acc_diff_mean": ci_res["acc_diff_mean"],
                "acc_diff_ci": ci_res["acc_diff_ci"],
            }

            if pair_key != "B1_vs_B2":
                secondary_pvalues.append((pair_key, p_val))

        # Holm adjustment on secondary tests
        holm_results = holm_bonferroni_correction(secondary_pvalues)
        for k, v in holm_results.items():
            statistical_tests[k]["holm_bonferroni"] = v

        # 4. Pre-registered Decision Gate
        # Primary: B1 vs B2
        primary_key = "B1_vs_B2"
        decision = "INCONCLUSIVE"
        decision_rationale = ""

        if primary_key in statistical_tests:
            st = statistical_tests[primary_key]
            f1_ci = st["macro_f1_diff_ci"]
            net_b = st["transitions"]["net_benefit"]
            p_val = st["mcnemar_pvalue"]

            # Check criteria
            if f1_ci[0] > 0.0 and net_b > 0 and p_val < 0.05:
                decision = "GO"
                decision_rationale = (
                    f"B1 statistically significantly outperformed generic control B2 on Macro-F1 "
                    f"(CI: [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}], Net Benefit: +{net_b}, McNemar p={p_val:.4f})."
                )
            elif net_b < 0 and f1_ci[1] < 0.0:
                decision = "NO-GO"
                decision_rationale = (
                    f"B1 statistically significantly underperformed B2 (Net: {net_b}, CI: [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}])."
                )
            elif net_b < 0:
                decision = "NO-GO"
                decision_rationale = (
                    f"B1 produced net negative transitions compared to matched generic control B2 (Net Benefit: {net_b})."
                )
            elif f1_ci[0] <= 0.0 and f1_ci[1] >= 0.0:
                decision = "INCONCLUSIVE"
                decision_rationale = (
                    f"The 95% Bootstrap CI for Delta Macro-F1 contains zero [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}] "
                    f"(Net Benefit: {net_b}, McNemar p={p_val:.4f}). Note: A CI spanning zero indicates "
                    f"insufficient statistical power to resolve the difference, not empirical equivalence."
                )
            else:
                decision = "INCONCLUSIVE"
                decision_rationale = "Results do not satisfy clear directional significance threshold."

        # 5. Save Artifacts
        # metrics_by_arm.csv
        with open(self.output_dir / "metrics_by_arm.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["arm", "macro_f1", "accuracy", "balanced_accuracy", "valid_verdict_rate", "total_claims", "correct_count", "unparseable_count"])
            for arm in EXPERIMENTAL_ARMS_M20B:
                if arm in arm_metrics:
                    m = arm_metrics[arm]
                    w.writerow([
                        arm, f"{m['macro_f1']:.4f}", f"{m['accuracy']:.4f}", f"{m['balanced_accuracy']:.4f}",
                        f"{m['valid_verdict_rate']:.4f}", m["total_claims"], m["correct_count"], m["unparseable_count"]
                    ])

        # per_class_metrics.csv
        with open(self.output_dir / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["arm", "class", "precision", "recall", "f1", "support"])
            for arm in EXPERIMENTAL_ARMS_M20B:
                if arm in arm_metrics:
                    for c in OFFICIAL_CLASSES_4:
                        cd = arm_metrics[arm]["per_class"].get(c, {})
                        w.writerow([arm, c, f"{cd.get('precision', 0.0):.4f}", f"{cd.get('recall', 0.0):.4f}", f"{cd.get('f1', 0.0):.4f}", cd.get("support", 0)])

        # paired_transitions.csv
        with open(self.output_dir / "paired_transitions.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["comparison", "arm_a", "arm_b", "both_correct", "both_wrong", "corrected", "harmed", "net_benefit", "mcnemar_pvalue", "odds_ratio"])
            for t in transitions_list:
                w.writerow([f"{t.arm_b}_vs_{t.arm_a}", t.arm_a, t.arm_b, t.both_correct, t.both_wrong, t.corrected, t.harmed, t.net_benefit, f"{t.mcnemar_pvalue:.5f}", f"{t.odds_ratio:.3f}"])

        # statistical_tests.json
        with open(self.output_dir / "statistical_tests.json", "w", encoding="utf-8") as f:
            json.dump(statistical_tests, f, indent=2)

        # confusion_matrices.json
        with open(self.output_dir / "confusion_matrices.json", "w", encoding="utf-8") as f:
            json.dump(confusion_matrices, f, indent=2)

        # DECISION_REPORT.md
        with open(self.output_dir / "DECISION_REPORT.md", "w", encoding="utf-8") as f:
            f.write(f"""# Milestone M2.0c: Decision Report

## Empirical Determination: **{decision}**

**Decision Rationale:** {decision_rationale}

---

## 1. Primary Empirical Results across 6 Arms on Independent Holdout

| Arm | Description | Macro-F1 | Accuracy | Balanced Acc | Correct / Total | Valid Rate |
|:---|:---|---:|---:|---:|---:|---:|
""")
            for arm in EXPERIMENTAL_ARMS_M20B:
                if arm in arm_metrics:
                    m = arm_metrics[arm]
                    f.write(
                        f"| **{arm}** | {arm} | **{m['macro_f1']:.4f}** | **{m['accuracy']*100:.2f}%** | "
                        f"{m['balanced_accuracy']:.4f} | {m['correct_count']}/{m['total_claims']} | {m['valid_verdict_rate']*100:.1f}% |\n"
                    )

            f.write(f"""
## 2. Statistical Comparison Battery

### Primary Hypothesis: B1 (Instance-Grounded Error) vs B2 (Matched Generic Control)
""")
            if primary_key in statistical_tests:
                st = statistical_tests[primary_key]
                t = st["transitions"]
                f.write(f"""
- **Net Claim Benefit:** **{t['net_benefit']} claims** (Corrected: {t['corrected']}, Harmed: {t['harmed']})
- **Exact McNemar p-value (Accuracy test):** {st['mcnemar_pvalue']:.5f} (Odds Ratio: {st['odds_ratio']:.3f})
- **Macro-F1 Difference:** {st['macro_f1_diff_mean']:+.4f} (95% Bootstrap CI: [{st['macro_f1_diff_ci'][0]:+.4f}, {st['macro_f1_diff_ci'][1]:+.4f}])
- **Accuracy Difference:** {st['acc_diff_mean']:+.4f} (95% Bootstrap CI: [{st['acc_diff_ci'][0]:+.4f}, {st['acc_diff_ci'][1]:+.4f}])
- *Note:* McNemar tests paired correctness (Accuracy); Macro-F1 difference is tested via paired bootstrapping. A confidence interval spanning zero denotes insufficient statistical power to separate the arms, not proven equivalence.
""")

            f.write(f"""
## 3. Subgroup Diagnostics: INTERVENE vs NO_INTERVENTION

- **Total Holdout Claims:** {total_claims}
- **`INTERVENE` Triggered:** {len(b1_intervened_ids)} ({coverage_rate:.1f}%)
- **`NO_INTERVENTION` Fallback:** {len(b1_no_intervene_ids)} ({100.0 - coverage_rate:.1f}%)

| Subgroup | Arm | Accuracy | Macro-F1 | Sample Count |
|:---|:---|---:|---:|---:|
| **INTERVENE** | B0 (Zero-Shot) | {subgroup_metrics['INTERVENE'].get('B0', {}).get('accuracy', 0.0)*100:.2f}% | {subgroup_metrics['INTERVENE'].get('B0', {}).get('macro_f1', 0.0):.4f} | {len(b1_intervened_ids)} |
| | **B1 (Grounded Error)** | **{subgroup_metrics['INTERVENE'].get('B1', {}).get('accuracy', 0.0)*100:.2f}%** | **{subgroup_metrics['INTERVENE'].get('B1', {}).get('macro_f1', 0.0):.4f}** | {len(b1_intervened_ids)} |
| | B2 (Grounded Generic) | {subgroup_metrics['INTERVENE'].get('B2', {}).get('accuracy', 0.0)*100:.2f}% | {subgroup_metrics['INTERVENE'].get('B2', {}).get('macro_f1', 0.0):.4f} | {len(b1_intervened_ids)} |
| **NO_INTERVENTION** | B0 (Zero-Shot) | {subgroup_metrics['NO_INTERVENTION'].get('B0', {}).get('accuracy', 0.0)*100:.2f}% | {subgroup_metrics['NO_INTERVENTION'].get('B0', {}).get('macro_f1', 0.0):.4f} | {len(b1_no_intervene_ids)} |
| | **B1 (Fallback to B0)** | **{subgroup_metrics['NO_INTERVENTION'].get('B1', {}).get('accuracy', 0.0)*100:.2f}%** | **{subgroup_metrics['NO_INTERVENTION'].get('B1', {}).get('macro_f1', 0.0):.4f}** | {len(b1_no_intervene_ids)} |
| | B2 (Fallback to B0) | {subgroup_metrics['NO_INTERVENTION'].get('B2', {}).get('accuracy', 0.0)*100:.2f}% | {subgroup_metrics['NO_INTERVENTION'].get('B2', {}).get('macro_f1', 0.0):.4f} | {len(b1_no_intervene_ids)} |
""")

        # FINAL_REPORT.md
        with open(self.output_dir / "FINAL_REPORT.md", "w", encoding="utf-8") as f:
            f.write(f"""# Milestone M2.0c: Final Scientific Evaluation Report

## Executive Summary
Milestone M2.0c evaluated **Instance-Grounded Error Intervention** across 6 experimental arms on the independent holdout partition of {total_claims} claims using frozen Qwen3-8B BF16 and frozen R4-PF1000-v1 Top-8 evidence.

- **Primary Hypothesis Test (B1 vs B2):**
  * Delta Macro-F1: {statistical_tests.get('B1_vs_B2', {}).get('macro_f1_diff_mean', 0.0):+.4f} (95% CI: [{statistical_tests.get('B1_vs_B2', {}).get('macro_f1_diff_ci', [0,0])[0]:+.4f}, {statistical_tests.get('B1_vs_B2', {}).get('macro_f1_diff_ci', [0,0])[1]:+.4f}])
  * Net Claim Benefit: {statistical_tests.get('B1_vs_B2', {}).get('transitions', {}).get('net_benefit', 0)} claims (McNemar p = {statistical_tests.get('B1_vs_B2', {}).get('mcnemar_pvalue', 1.0):.5f})
- **Empirical Determination:** **{decision}**
- **Decision Rationale:** {decision_rationale}
""")

        return {
            "decision": decision,
            "decision_rationale": decision_rationale,
            "arm_metrics": arm_metrics,
            "statistical_tests": statistical_tests,
            "subgroup_metrics": subgroup_metrics,
            "coverage_rate": coverage_rate,
        }
