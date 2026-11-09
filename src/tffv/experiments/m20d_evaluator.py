"""Milestone M2.0d: Phase H & I â€” Statistical Evaluator & Decision Battery.

Evaluates 6 Track C arms:
- Computes Macro-F1, Accuracy, Balanced Accuracy, Per-class F1, Attribution validity.
- Primary comparison: C2 vs C3 (Paired Bootstrap 95% CI with >= 5,000 resamples, Exact McNemar test).
- Secondary comparisons: C2 vs B0, G2, C1, C4 with Holm-Bonferroni correction.
- Subgroup diagnostics: PATTERN_SELECTED vs NO_PATTERN_SELECTED.
- Emits CSVs, JSONs, FINAL_REPORT.md, DECISION_REPORT.md.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from tffv.experiments.m20d_schemas import M20dPredictionRecord

ALL_CLASSES = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]


class M20dEvaluator:
    def __init__(
        self,
        records: List[M20dPredictionRecord],
        output_dir: str | Path = "outputs/m20d-contrastive-verification-v1",
        n_bootstraps: int = 5000,
        seed: int = 42,
    ):
        self.records = records
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_bootstraps = n_bootstraps
        self.seed = seed

    def evaluate_all(self) -> Dict[str, Any]:
        """Runs complete evaluation, statistical tests, and decision determination."""
        # 1. Group records by arm and by sample_id
        arms_records: Dict[str, List[M20dPredictionRecord]] = {}
        for r in self.records:
            arms_records.setdefault(r.arm, []).append(r)

        # 2. Compute metrics by arm
        metrics_by_arm: Dict[str, Dict[str, Any]] = {}
        per_class_by_arm: Dict[str, Dict[str, Dict[str, float]]] = {}
        confusion_by_arm: Dict[str, Dict[str, Dict[str, int]]] = {}

        for arm, recs in sorted(arms_records.items()):
            arm_metrics, per_class, cm = self._compute_arm_metrics(recs)
            metrics_by_arm[arm] = arm_metrics
            per_class_by_arm[arm] = per_class
            confusion_by_arm[arm] = cm

        # 3. Statistical Comparison Battery
        stat_results = self._run_statistical_battery(arms_records)

        # 4. Subgroup diagnostics
        subgroup_results = self._run_subgroup_diagnostics(arms_records)

        # 5. Determine Pre-registered Decision
        decision, rationale = self._determine_decision(stat_results, metrics_by_arm)

        # 6. Save artifacts
        self._save_csvs(metrics_by_arm, per_class_by_arm, stat_results)
        self._save_jsons(confusion_by_arm, stat_results)
        self._save_markdown_reports(metrics_by_arm, stat_results, subgroup_results, decision, rationale)

        return {
            "decision": decision,
            "decision_rationale": rationale,
            "metrics_by_arm": metrics_by_arm,
            "statistical_tests": stat_results,
            "subgroups": subgroup_results,
        }

    def _compute_arm_metrics(
        self,
        recs: List[M20dPredictionRecord],
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]], Dict[str, Dict[str, int]]]:
        total = len(recs)
        correct = sum(1 for r in recs if r.parsed_verdict == r.gold_label and r.gold_label is not None)
        unparseable = sum(1 for r in recs if r.parse_status != "SUCCESS" or r.parsed_verdict is None)
        valid_attr = sum(1 for r in recs if r.attribution and r.attribution.is_valid_attribution)

        accuracy = correct / total if total > 0 else 0.0
        valid_verdict_rate = (total - unparseable) / total if total > 0 else 0.0
        attr_valid_rate = valid_attr / total if total > 0 else 0.0

        # Confusion matrix: gold -> pred -> count
        cm: Dict[str, Dict[str, int]] = {g: {p: 0 for p in ALL_CLASSES} for g in ALL_CLASSES}
        for g in ALL_CLASSES:
            cm[g]["UNPARSEABLE"] = 0

        for r in recs:
            g = r.gold_label
            p = r.parsed_verdict
            if g in ALL_CLASSES:
                if p in ALL_CLASSES:
                    cm[g][p] += 1
                else:
                    cm[g]["UNPARSEABLE"] += 1

        # Per-class metrics
        per_class: Dict[str, Dict[str, float]] = {}
        recalls = []
        f1s = []

        for c in ALL_CLASSES:
            tp = cm[c].get(c, 0)
            fn = sum(cm[c][p] for p in cm[c] if p != c)
            fp = sum(cm[g].get(c, 0) for g in ALL_CLASSES if g != c)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            per_class[c] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "support": tp + fn,
            }
            recalls.append(rec)
            f1s.append(f1)

        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        balanced_acc = sum(recalls) / len(recalls) if recalls else 0.0

        summary = {
            "macro_f1": round(macro_f1, 4),
            "accuracy": round(accuracy, 4),
            "balanced_accuracy": round(balanced_acc, 4),
            "valid_verdict_rate": round(valid_verdict_rate, 4),
            "attribution_valid_rate": round(attr_valid_rate, 4),
            "total_claims": total,
            "correct_count": correct,
            "unparseable_count": unparseable,
        }
        return summary, per_class, cm

    def _run_statistical_battery(
        self,
        arms_records: Dict[str, List[M20dPredictionRecord]],
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        # Build map by sample_id for each arm
        arm_maps: Dict[str, Dict[str, M20dPredictionRecord]] = {}
        for arm, recs in arms_records.items():
            arm_maps[arm] = {r.sample_id: r for r in recs}

        # Primary comparison: C2 vs C3
        if "C2" in arm_maps and "C3" in arm_maps:
            results["C2_vs_C3"] = self._paired_test(arm_maps["C3"], arm_maps["C2"], "C3", "C2")

        # Secondary comparisons: C2 vs B0, G2, C1, C4
        secondaries = [("C2_vs_B0", "B0"), ("C2_vs_G2", "G2"), ("C2_vs_C1", "C1"), ("C2_vs_C4", "C4")]
        sec_tests = {}
        for comp_name, base_arm in secondaries:
            if "C2" in arm_maps and base_arm in arm_maps:
                sec_tests[comp_name] = self._paired_test(arm_maps[base_arm], arm_maps["C2"], base_arm, "C2")

        # Apply Holm-Bonferroni correction on secondary McNemar p-values
        sorted_sec = sorted(sec_tests.items(), key=lambda kv: kv[1]["mcnemar_pvalue"])
        m = len(sorted_sec)
        alpha = 0.05
        for rank, (k, v) in enumerate(sorted_sec):
            p = v["mcnemar_pvalue"]
            thresh = alpha / (m - rank)
            v["holm_bonferroni"] = {
                "raw_pvalue": p,
                "rank": rank + 1,
                "alpha_threshold": thresh,
                "is_significant_holm": bool(p <= thresh),
            }
            results[k] = v

        return results

    def _paired_test(
        self,
        map_a: Dict[str, M20dPredictionRecord],
        map_b: Dict[str, M20dPredictionRecord],
        arm_a_name: str,
        arm_b_name: str,
    ) -> Dict[str, Any]:
        common_ids = sorted(list(set(map_a.keys()).intersection(set(map_b.keys()))))
        both_correct = 0
        both_wrong = 0
        corrected = 0  # B correct, A wrong
        harmed = 0     # A correct, B wrong

        gold_labels = []
        preds_a = []
        preds_b = []

        for sid in common_ids:
            rec_a = map_a[sid]
            rec_b = map_b[sid]
            gold = rec_a.gold_label
            p_a = rec_a.parsed_verdict
            p_b = rec_b.parsed_verdict

            gold_labels.append(gold)
            preds_a.append(p_a)
            preds_b.append(p_b)

            correct_a = (p_a == gold and gold is not None)
            correct_b = (p_b == gold and gold is not None)

            if correct_a and correct_b:
                both_correct += 1
            elif not correct_a and not correct_b:
                both_wrong += 1
            elif correct_b and not correct_a:
                corrected += 1
            elif correct_a and not correct_b:
                harmed += 1

        net_benefit = corrected - harmed

        # Exact McNemar
        b = corrected
        c = harmed
        n = b + c
        if n == 0:
            mcnemar_p = 1.0
        else:
            k = min(b, c)
            binom_sum = sum(math.comb(n, i) for i in range(k + 1))
            mcnemar_p = min(1.0, 2.0 * binom_sum * (0.5 ** n))

        odds_ratio = (b / c) if c > 0 else (float("inf") if b > 0 else 1.0)

        # Paired Bootstrap for Macro-F1 and Accuracy differences
        f1_diffs, acc_diffs = self._paired_bootstrap(gold_labels, preds_a, preds_b)
        f1_ci = (float(np.percentile(f1_diffs, 2.5)), float(np.percentile(f1_diffs, 97.5)))
        acc_ci = (float(np.percentile(acc_diffs, 2.5)), float(np.percentile(acc_diffs, 97.5)))

        return {
            "arm_a": arm_a_name,
            "arm_b": arm_b_name,
            "transitions": {
                "arm_a": arm_a_name,
                "arm_b": arm_b_name,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "corrected": corrected,
                "harmed": harmed,
                "net_benefit": net_benefit,
                "mcnemar_pvalue": round(mcnemar_p, 5),
                "odds_ratio": round(odds_ratio, 3) if odds_ratio != float("inf") else 999.0,
                "total_claims": len(common_ids),
            },
            "mcnemar_pvalue": round(mcnemar_p, 5),
            "odds_ratio": round(odds_ratio, 3) if odds_ratio != float("inf") else 999.0,
            "macro_f1_diff_mean": float(np.mean(f1_diffs)),
            "macro_f1_diff_ci": [round(f1_ci[0], 4), round(f1_ci[1], 4)],
            "acc_diff_mean": float(np.mean(acc_diffs)),
            "acc_diff_ci": [round(acc_ci[0], 4), round(acc_ci[1], 4)],
        }

    def _paired_bootstrap(
        self,
        gold_labels: List[Optional[str]],
        preds_a: List[Optional[str]],
        preds_b: List[Optional[str]],
    ) -> Tuple[List[float], List[float]]:
        n = len(gold_labels)
        rng = np.random.default_rng(self.seed)
        f1_diffs = []
        acc_diffs = []

        for _ in range(self.n_bootstraps):
            indices = rng.integers(0, n, size=n)
            sub_gold = [gold_labels[i] for i in indices]
            sub_a = [preds_a[i] for i in indices]
            sub_b = [preds_b[i] for i in indices]

            acc_a = sum(1 for i in range(n) if sub_a[i] == sub_gold[i] and sub_gold[i] is not None) / n
            acc_b = sum(1 for i in range(n) if sub_b[i] == sub_gold[i] and sub_gold[i] is not None) / n
            acc_diffs.append(acc_b - acc_a)

            f1_a = self._calc_macro_f1(sub_gold, sub_a)
            f1_b = self._calc_macro_f1(sub_gold, sub_b)
            f1_diffs.append(f1_b - f1_a)

        return f1_diffs, acc_diffs

    def _calc_macro_f1(self, gold: List[Optional[str]], pred: List[Optional[str]]) -> float:
        f1s = []
        for c in ALL_CLASSES:
            tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
            fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
            fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
        return sum(f1s) / len(f1s) if f1s else 0.0

    def _run_subgroup_diagnostics(
        self,
        arms_records: Dict[str, List[M20dPredictionRecord]],
    ) -> Dict[str, Any]:
        """Analyzes performance split by pattern selection activation."""
        c2_recs = arms_records.get("C2", [])
        if not c2_recs:
            return {}

        selected_sids = {r.sample_id for r in c2_recs if r.pattern_decision == "PATTERN_SELECTED"}
        fallback_sids = {r.sample_id for r in c2_recs if r.pattern_decision == "NO_PATTERN_SELECTED"}

        subgroups = {
            "PATTERN_SELECTED": {"count": len(selected_sids), "arms": {}},
            "NO_PATTERN_SELECTED": {"count": len(fallback_sids), "arms": {}},
        }

        for arm, recs in arms_records.items():
            sel_recs = [r for r in recs if r.sample_id in selected_sids]
            fall_recs = [r for r in recs if r.sample_id in fallback_sids]

            m_sel, _, _ = self._compute_arm_metrics(sel_recs)
            m_fall, _, _ = self._compute_arm_metrics(fall_recs)

            subgroups["PATTERN_SELECTED"]["arms"][arm] = m_sel
            subgroups["NO_PATTERN_SELECTED"]["arms"][arm] = m_fall

        return subgroups

    def _determine_decision(
        self,
        stat_results: Dict[str, Any],
        metrics_by_arm: Dict[str, Dict[str, Any]],
    ) -> Tuple[str, str]:
        """Pre-registered Decision Logic."""
        c2_c3 = stat_results.get("C2_vs_C3")
        if not c2_c3:
            return "INCONCLUSIVE", "Missing primary comparison between C2 and C3."

        trans = c2_c3.get("transitions", {})
        net_benefit = trans.get("net_benefit", 0)
        f1_ci = c2_c3.get("macro_f1_diff_ci", [0.0, 0.0])
        p_val = c2_c3.get("mcnemar_pvalue", 1.0)

        # GO: Reliable positive benefit over matched control
        if net_benefit > 0 and (f1_ci[0] > 0.0 or (p_val < 0.05 and net_benefit >= 5)):
            return "GO", f"C2 demonstrated statistically reliable superiority over matched control C3 (Net Benefit: +{net_benefit}, 95% CI: {f1_ci})."

        # NO-GO: Net zero or negative benefit, or negative CI
        if net_benefit <= 0:
            return "NO-GO", f"C2 produced net negative or zero claim transitions over matched generic control C3 (Net Benefit: {net_benefit}, McNemar p={p_val})."

        # INCONCLUSIVE: Positive net benefit but broad CI spanning zero
        if net_benefit > 0 and f1_ci[0] <= 0.0:
            return "INCONCLUSIVE", (
                f"C2 showed positive transition direction (Net Benefit: +{net_benefit}), "
                f"but Macro-F1 95% bootstrap CI spans zero ({f1_ci}) and McNemar p={p_val} is not statistically significant."
            )

        return "NO-GO", "No evidence of incremental benefit from historical error patterns."

    def _save_csvs(
        self,
        metrics_by_arm: Dict[str, Dict[str, Any]],
        per_class_by_arm: Dict[str, Dict[str, Dict[str, float]]],
        stat_results: Dict[str, Any],
    ) -> None:
        # 1. metrics_by_arm.csv
        metrics_csv = self.output_dir / "metrics_by_arm.csv"
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["arm", "macro_f1", "accuracy", "balanced_accuracy", "valid_verdict_rate", "attribution_valid_rate", "total_claims", "correct_count", "unparseable_count"])
            for arm, m in metrics_by_arm.items():
                writer.writerow([
                    arm,
                    f"{m['macro_f1']:.4f}",
                    f"{m['accuracy']:.4f}",
                    f"{m['balanced_accuracy']:.4f}",
                    f"{m['valid_verdict_rate']:.4f}",
                    f"{m['attribution_valid_rate']:.4f}",
                    m["total_claims"],
                    m["correct_count"],
                    m["unparseable_count"],
                ])

        # 2. per_class_metrics.csv
        per_class_csv = self.output_dir / "per_class_metrics.csv"
        with open(per_class_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["arm", "class", "precision", "recall", "f1", "support"])
            for arm, classes in per_class_by_arm.items():
                for c, m in classes.items():
                    writer.writerow([arm, c, f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}", m["support"]])

        # 3. paired_transitions.csv
        trans_csv = self.output_dir / "paired_transitions.csv"
        with open(trans_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["comparison", "arm_a", "arm_b", "both_correct", "both_wrong", "corrected", "harmed", "net_benefit", "mcnemar_pvalue", "odds_ratio"])
            for comp, res in stat_results.items():
                t = res.get("transitions", {})
                writer.writerow([
                    comp,
                    t.get("arm_a"),
                    t.get("arm_b"),
                    t.get("both_correct"),
                    t.get("both_wrong"),
                    t.get("corrected"),
                    t.get("harmed"),
                    t.get("net_benefit"),
                    f"{t.get('mcnemar_pvalue', 1.0):.5f}",
                    f"{t.get('odds_ratio', 1.0):.3f}",
                ])

    def _save_jsons(
        self,
        confusion_by_arm: Dict[str, Dict[str, Dict[str, int]]],
        stat_results: Dict[str, Any],
    ) -> None:
        with open(self.output_dir / "confusion_matrices.json", "w", encoding="utf-8") as f:
            json.dump(confusion_by_arm, f, indent=2)

        with open(self.output_dir / "statistical_tests.json", "w", encoding="utf-8") as f:
            json.dump(stat_results, f, indent=2)

    def _save_markdown_reports(
        self,
        metrics_by_arm: Dict[str, Dict[str, Any]],
        stat_results: Dict[str, Any],
        subgroups: Dict[str, Any],
        decision: str,
        rationale: str,
    ) -> None:
        # DECISION_REPORT.md
        dec_path = self.output_dir / "DECISION_REPORT.md"
        with open(dec_path, "w", encoding="utf-8") as f:
            f.write(f"""# Milestone M2.0d: Decision Report

## Empirical Determination: **{decision}**

**Decision Rationale:** {rationale}

---

## 1. Primary Empirical Results across 6 Arms on Benchmark Evaluation Subset

| Arm | Description | Macro-F1 | Accuracy | Balanced Acc | Correct / Total | Valid Rate | Attr Valid |
|:---|:---|---:|---:|---:|---:|---:|---:|
""")
            for arm, m in metrics_by_arm.items():
                f.write(f"| **{arm}** | {arm} | **{m['macro_f1']:.4f}** | **{m['accuracy']*100:.2f}%** | {m['balanced_accuracy']:.4f} | {m['correct_count']}/{m['total_claims']} | {m['valid_verdict_rate']*100:.1f}% | {m['attribution_valid_rate']*100:.1f}% |\n")

            f.write("\n## 2. Statistical Comparison Battery\n\n")
            c2_c3 = stat_results.get("C2_vs_C3", {})
            t = c2_c3.get("transitions", {})
            f.write(f"""### Primary Hypothesis: C2 (Grounded Error Contrastive) vs C3 (Matched Generic Control)

- **Net Claim Benefit:** **{t.get('net_benefit', 0)} claims** (Corrected: {t.get('corrected', 0)}, Harmed: {t.get('harmed', 0)})
- **Exact McNemar p-value (Accuracy test):** {t.get('mcnemar_pvalue', 1.0):.5f} (Odds Ratio: {t.get('odds_ratio', 1.0):.3f})
- **Macro-F1 Difference:** {c2_c3.get('macro_f1_diff_mean', 0.0):+.4f} (95% Bootstrap CI: {c2_c3.get('macro_f1_diff_ci')})
- **Accuracy Difference:** {c2_c3.get('acc_diff_mean', 0.0):+.4f} (95% Bootstrap CI: {c2_c3.get('acc_diff_ci')})
- *Note:* McNemar tests paired correctness (Accuracy); Macro-F1 difference is tested via paired bootstrapping. A confidence interval spanning zero denotes insufficient statistical power to separate the arms, not proven equivalence.

---

## 3. Subgroup Diagnostics: PATTERN_SELECTED vs NO_PATTERN_SELECTED
""")
            if subgroups:
                sel = subgroups.get("PATTERN_SELECTED", {})
                fall = subgroups.get("NO_PATTERN_SELECTED", {})
                f.write(f"- **PATTERN_SELECTED Triggered:** {sel.get('count', 0)}\n")
                f.write(f"- **NO_PATTERN_SELECTED Fallback:** {fall.get('count', 0)}\n\n")
                f.write("| Subgroup | Arm | Accuracy | Macro-F1 | Sample Count |\n|:---|:---|---:|---:|---:|\n")
                for arm, m in sel.get("arms", {}).items():
                    f.write(f"| **PATTERN_SELECTED** | {arm} | {m['accuracy']*100:.2f}% | {m['macro_f1']:.4f} | {sel.get('count', 0)} |\n")
                for arm, m in fall.get("arms", {}).items():
                    f.write(f"| **NO_PATTERN_SELECTED** | {arm} | {m['accuracy']*100:.2f}% | {m['macro_f1']:.4f} | {fall.get('count', 0)} |\n")

        # FINAL_REPORT.md
        final_path = self.output_dir / "FINAL_REPORT.md"
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(f"""# Milestone M2.0d: Final Evaluation Report

- **Milestone:** M2.0d Contrastive Evidence Verification (Track C)
- **Status:** Complete
- **Pre-registered Determination:** {decision}
- **Primary Finding:** {rationale}

Deliverables sealed in: `{self.output_dir}`
""")
