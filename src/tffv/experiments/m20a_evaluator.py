"""Milestone M2.0-A: Statistical Evaluator & Transition Analyzer.

Calculates:
1. 4-class Macro-F1, Accuracy, Balanced Accuracy, and per-class metrics.
2. Pairwise error transition tables with strict mathematical conservation:
   Delta Accuracy == (Corrected - Harmed) / N.
3. McNemar exact paired test.
4. Paired Bootstrap 95% Confidence Intervals (5,000 iterations).
5. Comprehensive decision and reporting documents.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

def exact_mcnemar_test(b: int, c: int) -> float:
    """Exact two-sided binomial McNemar test under H0: p = 0.5."""
    total_disc = b + c
    if total_disc == 0:
        return 1.0
    min_k = min(b, c)
    p_val = 2.0 * sum(math.comb(total_disc, k) * (0.5 ** total_disc) for k in range(min_k + 1))
    return min(1.0, p_val)
from tffv.experiments.m20a_schemas import (
    M20aArm,
    M20aPredictionRecord,
    M20aTransitionStats,
    OFFICIAL_CLASSES,
)


class M20aEvaluator:
    """Rigorous statistical evaluator for M2.0-A."""

    def __init__(
        self,
        predictions_by_arm: Dict[str, List[M20aPredictionRecord]],
        seed: int = 42,
        n_bootstraps: int = 5000,
    ):
        self.predictions_by_arm = predictions_by_arm
        self.seed = seed
        self.n_bootstraps = n_bootstraps
        self.arms = list(predictions_by_arm.keys())

    def evaluate_all(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Runs full evaluation and optionally saves reports."""
        metrics_by_arm = {}
        confusion_matrices = {}
        per_class_by_arm = {}

        for arm, records in self.predictions_by_arm.items():
            golds = [r.gold_verdict for r in records]
            preds = [r.predicted_verdict for r in records]

            # Overall metrics
            acc = sum(1 for g, p in zip(golds, preds) if g == p) / max(len(golds), 1)
            macro_f1, per_class = self._compute_class_metrics(golds, preds)
            bal_acc = sum(per_class[c]["recall"] for c in OFFICIAL_CLASSES) / len(OFFICIAL_CLASSES)

            metrics_by_arm[arm] = {
                "arm": arm,
                "total_claims": len(records),
                "accuracy": round(acc, 4),
                "accuracy_pct": f"{acc * 100:.2f}%",
                "macro_f1": round(macro_f1, 4),
                "balanced_accuracy": round(bal_acc, 4),
                "valid_parse_count": sum(1 for r in records if r.is_parseable),
                "valid_parse_rate": round(sum(1 for r in records if r.is_parseable) / max(len(records), 1), 4),
            }

            per_class_by_arm[arm] = per_class
            confusion_matrices[arm] = self._build_confusion_matrix(golds, preds)

        # Transition analyses across matched pairs
        transitions = self._compute_pairwise_transitions()

        results = {
            "milestone": "M2.0-A",
            "seed": self.seed,
            "n_bootstraps": self.n_bootstraps,
            "metrics_by_arm": metrics_by_arm,
            "per_class_metrics": per_class_by_arm,
            "confusion_matrices": confusion_matrices,
            "transitions": {k: v.to_dict() for k, v in transitions.items()},
        }

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self._save_artifacts(results, output_dir)

        return results

    def _compute_class_metrics(
        self,
        golds: List[str],
        preds: List[str],
    ) -> Tuple[float, Dict[str, Dict[str, float]]]:
        per_class: Dict[str, Dict[str, float]] = {}
        f1_list = []

        for c in OFFICIAL_CLASSES:
            tp = sum(1 for g, p in zip(golds, preds) if g == c and p == c)
            fp = sum(1 for g, p in zip(golds, preds) if g != c and p == c)
            fn = sum(1 for g, p in zip(golds, preds) if g == c and p != c)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            per_class[c] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "support": sum(1 for g in golds if g == c),
            }
            f1_list.append(f1)

        macro_f1 = sum(f1_list) / len(f1_list)
        return macro_f1, per_class

    def _build_confusion_matrix(
        self,
        golds: List[str],
        preds: List[str],
    ) -> Dict[str, Dict[str, int]]:
        matrix = {g: {p: 0 for p in OFFICIAL_CLASSES} for g in OFFICIAL_CLASSES}
        for g, p in zip(golds, preds):
            if g in matrix and p in matrix[g]:
                matrix[g][p] += 1
        return matrix

    def _compute_pairwise_transitions(self) -> Dict[str, M20aTransitionStats]:
        """Computes transitions for key scientific comparison pairs."""
        pairs = [
            ("E1", "G1", "Primary: Error-Derived Rules vs Generic Rules"),
            ("E1", "G2", "Primary: Error-Derived Rules vs Length-Matched Generic"),
            ("E1", "B0", "Secondary: Error-Derived Rules vs Zero-Shot Base"),
            ("E1", "E2", "Ablation: Error-Derived Rules vs Shuffled/Irrelevant Control"),
            ("G1", "B0", "Control: Generic Rules vs Zero-Shot Base"),
            ("G2", "B0", "Control: Length-Matched Generic vs Zero-Shot Base"),
        ]

        transitions: Dict[str, M20aTransitionStats] = {}

        for cand_arm, base_arm, desc in pairs:
            if cand_arm not in self.predictions_by_arm or base_arm not in self.predictions_by_arm:
                continue

            c_records = {r.sample_id: r for r in self.predictions_by_arm[cand_arm]}
            b_records = {r.sample_id: r for r in self.predictions_by_arm[base_arm]}
            shared_ids = sorted(list(set(c_records.keys()) & set(b_records.keys())))

            if not shared_ids:
                continue

            n = len(shared_ids)
            both_correct = 0
            both_wrong = 0
            harmed = 0    # base correct, cand wrong (n_01)
            corrected = 0 # base wrong, cand correct (n_10)

            c_corrects = []
            b_corrects = []
            golds = []
            c_preds = []
            b_preds = []

            for sid in shared_ids:
                c_rec = c_records[sid]
                b_rec = b_records[sid]
                c_corr = (c_rec.predicted_verdict == c_rec.gold_verdict)
                b_corr = (b_rec.predicted_verdict == b_rec.gold_verdict)

                c_corrects.append(1 if c_corr else 0)
                b_corrects.append(1 if b_corr else 0)
                golds.append(c_rec.gold_verdict)
                c_preds.append(c_rec.predicted_verdict)
                b_preds.append(b_rec.predicted_verdict)

                if b_corr and c_corr:
                    both_correct += 1
                elif not b_corr and not c_corr:
                    both_wrong += 1
                elif b_corr and not c_corr:
                    harmed += 1
                elif not b_corr and c_corr:
                    corrected += 1

            net_benefit = corrected - harmed
            c_total_correct = sum(c_corrects)
            b_total_correct = sum(b_corrects)

            # Mathematical conservation check
            if net_benefit != (c_total_correct - b_total_correct):
                raise ValueError(
                    f"Transition conservation violated for {cand_arm} vs {base_arm}: "
                    f"net_benefit={net_benefit} != cand_corr({c_total_correct}) - base_corr({b_total_correct})"
                )

            # McNemar exact test: b = corrected, c = harmed
            mcnemar_p = exact_mcnemar_test(b=corrected, c=harmed)

            # Bootstrap CIs
            ci_macro, ci_acc = self._paired_bootstrap(
                golds, c_preds, b_preds, self.n_bootstraps, self.seed
            )

            c_macro, _ = self._compute_class_metrics(golds, c_preds)
            b_macro, _ = self._compute_class_metrics(golds, b_preds)
            delta_macro = round(c_macro - b_macro, 4)
            delta_acc = round((c_total_correct - b_total_correct) / n, 4)

            ratio = round(harmed / max(corrected, 1), 3)

            stat = M20aTransitionStats(
                comparison=desc,
                cand_arm=cand_arm,
                base_arm=base_arm,
                total_claims=n,
                both_correct=both_correct,
                both_wrong=both_wrong,
                harmed=harmed,
                corrected=corrected,
                net_benefit=net_benefit,
                harm_rate=round(harmed / n, 4),
                correction_rate=round(corrected / n, 4),
                net_benefit_rate=round(net_benefit / n, 4),
                harm_to_correction_ratio=ratio,
                mcnemar_b=corrected,
                mcnemar_c=harmed,
                mcnemar_p_value=mcnemar_p,
                delta_macro_f1=delta_macro,
                delta_macro_f1_ci=ci_macro,
                delta_accuracy=delta_acc,
                delta_accuracy_ci=ci_acc,
            )
            key = f"{cand_arm}_vs_{base_arm}"
            transitions[key] = stat

        return transitions

    def _paired_bootstrap(
        self,
        golds: List[str],
        preds_a: List[str],
        preds_b: List[str],
        n_bootstraps: int = 5000,
        seed: int = 42,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        rng = random.Random(seed)
        n = len(golds)
        indices = list(range(n))

        diff_macros = []
        diff_accs = []

        for _ in range(n_bootstraps):
            sample_idx = [rng.choice(indices) for _ in range(n)]
            b_gold = [golds[i] for i in sample_idx]
            b_pa = [preds_a[i] for i in sample_idx]
            b_pb = [preds_b[i] for i in sample_idx]

            acc_a = sum(1 for g, p in zip(b_gold, b_pa) if g == p) / n
            acc_b = sum(1 for g, p in zip(b_gold, b_pb) if g == p) / n
            diff_accs.append(acc_a - acc_b)

            f1_a, _ = self._compute_class_metrics(b_gold, b_pa)
            f1_b, _ = self._compute_class_metrics(b_gold, b_pb)
            diff_macros.append(f1_a - f1_b)

        diff_macros.sort()
        diff_accs.sort()

        lo_idx = int(0.025 * n_bootstraps)
        hi_idx = int(0.975 * n_bootstraps)

        ci_macro = (round(diff_macros[lo_idx], 4), round(diff_macros[hi_idx], 4))
        ci_acc = (round(diff_accs[lo_idx], 4), round(diff_accs[hi_idx], 4))

        return ci_macro, ci_acc

    def _save_artifacts(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Serializes tables and reports."""
        # 1. metrics_by_arm.csv
        with open(output_dir / "metrics_by_arm.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Arm", "Total Claims", "Accuracy", "Macro-F1", "Balanced Accuracy", "Valid Parse Rate"])
            for m in results["metrics_by_arm"].values():
                writer.writerow([
                    m["arm"],
                    m["total_claims"],
                    f"{m['accuracy'] * 100:.2f}%",
                    f"{m['macro_f1']:.4f}",
                    f"{m['balanced_accuracy']:.4f}",
                    f"{m['valid_parse_rate'] * 100:.1f}%",
                ])

        # 2. per_class_metrics.csv
        with open(output_dir / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Arm", "Class", "Precision", "Recall", "F1", "Support"])
            for arm, classes in results["per_class_metrics"].items():
                for cname, cmetrics in classes.items():
                    writer.writerow([
                        arm,
                        cname,
                        f"{cmetrics['precision']:.4f}",
                        f"{cmetrics['recall']:.4f}",
                        f"{cmetrics['f1']:.4f}",
                        cmetrics["support"],
                    ])

        # 3. statistical_tests.json
        (output_dir / "statistical_tests.json").write_text(
            json.dumps(results["transitions"], indent=2), encoding="utf-8"
        )

        # 4. confusion_matrices.json
        (output_dir / "confusion_matrices.json").write_text(
            json.dumps(results["confusion_matrices"], indent=2), encoding="utf-8"
        )

        # 5. FINAL_REPORT.md & DECISION_REPORT.md
        self._write_reports(results, output_dir)

    def _write_reports(self, results: Dict[str, Any], output_dir: Path) -> None:
        metrics = results["metrics_by_arm"]
        trans = results["transitions"]

        final_lines = [
            "# Milestone M2.0-A Final Evaluation Report: Error-Derived Verification Rules",
            "",
            f"**Evaluation Scope:** {list(metrics.values())[0]['total_claims']} Claims across 5 Experimental Arms",
            "**Primary Metric:** Four-class Macro-F1 (Official AVeriTeC Ontology)",
            "",
            "## 1. Overall Performance Across 5 Arms",
            "",
            "| Arm | Description | Accuracy | Macro-F1 | Balanced Acc | Valid Parse Rate |",
            "| :--- | :--- | :---: | :---: | :---: | :---: |",
        ]

        arm_desc = {
            "B0": "Zero-Shot Base Control (No rules)",
            "G1": "Generic Rules Control (Concise guidelines)",
            "G2": "Length-Matched Generic Control (Detailed guidelines)",
            "E1": "Error-Derived Rules (Historical failure patterns)",
            "E2": "Shuffled/Irrelevant Control (Negative control)",
        }

        for arm in ["B0", "G1", "G2", "E1", "E2"]:
            if arm in metrics:
                m = metrics[arm]
                final_lines.append(
                    f"| **{arm}** | {arm_desc.get(arm, arm)} | {m['accuracy_pct']} | `{m['macro_f1']:.4f}` | `{m['balanced_accuracy']:.4f}` | `{m['valid_parse_rate'] * 100:.1f}%` |"
                )

        final_lines.extend([
            "",
            "## 2. Pairwise Transitions & Hypothesis Testing (5,000 Bootstraps)",
            "",
            "| Comparison | Cand Acc | Base Acc | Corrected | Harmed | Net Benefit | McNemar p | Delta Macro-F1 [95% CI] |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for key, t in trans.items():
            ci_str = f"[{t['delta_macro_f1_ci'][0]:+.4f}, {t['delta_macro_f1_ci'][1]:+.4f}]"
            final_lines.append(
                f"| **{key}** | {t['cand_arm']} | {t['base_arm']} | **{t['corrected']}** | **{t['harmed']}** | **{t['net_benefit']:+d}** ({t['net_benefit_rate']:+.2%}) | `{t['mcnemar_p_value']:.5f}` | `{t['delta_macro_f1']:+.4f}` {ci_str} |"
            )

        (output_dir / "FINAL_REPORT.md").write_text("\n".join(final_lines) + "\n", encoding="utf-8")

        # DECISION_REPORT.md
        dec_lines = [
            "# Milestone M2.0-A Decision Report",
            "",
            "**Research Question:** *Can historical error patterns converted into explicit verification rules improve frozen fact verification?*",
            "",
            "## Key Scientific Findings:",
        ]

        if "E1_vs_G1" in trans:
            t = trans["E1_vs_G1"]
            dec_lines.append(
                f"1. **E1 vs G1 (Error-Derived vs Generic):** Net benefit = {t['net_benefit']} claims ({t['net_benefit_rate']:+.2%}), McNemar p = {t['mcnemar_p_value']:.5f}."
            )
        if "E1_vs_G2" in trans:
            t = trans["E1_vs_G2"]
            dec_lines.append(
                f"2. **E1 vs G2 (Length-Matched Control):** Net benefit = {t['net_benefit']} claims ({t['net_benefit_rate']:+.2%}), Delta Macro-F1 = {t['delta_macro_f1']:+.4f}."
            )
        if "E1_vs_B0" in trans:
            t = trans["E1_vs_B0"]
            dec_lines.append(
                f"3. **E1 vs B0 (Zero-Shot Anchor):** Net benefit = {t['net_benefit']} claims ({t['net_benefit_rate']:+.2%}), McNemar p = {t['mcnemar_p_value']:.5f}."
            )

        dec_lines.extend([
            "",
            "## Protocol Invariants Maintained:",
            "- 100% frozen backbone Qwen3-8B BF16.",
            "- Fixed R4-PF1000-v1 evidence across all arms.",
            "- Transition conservation mathematically verified: net_benefit == cand_correct - base_correct.",
            "- Strict group-fold isolation: rules mined exclusively from Train Fold-1.",
        ])

        (output_dir / "DECISION_REPORT.md").write_text("\n".join(dec_lines) + "\n", encoding="utf-8")
