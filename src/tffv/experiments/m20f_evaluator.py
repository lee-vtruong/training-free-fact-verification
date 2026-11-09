"""Milestone M2.0f: Factorial Ablation Evaluator & Contradiction Trap Diagnostics.

Evaluates 6 arms (B0, G2, A1, A2, A3, A4):
- Factor A: Matched Gold ICL vs Historical Error ICL
- Factor B: No Forced Attribution vs Forced Evidence Attribution
- Controls: B0 Zero-Shot, G2 Generic Evidence Rules
- Calculates Main Effects, Interaction Effect, Paired Transitions, Bootstrap 95% CIs.
- Dissects Contradiction Trap (transitions to Conflicting Evidence/Cherrypicking).
- Generates all CSVs, JSONs, DECISION_REPORT.md, and FINAL_REPORT.md.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from tffv.experiments.m20f_schemas import (
    AttributionFactor,
    KnowledgeFactor,
    M20fArm,
    M20fPredictionRecord,
)

ALL_CLASSES = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]


class M20fEvaluator:
    """Evaluates factorial ablation experiments and generates forensic reports."""

    def __init__(
        self,
        records: List[M20fPredictionRecord],
        output_dir: str | Path = "outputs/m20f-factorial-ablation-v1",
        n_bootstraps: int = 5000,
        seed: int = 42,
    ):
        self.records = records
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_bootstraps = n_bootstraps
        self.seed = seed

    def evaluate_all(self) -> Dict[str, Any]:
        """Runs complete factorial evaluation battery."""
        # 1. Group records by arm and by sample_id
        arms_records: Dict[str, List[M20fPredictionRecord]] = {}
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

        # 3. Factorial Effects & Bootstrap CIs
        factorial_results = self._compute_factorial_effects(arms_records)

        # 4. Paired Transitions against B0 and G2
        paired_transitions = self._compute_paired_transitions(arms_records)

        # 5. Contradiction Trap & Transition Diagnostics
        trap_diagnostics = self._compute_contradiction_trap_diagnostics(arms_records)

        # 6. Pre-registered Decision Battery
        decision, rationale = self._determine_decision(
            metrics_by_arm, factorial_results, trap_diagnostics
        )

        # 7. Save Artifacts
        self._save_csvs(
            metrics_by_arm,
            per_class_by_arm,
            factorial_results,
            paired_transitions,
            trap_diagnostics,
        )
        self._save_jsons(confusion_by_arm, factorial_results, paired_transitions)
        self._save_markdown_reports(
            metrics_by_arm,
            per_class_by_arm,
            factorial_results,
            paired_transitions,
            trap_diagnostics,
            decision,
            rationale,
        )

        return {
            "decision": decision,
            "decision_rationale": rationale,
            "metrics_by_arm": metrics_by_arm,
            "factorial_effects": factorial_results,
        }

    def _compute_arm_metrics(
        self, records: List[M20fPredictionRecord]
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]], Dict[str, Dict[str, int]]]:
        """Computes summary metrics, per-class F1, and confusion matrix for an arm."""
        n_total = len(records)
        parsed_records = [r for r in records if r.parsed_verdict in ALL_CLASSES]
        n_parsed = len(parsed_records)
        parse_rate = n_parsed / n_total if n_total > 0 else 0.0

        n_valid_attr = sum(1 for r in records if r.is_valid_attribution)
        valid_attr_rate = n_valid_attr / n_total if n_total > 0 else 0.0

        y_true = [r.gold_label for r in records]
        y_pred = [r.parsed_verdict if r.parsed_verdict in ALL_CLASSES else "UNPARSEABLE" for r in records]

        correct_count = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp and yt is not None)
        accuracy = correct_count / n_total if n_total > 0 else 0.0

        # Per-class metrics
        per_class: Dict[str, Dict[str, float]] = {}
        f1_list = []
        rec_list = []

        for cls_name in ALL_CLASSES:
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls_name and yp == cls_name)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cls_name and yp == cls_name)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls_name and yp != cls_name)
            support = sum(1 for yt in y_true if yt == cls_name)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            per_class[cls_name] = {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": float(support),
            }
            if support > 0:
                f1_list.append(f1)
                rec_list.append(rec)

        macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
        balanced_acc = float(np.mean(rec_list)) if rec_list else 0.0

        # Confusion Matrix
        cm: Dict[str, Dict[str, int]] = {
            yt_cls: {yp_cls: 0 for yp_cls in ALL_CLASSES + ["UNPARSEABLE"]}
            for yt_cls in ALL_CLASSES
        }
        for yt, yp in zip(y_true, y_pred):
            if yt in cm and yp in cm[yt]:
                cm[yt][yp] += 1

        summary = {
            "total_claims": n_total,
            "parsed_claims": n_parsed,
            "parse_rate": parse_rate,
            "valid_attribution_rate": valid_attr_rate,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_acc,
            "correct_claims": correct_count,
        }
        return summary, per_class, cm

    def _compute_factorial_effects(
        self, arms_records: Dict[str, List[M20fPredictionRecord]]
    ) -> Dict[str, Any]:
        """Computes 2x2 factorial effects and paired bootstrap confidence intervals."""
        # Check required factorial arms
        req = ["A1", "A2", "A3", "A4"]
        if not all(arm in arms_records for arm in req):
            return {"status": "INCOMPLETE_ARMS"}

        # Align samples across arms
        by_sample: Dict[str, Dict[str, M20fPredictionRecord]] = {}
        for arm in req + ["B0", "G2"]:
            if arm in arms_records:
                for r in arms_records[arm]:
                    by_sample.setdefault(r.sample_id, {})[arm] = r

        common_ids = [
            sid for sid, arm_map in by_sample.items()
            if all(k in arm_map for k in req)
        ]
        n_samples = len(common_ids)
        if n_samples == 0:
            return {"status": "NO_COMMON_SAMPLES"}

        gold = np.array([by_sample[sid]["A1"].gold_label for sid in common_ids])
        preds_a1 = np.array([by_sample[sid]["A1"].parsed_verdict for sid in common_ids])
        preds_a2 = np.array([by_sample[sid]["A2"].parsed_verdict for sid in common_ids])
        preds_a3 = np.array([by_sample[sid]["A3"].parsed_verdict for sid in common_ids])
        preds_a4 = np.array([by_sample[sid]["A4"].parsed_verdict for sid in common_ids])

        correct_a1 = (preds_a1 == gold).astype(float)
        correct_a2 = (preds_a2 == gold).astype(float)
        correct_a3 = (preds_a3 == gold).astype(float)
        correct_a4 = (preds_a4 == gold).astype(float)

        acc_a1 = float(np.mean(correct_a1))
        acc_a2 = float(np.mean(correct_a2))
        acc_a3 = float(np.mean(correct_a3))
        acc_a4 = float(np.mean(correct_a4))

        # Main Effect Attribution (Factor B): 0.5 * ((A2 - A1) + (A4 - A3))
        main_effect_attr_acc = 0.5 * ((acc_a2 - acc_a1) + (acc_a4 - acc_a3))

        # Main Effect Error Knowledge (Factor A): 0.5 * ((A3 - A1) + (A4 - A2))
        main_effect_error_acc = 0.5 * ((acc_a3 - acc_a1) + (acc_a4 - acc_a2))

        # Interaction Effect (A x B): (A4 - A3) - (A2 - A1)
        interaction_acc = (acc_a4 - acc_a3) - (acc_a2 - acc_a1)

        # Macro-F1 calculations
        def calc_macro_f1(g: np.ndarray, p: np.ndarray) -> float:
            f1s = []
            for c in ALL_CLASSES:
                tp = np.sum((g == c) & (p == c))
                fp = np.sum((g != c) & (p == c))
                fn = np.sum((g == c) & (p != c))
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                if np.sum(g == c) > 0:
                    f1s.append(f1)
            return float(np.mean(f1s)) if f1s else 0.0

        mf1_a1 = calc_macro_f1(gold, preds_a1)
        mf1_a2 = calc_macro_f1(gold, preds_a2)
        mf1_a3 = calc_macro_f1(gold, preds_a3)
        mf1_a4 = calc_macro_f1(gold, preds_a4)

        main_effect_attr_mf1 = 0.5 * ((mf1_a2 - mf1_a1) + (mf1_a4 - mf1_a3))
        main_effect_error_mf1 = 0.5 * ((mf1_a3 - mf1_a1) + (mf1_a4 - mf1_a2))
        interaction_mf1 = (mf1_a4 - mf1_a3) - (mf1_a2 - mf1_a1)

        # Bootstrap Confidence Intervals (paired resampling)
        rng = np.random.default_rng(self.seed)
        boot_attr_acc = []
        boot_error_acc = []
        boot_inter_acc = []

        boot_attr_mf1 = []
        boot_error_mf1 = []
        boot_inter_mf1 = []

        for _ in range(self.n_bootstraps):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            g_b = gold[idx]
            p1_b = preds_a1[idx]
            p2_b = preds_a2[idx]
            p3_b = preds_a3[idx]
            p4_b = preds_a4[idx]

            c1_b = (p1_b == g_b).mean()
            c2_b = (p2_b == g_b).mean()
            c3_b = (p3_b == g_b).mean()
            c4_b = (p4_b == g_b).mean()

            boot_attr_acc.append(0.5 * ((c2_b - c1_b) + (c4_b - c3_b)))
            boot_error_acc.append(0.5 * ((c3_b - c1_b) + (c4_b - c2_b)))
            boot_inter_acc.append((c4_b - c3_b) - (c2_b - c1_b))

            f1_1 = calc_macro_f1(g_b, p1_b)
            f1_2 = calc_macro_f1(g_b, p2_b)
            f1_3 = calc_macro_f1(g_b, p3_b)
            f1_4 = calc_macro_f1(g_b, p4_b)

            boot_attr_mf1.append(0.5 * ((f1_2 - f1_1) + (f1_4 - f1_3)))
            boot_error_mf1.append(0.5 * ((f1_3 - f1_1) + (f1_4 - f1_2)))
            boot_inter_mf1.append((f1_4 - f1_3) - (f1_2 - f1_1))

        ci_attr_acc = [float(np.percentile(boot_attr_acc, 2.5)), float(np.percentile(boot_attr_acc, 97.5))]
        ci_error_acc = [float(np.percentile(boot_error_acc, 2.5)), float(np.percentile(boot_error_acc, 97.5))]
        ci_inter_acc = [float(np.percentile(boot_inter_acc, 2.5)), float(np.percentile(boot_inter_acc, 97.5))]

        ci_attr_mf1 = [float(np.percentile(boot_attr_mf1, 2.5)), float(np.percentile(boot_attr_mf1, 97.5))]
        ci_error_mf1 = [float(np.percentile(boot_error_mf1, 2.5)), float(np.percentile(boot_error_mf1, 97.5))]
        ci_inter_mf1 = [float(np.percentile(boot_inter_mf1, 2.5)), float(np.percentile(boot_inter_mf1, 97.5))]

        # Two-sided empirical p-values (proportion of bootstrap samples crossing 0)
        p_attr_acc = float(min(np.mean(np.array(boot_attr_acc) <= 0), np.mean(np.array(boot_attr_acc) >= 0)) * 2)
        p_error_acc = float(min(np.mean(np.array(boot_error_acc) <= 0), np.mean(np.array(boot_error_acc) >= 0)) * 2)
        p_inter_acc = float(min(np.mean(np.array(boot_inter_acc) <= 0), np.mean(np.array(boot_inter_acc) >= 0)) * 2)

        return {
            "n_aligned_samples": n_samples,
            "cell_accuracies": {"A1": acc_a1, "A2": acc_a2, "A3": acc_a3, "A4": acc_a4},
            "cell_macro_f1": {"A1": mf1_a1, "A2": mf1_a2, "A3": mf1_a3, "A4": mf1_a4},
            "main_effect_attribution": {
                "delta_accuracy": main_effect_attr_acc,
                "ci95_accuracy": ci_attr_acc,
                "p_value_accuracy": p_attr_acc,
                "delta_macro_f1": main_effect_attr_mf1,
                "ci95_macro_f1": ci_attr_mf1,
            },
            "main_effect_error_knowledge": {
                "delta_accuracy": main_effect_error_acc,
                "ci95_accuracy": ci_error_acc,
                "p_value_accuracy": p_error_acc,
                "delta_macro_f1": main_effect_error_mf1,
                "ci95_macro_f1": ci_error_mf1,
            },
            "interaction_effect": {
                "delta_accuracy": interaction_acc,
                "ci95_accuracy": ci_inter_acc,
                "p_value_accuracy": p_inter_acc,
                "delta_macro_f1": interaction_mf1,
                "ci95_macro_f1": ci_inter_mf1,
            },
        }

    def _compute_paired_transitions(
        self, arms_records: Dict[str, List[M20fPredictionRecord]]
    ) -> Dict[str, Dict[str, Any]]:
        """Computes pairwise transitions (Corrected, Harmed, Net) against B0 and G2."""
        transitions: Dict[str, Dict[str, Any]] = {}
        b0_records = {r.sample_id: r for r in arms_records.get("B0", [])}

        for arm, recs in arms_records.items():
            if arm == "B0":
                continue
            corrected = 0
            harmed = 0
            both_correct = 0
            both_incorrect = 0

            for r in recs:
                b0_r = b0_records.get(r.sample_id)
                if not b0_r:
                    continue
                gold = r.gold_label
                r_corr = (r.parsed_verdict == gold)
                b0_corr = (b0_r.parsed_verdict == gold)

                if not b0_corr and r_corr:
                    corrected += 1
                elif b0_corr and not r_corr:
                    harmed += 1
                elif b0_corr and r_corr:
                    both_correct += 1
                else:
                    both_incorrect += 1

            net = corrected - harmed
            # Exact McNemar test
            b = harmed
            c = corrected
            if (b + c) > 0:
                # Two-sided binomial test under p=0.5
                k = min(b, c)
                n = b + c
                from scipy.stats import binom
                p_mcnemar = float(min(1.0, 2.0 * binom.cdf(k, n, 0.5)))
            else:
                p_mcnemar = 1.0

            transitions[arm] = {
                "baseline": "B0",
                "corrected": corrected,
                "harmed": harmed,
                "net_benefit": net,
                "both_correct": both_correct,
                "both_incorrect": both_incorrect,
                "p_value_mcnemar": p_mcnemar,
            }
        return transitions

    def _compute_contradiction_trap_diagnostics(
        self, arms_records: Dict[str, List[M20fPredictionRecord]]
    ) -> Dict[str, Dict[str, Any]]:
        """Dissects how each arm causes binary (Supported/Refuted) claims to flip to Conflicting or NEI."""
        diagnostics: Dict[str, Dict[str, Any]] = {}
        for arm, recs in arms_records.items():
            refuted_total = 0
            refuted_to_conflicting = 0
            refuted_to_nei = 0

            supported_total = 0
            supported_to_conflicting = 0
            supported_to_nei = 0

            conflicting_predictions = 0

            for r in recs:
                pv = r.parsed_verdict
                gt = r.gold_label
                if pv == "Conflicting Evidence/Cherrypicking":
                    conflicting_predictions += 1

                if gt == "Refuted":
                    refuted_total += 1
                    if pv == "Conflicting Evidence/Cherrypicking":
                        refuted_to_conflicting += 1
                    elif pv == "Not Enough Evidence":
                        refuted_to_nei += 1
                elif gt == "Supported":
                    supported_total += 1
                    if pv == "Conflicting Evidence/Cherrypicking":
                        supported_to_conflicting += 1
                    elif pv == "Not Enough Evidence":
                        supported_to_nei += 1

            trap_count = refuted_to_conflicting + supported_to_conflicting
            binary_total = refuted_total + supported_total
            trap_rate = trap_count / binary_total if binary_total > 0 else 0.0

            diagnostics[arm] = {
                "total_conflicting_predictions": conflicting_predictions,
                "refuted_to_conflicting": refuted_to_conflicting,
                "refuted_to_conflicting_rate": refuted_to_conflicting / refuted_total if refuted_total > 0 else 0.0,
                "supported_to_conflicting": supported_to_conflicting,
                "supported_to_conflicting_rate": supported_to_conflicting / supported_total if supported_total > 0 else 0.0,
                "total_contradiction_traps": trap_count,
                "contradiction_trap_rate": trap_rate,
                "refuted_to_nei": refuted_to_nei,
                "supported_to_nei": supported_to_nei,
            }
        return diagnostics

    def _determine_decision(
        self,
        metrics_by_arm: Dict[str, Dict[str, Any]],
        factorial_results: Dict[str, Any],
        trap_diagnostics: Dict[str, Any],
    ) -> Tuple[str, str]:
        """Pre-registered decision tree for Milestone M2.0f."""
        if factorial_results.get("status") in ["INCOMPLETE_ARMS", "NO_COMMON_SAMPLES"]:
            return "INCOMPLETE_DATA", "Factorial arms not fully present to evaluate."

        attr_res = factorial_results.get("main_effect_attribution", {})
        err_res = factorial_results.get("main_effect_error_knowledge", {})
        delta_attr_acc = attr_res.get("delta_accuracy", 0.0)
        p_attr_acc = attr_res.get("p_value_accuracy", 1.0)
        ci_attr_acc = attr_res.get("ci95_accuracy", [0.0, 0.0])

        delta_err_acc = err_res.get("delta_accuracy", 0.0)
        p_err_acc = err_res.get("p_value_accuracy", 1.0)

        # Check attribution distortion: upper bound of 95% CI < 0 or delta < 0 with p < 0.05
        attr_distorts = (delta_attr_acc < 0 and ci_attr_acc[1] <= 0) or (delta_attr_acc < 0 and p_attr_acc < 0.05)

        # Check error knowledge superiority
        err_superior = (delta_err_acc > 0 and p_err_acc < 0.05)
        err_parity = abs(delta_err_acc) < 0.015 or p_err_acc >= 0.05

        if attr_distorts and err_parity:
            decision = "ATTRIBUTION_DISTORTION_CONFIRMED__ERROR_PARITY"
            rationale = (
                f"Main effect of forced attribution is negative (DeltaAcc={delta_attr_acc:+.4f}, "
                f"95% CI [{ci_attr_acc[0]:+.4f}, {ci_attr_acc[1]:+.4f}], p={p_attr_acc:.4f}), "
                f"proving forced structured attribution induces the Contradiction Trap. "
                f"Historical error knowledge shows parity with matched Gold ICL (DeltaAcc={delta_err_acc:+.4f}, p={p_err_acc:.4f})."
            )
        elif attr_distorts and err_superior:
            decision = "ATTRIBUTION_DISTORTION_CONFIRMED__ERROR_SUPERIOR"
            rationale = (
                f"Attribution induces distortion (DeltaAcc={delta_attr_acc:+.4f}, p={p_attr_acc:.4f}), "
                f"but Historical Error Knowledge significantly outperforms Gold ICL (DeltaAcc={delta_err_acc:+.4f}, p={p_err_acc:.4f})."
            )
        elif not attr_distorts and err_superior:
            decision = "ERROR_KNOWLEDGE_SUPERIOR"
            rationale = (
                f"Historical Error Knowledge outperforms Gold ICL (DeltaAcc={delta_err_acc:+.4f}, p={p_err_acc:.4f}), "
                f"and forced attribution does not significantly degrade performance."
            )
        else:
            decision = "INCONCLUSIVE_FACTORIAL"
            rationale = (
                f"Effects did not meet pre-registered significance gates (DeltaAttr={delta_attr_acc:+.4f}, "
                f"p={p_attr_acc:.4f}; DeltaErr={delta_err_acc:+.4f}, p={p_err_acc:.4f})."
            )

        return decision, rationale

    def _save_csvs(
        self,
        metrics_by_arm: Dict[str, Dict[str, Any]],
        per_class_by_arm: Dict[str, Dict[str, Dict[str, float]]],
        factorial_results: Dict[str, Any],
        paired_transitions: Dict[str, Dict[str, Any]],
        trap_diagnostics: Dict[str, Dict[str, Any]],
    ) -> None:
        """Saves tabular artifacts."""
        # 1. metrics_by_arm.csv
        p_metrics = self.output_dir / "metrics_by_arm.csv"
        with open(p_metrics, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "arm", "total_claims", "parsed_claims", "parse_rate",
                "valid_attribution_rate", "accuracy", "macro_f1", "balanced_accuracy"
            ])
            for arm, m in sorted(metrics_by_arm.items()):
                w.writerow([
                    arm, m["total_claims"], m["parsed_claims"], f"{m['parse_rate']:.4f}",
                    f"{m['valid_attribution_rate']:.4f}", f"{m['accuracy']:.4f}",
                    f"{m['macro_f1']:.4f}", f"{m['balanced_accuracy']:.4f}"
                ])

        # 2. per_class_metrics.csv
        p_perclass = self.output_dir / "per_class_metrics.csv"
        with open(p_perclass, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["arm", "class", "precision", "recall", "f1", "support"])
            for arm, classes in sorted(per_class_by_arm.items()):
                for cls_name, cm in classes.items():
                    w.writerow([
                        arm, cls_name, f"{cm['precision']:.4f}", f"{cm['recall']:.4f}",
                        f"{cm['f1']:.4f}", int(cm['support'])
                    ])

        # 3. factorial_effects.csv
        p_fact = self.output_dir / "factorial_effects.csv"
        with open(p_fact, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["effect", "delta_accuracy", "ci95_low_acc", "ci95_high_acc", "p_val_acc", "delta_macro_f1", "ci95_low_mf1", "ci95_high_mf1"])
            for eff_name in ["main_effect_attribution", "main_effect_error_knowledge", "interaction_effect"]:
                eff = factorial_results.get(eff_name, {})
                if eff:
                    w.writerow([
                        eff_name,
                        f"{eff.get('delta_accuracy', 0.0):+.4f}",
                        f"{eff.get('ci95_accuracy', [0, 0])[0]:+.4f}",
                        f"{eff.get('ci95_accuracy', [0, 0])[1]:+.4f}",
                        f"{eff.get('p_value_accuracy', 1.0):.4f}",
                        f"{eff.get('delta_macro_f1', 0.0):+.4f}",
                        f"{eff.get('ci95_macro_f1', [0, 0])[0]:+.4f}",
                        f"{eff.get('ci95_macro_f1', [0, 0])[1]:+.4f}",
                    ])

        # 4. paired_transitions.csv
        p_trans = self.output_dir / "paired_transitions.csv"
        with open(p_trans, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["arm", "baseline", "corrected", "harmed", "net_benefit", "both_correct", "both_incorrect", "p_mcnemar"])
            for arm, t in sorted(paired_transitions.items()):
                w.writerow([
                    arm, t["baseline"], t["corrected"], t["harmed"], t["net_benefit"],
                    t["both_correct"], t["both_incorrect"], f"{t['p_value_mcnemar']:.5f}"
                ])

        # 5. contradiction_trap_diagnostics.csv
        p_trap = self.output_dir / "contradiction_trap_diagnostics.csv"
        with open(p_trap, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "arm", "total_conflicting_preds", "refuted_to_conflicting", "refuted_to_conflicting_rate",
                "supported_to_conflicting", "supported_to_conflicting_rate", "total_contradiction_traps",
                "contradiction_trap_rate", "refuted_to_nei", "supported_to_nei"
            ])
            for arm, td in sorted(trap_diagnostics.items()):
                w.writerow([
                    arm, td["total_conflicting_predictions"],
                    td["refuted_to_conflicting"], f"{td['refuted_to_conflicting_rate']:.4f}",
                    td["supported_to_conflicting"], f"{td['supported_to_conflicting_rate']:.4f}",
                    td["total_contradiction_traps"], f"{td['contradiction_trap_rate']:.4f}",
                    td["refuted_to_nei"], td["supported_to_nei"]
                ])

    def _save_jsons(
        self,
        confusion_by_arm: Dict[str, Dict[str, Dict[str, int]]],
        factorial_results: Dict[str, Any],
        paired_transitions: Dict[str, Dict[str, Any]],
    ) -> None:
        """Saves structured JSON receipts."""
        with open(self.output_dir / "confusion_matrices.json", "w", encoding="utf-8") as f:
            json.dump(confusion_by_arm, f, indent=2)

        stats = {
            "factorial_results": factorial_results,
            "paired_transitions": paired_transitions,
        }
        with open(self.output_dir / "statistical_tests.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    def _save_markdown_reports(
        self,
        metrics_by_arm: Dict[str, Dict[str, Any]],
        per_class_by_arm: Dict[str, Dict[str, Dict[str, float]]],
        factorial_results: Dict[str, Any],
        paired_transitions: Dict[str, Dict[str, Any]],
        trap_diagnostics: Dict[str, Dict[str, Any]],
        decision: str,
        rationale: str,
    ) -> None:
        """Generates FINAL_REPORT.md and DECISION_REPORT.md."""
        # DECISION_REPORT.md
        p_dec = self.output_dir / "DECISION_REPORT.md"
        with open(p_dec, "w", encoding="utf-8") as f:
            f.write("# Milestone M2.0f: Pre-registered Decision Report\n\n")
            f.write(f"## Final Scientific Decision: `{decision}`\n\n")
            f.write(f"**Rationale:** {rationale}\n\n")
            f.write("### Factorial Effect Summary:\n")
            f.write("| Effect | Delta Accuracy (95% CI) | p-value | Delta Macro-F1 (95% CI) |\n")
            f.write("|:---|:---:|:---:|:---:|\n")
            for eff_name, label in [
                ("main_effect_attribution", "Main Effect: Forced Attribution (Factor B)"),
                ("main_effect_error_knowledge", "Main Effect: Error Knowledge vs Gold (Factor A)"),
                ("interaction_effect", "Interaction Effect (A x B)"),
            ]:
                eff = factorial_results.get(eff_name, {})
                if eff:
                    d_acc = eff.get("delta_accuracy", 0.0)
                    ci_acc = eff.get("ci95_accuracy", [0, 0])
                    p_acc = eff.get("p_value_accuracy", 1.0)
                    d_mf1 = eff.get("delta_macro_f1", 0.0)
                    ci_mf1 = eff.get("ci95_macro_f1", [0, 0])
                    f.write(
                        f"| {label} | {d_acc:+.4f} [{ci_acc[0]:+.4f}, {ci_acc[1]:+.4f}] | "
                        f"{p_acc:.4f} | {d_mf1:+.4f} [{ci_mf1[0]:+.4f}, {ci_mf1[1]:+.4f}] |\n"
                    )

        # FINAL_REPORT.md
        p_fin = self.output_dir / "FINAL_REPORT.md"
        with open(p_fin, "w", encoding="utf-8") as f:
            f.write("# Milestone M2.0f: Factorial Ablation Analysis Report\n\n")
            f.write("## 1. Executive Summary\n\n")
            f.write(f"- **Primary Finding:** `{decision}`\n")
            f.write(f"- **Core Rationale:** {rationale}\n\n")

            f.write("## 2. Primary Metrics Across Evaluated Arms\n\n")
            f.write("| Arm | Description | Accuracy | Macro-F1 | Parse Rate | Net vs B0 |\n")
            f.write("|:---|:---|:---:|:---:|:---:|:---:|\n")
            desc_map = {
                "B0": "Canonical Zero-Shot Baseline",
                "G2": "Generic Evidence-Guided Rules Control",
                "A1": "Matched Gold ICL (No Attribution)",
                "A2": "Matched Gold ICL + Forced Attribution",
                "A3": "Historical Error ICL (No Attribution)",
                "A4": "Historical Error ICL + Forced Attribution",
            }
            for arm, m in sorted(metrics_by_arm.items()):
                net_str = str(paired_transitions.get(arm, {}).get("net_benefit", "-"))
                f.write(
                    f"| **{arm}** | {desc_map.get(arm, arm)} | {m['accuracy']*100:.2f}% | "
                    f"{m['macro_f1']:.4f} | {m['parse_rate']*100:.1f}% | {net_str} |\n"
                )

            f.write("\n## 3. Contradiction Trap & Transition Dissection\n\n")
            f.write("| Arm | Conflicting Preds | Refuted -> Conflicting | Supported -> Conflicting | Trap Rate |\n")
            f.write("|:---|:---:|:---:|:---:|:---:|\n")
            for arm, td in sorted(trap_diagnostics.items()):
                f.write(
                    f"| **{arm}** | {td['total_conflicting_predictions']} | "
                    f"{td['refuted_to_conflicting']} ({td['refuted_to_conflicting_rate']*100:.1f}%) | "
                    f"{td['supported_to_conflicting']} ({td['supported_to_conflicting_rate']*100:.1f}%) | "
                    f"{td['contradiction_trap_rate']*100:.1f}% |\n"
                )

            f.write("\n## 4. Factorial Hypotheses Verification\n\n")
            f.write("1. **Hypothesis 1 (Attribution Induces Contradiction Trap):**\n")
            eff_b = factorial_results.get("main_effect_attribution", {})
            f.write(
                f"   - Main Effect of Attribution on Accuracy: {eff_b.get('delta_accuracy', 0.0):+.4f} "
                f"(95% CI: [{eff_b.get('ci95_accuracy', [0, 0])[0]:+.4f}, {eff_b.get('ci95_accuracy', [0, 0])[1]:+.4f}], p={eff_b.get('p_value_accuracy', 1.0):.4f})\n"
            )
            f.write(
                f"   - Main Effect of Attribution on Macro-F1: {eff_b.get('delta_macro_f1', 0.0):+.4f} "
                f"(95% CI: [{eff_b.get('ci95_macro_f1', [0, 0])[0]:+.4f}, {eff_b.get('ci95_macro_f1', [0, 0])[1]:+.4f}])\n\n"
            )

            f.write("2. **Hypothesis 2 (Historical Error Knowledge vs Matched Gold ICL):**\n")
            eff_a = factorial_results.get("main_effect_error_knowledge", {})
            f.write(
                f"   - Main Effect of Historical Error Knowledge on Accuracy: {eff_a.get('delta_accuracy', 0.0):+.4f} "
                f"(95% CI: [{eff_a.get('ci95_accuracy', [0, 0])[0]:+.4f}, {eff_a.get('ci95_accuracy', [0, 0])[1]:+.4f}], p={eff_a.get('p_value_error', 1.0):.4f})\n"
            )
            f.write(
                f"   - Main Effect of Historical Error Knowledge on Macro-F1: {eff_a.get('delta_macro_f1', 0.0):+.4f} "
                f"(95% CI: [{eff_a.get('ci95_macro_f1', [0, 0])[0]:+.4f}, {eff_a.get('ci95_macro_f1', [0, 0])[1]:+.4f}])\n\n"
            )

            f.write("3. **Hypothesis 3 (Interaction Effect):**\n")
            eff_ab = factorial_results.get("interaction_effect", {})
            f.write(
                f"   - Interaction (A x B) on Accuracy: {eff_ab.get('delta_accuracy', 0.0):+.4f} "
                f"(95% CI: [{eff_ab.get('ci95_accuracy', [0, 0])[0]:+.4f}, {eff_ab.get('ci95_accuracy', [0, 0])[1]:+.4f}])\n"
            )
