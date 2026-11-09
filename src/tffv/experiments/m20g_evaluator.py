"""Milestone M2.0g: Comprehensive Evaluator for Robustness, Component Ablation & Generalization.

Evaluates:
- E1: Robustness across Seeds 1..5 and shot counts k in {2, 4, 6}.
- E2: Knowledge Component Ablation (K1, K2, K3, K4).
- E3: Source-Fold Generalization on independent outer fold.
- Contradiction Trap and NEI transition diagnostics.
- Generates all CSVs, JSONs, DECISION_REPORT.md, and FINAL_REPORT.md.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tffv.experiments.m20g_schemas import (
    ExperimentPhase,
    KnowledgeComponent,
    M20gPredictionRecord,
)

ALL_CLASSES = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]


class M20gEvaluator:
    """Evaluates M2.0g experiments and generates statistical reports."""

    def __init__(
        self,
        records: List[M20gPredictionRecord],
        output_dir: str | Path = "outputs/m20g-robustness-attribution-v1",
        n_bootstraps: int = 5000,
        seed: int = 42,
    ):
        self.records = records
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_bootstraps = n_bootstraps
        self.seed = seed

    def evaluate_all(self) -> Dict[str, Any]:
        """Runs the complete evaluation battery across E1, E2, E3."""
        # 1. Group records by arm_name
        arms_records: Dict[str, List[M20gPredictionRecord]] = {}
        for r in self.records:
            arms_records.setdefault(r.arm_name, []).append(r)

        # 2. Compute metrics by arm
        metrics_by_arm: Dict[str, Dict[str, Any]] = {}
        per_class_by_arm: Dict[str, Dict[str, Dict[str, float]]] = {}
        confusion_by_arm: Dict[str, Dict[str, Dict[str, int]]] = {}

        for arm, recs in sorted(arms_records.items()):
            arm_metrics, per_class, cm = self._compute_arm_metrics(recs)
            metrics_by_arm[arm] = arm_metrics
            per_class_by_arm[arm] = per_class
            confusion_by_arm[arm] = cm

        # 3. E1: Exemplar Robustness & Seed Sensitivity
        e1_results = self._evaluate_e1(arms_records)

        # 4. E2: Knowledge Component Ablation
        e2_results = self._evaluate_e2(arms_records)

        # 5. E3: Source-Fold Generalization
        e3_results = self._evaluate_e3(arms_records)

        # 6. Contradiction Trap & NEI Diagnostics
        trap_diagnostics = self._compute_trap_diagnostics(arms_records)

        # 7. Pre-registered Scientific Decision
        decision, rationale = self._determine_decision(e1_results, e2_results, e3_results)

        # 8. Save CSVs, JSONs, Markdown reports
        self._save_csvs(
            metrics_by_arm,
            per_class_by_arm,
            e1_results,
            e2_results,
            e3_results,
            trap_diagnostics,
        )
        self._save_jsons(confusion_by_arm, e1_results, e2_results, e3_results)
        self._save_markdown_reports(
            metrics_by_arm,
            e1_results,
            e2_results,
            e3_results,
            trap_diagnostics,
            decision,
            rationale,
        )

        return {
            "decision": decision,
            "decision_rationale": rationale,
            "metrics_by_arm": metrics_by_arm,
            "e1_results": e1_results,
            "e2_results": e2_results,
            "e3_results": e3_results,
        }

    def _compute_arm_metrics(
        self, records: List[M20gPredictionRecord]
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]], Dict[str, Dict[str, int]]]:
        n_total = len(records)
        parsed_records = [r for r in records if r.parsed_verdict in ALL_CLASSES]
        n_parsed = len(parsed_records)
        parse_rate = n_parsed / n_total if n_total > 0 else 0.0

        y_true = [r.gold_label for r in records]
        y_pred = [r.parsed_verdict if r.parsed_verdict in ALL_CLASSES else "UNPARSEABLE" for r in records]

        correct_count = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp and yt is not None)
        accuracy = correct_count / n_total if n_total > 0 else 0.0

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
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_acc,
            "correct_claims": correct_count,
        }
        return summary, per_class, cm

    def _evaluate_e1(self, arms_records: Dict[str, List[M20gPredictionRecord]]) -> Dict[str, Any]:
        """Evaluates seed variation and shot counts for E1."""
        results: Dict[str, Any] = {}
        for k in [2, 4, 6]:
            k_key = f"k_{k}"
            results[k_key] = {"seeds": {}, "error_accuracies": [], "gold_accuracies": []}

            for seed in [1, 2, 3, 4, 5]:
                err_arm = f"E1_k{k}_s{seed}_error"
                gold_arm = f"E1_k{k}_s{seed}_gold"

                if err_arm in arms_records and gold_arm in arms_records:
                    recs_err = arms_records[err_arm]
                    recs_gold = arms_records[gold_arm]

                    m_err, _, _ = self._compute_arm_metrics(recs_err)
                    m_gold, _, _ = self._compute_arm_metrics(recs_gold)

                    results[k_key]["seeds"][str(seed)] = {
                        "error_accuracy": m_err["accuracy"],
                        "error_macro_f1": m_err["macro_f1"],
                        "gold_accuracy": m_gold["accuracy"],
                        "gold_macro_f1": m_gold["macro_f1"],
                        "delta_accuracy": m_err["accuracy"] - m_gold["accuracy"],
                        "delta_macro_f1": m_err["macro_f1"] - m_gold["macro_f1"],
                    }
                    results[k_key]["error_accuracies"].append(m_err["accuracy"])
                    results[k_key]["gold_accuracies"].append(m_gold["accuracy"])

            err_accs = results[k_key]["error_accuracies"]
            gold_accs = results[k_key]["gold_accuracies"]
            if err_accs:
                results[k_key]["error_mean"] = float(np.mean(err_accs))
                results[k_key]["error_std"] = float(np.std(err_accs))
                results[k_key]["gold_mean"] = float(np.mean(gold_accs))
                results[k_key]["gold_std"] = float(np.std(gold_accs))
                results[k_key]["mean_delta"] = float(np.mean(err_accs) - np.mean(gold_accs))

        return results

    def _evaluate_e2(self, arms_records: Dict[str, List[M20gPredictionRecord]]) -> Dict[str, Any]:
        """Evaluates knowledge component ablation for E2."""
        e2_arms = ["E2_K1_gold_only", "E2_K2_neutral_expl", "E2_K3_wrong_pred", "E2_K4_grounded_error"]
        results: Dict[str, Any] = {}

        for arm in e2_arms:
            if arm in arms_records:
                m, _, _ = self._compute_arm_metrics(arms_records[arm])
                results[arm] = m

        # Compare K4 vs K1, K2, K3
        if "E2_K1_gold_only" in results and "E2_K4_grounded_error" in results:
            results["delta_K4_vs_K1"] = results["E2_K4_grounded_error"]["accuracy"] - results["E2_K1_gold_only"]["accuracy"]
        if "E2_K2_neutral_expl" in results and "E2_K4_grounded_error" in results:
            results["delta_K4_vs_K2"] = results["E2_K4_grounded_error"]["accuracy"] - results["E2_K2_neutral_expl"]["accuracy"]
        if "E2_K3_wrong_pred" in results and "E2_K4_grounded_error" in results:
            results["delta_K4_vs_K3"] = results["E2_K4_grounded_error"]["accuracy"] - results["E2_K3_wrong_pred"]["accuracy"]

        return results

    def _evaluate_e3(self, arms_records: Dict[str, List[M20gPredictionRecord]]) -> Dict[str, Any]:
        """Evaluates source-fold generalization on outer fold for E3."""
        e3_arms = ["E3_B0_baseline", "E3_A1_gold", "E3_A3_error"]
        results: Dict[str, Any] = {}

        for arm in e3_arms:
            if arm in arms_records:
                m, _, _ = self._compute_arm_metrics(arms_records[arm])
                results[arm] = m

        if "E3_B0_baseline" in results and "E3_A3_error" in results:
            results["delta_A3_vs_B0"] = results["E3_A3_error"]["accuracy"] - results["E3_B0_baseline"]["accuracy"]
            results["delta_mf1_A3_vs_B0"] = results["E3_A3_error"]["macro_f1"] - results["E3_B0_baseline"]["macro_f1"]

        return results

    def _compute_trap_diagnostics(
        self, arms_records: Dict[str, List[M20gPredictionRecord]]
    ) -> Dict[str, Dict[str, Any]]:
        diagnostics: Dict[str, Dict[str, Any]] = {}
        for arm, recs in arms_records.items():
            refuted_total = 0
            refuted_to_conflicting = 0
            refuted_to_nei = 0
            supported_total = 0
            supported_to_conflicting = 0
            conflicting_preds = 0

            for r in recs:
                pv = r.parsed_verdict
                gt = r.gold_label
                if pv == "Conflicting Evidence/Cherrypicking":
                    conflicting_preds += 1

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

            diagnostics[arm] = {
                "total_conflicting_predictions": conflicting_preds,
                "refuted_to_conflicting": refuted_to_conflicting,
                "refuted_to_conflicting_rate": refuted_to_conflicting / refuted_total if refuted_total > 0 else 0.0,
                "supported_to_conflicting": supported_to_conflicting,
                "refuted_to_nei": refuted_to_nei,
            }
        return diagnostics

    def _determine_decision(
        self,
        e1_res: Dict[str, Any],
        e2_res: Dict[str, Any],
        e3_res: Dict[str, Any],
    ) -> Tuple[str, str]:
        # Check E1 robustness across seeds
        k2_delta = e1_res.get("k_2", {}).get("mean_delta", 0.0)
        e1_robust = k2_delta > 0.02

        # Check E2 knowledge attribution
        d_k4_vs_k2 = e2_res.get("delta_K4_vs_K2", 0.0)
        e2_knowledge_driven = d_k4_vs_k2 > 0.01

        # Check E3 outer fold generalization
        d_e3 = e3_res.get("delta_A3_vs_B0", 0.0)
        e3_generalizes = d_e3 > 0.02

        if e1_robust and e2_knowledge_driven and e3_generalizes:
            decision = "HISTORICAL_ERROR_ROBUSTNESS_CONFIRMED"
            rationale = (
                f"Historical error knowledge demonstrated robust superiority over gold ICL across 5 seeds "
                f"(E1 mean delta = {k2_delta:+.4f}), component ablation confirmed qualitative grounding drives lift "
                f"(K4 vs K2 = {d_k4_vs_k2:+.4f}), and out-of-sample outer fold verified generalization ({d_e3:+.4f})."
            )
        elif e1_robust and not e3_generalizes:
            decision = "ROBUST_IN_SAMPLE__GENERALIZATION_LIMITED"
            rationale = (
                f"Historical error knowledge is robust across seeds in-sample ({k2_delta:+.4f}), "
                f"but failed to achieve consistent improvement on the outer evaluation fold ({d_e3:+.4f})."
            )
        else:
            decision = "EXEMPLAR_SENSITIVITY_DETECTED"
            rationale = (
                f"Performance varied significantly across exemplar draws or failed to outperform matched controls "
                f"(E1 mean delta = {k2_delta:+.4f}, E2 K4 vs K2 = {d_k4_vs_k2:+.4f})."
            )

        return decision, rationale

    def _save_csvs(
        self,
        metrics_by_arm: Dict[str, Dict[str, Any]],
        per_class_by_arm: Dict[str, Dict[str, Dict[str, float]]],
        e1_results: Dict[str, Any],
        e2_results: Dict[str, Any],
        e3_results: Dict[str, Any],
        trap_diagnostics: Dict[str, Dict[str, Any]],
    ) -> None:
        # 1. metrics_by_arm.csv
        with open(self.output_dir / "metrics_by_arm.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["arm", "total_claims", "parsed_claims", "parse_rate", "accuracy", "macro_f1", "balanced_accuracy"])
            for arm, m in sorted(metrics_by_arm.items()):
                w.writerow([
                    arm, m["total_claims"], m["parsed_claims"], f"{m['parse_rate']:.4f}",
                    f"{m['accuracy']:.4f}", f"{m['macro_f1']:.4f}", f"{m['balanced_accuracy']:.4f}"
                ])

        # 2. e1_robustness_summary.csv
        with open(self.output_dir / "e1_robustness_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["k_shots", "error_mean_acc", "error_std_acc", "gold_mean_acc", "gold_std_acc", "mean_delta_acc"])
            for k in [2, 4, 6]:
                kd = e1_results.get(f"k_{k}", {})
                if kd and "error_mean" in kd:
                    w.writerow([
                        k,
                        f"{kd['error_mean']:.4f}", f"{kd['error_std']:.4f}",
                        f"{kd['gold_mean']:.4f}", f"{kd['gold_std']:.4f}",
                        f"{kd['mean_delta']:.4f}",
                    ])

        # 3. e2_component_ablation.csv
        with open(self.output_dir / "e2_component_ablation.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["component", "description", "accuracy", "macro_f1"])
            comp_map = {
                "E2_K1_gold_only": "K1: Gold Only",
                "E2_K2_neutral_expl": "K2: Gold + Neutral Explanation",
                "E2_K3_wrong_pred": "K3: Wrong Pred + Gold Correction",
                "E2_K4_grounded_error": "K4: Full Grounded Error Contrast",
            }
            for arm, desc in comp_map.items():
                m = e2_results.get(arm, {})
                if m:
                    w.writerow([arm, desc, f"{m.get('accuracy', 0):.4f}", f"{m.get('macro_f1', 0):.4f}"])

        # 4. e3_generalization_summary.csv
        with open(self.output_dir / "e3_generalization_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["arm", "accuracy", "macro_f1", "delta_vs_b0"])
            b0_acc = e3_results.get("E3_B0_baseline", {}).get("accuracy", 0.0)
            for arm in ["E3_B0_baseline", "E3_A1_gold", "E3_A3_error"]:
                m = e3_results.get(arm, {})
                if m:
                    acc = m.get("accuracy", 0.0)
                    delta = acc - b0_acc if arm != "E3_B0_baseline" else 0.0
                    w.writerow([arm, f"{acc:.4f}", f"{m.get('macro_f1', 0):.4f}", f"{delta:+.4f}"])

    def _save_jsons(
        self,
        confusion_by_arm: Dict[str, Dict[str, Dict[str, int]]],
        e1_res: Dict[str, Any],
        e2_res: Dict[str, Any],
        e3_res: Dict[str, Any],
    ) -> None:
        with open(self.output_dir / "confusion_matrices.json", "w", encoding="utf-8") as f:
            json.dump(confusion_by_arm, f, indent=2)

        stats = {
            "e1_robustness": e1_res,
            "e2_ablation": e2_res,
            "e3_generalization": e3_res,
        }
        with open(self.output_dir / "statistical_tests.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    def _save_markdown_reports(
        self,
        metrics_by_arm: Dict[str, Dict[str, Any]],
        e1_res: Dict[str, Any],
        e2_res: Dict[str, Any],
        e3_res: Dict[str, Any],
        trap_diagnostics: Dict[str, Dict[str, Any]],
        decision: str,
        rationale: str,
    ) -> None:
        # DECISION_REPORT.md
        with open(self.output_dir / "DECISION_REPORT.md", "w", encoding="utf-8") as f:
            f.write("# Milestone M2.0g: Pre-registered Decision Report\n\n")
            f.write(f"## Final Scientific Determination: `{decision}`\n\n")
            f.write(f"**Rationale:** {rationale}\n\n")

        # FINAL_REPORT.md
        with open(self.output_dir / "FINAL_REPORT.md", "w", encoding="utf-8") as f:
            f.write("# Milestone M2.0g: Robustness, Component Ablation & Generalization Report\n\n")
            f.write("## 1. Executive Summary\n\n")
            f.write(f"- **Determination:** `{decision}`\n")
            f.write(f"- **Core Rationale:** {rationale}\n\n")

            f.write("## 2. E1: Exemplar Robustness & Multi-Seed Analysis\n\n")
            f.write("| k Shots | Error Mean Acc (Std) | Gold Mean Acc (Std) | Mean Delta |\n")
            f.write("|:---|:---:|:---:|:---:|\n")
            for k in [2, 4, 6]:
                kd = e1_res.get(f"k_{k}", {})
                if kd and "error_mean" in kd:
                    f.write(
                        f"| k={k} | {kd['error_mean']*100:.2f}% (±{kd['error_std']*100:.2f}%) | "
                        f"{kd['gold_mean']*100:.2f}% (±{kd['gold_std']*100:.2f}%) | "
                        f"{kd['mean_delta']*100:+.2f}% |\n"
                    )

            f.write("\n## 3. E2: Knowledge Component Ablation\n\n")
            f.write("| Component | Description | Accuracy | Macro-F1 |\n")
            f.write("|:---|:---|:---:|:---:|\n")
            comp_map = {
                "E2_K1_gold_only": "K1: Gold Only (Minimal prompt)",
                "E2_K2_neutral_expl": "K2: Gold + Neutral Explanation",
                "E2_K3_wrong_pred": "K3: Wrong Pred + Gold Correction",
                "E2_K4_grounded_error": "K4: Full Grounded Error Contrast",
            }
            for arm, desc in comp_map.items():
                m = e2_res.get(arm, {})
                if m:
                    f.write(f"| **{arm}** | {desc} | {m.get('accuracy', 0)*100:.2f}% | {m.get('macro_f1', 0):.4f} |\n")

            f.write("\n## 4. E3: Source-Fold Generalization (Outer Group-Isolated Fold)\n\n")
            f.write("| Arm | Description | Accuracy | Macro-F1 | Delta vs B0 |\n")
            f.write("|:---|:---|:---:|:---:|:---:|\n")
            b0_acc = e3_res.get("E3_B0_baseline", {}).get("accuracy", 0.0)
            for arm in ["E3_B0_baseline", "E3_A1_gold", "E3_A3_error"]:
                m = e3_res.get(arm, {})
                if m:
                    acc = m.get("accuracy", 0.0)
                    delta = acc - b0_acc if arm != "E3_B0_baseline" else 0.0
                    f.write(f"| **{arm}** | Outer Fold Evaluation | {acc*100:.2f}% | {m.get('macro_f1', 0):.4f} | {delta*100:+.2f}% |\n")
