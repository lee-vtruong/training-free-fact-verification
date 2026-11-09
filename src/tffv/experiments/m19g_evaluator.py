"""Milestone M1.9g: Statistical Evaluator & Hypothesis Testing.

Includes:
1. Hardened verdict parsing (immune to template schema echoing).
2. Four-class Macro-F1, Accuracy, and Balanced Accuracy computation.
3. 5,000-sample paired bootstrap confidence intervals.
4. Exact McNemar significance tests with Holm-Bonferroni correction.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tffv.experiments.m19g_guardrails import AVERITEC_CLASSES, parse_verdict_hardened

CLASS_LIST = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]


@dataclass
class MetricSummary:
    macro_f1: float
    accuracy: float
    balanced_accuracy: float
    valid_verdict_rate: float
    per_class: Dict[str, Dict[str, float]]
    confusion_matrix: Dict[str, Dict[str, int]]
    total_claims: int
    correct_count: int
    unparseable_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "macro_f1": round(self.macro_f1, 4),
            "accuracy": round(self.accuracy, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "valid_verdict_rate": round(self.valid_verdict_rate, 4),
            "total_claims": self.total_claims,
            "correct_count": self.correct_count,
            "unparseable_count": self.unparseable_count,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
        }


class M19gEvaluator:
    def __init__(self, bootstrap_samples: int = 5000, seed: int = 42):
        self.bootstrap_samples = bootstrap_samples
        self.seed = seed

    def compute_arm_metrics(self, predictions: List[Dict[str, Any]]) -> MetricSummary:
        """Computes comprehensive 4-class evaluation metrics."""
        total = len(predictions)
        if total == 0:
            raise ValueError("No predictions provided for evaluation!")

        # Initialize confusion matrix
        cm: Dict[str, Dict[str, int]] = {
            gold: {pred: 0 for pred in CLASS_LIST + ["UNPARSEABLE"]}
            for gold in CLASS_LIST
        }

        correct = 0
        unparseable = 0

        for p in predictions:
            gold = p.get("gold_verdict", "")
            pred = p.get("predicted_verdict")

            if gold not in CLASS_LIST:
                continue

            if pred is None or pred not in CLASS_LIST:
                unparseable += 1
                cm[gold]["UNPARSEABLE"] += 1
            else:
                cm[gold][pred] += 1
                if pred == gold:
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0
        valid_rate = (total - unparseable) / total if total > 0 else 0.0

        # Per-class metrics
        per_class: Dict[str, Dict[str, float]] = {}
        f1_scores: List[float] = []
        recalls: List[float] = []

        for c in CLASS_LIST:
            tp = cm[c].get(c, 0)
            fn = sum(cm[c].get(other, 0) for other in CLASS_LIST + ["UNPARSEABLE"] if other != c)
            fp = sum(cm[other].get(c, 0) for other in CLASS_LIST if other != c)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            support = tp + fn
            per_class[c] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "support": support,
            }
            f1_scores.append(f1)
            recalls.append(rec)

        macro_f1 = float(np.mean(f1_scores))
        balanced_acc = float(np.mean(recalls))

        return MetricSummary(
            macro_f1=macro_f1,
            accuracy=accuracy,
            balanced_accuracy=balanced_acc,
            valid_verdict_rate=valid_rate,
            per_class=per_class,
            confusion_matrix=cm,
            total_claims=total,
            correct_count=correct,
            unparseable_count=unparseable,
        )

    def paired_bootstrap_test(
        self,
        preds_a: List[Dict[str, Any]],
        preds_b: List[Dict[str, Any]],
    ) -> Tuple[float, Tuple[float, float]]:
        """Paired bootstrap for difference in Macro-F1 (A - B)."""
        rng = np.random.RandomState(self.seed)
        n = len(preds_a)
        if len(preds_b) != n:
            raise ValueError(f"Arm sizes must match for paired bootstrap! {n} != {len(preds_b)}")

        summary_a = self.compute_arm_metrics(preds_a)
        summary_b = self.compute_arm_metrics(preds_b)
        obs_delta = summary_a.macro_f1 - summary_b.macro_f1

        deltas = []
        indices = np.arange(n)
        for _ in range(self.bootstrap_samples):
            boot_idx = rng.choice(indices, size=n, replace=True)
            boot_a = [preds_a[i] for i in boot_idx]
            boot_b = [preds_b[i] for i in boot_idx]

            ma = self.compute_arm_metrics(boot_a).macro_f1
            mb = self.compute_arm_metrics(boot_b).macro_f1
            deltas.append(ma - mb)

        deltas = np.array(deltas)
        ci_lower = float(np.percentile(deltas, 2.5))
        ci_upper = float(np.percentile(deltas, 97.5))

        return obs_delta, (ci_lower, ci_upper)

    def mcnemar_exact_test(
        self,
        preds_a: List[Dict[str, Any]],
        preds_b: List[Dict[str, Any]],
    ) -> Tuple[int, int, float]:
        """Exact binomial McNemar test between two paired binary outcome vectors."""
        n = len(preds_a)
        b = 0  # A correct, B incorrect
        c = 0  # A incorrect, B correct

        for i in range(n):
            a_correct = (preds_a[i].get("predicted_verdict") == preds_a[i].get("gold_verdict"))
            b_correct = (preds_b[i].get("predicted_verdict") == preds_b[i].get("gold_verdict"))

            if a_correct and not b_correct:
                b += 1
            elif not a_correct and b_correct:
                c += 1

        total_disc = b + c
        if total_disc == 0:
            return b, c, 1.0

        # Exact two-sided binomial test under H0: p = 0.5
        min_k = min(b, c)
        p_val = 2.0 * sum(math.comb(total_disc, k) * (0.5 ** total_disc) for k in range(min_k + 1))
        p_val = min(1.0, p_val)

        return b, c, p_val

    def run_hypothesis_battery(
        self,
        arm_predictions: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Runs paired statistical hypothesis battery with Holm-Bonferroni correction."""
        comparisons = [
            ("Primary: Error-History ICL vs Matched Gold-Only ICL", "H2", "B1"),
            ("Secondary: Correct-History ICL vs Zero-Shot Base", "H1", "B0"),
            ("Secondary: Error-History ICL vs Zero-Shot Base", "H2", "B0"),
            ("Secondary: Mixed-History ICL vs Zero-Shot Base", "H3", "B0"),
            ("Secondary: Mixed-History ICL vs Matched Gold-Only ICL", "H3", "B2"),
        ]

        test_results = []
        for desc, arm_a, arm_b in comparisons:
            pa = arm_predictions.get(arm_a, [])
            pb = arm_predictions.get(arm_b, [])

            if not pa or not pb:
                continue

            obs_delta, (ci_l, ci_u) = self.paired_bootstrap_test(pa, pb)
            b, c, p_val = self.mcnemar_exact_test(pa, pb)

            test_results.append({
                "description": desc,
                "arm_a": arm_a,
                "arm_b": arm_b,
                "observed_delta_macro_f1": round(obs_delta, 4),
                "macro_f1_95_ci": [round(ci_l, 4), round(ci_u, 4)],
                "mcnemar_b": b,
                "mcnemar_c": c,
                "mcnemar_exact_p_value": round(p_val, 5),
                "is_significant_at_05": p_val < 0.05 and ci_l > 0,
            })

        # Holm-Bonferroni correction
        test_results.sort(key=lambda x: x["mcnemar_exact_p_value"])
        m = len(test_results)
        for rank, res in enumerate(test_results, 1):
            threshold = 0.05 / (m - rank + 1)
            res["holm_threshold"] = round(threshold, 5)
            res["holm_significant"] = res["mcnemar_exact_p_value"] < threshold and res["observed_delta_macro_f1"] > 0

        return {
            "bootstrap_samples": self.bootstrap_samples,
            "seed": self.seed,
            "hypotheses_evaluated": test_results,
        }
