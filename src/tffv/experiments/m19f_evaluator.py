"""Milestone M1.9f: Statistical Evaluator & Metric Battery.

Computes 4-class Macro-F1, Accuracy, Balanced Accuracy, Per-Class metrics,
Confusion Matrices, paired bootstrap 95% CIs, exact McNemar tests, and Holm adjustments.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

AVERITEC_CLASSES = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]


def parse_verdict(raw_output: str) -> Tuple[str, bool]:
    """Parse verdict from verifier raw completion."""
    if not raw_output:
        return ("Not Enough Evidence", False)

    # Check for explicit 'Verdict:' field
    m = re.search(r"Verdict:\s*([A-Za-z\s/]+)", raw_output, re.IGNORECASE)
    if m:
        c = m.group(1).strip()
        for cls in AVERITEC_CLASSES:
            if cls.lower() in c.lower():
                return (cls, True)

    # Fallback to direct label scan
    for cls in AVERITEC_CLASSES:
        if cls.lower() in raw_output.lower():
            return (cls, True)

    return ("Not Enough Evidence", False)


@dataclass
class MetricSummary:
    arm: str
    macro_f1: float
    accuracy: float
    balanced_accuracy: float
    valid_verdict_rate: float
    total_claims: int
    correct_count: int
    unparseable_count: int
    per_class: Dict[str, Dict[str, float]]
    confusion_matrix: Dict[str, Dict[str, int]]


class M19fEvaluator:
    def __init__(self, bootstrap_resamples: int = 5000, seed: int = 42):
        self.bootstrap_resamples = bootstrap_resamples
        self.seed = seed

    def compute_arm_metrics(
        self,
        arm_name: str,
        predictions: List[Dict[str, Any]],
    ) -> MetricSummary:
        total = len(predictions)
        if total == 0:
            return MetricSummary(
                arm=arm_name,
                macro_f1=0.0,
                accuracy=0.0,
                balanced_accuracy=0.0,
                valid_verdict_rate=0.0,
                total_claims=0,
                correct_count=0,
                unparseable_count=0,
                per_class={},
                confusion_matrix={},
            )

        cm = {g: {p: 0 for p in AVERITEC_CLASSES} for g in AVERITEC_CLASSES}
        correct_count = 0
        unparseable_count = 0

        for p in predictions:
            g = p["gold_verdict"]
            pr = p["predicted_verdict"]
            is_valid = p.get("is_parseable", True)

            if not is_valid:
                unparseable_count += 1

            if g in cm and pr in cm[g]:
                cm[g][pr] += 1

            if g == pr:
                correct_count += 1

        acc = correct_count / total
        valid_rate = (total - unparseable_count) / total

        # Per-class metrics
        per_class = {}
        f1_list = []
        recall_list = []

        for cls in AVERITEC_CLASSES:
            tp = cm[cls][cls]
            fp = sum(cm[o][cls] for o in AVERITEC_CLASSES if o != cls)
            fn = sum(cm[cls][o] for o in AVERITEC_CLASSES if o != cls)
            support = sum(cm[cls].values())

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            per_class[cls] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "support": support,
            }
            f1_list.append(f1)
            recall_list.append(rec)

        macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
        balanced_acc = sum(recall_list) / len(recall_list) if recall_list else 0.0

        return MetricSummary(
            arm=arm_name,
            macro_f1=round(macro_f1, 4),
            accuracy=round(acc, 4),
            balanced_accuracy=round(balanced_acc, 4),
            valid_verdict_rate=round(valid_rate, 4),
            total_claims=total,
            correct_count=correct_count,
            unparseable_count=unparseable_count,
            per_class=per_class,
            confusion_matrix=cm,
        )

    def _exact_mcnemar_p_value(self, b: int, c: int) -> float:
        n = b + c
        if n == 0:
            return 1.0
        k = min(b, c)
        prob_sum = 0.0
        for i in range(k + 1):
            prob_sum += math.comb(n, i) * (0.5 ** n)
        return min(1.0, 2.0 * prob_sum)

    def paired_bootstrap_test(
        self,
        preds_arm_a: List[Dict[str, Any]],
        preds_arm_b: List[Dict[str, Any]],
        arm_a_name: str,
        arm_b_name: str,
    ) -> Dict[str, Any]:
        n = len(preds_arm_a)
        if n == 0 or len(preds_arm_b) != n:
            raise ValueError(f"Mismatched prediction lengths: {len(preds_arm_a)} vs {len(preds_arm_b)}")

        b_discordant = 0
        c_discordant = 0
        a_concordant_correct = 0
        d_concordant_wrong = 0

        for a, b in zip(preds_arm_a, preds_arm_b):
            c_a = a["is_correct"]
            c_b = b["is_correct"]
            if c_a and c_b:
                a_concordant_correct += 1
            elif c_a and not c_b:
                b_discordant += 1
            elif not c_a and c_b:
                c_discordant += 1
            else:
                d_concordant_wrong += 1

        mcnemar_p = self._exact_mcnemar_p_value(b_discordant, c_discordant)

        base_a = self.compute_arm_metrics(arm_a_name, preds_arm_a)
        base_b = self.compute_arm_metrics(arm_b_name, preds_arm_b)
        obs_delta_f1 = base_a.macro_f1 - base_b.macro_f1
        obs_delta_acc = base_a.accuracy - base_b.accuracy

        rng = random.Random(self.seed)
        boot_delta_f1: List[float] = []
        boot_delta_acc: List[float] = []

        for _ in range(self.bootstrap_resamples):
            indices = [rng.randint(0, n - 1) for _ in range(n)]
            b_a = [preds_arm_a[i] for i in indices]
            b_b = [preds_arm_b[i] for i in indices]

            m_a = self.compute_arm_metrics(arm_a_name, b_a)
            m_b = self.compute_arm_metrics(arm_b_name, b_b)

            boot_delta_f1.append(m_a.macro_f1 - m_b.macro_f1)
            boot_delta_acc.append(m_a.accuracy - m_b.accuracy)

        boot_delta_f1.sort()
        boot_delta_acc.sort()

        low_idx = int(0.025 * self.bootstrap_resamples)
        high_idx = int(0.975 * self.bootstrap_resamples)

        f1_ci = [round(boot_delta_f1[low_idx], 4), round(boot_delta_f1[high_idx], 4)]
        acc_ci = [round(boot_delta_acc[low_idx], 4), round(boot_delta_acc[high_idx], 4)]

        return {
            "comparison": f"{arm_a_name} vs {arm_b_name}",
            "arm_a": arm_a_name,
            "arm_b": arm_b_name,
            "total_claims": n,
            "observed_delta_macro_f1": round(obs_delta_f1, 4),
            "macro_f1_95_ci": f1_ci,
            "observed_delta_accuracy": round(obs_delta_acc, 4),
            "accuracy_95_ci": acc_ci,
            "contingency_table": {
                "both_correct": a_concordant_correct,
                "arm_a_correct_arm_b_wrong": b_discordant,
                "arm_a_wrong_arm_b_correct": c_discordant,
                "both_wrong": d_concordant_wrong,
            },
            "mcnemar_exact_p_value": round(mcnemar_p, 5),
            "is_significant_at_05": (f1_ci[0] > 0 or f1_ci[1] < 0) and mcnemar_p < 0.05,
        }

    def run_full_battery(
        self,
        all_arm_preds: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        hypothesis_pairs = [
            ("H2", "B1", "Primary: Error-History ICL vs Matched Gold-Only ICL"),
            ("H1", "B0", "Secondary: Correct-History ICL vs Zero-Shot Base"),
            ("H2", "B0", "Secondary: Error-History ICL vs Zero-Shot Base"),
            ("H3", "B2", "Secondary: Mixed-History ICL vs Matched Gold-Only ICL"),
            ("H3", "B0", "Secondary: Mixed-History ICL vs Zero-Shot Base"),
        ]

        raw_tests = []
        for a_name, b_name, desc in hypothesis_pairs:
            if a_name in all_arm_preds and b_name in all_arm_preds:
                t = self.paired_bootstrap_test(all_arm_preds[a_name], all_arm_preds[b_name], a_name, b_name)
                t["description"] = desc
                raw_tests.append(t)

        raw_tests.sort(key=lambda x: x["mcnemar_exact_p_value"])
        m = len(raw_tests)
        adjusted_tests = []

        for rank, t in enumerate(raw_tests):
            alpha_adj = 0.05 / (m - rank)
            p = t["mcnemar_exact_p_value"]
            t["holm_threshold"] = round(alpha_adj, 5)
            t["holm_significant"] = p < alpha_adj
            adjusted_tests.append(t)

        return {
            "battery": "M1.9f Statistical Battery (Paired Bootstrap + McNemar + Holm)",
            "hypotheses_evaluated": adjusted_tests,
        }
