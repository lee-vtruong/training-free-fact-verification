"""
Milestone M1.9: Offline Evaluator & Statistical Inference Suite.

Computes 4-class Macro-F1, Accuracy, Per-Class metrics, Confusion Matrices,
and rigorous paired statistical inference (Bootstrap 95% CIs, exact McNemar test,
Holm-Bonferroni correction).
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
    per_class_f1: Dict[str, float]
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_support: Dict[str, int]
    confusion_matrix: Dict[str, Dict[str, int]]


class M19Evaluator:
    """Computes full evaluation metrics and statistical hypothesis tests."""

    def __init__(self, bootstrap_resamples: int = 5000, seed: int = 42):
        self.bootstrap_resamples = bootstrap_resamples
        self.seed = seed

    def compute_arm_metrics(
        self,
        arm: str,
        predictions: List[Dict[str, Any]],
    ) -> MetricSummary:
        """
        predictions: list of dicts with:
        - sample_id
        - gold_verdict
        - predicted_verdict
        - is_parseable
        """
        total = len(predictions)
        if total == 0:
            raise ValueError(f"No predictions provided for arm {arm}")

        conf_mat: Dict[str, Dict[str, int]] = {
            g: {p: 0 for p in AVERITEC_CLASSES} for g in AVERITEC_CLASSES
        }
        correct = 0
        unparseable = 0

        for r in predictions:
            gold = r.get("gold_verdict")
            pred = r.get("predicted_verdict")
            parseable = r.get("is_parseable", True)

            if not parseable or pred not in AVERITEC_CLASSES:
                unparseable += 1

            if gold in conf_mat and pred in conf_mat[gold]:
                conf_mat[gold][pred] += 1

            if gold == pred:
                correct += 1

        accuracy = correct / total
        valid_rate = (total - unparseable) / total

        # Per-class P, R, F1
        p_dict: Dict[str, float] = {}
        r_dict: Dict[str, float] = {}
        f1_dict: Dict[str, float] = {}
        support_dict: Dict[str, int] = {}

        for cls in AVERITEC_CLASSES:
            tp = conf_mat[cls][cls]
            fn = sum(conf_mat[cls][p] for p in AVERITEC_CLASSES if p != cls)
            fp = sum(conf_mat[g][cls] for g in AVERITEC_CLASSES if g != cls)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            p_dict[cls] = round(prec, 4)
            r_dict[cls] = round(rec, 4)
            f1_dict[cls] = round(f1, 4)
            support_dict[cls] = tp + fn

        macro_f1 = sum(f1_dict.values()) / len(AVERITEC_CLASSES)
        bal_acc = sum(r_dict.values()) / len(AVERITEC_CLASSES)

        return MetricSummary(
            arm=arm,
            macro_f1=round(macro_f1, 4),
            accuracy=round(accuracy, 4),
            balanced_accuracy=round(bal_acc, 4),
            valid_verdict_rate=round(valid_rate, 4),
            total_claims=total,
            correct_count=correct,
            unparseable_count=unparseable,
            per_class_f1=f1_dict,
            per_class_precision=p_dict,
            per_class_recall=r_dict,
            per_class_support=support_dict,
            confusion_matrix=conf_mat,
        )

    def _exact_mcnemar_p_value(self, b: int, c: int) -> float:
        """
        Exact two-tailed binomial test for paired discordant pairs.
        H0: p = 0.5 under Binomial(n = b + c, 0.5)
        """
        n = b + c
        if n == 0:
            return 1.0
        k = min(b, c)
        prob_sum = 0.0
        for i in range(k + 1):
            prob_sum += math.comb(n, i) * (0.5 ** n)
        two_tailed_p = min(1.0, 2.0 * prob_sum)
        return round(two_tailed_p, 5)

    def paired_bootstrap_test(
        self,
        preds_arm_a: List[Dict[str, Any]],
        preds_arm_b: List[Dict[str, Any]],
        arm_a_name: str,
        arm_b_name: str,
    ) -> Dict[str, Any]:
        """
        Computes paired bootstrap 95% CIs for ΔMacro-F1 and ΔAccuracy (Arm A - Arm B),
        and exact McNemar test on paired correctness.
        """
        if len(preds_arm_a) != len(preds_arm_b):
            raise ValueError(f"Mismatched prediction lengths: {len(preds_arm_a)} vs {len(preds_arm_b)}")

        n = len(preds_arm_a)
        # Verify alignment by sample_id
        paired = []
        b_discordant = 0  # A correct, B wrong
        c_discordant = 0  # A wrong, B correct
        a_concordant_correct = 0
        d_concordant_wrong = 0

        for r_a, r_b in zip(preds_arm_a, preds_arm_b):
            if r_a["sample_id"] != r_b["sample_id"]:
                raise ValueError(f"Unaligned sample IDs: {r_a['sample_id']} vs {r_b['sample_id']}")

            gold = r_a["gold_verdict"]
            p_a = r_a["predicted_verdict"]
            p_b = r_b["predicted_verdict"]

            is_correct_a = (p_a == gold)
            is_correct_b = (p_b == gold)

            if is_correct_a and not is_correct_b:
                b_discordant += 1
            elif not is_correct_a and is_correct_b:
                c_discordant += 1
            elif is_correct_a and is_correct_b:
                a_concordant_correct += 1
            else:
                d_concordant_wrong += 1

            paired.append((gold, p_a, p_b))

        mcnemar_p = self._exact_mcnemar_p_value(b_discordant, c_discordant)

        # Baseline observed deltas
        base_a = self.compute_arm_metrics(arm_a_name, preds_arm_a)
        base_b = self.compute_arm_metrics(arm_b_name, preds_arm_b)
        obs_delta_f1 = base_a.macro_f1 - base_b.macro_f1
        obs_delta_acc = base_a.accuracy - base_b.accuracy

        # Paired bootstrap
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

        f1_ci = (round(boot_delta_f1[low_idx], 4), round(boot_delta_f1[high_idx], 4))
        acc_ci = (round(boot_delta_acc[low_idx], 4), round(boot_delta_acc[high_idx], 4))

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
            "mcnemar_exact_p_value": mcnemar_p,
            "is_significant_at_05": (f1_ci[0] > 0 or f1_ci[1] < 0) and mcnemar_p < 0.05,
        }

    def run_full_statistical_battery(
        self,
        all_arm_preds: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Runs primary and secondary hypothesis tests with Holm-Bonferroni correction."""
        hypothesis_pairs = [
            ("H2", "B1", "Primary: Error-History ICL vs Matched Gold-Only ICL"),
            ("H2", "B0", "Secondary: Error-History ICL vs Zero-Shot Base"),
            ("H3", "B2", "Secondary: Mixed-History ICL vs Matched Gold-Only ICL"),
            ("H3", "B0", "Secondary: Mixed-History ICL vs Zero-Shot Base"),
            ("H1", "B0", "Secondary: Correct-History ICL vs Zero-Shot Base"),
        ]

        raw_tests = []
        for a_name, b_name, desc in hypothesis_pairs:
            if a_name in all_arm_preds and b_name in all_arm_preds:
                t = self.paired_bootstrap_test(all_arm_preds[a_name], all_arm_preds[b_name], a_name, b_name)
                t["description"] = desc
                raw_tests.append(t)

        # Holm-Bonferroni correction on p-values
        raw_tests.sort(key=lambda x: x["mcnemar_exact_p_value"])
        m = len(raw_tests)
        adjusted_tests = []

        for rank, t in enumerate(raw_tests):
            alpha_adj = 0.05 / (m - rank)
            p = t["mcnemar_exact_p_value"]
            t["holm_alpha_threshold"] = round(alpha_adj, 5)
            t["holm_significant"] = p < alpha_adj
            adjusted_tests.append(t)

        return {
            "battery": "M1.9 Statistical Battery (Paired Bootstrap + McNemar + Holm)",
            "hypotheses_evaluated": adjusted_tests,
        }
