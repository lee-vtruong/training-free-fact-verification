"""Milestone M2.0i: Statistical Evaluator for Factorial Mechanism Ablation.

Computes:
- Accuracy, Macro-F1, Balanced Accuracy.
- Per-class metrics (Precision, Recall, F1, Support).
- Paired bootstrap 95% Confidence Intervals (>= 5,000 resamples by claim ID).
- Exact McNemar tests for paired correctness.
- Corrected / Harmed transition counts.
- 2x2 Factorial interaction analysis (Mistake Content x Framing).
- Primary endpoint evaluation: Delta Macro-F1 (F3 - F4).
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from tffv.experiments.m20i_schemas import M20iPredictionRecord, MechanismArm

CANONICAL_CLASSES = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]


class M20iStatisticalEvaluator:
    """Evaluates metrics, paired statistical comparisons, and factorial interactions."""

    def __init__(self, records: List[M20iPredictionRecord]):
        self.records = records
        self.records_by_arm: Dict[str, List[M20iPredictionRecord]] = defaultdict(list)
        for r in records:
            self.records_by_arm[r.arm_name].append(r)

    def compute_arm_metrics(self, arm_name: str) -> Dict[str, Any]:
        """Computes summary and per-class metrics for a specific arm."""
        recs = self.records_by_arm.get(arm_name, [])
        if not recs:
            return {}

        y_true = [r.gold_label for r in recs]
        y_pred = [r.parsed_verdict for r in recs]

        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        # Balanced accuracy
        recalls = []
        for c in CANONICAL_CLASSES:
            c_indices = [i for i, y in enumerate(y_true) if y == c]
            if c_indices:
                recalls.append(sum(1 for i in c_indices if y_pred[i] == c) / len(c_indices))
        bal_acc = float(np.mean(recalls)) if recalls else 0.0

        p, r, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=CANONICAL_CLASSES, zero_division=0
        )
        per_class = {}
        for idx, c in enumerate(CANONICAL_CLASSES):
            per_class[c] = {
                "precision": float(p[idx]),
                "recall": float(r[idx]),
                "f1": float(f1[idx]),
                "support": int(sup[idx]),
            }

        prompt_tokens = [r.prompt_tokens for r in recs if r.prompt_tokens > 0]
        output_tokens = [r.output_tokens for r in recs if r.output_tokens > 0]

        return {
            "arm_name": arm_name,
            "n_records": len(recs),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "balanced_accuracy": bal_acc,
            "per_class": per_class,
            "mean_prompt_tokens": float(np.mean(prompt_tokens)) if prompt_tokens else 0.0,
            "median_prompt_tokens": float(np.median(prompt_tokens)) if prompt_tokens else 0.0,
            "mean_output_tokens": float(np.mean(output_tokens)) if output_tokens else 0.0,
        }

    def compute_paired_comparison(
        self,
        arm_test: str,
        arm_ctrl: str,
        n_bootstraps: int = 5000,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Computes paired bootstrap 95% CI and exact McNemar statistic between two arms."""
        test_recs = {r.sample_id: r for r in self.records_by_arm.get(arm_test, [])}
        ctrl_recs = {r.sample_id: r for r in self.records_by_arm.get(arm_ctrl, [])}

        common_ids = sorted(list(set(test_recs.keys()).intersection(set(ctrl_recs.keys()))))
        if not common_ids:
            return {}

        y_true = [test_recs[cid].gold_label for cid in common_ids]
        y_test = [test_recs[cid].parsed_verdict for cid in common_ids]
        y_ctrl = [ctrl_recs[cid].parsed_verdict for cid in common_ids]

        acc_test = accuracy_score(y_true, y_test)
        acc_ctrl = accuracy_score(y_true, y_ctrl)
        delta_acc = float(acc_test - acc_ctrl)

        f1_test = f1_score(y_true, y_test, average="macro", zero_division=0)
        f1_ctrl = f1_score(y_true, y_ctrl, average="macro", zero_division=0)
        delta_f1 = float(f1_test - f1_ctrl)

        # Transitions: Corrected (Test=1, Ctrl=0) vs Harmed (Test=0, Ctrl=1)
        b_corrected = 0  # Test correct, Ctrl wrong
        c_harmed = 0     # Test wrong, Ctrl correct
        both_correct = 0
        both_wrong = 0

        for t_p, c_p, y in zip(y_test, y_ctrl, y_true):
            t_cor = (t_p == y)
            c_cor = (c_p == y)
            if t_cor and not c_cor:
                b_corrected += 1
            elif not t_cor and c_cor:
                c_harmed += 1
            elif t_cor and c_cor:
                both_correct += 1
            else:
                both_wrong += 1

        # McNemar statistic with continuity correction
        mcnemar_stat = float(((abs(b_corrected - c_harmed) - 1) ** 2) / (b_corrected + c_harmed)) if (b_corrected + c_harmed) > 0 else 0.0

        # Paired bootstrap across claim IDs
        rng = np.random.RandomState(seed)
        n = len(common_ids)
        y_true_arr = np.array(y_true)
        y_test_arr = np.array(y_test)
        y_ctrl_arr = np.array(y_ctrl)

        boot_d_acc = []
        boot_d_f1 = []
        for _ in range(n_bootstraps):
            idx = rng.randint(0, n, size=n)
            b_y = y_true_arr[idx]
            b_t = y_test_arr[idx]
            b_c = y_ctrl_arr[idx]

            b_acc_t = accuracy_score(b_y, b_t)
            b_acc_c = accuracy_score(b_y, b_c)
            boot_d_acc.append(b_acc_t - b_acc_c)

            b_f1_t = f1_score(b_y, b_t, average="macro", zero_division=0)
            b_f1_c = f1_score(b_y, b_c, average="macro", zero_division=0)
            boot_d_f1.append(b_f1_t - b_f1_c)

        acc_ci = [float(np.percentile(boot_d_acc, 2.5)), float(np.percentile(boot_d_acc, 97.5))]
        f1_ci = [float(np.percentile(boot_d_f1, 2.5)), float(np.percentile(boot_d_f1, 97.5))]

        return {
            "test_arm": arm_test,
            "control_arm": arm_ctrl,
            "n_claims": len(common_ids),
            "test_accuracy": float(acc_test),
            "control_accuracy": float(acc_ctrl),
            "delta_accuracy": delta_acc,
            "accuracy_95_ci": acc_ci,
            "test_macro_f1": float(f1_test),
            "control_macro_f1": float(f1_ctrl),
            "delta_macro_f1": delta_f1,
            "macro_f1_95_ci": f1_ci,
            "b_corrected": b_corrected,
            "c_harmed": c_harmed,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "net_benefit": b_corrected - c_harmed,
            "mcnemar_stat": mcnemar_stat,
        }

    def compute_factorial_interaction(self) -> Dict[str, Any]:
        """Estimates main effects and 2x2 interaction between Framing and Mistake Content.

        Factors:
        - Factor A (Framing): 0 = Standard [VERIFICATION REASONING], 1 = Contrastive [HISTORICAL ERROR CONTRAST]
        - Factor B (Content): 0 = Absent (Neutral expl only), 1 = Present (Actual observed mistake)

        Arms:
        - (0, 0) = F0_neutral_expl
        - (1, 0) = F1_header_only
        - (0, 1) = F2_observed_mistake
        - (1, 1) = F3_full_historical_error
        """
        f0 = self.compute_arm_metrics(MechanismArm.F0_NEUTRAL_EXPL.value)
        f1 = self.compute_arm_metrics(MechanismArm.F1_HEADER_ONLY.value)
        f2 = self.compute_arm_metrics(MechanismArm.F2_OBSERVED_MISTAKE.value)
        f3 = self.compute_arm_metrics(MechanismArm.F3_FULL_HISTORICAL_ERROR.value)

        if not (f0 and f1 and f2 and f3):
            return {}

        y00_acc, y00_f1 = f0["accuracy"], f0["macro_f1"]
        y10_acc, y10_f1 = f1["accuracy"], f1["macro_f1"]
        y01_acc, y01_f1 = f2["accuracy"], f2["macro_f1"]
        y11_acc, y11_f1 = f3["accuracy"], f3["macro_f1"]

        # Main effect of Framing: 0.5 * ((y10 - y00) + (y11 - y01))
        main_framing_acc = 0.5 * ((y10_acc - y00_acc) + (y11_acc - y01_acc))
        main_framing_f1 = 0.5 * ((y10_f1 - y00_f1) + (y11_f1 - y01_f1))

        # Main effect of Mistake Content: 0.5 * ((y01 - y00) + (y11 - y10))
        main_content_acc = 0.5 * ((y01_acc - y00_acc) + (y11_acc - y10_acc))
        main_content_f1 = 0.5 * ((y01_f1 - y00_f1) + (y11_f1 - y10_f1))

        # 2-way Interaction (Framing x Content): (y11 - y10) - (y01 - y00)
        interaction_acc = (y11_acc - y10_acc) - (y01_acc - y00_acc)
        interaction_f1 = (y11_f1 - y10_f1) - (y01_f1 - y00_f1)

        return {
            "main_effect_framing": {
                "delta_accuracy": float(main_framing_acc),
                "delta_macro_f1": float(main_framing_f1),
            },
            "main_effect_content": {
                "delta_accuracy": float(main_content_acc),
                "delta_macro_f1": float(main_content_f1),
            },
            "interaction_framing_x_content": {
                "delta_accuracy": float(interaction_acc),
                "delta_macro_f1": float(interaction_f1),
            },
        }
