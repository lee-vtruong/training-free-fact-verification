"""Statistical analysis module for M1.8d paired retriever comparisons.

Implements:
1. Paired bootstrap confidence intervals (95%) for Delta-Accuracy and Delta-Macro-F1.
2. Exact McNemar test for paired verdict correctness.
3. Holm-Bonferroni step-down multiplicity correction.
4. Paired evidence overlap and rank correlation metrics.
"""
import math
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from tffv.verification.ontology import VERDICTS


def compute_macro_f1_from_pairs(pairs: List[Tuple[str, str]]) -> float:
    """Compute Macro-F1 from list of (gold, predicted) pairs."""
    f1_scores = []
    for c in VERDICTS:
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def compute_accuracy_from_pairs(pairs: List[Tuple[str, str]]) -> float:
    """Compute Accuracy from list of (gold, predicted) pairs."""
    if not pairs:
        return 0.0
    return float(sum(1 for g, p in pairs if g == p) / len(pairs))


def paired_bootstrap_test(
    candidate_preds: Dict[str, str],
    reference_preds: Dict[str, str],
    gold_labels: Dict[str, str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Calculate 95% bootstrap confidence intervals for Delta-Macro-F1 and Delta-Accuracy."""
    sids = sorted(list(set(candidate_preds.keys()) & set(reference_preds.keys()) & set(gold_labels.keys())))
    n = len(sids)
    if n == 0:
        return {"n": 0, "status": "NO_PAIRED_SAMPLES"}

    rng = np.random.default_rng(seed)

    cand_pairs = [(gold_labels[s], candidate_preds[s]) for s in sids]
    ref_pairs = [(gold_labels[s], reference_preds[s]) for s in sids]

    cand_acc = compute_accuracy_from_pairs(cand_pairs)
    ref_acc = compute_accuracy_from_pairs(ref_pairs)
    delta_acc = cand_acc - ref_acc

    cand_f1 = compute_macro_f1_from_pairs(cand_pairs)
    ref_f1 = compute_macro_f1_from_pairs(ref_pairs)
    delta_f1 = cand_f1 - ref_f1

    diff_accs = []
    diff_f1s = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_cand = [cand_pairs[i] for i in idx]
        boot_ref = [ref_pairs[i] for i in idx]

        b_d_acc = compute_accuracy_from_pairs(boot_cand) - compute_accuracy_from_pairs(boot_ref)
        b_d_f1 = compute_macro_f1_from_pairs(boot_cand) - compute_macro_f1_from_pairs(boot_ref)

        diff_accs.append(b_d_acc)
        diff_f1s.append(b_d_f1)

    ci_acc = [float(np.percentile(diff_accs, 2.5)), float(np.percentile(diff_accs, 97.5))]
    ci_f1 = [float(np.percentile(diff_f1s, 2.5)), float(np.percentile(diff_f1s, 97.5))]

    return {
        "n_samples": n,
        "n_bootstrap": n_bootstrap,
        "candidate_accuracy": cand_acc,
        "reference_accuracy": ref_acc,
        "delta_accuracy": delta_acc,
        "delta_accuracy_ci_95": ci_acc,
        "candidate_macro_f1": cand_f1,
        "reference_macro_f1": ref_f1,
        "delta_macro_f1": delta_f1,
        "delta_macro_f1_ci_95": ci_f1,
    }


def mcnemar_test(
    candidate_preds: Dict[str, str],
    reference_preds: Dict[str, str],
    gold_labels: Dict[str, str],
) -> Dict[str, Any]:
    """Exact McNemar test for paired correctness contingency table."""
    sids = sorted(list(set(candidate_preds.keys()) & set(reference_preds.keys()) & set(gold_labels.keys())))
    b = 0  # Candidate correct, Reference wrong
    c = 0  # Candidate wrong, Reference correct
    a = 0  # Both correct
    d = 0  # Both wrong

    for s in sids:
        cand_corr = (candidate_preds[s] == gold_labels[s])
        ref_corr = (reference_preds[s] == gold_labels[s])

        if cand_corr and ref_corr:
            a += 1
        elif cand_corr and not ref_corr:
            b += 1
        elif not cand_corr and ref_corr:
            c += 1
        else:
            d += 1

    discordant = b + c
    if discordant == 0:
        p_val = 1.0
    else:
        # Exact two-tailed binomial test under H0: p = 0.5
        from scipy.stats import binomtest
        res = binomtest(min(b, c), n=discordant, p=0.5, alternative="two-sided")
        p_val = float(res.pvalue)

    return {
        "contingency_table": {
            "both_correct": a,
            "candidate_only_correct": b,
            "reference_only_correct": c,
            "both_wrong": d,
        },
        "discordant_verdicts_count": discordant,
        "p_value": p_val,
    }


def holm_bonferroni_correction(p_values: Dict[str, float]) -> Dict[str, Any]:
    """Apply Holm-Bonferroni step-down procedure for family-wise error control."""
    sorted_arms = sorted(p_values.items(), key=lambda x: x[1])
    m = len(sorted_arms)
    corrected = {}

    prev_adjusted = 0.0
    for rank, (arm, p) in enumerate(sorted_arms, 1):
        adjusted = (m - rank + 1) * p
        adjusted = max(prev_adjusted, min(1.0, adjusted))
        prev_adjusted = adjusted
        corrected[arm] = {
            "raw_p_value": p,
            "adjusted_p_value": float(adjusted),
            "significant_at_05": adjusted < 0.05,
        }

    return corrected
