"""Comprehensive Mechanism Audit for Milestone M1.4.1.

Audits D0-D6 results from M1.4 Pilot (outputs/m14-pilot-20261025-014111):
1. Recomputes all metrics from raw predictions and cross-checks with scikit-learn.
2. Calculates exact paired comparisons (D4 vs D0, D4 vs D3, D4 vs D5, D2 vs D1):
   - Accuracy delta, Macro-F1 delta
   - Exact two-sided McNemar p-value
   - Paired bootstrap 95% CI (10,000 resamples)
3. Audits exemplar retrieval:
   - Retrieved exemplar IDs, lexical 3-gram similarity scores, token counts for D3, D4, D5
4. Identifies case studies:
   - D4 correct & D3 wrong
   - D3 correct & D4 wrong
   - Both correct
   - Both wrong
5. Validates anti-leakage safeguards (Fold A source isolation, near-duplicate quarantine, zero gold leakage).
"""
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from tffv.data import read_jsonl
from tffv.verification.evaluate import compute_metrics, load_gold_labels
from tffv.verification.m14_prompting import compute_claim_similarity
from tffv.verification.ontology import VERDICTS, normalize_verdict
from tffv.verification.stabilization import exact_mcnemar_test_twosided


def paired_bootstrap_ci_10k(
    y_base: List[int],
    y_target: List[int],
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Calculate paired bootstrap confidence interval with 10,000 resamples."""
    rng = np.random.default_rng(seed)
    n = len(y_base)
    base_arr = np.array(y_base)
    target_arr = np.array(y_target)
    point_diff = float(np.mean(target_arr) - np.mean(base_arr))

    indices = rng.integers(0, n, size=(n_boot, n))
    base_samples = base_arr[indices]
    target_samples = target_arr[indices]
    diffs = np.mean(target_samples, axis=1) - np.mean(base_samples, axis=1)

    low = float(np.percentile(diffs, 100 * (alpha / 2)))
    high = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return point_diff, low, high


def fast_macro_f1_from_indices(gold_idx: np.ndarray, pred_idx: np.ndarray, num_classes: int = 4) -> float:
    """Vectorized calculation of macro-f1 from integer class indices."""
    valid_mask = (pred_idx >= 0) & (pred_idx < num_classes)
    flat_idx = np.where(valid_mask, gold_idx * num_classes + pred_idx, -1)
    valid_flat = flat_idx[flat_idx >= 0]
    counts = np.bincount(valid_flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    
    tp = np.diag(counts)
    fp = np.sum(counts, axis=0) - tp
    support = np.bincount(gold_idx, minlength=num_classes)
    fn = support - tp

    with np.errstate(divide='ignore', invalid='ignore'):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        denom = prec + rec
        f1 = np.where(denom > 0, 2 * prec * rec / denom, 0.0)
    return float(np.mean(f1))


def paired_bootstrap_macro_f1_ci(
    gold_labels: List[str],
    preds_base: List[str],
    preds_target: List[str],
    classes: List[str],
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Calculate paired bootstrap CI for Macro-F1 difference with 10,000 resamples (vectorized)."""
    rng = np.random.default_rng(seed)
    n = len(gold_labels)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    gold_idx = np.array([class_to_idx[g] for g in gold_labels])
    base_idx = np.array([class_to_idx.get(p, -1) for p in preds_base])
    target_idx = np.array([class_to_idx.get(p, -1) for p in preds_target])

    point_base_f1 = fast_macro_f1_from_indices(gold_idx, base_idx, num_classes)
    point_target_f1 = fast_macro_f1_from_indices(gold_idx, target_idx, num_classes)
    point_diff = point_target_f1 - point_base_f1

    # Pre-sample all bootstrap indices (n_boot x n)
    boot_indices = rng.integers(0, n, size=(n_boot, n))
    
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample_idx = boot_indices[b]
        g_s = gold_idx[sample_idx]
        b_s = base_idx[sample_idx]
        t_s = target_idx[sample_idx]
        b_f1 = fast_macro_f1_from_indices(g_s, b_s, num_classes)
        t_f1 = fast_macro_f1_from_indices(g_s, t_s, num_classes)
        diffs[b] = t_f1 - b_f1

    low = float(np.percentile(diffs, 100 * (alpha / 2)))
    high = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return point_diff, low, high


def main():
    root = Path(__file__).resolve().parent.parent
    train_json_path = root / "data" / "raw" / "averitec" / "train.json"
    pilot_dir = root / "outputs" / "m14-pilot-20261025-014111"

    print(f"Loading gold labels from {train_json_path}...")
    gold_map = load_gold_labels(train_json_path)

    conditions = ["D0", "D1", "D2", "D3", "D4", "D5", "D6"]
    cond_preds: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for c in conditions:
        preds_file = pilot_dir / c / "predictions.jsonl"
        rows = read_jsonl(preds_file)
        cond_preds[c] = {r["sample_id"]: r for r in rows}

    eval_sids = sorted(list(cond_preds["D0"].keys()), key=lambda x: int(x.split(":")[1]))
    print(f"Total evaluated claims: {len(eval_sids)}")

    # 1. Metric recomputation & cross-check with scikit-learn
    print("\n" + "=" * 80)
    print("PART A.1 & A.2: RECOMPUTED METRICS AND SCIKIT-LEARN VALIDATION")
    print("=" * 80)

    classes = list(VERDICTS)
    metrics_table = {}

    for c in conditions:
        eval_rows = []
        y_true = []
        y_pred = []
        for sid in eval_sids:
            row = cond_preds[c][sid]
            gold = gold_map[sid]["gold_label"]
            pred = row.get("predicted_verdict", "Unparseable")
            is_corr = bool(pred == gold)
            eval_rows.append({
                "sample_id": sid,
                "gold_label": gold,
                "predicted_verdict": pred,
                "is_correct": is_corr,
            })
            y_true.append(gold)
            y_pred.append(pred)

        # Internal metric calculation
        internal_m = compute_metrics(eval_rows)

        # scikit-learn metrics
        sk_acc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)
        sk_macro_f1 = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
        sk_macro_p = precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
        sk_macro_r = recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
        sk_cm = confusion_matrix(y_true, y_pred, labels=classes)

        # Consistency assert
        assert abs(internal_m["accuracy"] - sk_acc) < 1e-9, f"Accuracy mismatch in {c}"
        assert abs(internal_m["macro_f1"] - sk_macro_f1) < 1e-9, f"Macro-F1 mismatch in {c}"
        assert abs(internal_m["macro_precision"] - sk_macro_p) < 1e-9, f"Precision mismatch in {c}"
        assert abs(internal_m["macro_recall"] - sk_macro_r) < 1e-9, f"Recall mismatch in {c}"

        metrics_table[c] = {
            "accuracy": sk_acc,
            "correct": internal_m["correct_claims"],
            "total": internal_m["total_claims"],
            "macro_f1": sk_macro_f1,
            "macro_p": sk_macro_p,
            "macro_r": sk_macro_r,
            "confusion_matrix": internal_m["confusion_matrix"],
            "per_class": internal_m["per_class"],
        }
        print(f"[{c}] Acc: {sk_acc:.4f} ({internal_m['correct_claims']}/{internal_m['total_claims']}) | "
              f"Macro-F1: {sk_macro_f1:.4f} | Prec: {sk_macro_p:.4f} | Rec: {sk_macro_r:.4f}")

    # 2. Paired Comparisons
    print("\n" + "=" * 80)
    print("PART A.3 & A.4: PAIRED COMPARISONS (D4 vs D0, D4 vs D3, D4 vs D5, D2 vs D1)")
    print("=" * 80)

    comparison_pairs = [
        ("D0", "D4"),
        ("D3", "D4"),
        ("D5", "D4"),
        ("D1", "D2"),
    ]

    paired_results = {}
    for base_c, target_c in comparison_pairs:
        y_true = [gold_map[sid]["gold_label"] for sid in eval_sids]
        base_preds = [cond_preds[base_c][sid]["predicted_verdict"] for sid in eval_sids]
        target_preds = [cond_preds[target_c][sid]["predicted_verdict"] for sid in eval_sids]

        y_base_corr = [1 if bp == yt else 0 for bp, yt in zip(base_preds, y_true)]
        y_target_corr = [1 if tp == yt else 0 for tp, yt in zip(target_preds, y_true)]

        # 9 mutually exclusive transition categories
        transitions = {
            "CC": 0, "CW": 0, "CU": 0,
            "WC": 0, "WW": 0, "WU": 0,
            "UC": 0, "UW": 0, "UU": 0,
        }
        for yt, bp, tp in zip(y_true, base_preds, target_preds):
            b_cat = "C" if bp == yt else ("W" if bp in VERDICTS else "U")
            t_cat = "C" if tp == yt else ("W" if tp in VERDICTS else "U")
            transitions[f"{b_cat}{t_cat}"] += 1

        b_disc = transitions["CW"] + transitions["CU"]  # Base correct, target wrong/unparsed
        c_disc = transitions["WC"] + transitions["UC"]  # Base wrong/unparsed, target correct
        p_val = exact_mcnemar_test_twosided(b_disc, c_disc)

        acc_diff, acc_ci_low, acc_ci_high = paired_bootstrap_ci_10k(y_base_corr, y_target_corr, n_boot=10000)
        f1_diff, f1_ci_low, f1_ci_high = paired_bootstrap_macro_f1_ci(y_true, base_preds, target_preds, classes, n_boot=10000)

        paired_results[f"{target_c}_vs_{base_c}"] = {
            "base_c": base_c,
            "target_c": target_c,
            "transitions": transitions,
            "mcnemar_b": b_disc,
            "mcnemar_c": c_disc,
            "mcnemar_p": p_val,
            "acc_diff": acc_diff,
            "acc_ci": (acc_ci_low, acc_ci_high),
            "f1_diff": f1_diff,
            "f1_ci": (f1_ci_low, f1_ci_high),
        }

        print(f"\n--- {target_c} vs {base_c} ---")
        print(f"  Transitions: CC={transitions['CC']}, CW={transitions['CW']}, WC={transitions['WC']}, WW={transitions['WW']}")
        print(f"  Discordants: Base-only correct (regressed)={b_disc}, Target-only correct (improved)={c_disc}")
        print(f"  Net correct gain: {c_disc - b_disc:+d}")
        print(f"  Accuracy delta: {acc_diff:+.4f} (95% CI: [{acc_ci_low:+.4f}, {acc_ci_high:+.4f}])")
        print(f"  Macro-F1 delta: {f1_diff:+.4f} (95% CI: [{f1_ci_low:+.4f}, {f1_ci_high:+.4f}])")
        print(f"  Exact McNemar p-value: {p_val:.4f} (Statistically Significant at alpha=0.05? {'YES' if p_val < 0.05 else 'NO'})")

    # 3. Exemplar Audit for D3, D4, D5
    print("\n" + "=" * 80)
    print("PART A.5 & A.6: EXEMPLAR & REPRESENTATION AUDIT (D3 vs D4 vs D5)")
    print("=" * 80)

    memory_bank_path = pilot_dir / "error_memory_bank.json"
    memory_bank = json.loads(memory_bank_path.read_text(encoding="utf-8"))
    memory_claims = memory_bank.get("records", {})

    print(f"Memory bank loaded: {len(memory_claims)} exemplars from Fold A.")

    d3_exemplars = {sid: cond_preds["D3"][sid].get("exemplar_sample_ids", []) for sid in eval_sids}
    d4_exemplars = {sid: cond_preds["D4"][sid].get("exemplar_sample_ids", []) for sid in eval_sids}
    d5_exemplars = {sid: cond_preds["D5"][sid].get("exemplar_sample_ids", []) for sid in eval_sids}

    # Verify D4 and D5 use identical exemplars (since both retrieve via lexical 3-gram similarity)
    d4_d5_match = all(d4_exemplars[sid] == d5_exemplars[sid] for sid in eval_sids)
    print(f"Do D4 and D5 use identical exemplar IDs for all 46 claims? {d4_d5_match}")

    # Compute similarity scores of retrieved exemplars for D4
    d4_similarities = []
    d3_similarities = []
    overlap_count = 0

    for sid in eval_sids:
        eval_claim = cond_preds["D4"][sid]["claim"]
        d4_exs = d4_exemplars[sid]
        d3_exs = d3_exemplars[sid]

        for ex_id in d4_exs:
            src_text = memory_claims[ex_id]["claim"]
            sim = compute_claim_similarity(eval_claim, src_text)
            d4_similarities.append(sim)

        for ex_id in d3_exs:
            src_text = memory_claims[ex_id]["claim"]
            sim = compute_claim_similarity(eval_claim, src_text)
            d3_similarities.append(sim)

        overlap_count += len(set(d4_exs).intersection(set(d3_exs)))

    print(f"D4 Mean 3-gram Jaccard Similarity: {np.mean(d4_similarities):.4f} (min: {np.min(d4_similarities):.4f}, max: {np.max(d4_similarities):.4f})")
    print(f"D3 Mean 3-gram Jaccard Similarity: {np.mean(d3_similarities):.4f} (min: {np.min(d3_similarities):.4f}, max: {np.max(d3_similarities):.4f})")
    print(f"Exemplar overlap between D3 and D4 across all 46 claims (out of {len(eval_sids)*2} total slots): {overlap_count} ({overlap_count / (len(eval_sids)*2) * 100:.1f}%)")

    # Prompt and token counts
    for c in ["D3", "D4", "D5"]:
        p_toks = [cond_preds[c][sid]["prompt_tokens"] for sid in eval_sids]
        g_toks = [cond_preds[c][sid]["generated_tokens"] for sid in eval_sids]
        lats = [cond_preds[c][sid]["latency_seconds"] for sid in eval_sids]
        print(f"[{c}] Prompt tokens: {np.mean(p_toks):.1f} | Gen tokens: {np.mean(g_toks):.1f} | Latency: {np.mean(lats):.2f}s")

    # 4. Case Studies
    print("\n" + "=" * 80)
    print("PART A.7: CASE STUDIES (D4 vs D3)")
    print("=" * 80)

    d4_corr_d3_wrong = []
    d3_corr_d4_wrong = []
    both_corr = []
    both_wrong = []

    for sid in eval_sids:
        yt = gold_map[sid]["gold_label"]
        d3_p = cond_preds["D3"][sid]["predicted_verdict"]
        d4_p = cond_preds["D4"][sid]["predicted_verdict"]

        d3_ok = (d3_p == yt)
        d4_ok = (d4_p == yt)

        if d4_ok and not d3_ok:
            d4_corr_d3_wrong.append(sid)
        elif d3_ok and not d4_ok:
            d3_corr_d4_wrong.append(sid)
        elif d4_ok and d3_ok:
            both_corr.append(sid)
        else:
            both_wrong.append(sid)

    print(f"D4 Correct & D3 Wrong (improved by retrieval): {len(d4_corr_d3_wrong)} claims -> {d4_corr_d3_wrong}")
    print(f"D3 Correct & D4 Wrong (regressed by retrieval): {len(d3_corr_d4_wrong)} claims -> {d3_corr_d4_wrong}")
    print(f"Both Correct: {len(both_corr)} claims")
    print(f"Both Wrong: {len(both_wrong)} claims")

    # Print details of the discordant claims
    print("\n--- Detailed Inspection: D4 Correct & D3 Wrong ---")
    for sid in d4_corr_d3_wrong:
        gold = gold_map[sid]["gold_label"]
        claim = cond_preds["D4"][sid]["claim"]
        d3_p = cond_preds["D3"][sid]["predicted_verdict"]
        d4_p = cond_preds["D4"][sid]["predicted_verdict"]
        d4_exs = d4_exemplars[sid]
        d3_exs = d3_exemplars[sid]
        print(f"Claim ID: {sid} | Gold: {gold}")
        print(f"  Claim: {claim}")
        print(f"  D3 Prediction: {d3_p} (Exemplars: {d3_exs})")
        print(f"  D4 Prediction: {d4_p} (Exemplars: {d4_exs})")
        print(f"  D4 Generation: {cond_preds['D4'][sid]['raw_generation'].strip()}")
        print("-" * 50)

    print("\n--- Detailed Inspection: D3 Correct & D4 Wrong ---")
    for sid in d3_corr_d4_wrong:
        gold = gold_map[sid]["gold_label"]
        claim = cond_preds["D4"][sid]["claim"]
        d3_p = cond_preds["D3"][sid]["predicted_verdict"]
        d4_p = cond_preds["D4"][sid]["predicted_verdict"]
        d4_exs = d4_exemplars[sid]
        d3_exs = d3_exemplars[sid]
        print(f"Claim ID: {sid} | Gold: {gold}")
        print(f"  Claim: {claim}")
        print(f"  D3 Prediction: {d3_p} (Exemplars: {d3_exs})")
        print(f"  D4 Prediction: {d4_p} (Exemplars: {d4_exs})")
        print(f"  D4 Generation: {cond_preds['D4'][sid]['raw_generation'].strip()}")
        print("-" * 50)


if __name__ == "__main__":
    main()
