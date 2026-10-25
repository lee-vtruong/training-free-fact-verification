"""Audit D2 positive signal, error rules, and confirmatory statistics on Fold C."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import math
from collections import Counter, defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from tffv.verification.ontology import VERDICTS, normalize_verdict
from tffv.verification.evaluate import load_gold_labels, compute_metrics
from tffv.verification.m13_audit import find_train_json
from tffv.verification.stabilization import exact_mcnemar_test_twosided, paired_bootstrap_ci
from tffv.verification.m141_confirmatory import paired_bootstrap_macro_f1_ci

def holm_bonferroni(p_values):
    """Compute Holm-Bonferroni adjusted p-values."""
    # p_values: dict of name -> p_val
    sorted_pairs = sorted(p_values.items(), key=lambda x: x[1])
    m = len(sorted_pairs)
    adjusted = {}
    cum_max = 0.0
    for k, (name, p) in enumerate(sorted_pairs):
        # multiplier is (m - k)
        adj_p = min(1.0, p * (m - k))
        cum_max = max(cum_max, adj_p)
        adjusted[name] = min(1.0, cum_max)
    return adjusted

def main():
    root = Path(__file__).resolve().parent.parent
    train_json_p = find_train_json(project_root=root)
    gold_map = load_gold_labels(train_json_p)

    out_base = root / "outputs" / "m141-confirmatory-20261025-101555"
    conditions = ["D0", "D1", "D2", "D3", "D4", "D5"]
    available_conds = [c for c in conditions if (out_base / c / "predictions.jsonl").is_file()]

    print(f"Auditing Confirmatory Run: {out_base}")
    print(f"Available conditions: {available_conds}")

    fold_c_sids = [f"train:{i}" for i in range(100, 150)]
    classes = list(VERDICTS)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    y_gold = [gold_map[sid]["gold_label"] for sid in fold_c_sids]
    print(f"Gold label counts across 50 claims: {Counter(y_gold)}")

    cond_preds = {}
    cond_parsed = {}
    cond_metrics = {}

    for cond in available_conds:
        preds_p = out_base / cond / "predictions.jsonl"
        lines = [json.loads(l) for l in preds_p.read_text(encoding="utf-8").splitlines()]
        by_sid = {r["sample_id"]: r for r in lines}
        
        preds = []
        raw_preds = []
        for sid in fold_c_sids:
            r = by_sid.get(sid, {})
            p = r.get("predicted_verdict", "Unparseable")
            preds.append(p)
            raw_preds.append(r)
        
        cond_preds[cond] = preds
        cond_parsed[cond] = raw_preds

        # Compute with compute_metrics
        eval_rows = [
            {"sample_id": sid, "gold_label": g, "predicted_verdict": p, "is_correct": bool(g == p)}
            for sid, g, p in zip(fold_c_sids, y_gold, preds)
        ]
        m = compute_metrics(eval_rows)

        # Cross-check with scikit-learn
        y_gold_arr = [g for g in y_gold]
        # In sklearn, replace Unparseable with dummy not in classes
        y_pred_sklearn = [p if p in classes else "Unparseable" for p in preds]
        sk_acc = accuracy_score(y_gold_arr, y_pred_sklearn)
        sk_p, sk_r, sk_f1, sk_supp = precision_recall_fscore_support(
            y_gold_arr, y_pred_sklearn, labels=classes, zero_division=0
        )
        sk_macro_f1 = np.mean(sk_f1)
        sk_macro_p = np.mean(sk_p)
        sk_macro_r = np.mean(sk_r)

        assert abs(m["accuracy"] - sk_acc) < 1e-9
        assert abs(m["macro_f1"] - sk_macro_f1) < 1e-9

        cond_metrics[cond] = {
            "acc": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "macro_p": m["macro_precision"],
            "macro_r": m["macro_recall"],
            "correct": sum(1 for g, p in zip(y_gold, preds) if g == p),
            "per_class": m["per_class"],
            "confusion_matrix": m["confusion_matrix"],
            "pred_counts": Counter(preds),
        }

    print("\n" + "="*80)
    print("TABLE 1: RECOMPUTED PERFORMANCE SUMMARY (FOLD C, N=50)")
    print("="*80)
    print(f"{'Cond':<6} {'Correct':<10} {'Accuracy':<12} {'Macro-F1':<12} {'Macro-P':<10} {'Macro-R':<10} {'Pred Counts'}")
    print("-" * 80)
    for cond in available_conds:
        cm = cond_metrics[cond]
        counts_str = ", ".join(f"{k}:{v}" for k, v in sorted(cm["pred_counts"].items()))
        print(f"{cond:<6} {cm['correct']}/50      {cm['acc']*100:.1f}%        {cm['macro_f1']:.4f}      {cm['macro_p']:.4f}     {cm['macro_r']:.4f}     {counts_str}")

    # Paired comparisons
    pairs = [
        ("D4", "D0"),
        ("D4", "D3"),
        ("D4", "D5"),
        ("D2", "D0"),
        ("D2", "D3"),
    ]

    print("\n" + "="*80)
    print("TABLE 2: PAIRED COMPARISONS, MCNEMAR & BOOTSTRAP CIS (10,000 RESAMPLES)")
    print("="*80)

    raw_p_values = {}
    pair_details = {}

    for target_c, base_c in pairs:
        if target_c not in available_conds or base_c not in available_conds:
            continue
        p_t = cond_preds[target_c]
        p_b = cond_preds[base_c]

        # 9 mutually exclusive categories
        categories = {
            "C_to_C": 0, "C_to_W": 0, "C_to_U": 0,
            "W_to_C": 0, "W_to_W": 0, "W_to_U": 0,
            "U_to_C": 0, "U_to_W": 0, "U_to_U": 0,
        }
        improved_sids = []
        regressed_sids = []

        bin_b = []
        bin_t = []

        for sid, g, b, t in zip(fold_c_sids, y_gold, p_b, p_t):
            b_corr = (b == g)
            t_corr = (t == g)
            b_unp = (b not in classes)
            t_unp = (t not in classes)

            bin_b.append(1 if b_corr else 0)
            bin_t.append(1 if t_corr else 0)

            if b_corr:
                if t_corr: categories["C_to_C"] += 1
                elif t_unp: categories["C_to_U"] += 1; regressed_sids.append(sid)
                else: categories["C_to_W"] += 1; regressed_sids.append(sid)
            elif b_unp:
                if t_corr: categories["U_to_C"] += 1; improved_sids.append(sid)
                elif t_unp: categories["U_to_U"] += 1
                else: categories["U_to_W"] += 1
            else: # b is wrong
                if t_corr: categories["W_to_C"] += 1; improved_sids.append(sid)
                elif t_unp: categories["W_to_U"] += 1
                else: categories["W_to_W"] += 1

        total_cat = sum(categories.values())
        assert total_cat == 50, f"Sum must be 50, got {total_cat}"

        b_disc = categories["C_to_W"] + categories["C_to_U"]
        c_disc = categories["W_to_C"] + categories["U_to_C"]
        mcnemar_p = exact_mcnemar_test_twosided(b_disc, c_disc)
        raw_p_values[f"{target_c}_vs_{base_c}"] = mcnemar_p

        acc_low, acc_high = paired_bootstrap_ci(bin_b, bin_t, n_boot=10000, seed=42)
        f1_diff, f1_low, f1_high = paired_bootstrap_macro_f1_ci(
            y_gold, p_b, p_t, classes=classes, n_boot=10000, seed=42
        )

        net = c_disc - b_disc
        pair_details[f"{target_c}_vs_{base_c}"] = {
            "categories": categories,
            "net": net,
            "b_disc": b_disc,
            "c_disc": c_disc,
            "p": mcnemar_p,
            "acc_ci": (acc_low, acc_high),
            "f1_ci": (f1_low, f1_high),
            "f1_diff": f1_diff,
            "improved_sids": improved_sids,
            "regressed_sids": regressed_sids,
        }

    # Holm-Bonferroni correction
    adjusted_p = holm_bonferroni(raw_p_values)

    print(f"{'Comparison':<12} {'Net':<6} {'C->C':<5} {'C->W':<5} {'W->C':<5} {'W->W':<5} {'Acc Delta [95% CI]':<26} {'F1 Delta [95% CI]':<26} {'Raw p':<10} {'Holm p':<10} {'Sig (0.05)'}")
    print("-" * 125)
    for pair_name, d in pair_details.items():
        c = d["categories"]
        acc_str = f"{d['acc_ci'][0]:+.2f}, {d['acc_ci'][1]:+.2f}"
        f1_str = f"{d['f1_diff']:+.4f} [{d['f1_ci'][0]:+.4f}, {d['f1_ci'][1]:+.4f}]"
        sig_str = "YES" if adjusted_p[pair_name] < 0.05 else "NO"
        raw_p_str = f"{d['p']:.4f}"
        adj_p_str = f"{adjusted_p[pair_name]:.4f}"
        print(f"{pair_name:<12} {d['net']:<+6} {c['C_to_C']:<5} {c['C_to_W']:<5} {c['W_to_C']:<5} {c['W_to_W']:<5} [{acc_str:<23}] {f1_str:<26} {raw_p_str:<10} {adj_p_str:<10} {sig_str}")

    # Inspect D2 vs D0 improvements & regressions
    print("\n" + "="*80)
    print("DETAILED CASE AUDIT: D2 vs D0 TRANSITIONS")
    print("="*80)
    d2_d0 = pair_details["D2_vs_D0"]
    print(f"Improved claims ({len(d2_d0['improved_sids'])}): {d2_d0['improved_sids']}")
    for sid in d2_d0["improved_sids"]:
        idx = fold_c_sids.index(sid)
        g = y_gold[idx]
        b = cond_preds["D0"][idx]
        t = cond_preds["D2"][idx]
        raw_b = cond_parsed["D0"][idx].get("raw_generation", "")[:80].replace("\n", " ")
        raw_t = cond_parsed["D2"][idx].get("raw_generation", "")[:80].replace("\n", " ")
        claim = gold_map[sid]["claim"][:60]
        print(f"  {sid}: Gold={g} | D0={b} -> D2={t} | Claim: \"{claim}...\"")
        print(f"     D0 text: {raw_b}")
        print(f"     D2 text: {raw_t}")

    print(f"\nRegressed claims ({len(d2_d0['regressed_sids'])}): {d2_d0['regressed_sids']}")
    for sid in d2_d0["regressed_sids"]:
        idx = fold_c_sids.index(sid)
        g = y_gold[idx]
        b = cond_preds["D0"][idx]
        t = cond_preds["D2"][idx]
        raw_b = cond_parsed["D0"][idx].get("raw_generation", "")[:80].replace("\n", " ")
        raw_t = cond_parsed["D2"][idx].get("raw_generation", "")[:80].replace("\n", " ")
        claim = gold_map[sid]["claim"][:60]
        print(f"  {sid}: Gold={g} | D0={b} -> D2={t} | Claim: \"{claim}...\"")
        print(f"     D0 text: {raw_b}")
        print(f"     D2 text: {raw_t}")

    # Per-class comparison D2 vs D0 vs D3
    print("\n" + "="*80)
    print("PER-CLASS METRICS: D0 vs D2 vs D3")
    print("="*80)
    for cls in classes:
        m0 = cond_metrics["D0"]["per_class"][cls]
        m2 = cond_metrics["D2"]["per_class"][cls]
        m3 = cond_metrics["D3"]["per_class"][cls]
        print(f"Class: {cls} (Support = {m0['support']})")
        print(f"  D0: P={m0['precision']:.4f}, R={m0['recall']:.4f}, F1={m0['f1']:.4f}")
        print(f"  D2: P={m2['precision']:.4f}, R={m2['recall']:.4f}, F1={m2['f1']:.4f}")
        print(f"  D3: P={m3['precision']:.4f}, R={m3['recall']:.4f}, F1={m3['f1']:.4f}")

    # Confusion matrices
    print("\n" + "="*80)
    print("CONFUSION MATRICES (Rows: Gold, Columns: Pred [Supp, Ref, NEI, Conf, Unp])")
    print("="*80)
    for cond in ["D0", "D2", "D3", "D4"]:
        print(f"\n--- Condition {cond} ---")
        cm = cond_metrics[cond]["confusion_matrix"]
        for g in classes:
            row_vals = [cm[g][p] for p in classes + ["Unparseable"]]
            print(f"  {g:<35}: {row_vals}")

if __name__ == "__main__":
    main()
