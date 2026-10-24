"""Evaluation and paired transition analysis for Frozen Base Stabilization (B0, B1, B2)."""
import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tffv.provenance import canonical, digest, file_hash, write_json
from tffv.verification.evaluate import compute_metrics, load_gold_labels, run_evaluation
from tffv.verification.ontology import VERDICTS


def _is_verdict_correct(row: Dict[str, Any]) -> bool:
    """Verify verdict correctness directly from predicted_verdict and gold_label.
    
    CRITICAL: Never trust raw 'is_correct' field directly from CSV without validation,
    because CSV parses values as strings where bool('False') evaluates to True!
    """
    pred = (row.get("predicted_verdict") or "").strip()
    gold = (row.get("gold_label") or "").strip()
    return bool(pred and pred in VERDICTS and pred == gold)


def _get_claim_status(row: Dict[str, Any]) -> str:
    """Get standardized status: 'correct', 'wrong', or 'unparsed'."""
    pred = (row.get("predicted_verdict") or "").strip()
    gold = (row.get("gold_label") or "").strip()
    if not pred or pred == "Unparseable" or pred not in VERDICTS:
        return "unparsed"
    if pred == gold:
        return "correct"
    return "wrong"


def exact_mcnemar_test_twosided(b: int, c: int) -> float:
    """Compute exact two-sided McNemar p-value via Binomial test under p=0.5."""
    n_disc = b + c
    if n_disc == 0:
        return 1.0
    m = min(b, c)
    prob_tails = sum(math.comb(n_disc, i) for i in range(m + 1)) * (0.5 ** n_disc)
    return min(1.0, 2.0 * prob_tails)


def paired_bootstrap_ci(
    y1: List[int],
    y2: List[int],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute paired bootstrap (1-alpha) confidence interval for accuracy difference (y2 - y1)."""
    if len(y1) != len(y2) or not y1:
        return (0.0, 0.0)
    
    diffs = [y2[i] - y1[i] for i in range(len(y1))]
    n = len(diffs)
    
    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_boot):
        sample = [rng.choice(diffs) for _ in range(n)]
        boot_means.append(sum(sample) / n)
        
    boot_means.sort()
    low_idx = int((alpha / 2.0) * n_boot)
    high_idx = int((1.0 - alpha / 2.0) * n_boot)
    return (boot_means[low_idx], boot_means[high_idx])


def analyze_paired_transitions(
    b0_eval_rows: List[Dict[str, Any]],
    target_eval_rows: List[Dict[str, Any]],
    target_name: str,
    baseline_name: str = "B0",
    n_bootstrap: int = 10000,
) -> Dict[str, Any]:
    """Compute paired per-claim verdict transitions from baseline to target configuration.
    
    Performs defensive verification on claim IDs, prevents string-boolean coercion bugs,
    and asserts mathematical invariants across transition categories.
    """
    # 1. Defensive Check: Duplicate Claim IDs
    b0_sids = [r["sample_id"] for r in b0_eval_rows]
    if len(b0_sids) != len(set(b0_sids)):
        raise ValueError(f"Duplicate sample_id detected in baseline {baseline_name} rows!")
    
    tgt_sids = [r["sample_id"] for r in target_eval_rows]
    if len(tgt_sids) != len(set(tgt_sids)):
        raise ValueError(f"Duplicate sample_id detected in target {target_name} rows!")

    b0_map = {r["sample_id"]: r for r in b0_eval_rows}
    target_map = {r["sample_id"]: r for r in target_eval_rows}

    # 2. Defensive Check: ID Alignment
    common_sids = sorted(
        set(b0_map.keys()) & set(target_map.keys()),
        key=lambda s: int(s.split(":")[-1]) if ":" in s and s.split(":")[-1].isdigit() else s,
    )
    if not common_sids:
        raise ValueError(f"No common sample_ids found between {baseline_name} and {target_name}!")
    
    if len(common_sids) != len(b0_sids) or len(common_sids) != len(tgt_sids):
        raise ValueError(
            f"ID alignment mismatch: {baseline_name} has {len(b0_sids)}, "
            f"{target_name} has {len(tgt_sids)}, intersection has {len(common_sids)}"
        )

    transition_counts = {
        "unparsed_to_correct": 0,
        "unparsed_to_wrong": 0,
        "wrong_to_correct": 0,
        "correct_to_wrong": 0,
        "correct_to_unparsed": 0,
        "wrong_to_unparsed": 0,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
        "unchanged_unparsed": 0,
    }

    improved_claims = []
    regressed_claims = []
    unchanged_correct = []
    unchanged_wrong = []
    unchanged_unparsed = []

    per_claim_transitions = []
    y_base = []
    y_tgt = []

    for sid in common_sids:
        b0_r = b0_map[sid]
        tgt_r = target_map[sid]

        gold = b0_r.get("gold_label", "").strip()
        b0_pred = (b0_r.get("predicted_verdict") or "").strip()
        tgt_pred = (tgt_r.get("predicted_verdict") or "").strip()

        # Re-derive verified correctness directly
        b0_correct = _is_verdict_correct(b0_r)
        tgt_correct = _is_verdict_correct(tgt_r)

        y_base.append(1 if b0_correct else 0)
        y_tgt.append(1 if tgt_correct else 0)

        b0_status = _get_claim_status(b0_r)
        tgt_status = _get_claim_status(tgt_r)

        if b0_status == tgt_status:
            trans_key = f"unchanged_{b0_status}"
        else:
            trans_key = f"{b0_status}_to_{tgt_status}"

        if trans_key in transition_counts:
            transition_counts[trans_key] += 1
        else:
            raise KeyError(f"Unexpected transition key: {trans_key}")

        is_improved = not b0_correct and tgt_correct
        is_regressed = b0_correct and not tgt_correct

        if is_improved:
            improved_claims.append(sid)
        elif is_regressed:
            regressed_claims.append(sid)
        elif b0_correct and tgt_correct:
            unchanged_correct.append(sid)
        elif b0_status == "wrong" and tgt_status == "wrong":
            unchanged_wrong.append(sid)
        elif b0_status == "unparsed" and tgt_status == "unparsed":
            unchanged_unparsed.append(sid)

        per_claim_transitions.append({
            "sample_id": sid,
            "gold_label": gold,
            "baseline_predicted": b0_pred,
            "target_predicted": tgt_pred,
            "baseline_correct": b0_correct,
            "target_correct": tgt_correct,
            "transition": trans_key,
            "is_improved": is_improved,
            "is_regressed": is_regressed,
        })

    # Mathematical Invariant 1: Sum of all categories == evaluated claims
    total_transition_claims = sum(transition_counts.values())
    if total_transition_claims != len(common_sids):
        raise AssertionError(
            f"Transition counts sum ({total_transition_claims}) does not match evaluated claims ({len(common_sids)})!"
        )

    # Mathematical Invariant 2: Net gain == Delta correct claims
    net_gain = len(improved_claims) - len(regressed_claims)
    delta_correct = sum(y_tgt) - sum(y_base)
    if net_gain != delta_correct:
        raise AssertionError(
            f"Net gain ({net_gain}) does not match delta correct claims ({delta_correct})!"
        )

    # 3. Statistical Significance
    b_discordant = sum(1 for a, b in zip(y_base, y_tgt) if a == 1 and b == 0)
    c_discordant = sum(1 for a, b in zip(y_base, y_tgt) if a == 0 and b == 1)
    p_value = exact_mcnemar_test_twosided(b_discordant, c_discordant)
    ci_low, ci_high = paired_bootstrap_ci(y_base, y_tgt, n_boot=n_bootstrap)

    return {
        "baseline_name": baseline_name,
        "target_name": target_name,
        "evaluated_claims": len(common_sids),
        "baseline_correct": sum(y_base),
        "target_correct": sum(y_tgt),
        "improved_count": len(improved_claims),
        "improved_claims": improved_claims,
        "regressed_count": len(regressed_claims),
        "regressed_claims": regressed_claims,
        "unchanged_correct_count": len(unchanged_correct),
        "unchanged_wrong_count": len(unchanged_wrong),
        "unchanged_unparsed_count": len(unchanged_unparsed),
        "net_gain": net_gain,
        "transition_counts": transition_counts,
        "discordant_pairs": {
            f"{baseline_name}_1_and_{target_name}_0": b_discordant,
            f"{baseline_name}_0_and_{target_name}_1": c_discordant,
        },
        "exact_mcnemar_p_value": p_value,
        "paired_bootstrap_95_ci": [ci_low, ci_high],
        "per_claim_transitions": per_claim_transitions,
    }


def build_comparison_report(
    eval_summaries: Dict[str, Dict[str, Any]],
    transitions: Dict[str, Dict[str, Any]],
) -> str:
    """Format markdown comparison report comparing B0, B1, and B2."""
    exp_order = ["B0", "B1", "B2"]
    available_exps = [e for e in exp_order if e in eval_summaries]

    lines = [
        "# M1.2.1 — Frozen Base Stabilization Comparison Report",
        "",
        "**Milestone:** M1.2.1  ",
        "**Backbone:** Frozen `Qwen/Qwen3-8B` (Revision `b968826d9c46dd6066d109eabc6255188de91218`)  ",
        "**Dataset:** 100 Exploratory Claims (`train:0` .. `train:99`)  ",
        "**Evidence:** M1.1 R4 Frozen Reranked Top-8 Passages  ",
        "",
        "---",
        "",
        "## 1. High-Level Performance Comparison",
        "",
        "| Configuration | Thinking Mode | Max New Tokens | Valid Verdicts | Accuracy (Overall) | Accuracy (Parsed) | Macro-F1 | Macro-Precision | Macro-Recall | Mean Latency | Mean Tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for exp_id in available_exps:
        s = eval_summaries[exp_id]
        m = s.get("metrics", {})
        gen_cfg = s.get("generation_config", {})
        runtime = s.get("runtime_stats", {})

        thinking_str = "Enabled" if gen_cfg.get("enable_thinking") is not False else "Disabled"
        max_tok = gen_cfg.get("max_new_tokens", "N/A")
        total = m.get("total_claims", 0)
        unparsed = sum(
            m.get("confusion_matrix", {}).get(c, {}).get("Unparseable", 0)
            for c in VERDICTS
        )
        valid = total - unparsed
        valid_pct = (valid / total * 100) if total else 0.0

        acc = m.get("accuracy", 0.0) * 100
        cond_acc = (m.get("correct_claims", 0) / valid * 100) if valid else 0.0
        f1 = m.get("macro_f1", 0.0)
        p = m.get("macro_precision", 0.0)
        r = m.get("macro_recall", 0.0)
        lat = runtime.get("mean_latency_seconds", 0.0)
        toks = runtime.get("mean_generated_tokens", 0.0)

        lines.append(
            f"| **{exp_id}** | {thinking_str} | {max_tok} | {valid}/{total} ({valid_pct:.1f}%) | "
            f"**{acc:.1f}%** | {cond_acc:.1f}% | **{f1:.4f}** | {p:.4f} | {r:.4f} | {lat:.2f}s | {toks:.1f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Per-Class F1-Score Breakdown",
        "",
        "| Class | Support | B0 F1 | B1 F1 | B2 F1 |",
        "|---|---:|---:|---:|---:|",
    ])

    for c in VERDICTS:
        b0_f1 = eval_summaries.get("B0", {}).get("metrics", {}).get("per_class", {}).get(c, {}).get("f1", 0.0)
        b1_f1 = eval_summaries.get("B1", {}).get("metrics", {}).get("per_class", {}).get(c, {}).get("f1", 0.0)
        b2_f1 = eval_summaries.get("B2", {}).get("metrics", {}).get("per_class", {}).get(c, {}).get("f1", 0.0)
        supp = eval_summaries.get("B0", {}).get("metrics", {}).get("per_class", {}).get(c, {}).get("support", 0)
        lines.append(f"| **{c}** | {supp} | {b0_f1:.4f} | {b1_f1:.4f} | {b2_f1:.4f} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Paired Verdict Transitions & Statistical Significance",
        "",
    ])

    for pair_key, trans in transitions.items():
        base_name = trans["baseline_name"]
        tgt_name = trans["target_name"]
        improved = trans["improved_count"]
        regressed = trans["regressed_count"]
        net = trans["net_gain"]
        tc = trans["transition_counts"]
        p_val = trans["exact_mcnemar_p_value"]
        ci = trans["paired_bootstrap_95_ci"]

        sig_note = "Statistically Significant (p < 0.05)" if p_val < 0.05 else "Not Statistically Significant (p >= 0.05)"

        lines.extend([
            f"### {base_name} $\\to$ {tgt_name}",
            "",
            f"- **Net Gain:** {net:+d} claims (Baseline: {trans['baseline_correct']}/100 $\\to$ Target: {trans['target_correct']}/100)",
            f"- **Total Improved:** {improved} (Unparsed $\\to$ Correct: {tc.get('unparsed_to_correct', 0)}, Wrong $\\to$ Correct: {tc.get('wrong_to_correct', 0)})",
            f"- **Total Regressed:** {regressed} (Correct $\\to$ Wrong: {tc.get('correct_to_wrong', 0)}, Correct $\\to$ Unparsed: {tc.get('correct_to_unparsed', 0)})",
            f"- **Unchanged Correct:** {tc.get('unchanged_correct', 0)} claims",
            f"- **Unchanged Wrong:** {tc.get('unchanged_wrong', 0)} claims",
            f"- **Unchanged Unparsed:** {tc.get('unchanged_unparsed', 0)} claims",
            f"- **Wrong $\\to$ Unparsed:** {tc.get('wrong_to_unparsed', 0)} claims",
            f"- **Exact McNemar p-value:** {p_val:.6e} ({p_val:.5f}) — *{sig_note}*",
            f"- **Paired Bootstrap 95% CI ($\\Delta$ Accuracy):** [{ci[0]*100:+.2f}%, {ci[1]*100:+.2f}%]",
            "",
            "| Transition Category | Count | Sample Claims (up to 5) |",
            "|---|---:|---|",
            f"| **Unparsed $\\to$ Correct** | {tc.get('unparsed_to_correct', 0)} | {', '.join([s for s in trans.get('improved_claims', []) if trans['per_claim_transitions'] and any(t['sample_id']==s and t['transition']=='unparsed_to_correct' for t in trans['per_claim_transitions'])][:5]) or 'None'} |",
            f"| **Wrong $\\to$ Correct** | {tc.get('wrong_to_correct', 0)} | {', '.join([s for s in trans.get('improved_claims', []) if trans['per_claim_transitions'] and any(t['sample_id']==s and t['transition']=='wrong_to_correct' for t in trans['per_claim_transitions'])][:5]) or 'None'} |",
            f"| **Correct $\\to$ Wrong** | {tc.get('correct_to_wrong', 0)} | {', '.join([s for s in trans.get('regressed_claims', []) if trans['per_claim_transitions'] and any(t['sample_id']==s and t['transition']=='correct_to_wrong' for t in trans['per_claim_transitions'])][:5]) or 'None'} |",
            f"| **Correct $\\to$ Unparsed** | {tc.get('correct_to_unparsed', 0)} | {', '.join([s for s in trans.get('regressed_claims', []) if trans['per_claim_transitions'] and any(t['sample_id']==s and t['transition']=='correct_to_unparsed' for t in trans['per_claim_transitions'])][:5]) or 'None'} |",
            f"| **Unchanged Correct** | {tc.get('unchanged_correct', 0)} | — |",
            f"| **Unchanged Wrong** | {tc.get('unchanged_wrong', 0)} | — |",
            f"| **Wrong $\\to$ Unparsed** | {tc.get('wrong_to_unparsed', 0)} | — |",
            f"| **Unchanged Unparsed** | {tc.get('unchanged_unparsed', 0)} | — |",
            "",
        ])

    return "\n".join(lines)


def run_stabilization_comparison(
    b0_eval_dir: str,
    b1_eval_dir: str,
    b2_eval_dir: str,
    output_dir: str,
) -> Path:
    """Load evaluation outputs for B0, B1, and B2, compute verified transitions, and persist comparison."""
    dest = Path(output_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    configs = [("B0", b0_eval_dir), ("B1", b1_eval_dir), ("B2", b2_eval_dir)]
    summaries = {}
    eval_rows_map = {}

    for name, path_str in configs:
        p = Path(path_str).expanduser().resolve()
        sum_p = p / "evaluation_summary.json"
        csv_p = p / "per_claim_evaluation.csv"

        if not sum_p.is_file():
            raise FileNotFoundError(f"Missing evaluation_summary.json in {p}")
        
        summary_data = json.loads(sum_p.read_text(encoding="utf-8"))
        
        # Load generation config from prediction manifest if available
        pred_dir = Path(summary_data.get("predictions_dir", ""))
        if (pred_dir / "manifest.json").is_file():
            pred_manifest = json.loads((pred_dir / "manifest.json").read_text(encoding="utf-8"))
            summary_data["generation_config"] = pred_manifest.get("generation_config", {})

        summaries[name] = summary_data

        with open(csv_p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            eval_rows_map[name] = list(reader)

    # Compute paired transitions: B0 -> B1, B0 -> B2, and B1 -> B2
    transitions = {}
    if "B0" in eval_rows_map and "B1" in eval_rows_map:
        transitions["B0_to_B1"] = analyze_paired_transitions(
            eval_rows_map["B0"], eval_rows_map["B1"], target_name="B1", baseline_name="B0"
        )
    if "B0" in eval_rows_map and "B2" in eval_rows_map:
        transitions["B0_to_B2"] = analyze_paired_transitions(
            eval_rows_map["B0"], eval_rows_map["B2"], target_name="B2", baseline_name="B0"
        )
    if "B1" in eval_rows_map and "B2" in eval_rows_map:
        transitions["B1_to_B2"] = analyze_paired_transitions(
            eval_rows_map["B1"], eval_rows_map["B2"], target_name="B2", baseline_name="B1"
        )

    # Generate comparison report
    report_md = build_comparison_report(summaries, transitions)
    (dest / "comparison_report.md").write_text(report_md, encoding="utf-8")

    # Save summary json
    summary_out = {
        "milestone": "M1.2.1",
        "phase": "frozen_base_stabilization_comparison",
        "summaries": summaries,
        "transitions": {
            k: {
                "baseline_name": v["baseline_name"],
                "target_name": v["target_name"],
                "evaluated_claims": v["evaluated_claims"],
                "baseline_correct": v["baseline_correct"],
                "target_correct": v["target_correct"],
                "improved_count": v["improved_count"],
                "regressed_count": v["regressed_count"],
                "net_gain": v["net_gain"],
                "transition_counts": v["transition_counts"],
                "discordant_pairs": v["discordant_pairs"],
                "exact_mcnemar_p_value": v["exact_mcnemar_p_value"],
                "paired_bootstrap_95_ci": v["paired_bootstrap_95_ci"],
                "improved_claims": v["improved_claims"],
                "regressed_claims": v["regressed_claims"],
            }
            for k, v in transitions.items()
        },
    }
    json_path = dest / "comparison_summary.json"
    if json_path.is_file():
        json_path.unlink()
    write_json(json_path, summary_out)

    return dest


def main():
    parser = argparse.ArgumentParser(description="M1.2.1 Stabilization Comparison Evaluator")
    parser.add_argument("--b0-eval", required=True, help="Path to B0 evaluation directory")
    parser.add_argument("--b1-eval", required=True, help="Path to B1 evaluation directory")
    parser.add_argument("--b2-eval", required=True, help="Path to B2 evaluation directory")
    parser.add_argument("--output", required=True, help="Output directory for comparison report")
    args = parser.parse_args()

    out_p = run_stabilization_comparison(
        b0_eval_dir=args.b0_eval,
        b1_eval_dir=args.b1_eval,
        b2_eval_dir=args.b2_eval,
        output_dir=args.output,
    )
    print(f"[stabilization] Comparison report generated in {out_p}")


if __name__ == "__main__":
    main()
