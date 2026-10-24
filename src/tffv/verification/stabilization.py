"""Evaluation and paired transition analysis for Frozen Base Stabilization (B0, B1, B2)."""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tffv.data import read_jsonl
from tffv.provenance import canonical, digest, file_hash, write_json
from tffv.verification.evaluate import compute_metrics, load_gold_labels, run_evaluation
from tffv.verification.ontology import VERDICTS


def analyze_paired_transitions(
    b0_eval_rows: List[Dict[str, Any]],
    target_eval_rows: List[Dict[str, Any]],
    target_name: str,
) -> Dict[str, Any]:
    """Compute paired per-claim verdict transitions from B0 to target configuration."""
    b0_map = {r["sample_id"]: r for r in b0_eval_rows}
    target_map = {r["sample_id"]: r for r in target_eval_rows}

    common_sids = sorted(
        set(b0_map.keys()) & set(target_map.keys()),
        key=lambda s: int(s.split(":")[-1]) if ":" in s else s,
    )

    improved_claims = []
    regressed_claims = []
    unchanged_correct = []
    unchanged_wrong = []

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

    per_claim_transitions = []

    for sid in common_sids:
        b0_r = b0_map[sid]
        tgt_r = target_map[sid]

        gold = b0_r["gold_label"]
        b0_pred = b0_r.get("predicted_verdict") or ""
        tgt_pred = tgt_r.get("predicted_verdict") or ""

        b0_correct = bool(b0_r.get("is_correct", False))
        tgt_correct = bool(tgt_r.get("is_correct", False))

        b0_status = "correct" if b0_correct else ("unparsed" if not b0_pred else "wrong")
        tgt_status = "correct" if tgt_correct else ("unparsed" if not tgt_pred else "wrong")

        transition_type = f"{b0_status}_to_{tgt_status}"
        if b0_status == tgt_status:
            transition_type = f"unchanged_{b0_status}"

        if transition_type in transition_counts:
            transition_counts[transition_type] += 1

        is_improved = not b0_correct and tgt_correct
        is_regressed = b0_correct and not tgt_correct

        if is_improved:
            improved_claims.append(sid)
        elif is_regressed:
            regressed_claims.append(sid)
        elif b0_correct and tgt_correct:
            unchanged_correct.append(sid)
        else:
            unchanged_wrong.append(sid)

        per_claim_transitions.append({
            "sample_id": sid,
            "gold_label": gold,
            "b0_predicted": b0_pred,
            "target_predicted": tgt_pred,
            "b0_correct": b0_correct,
            "target_correct": tgt_correct,
            "transition": transition_type,
            "is_improved": is_improved,
            "is_regressed": is_regressed,
        })

    net_gain = len(improved_claims) - len(regressed_claims)

    return {
        "target_name": target_name,
        "evaluated_claims": len(common_sids),
        "improved_count": len(improved_claims),
        "improved_claims": improved_claims,
        "regressed_count": len(regressed_claims),
        "regressed_claims": regressed_claims,
        "unchanged_correct_count": len(unchanged_correct),
        "unchanged_wrong_count": len(unchanged_wrong),
        "net_gain": net_gain,
        "transition_counts": transition_counts,
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
        "## 3. Paired Verdict Transitions vs Baseline B0",
        "",
    ])

    for tgt_name, trans in transitions.items():
        improved = trans["improved_count"]
        regressed = trans["regressed_count"]
        net = trans["net_gain"]
        tc = trans["transition_counts"]

        lines.extend([
            f"### Baseline B0 $\\to$ {tgt_name}",
            "",
            f"- **Net Gain:** {net:+d} claims (Improved: {improved}, Regressed: {regressed})",
            f"- **Unparsed Recovery (Unparsed $\\to$ Correct):** {tc.get('unparsed_to_correct', 0)} claims",
            f"- **Unparsed Resolution (Unparsed $\\to$ Wrong):** {tc.get('unparsed_to_wrong', 0)} claims",
            f"- **False Flips (Correct $\\to$ Wrong/Unparsed):** {tc.get('correct_to_wrong', 0) + tc.get('correct_to_unparsed', 0)} claims",
            "",
            "| Transition Category | Count | Sample Claims (up to 5) |",
            "|---|---:|---|",
            f"| **Unparsed $\\to$ Correct** | {tc.get('unparsed_to_correct', 0)} | {', '.join(trans.get('improved_claims', [])[:5]) or 'None'} |",
            f"| **Wrong $\\to$ Correct** | {tc.get('wrong_to_correct', 0)} | — |",
            f"| **Correct $\\to$ Wrong/Unparsed** | {regressed} | {', '.join(trans.get('regressed_claims', [])[:5]) or 'None'} |",
            f"| **Unchanged Correct** | {trans.get('unchanged_correct_count', 0)} | — |",
            f"| **Unchanged Wrong** | {trans.get('unchanged_wrong_count', 0)} | — |",
            "",
        ])

    return "\n".join(lines)


def run_stabilization_comparison(
    b0_eval_dir: str,
    b1_eval_dir: str,
    b2_eval_dir: str,
    output_dir: str,
) -> Path:
    """Load evaluation outputs for B0, B1, and B2, compute transitions, and persist comparison."""
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

        # Load per-claim csv rows
        import csv
        with open(csv_p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            eval_rows_map[name] = list(reader)

    # Compute transitions vs B0
    transitions = {}
    if "B0" in eval_rows_map and "B1" in eval_rows_map:
        transitions["B1"] = analyze_paired_transitions(eval_rows_map["B0"], eval_rows_map["B1"], "B1")
    if "B0" in eval_rows_map and "B2" in eval_rows_map:
        transitions["B2"] = analyze_paired_transitions(eval_rows_map["B0"], eval_rows_map["B2"], "B2")

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
                "evaluated_claims": v["evaluated_claims"],
                "improved_count": v["improved_count"],
                "regressed_count": v["regressed_count"],
                "net_gain": v["net_gain"],
                "transition_counts": v["transition_counts"],
                "improved_claims": v["improved_claims"],
                "regressed_claims": v["regressed_claims"],
            }
            for k, v in transitions.items()
        },
    }
    write_json(dest / "comparison_summary.json", summary_out)

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
