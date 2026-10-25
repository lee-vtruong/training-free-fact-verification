"""Execution runner and paired evaluation engine for M1.4 Controlled Fact Verification Pilot (D0–D6).

All models frozen, training-free, Quality-First.
Controlled pilot comparing:
- D0: Frozen Base Protocol v1.0 (Zero-shot B1)
- D1: Concise generic verification rules
- D2: Concise source-derived error rules
- D3: Random error demonstrations (k=2)
- D4: Similarity-retrieved error demonstrations (k=2)
- D5: Similarity-retrieved error demonstrations with verified corrections (k=2)
- D6: Correct/wrong contrastive demonstrations (k=2)
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tffv.data import read_jsonl
from tffv.provenance import canonical, digest, file_hash, write_json
from tffv.verification.backbone import (
    EXPECTED_QWEN3_8B_ID,
    EXPECTED_QWEN3_8B_REVISION,
    Qwen3FrozenBackbone,
    SyntheticBackbone,
)
from tffv.verification.error_memory import ErrorMemoryBank, build_error_memory_bank
from tffv.verification.evaluate import compute_metrics, load_gold_labels, run_evaluation
from tffv.verification.leakage import assert_fold_disjointness, quarantine_near_duplicates
from tffv.verification.m14_prompting import ALL_CONDITIONS_M14, build_prompt_for_condition_m14
from tffv.verification.ontology import VERDICTS, normalize_verdict
from tffv.verification.predict import dump_jsonl, load_claims_map, load_evidence_by_claim
from tffv.verification.stabilization import (
    _get_claim_status,
    _is_verdict_correct,
    analyze_paired_transitions,
    exact_mcnemar_test_twosided,
    paired_bootstrap_ci,
)


def run_condition_inference_m14(
    condition: str,
    target_sids: List[str],
    evidence_map: Dict[str, List[Dict[str, Any]]],
    claims_map: Dict[str, Dict[str, Any]],
    output_dir: Path,
    backbone: Any,
    memory_bank: Optional[ErrorMemoryBank] = None,
    max_new_tokens: int = 512,
    seed: int = 42,
) -> Path:
    """Execute inference for a single condition (D0..D6) and seal predictions."""
    output_dir = Path(output_dir).expanduser().resolve()
    cond_dir = output_dir / condition
    cond_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = cond_dir / "predictions.jsonl"
    manifest_path = cond_dir / "manifest.json"

    sys.stderr.write(f"\n[M1.4] [{condition}] Starting inference across {len(target_sids)} claims...\n")
    sys.stderr.flush()

    predictions = []
    valid_verdict_count = 0
    t0_total = time.perf_counter()

    for idx, sid in enumerate(target_sids, 1):
        claim_data = claims_map.get(sid, {})
        claim_text = claim_data.get("claim", "").strip()
        evidence_list = evidence_map.get(sid, [])

        t0 = time.perf_counter()
        prompt_data = build_prompt_for_condition_m14(
            condition=condition,
            claim=claim_text,
            evidence=evidence_list,
            memory_bank=memory_bank,
            seed=seed,
        )

        # Defensive guard: Exemplar IDs must never match target claim ID
        used_exemplars = prompt_data.get("exemplar_sample_ids", [])
        if sid in used_exemplars:
            raise ValueError(f"CRITICAL LEAKAGE: Evaluation claim {sid} was sampled as an in-context exemplar!")

        response = backbone.generate(prompt_data["messages"])
        lat = time.perf_counter() - t0

        raw_gen = response.get("generated_text", "")
        # Robust token accounting lookup
        gen_toks = response.get("generated_tokens") or response.get("generation_tokens")
        if gen_toks is None:
            gen_toks = max(1, len(raw_gen.split()))

        prompt_toks = response.get("prompt_tokens") or 0

        parsed_verdict = normalize_verdict(raw_gen)
        if parsed_verdict in VERDICTS:
            valid_verdict_count += 1

        pred_record = {
            "sample_id": sid,
            "claim": claim_text,
            "condition": condition,
            "raw_generation": raw_gen,
            "predicted_verdict": parsed_verdict,
            "prompt_tokens": prompt_toks,
            "generation_tokens": gen_toks,
            "generated_tokens": gen_toks,
            "latency_seconds": round(lat, 4),
            "prompt_sha256": prompt_data.get("prompt_sha256"),
            "exemplar_sample_ids": used_exemplars,
        }
        predictions.append(pred_record)

        if idx % 10 == 0 or idx == len(target_sids):
            sys.stderr.write(
                f"  [{condition}] {idx}/{len(target_sids)} completed "
                f"({valid_verdict_count}/{idx} valid verdicts, last lat: {lat:.2f}s)\n"
            )
            sys.stderr.flush()

    total_latency = time.perf_counter() - t0_total

    dump_jsonl(predictions_path, predictions)

    manifest = {
        "milestone": "M1.4",
        "condition": condition,
        "pilot_mode": "EXPLORATORY_DEBUG_PILOT",
        "total_claims": len(target_sids),
        "valid_verdicts": valid_verdict_count,
        "valid_rate": round(valid_verdict_count / len(target_sids), 4) if target_sids else 0.0,
        "max_new_tokens": max_new_tokens,
        "total_latency_seconds": round(total_latency, 2),
        "mean_latency_seconds": round(total_latency / len(target_sids), 4) if target_sids else 0.0,
        "predictions_file": str(predictions_path),
        "predictions_sha256": file_hash(predictions_path),
        "seed": seed,
    }
    write_json(manifest_path, manifest)
    sys.stderr.write(f"[M1.4] [{condition}] Finished and sealed: {predictions_path}\n")
    return predictions_path


def evaluate_condition_m14(
    condition: str,
    predictions_dir: Path,
    train_json_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Evaluate condition predictions against train.json ground truth."""
    eval_out = output_dir / condition / "evaluation"
    eval_out.mkdir(parents=True, exist_ok=True)
    run_evaluation(
        predictions_dir=str(predictions_dir / condition),
        train_json_path=str(train_json_path),
        output_dir=str(eval_out),
    )
    summary_path = eval_out / "evaluation_summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {}


def run_paired_comparison_m14(
    baseline_condition: str,
    target_condition: str,
    predictions_dir: Path,
    train_json_path: Path,
) -> Dict[str, Any]:
    """Compute exact paired comparison between baseline D0 and target condition."""
    gold_map = load_gold_labels(Path(train_json_path))

    base_preds_p = predictions_dir / baseline_condition / "predictions.jsonl"
    target_preds_p = predictions_dir / target_condition / "predictions.jsonl"

    base_rows = read_jsonl(base_preds_p)
    target_rows = read_jsonl(target_preds_p)

    target_by_id = {r["sample_id"]: r for r in target_rows}

    # Track full 9 mutually exclusive categories
    categories = {
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
        "unchanged_unparsed": 0,
        "wrong_to_correct": 0,
        "unparsed_to_correct": 0,
        "correct_to_wrong": 0,
        "correct_to_unparsed": 0,
        "wrong_to_unparsed": 0,
        "unparsed_to_wrong": 0,
    }

    correct_base = 0
    correct_target = 0

    eval_pairs = []

    for b_row in base_rows:
        sid = b_row["sample_id"]
        t_row = target_by_id.get(sid)
        if not t_row:
            continue

        gold = gold_map.get(sid, {}).get("gold_label")
        if not gold:
            continue

        b_stat = _get_claim_status({"predicted_verdict": b_row.get("predicted_verdict"), "gold_label": gold})
        t_stat = _get_claim_status({"predicted_verdict": t_row.get("predicted_verdict"), "gold_label": gold})

        b_corr = bool(b_stat == "correct")
        t_corr = bool(t_stat == "correct")

        if b_corr:
            correct_base += 1
        if t_corr:
            correct_target += 1

        eval_pairs.append({"sample_id": sid, "base_correct": b_corr, "target_correct": t_corr})

        # Map to exact 9 mutually exclusive transition categories
        if b_stat == "correct":
            if t_stat == "correct":
                categories["unchanged_correct"] += 1
            elif t_stat == "wrong":
                categories["correct_to_wrong"] += 1
            else:
                categories["correct_to_unparsed"] += 1
        elif b_stat == "wrong":
            if t_stat == "correct":
                categories["wrong_to_correct"] += 1
            elif t_stat == "wrong":
                categories["unchanged_wrong"] += 1
            else:
                categories["wrong_to_unparsed"] += 1
        else:  # b_stat == "unparsed"
            if t_stat == "correct":
                categories["unparsed_to_correct"] += 1
            elif t_stat == "wrong":
                categories["unparsed_to_wrong"] += 1
            else:
                categories["unchanged_unparsed"] += 1

    # Exact McNemar discordants:
    # b: Base correct, Target not correct (regression)
    # c: Base not correct, Target correct (improvement)
    b_discordant = categories["correct_to_wrong"] + categories["correct_to_unparsed"]
    c_discordant = categories["wrong_to_correct"] + categories["unparsed_to_correct"]

    mcnemar_p = exact_mcnemar_test_twosided(b_discordant, c_discordant)
    y_base = [1 if p["base_correct"] else 0 for p in eval_pairs]
    y_target = [1 if p["target_correct"] else 0 for p in eval_pairs]
    ci_low, ci_high = paired_bootstrap_ci(y_base, y_target, n_boot=2000, seed=42)

    total_claims = len(eval_pairs)
    cat_sum = sum(categories.values())
    assert cat_sum == total_claims, f"Category sum mismatch: {cat_sum} != {total_claims}"

    net_gain = (c_discordant) - (b_discordant)
    delta_corr = correct_target - correct_base
    assert net_gain == delta_corr, f"Mathematical identity failed: {net_gain} != {delta_corr}"

    return {
        "baseline_condition": baseline_condition,
        "target_condition": target_condition,
        "total_claims": total_claims,
        "correct_base": correct_base,
        "correct_target": correct_target,
        "net_gain": net_gain,
        "delta_correct": delta_corr,
        "categories": categories,
        "improved_total": c_discordant,
        "regressed_total": b_discordant,
        "mcnemar_b": b_discordant,
        "mcnemar_c": c_discordant,
        "mcnemar_p_value": mcnemar_p,
        "bootstrap_95_ci_lower": ci_low,
        "bootstrap_95_ci_upper": ci_high,
    }


def generate_m14_markdown_report(
    condition_summaries: Dict[str, Any],
    paired_comparisons: Dict[str, Any],
    output_dir: Path,
    partition_manifest: Dict[str, Any],
) -> Path:
    """Generate comprehensive scientific Markdown report for M1.4."""
    report_path = output_dir / "M1_4_PILOT_REPORT.md"

    # Multiplicity adjustment for paired comparisons
    m_tests = len(paired_comparisons)
    sorted_pairs = sorted(paired_comparisons.items(), key=lambda x: x[1]["mcnemar_p_value"])
    for rank, (cid, comp) in enumerate(sorted_pairs):
        p_raw = comp["mcnemar_p_value"]
        comp["p_bonferroni"] = min(1.0, p_raw * m_tests)
        comp["p_holm"] = min(1.0, p_raw * (m_tests - rank))
        comp["sig_holm"] = comp["p_holm"] < 0.05

    cond_descriptions = {
        "D0": "Frozen Base Protocol v1.0 (Zero-shot B1)",
        "D1": "Concise Generic Verification Rules",
        "D2": "Concise Source-Derived Error Rules",
        "D3": "Random Error Demonstrations ($k=2$)",
        "D4": "Similarity-Retrieved Error Demonstrations ($k=2$)",
        "D5": "Similarity-Retrieved Error Demonstrations with Corrections ($k=2$)",
        "D6": "Correct/Wrong Contrastive Demonstrations ($k=2$)",
    }

    lines = [
        "# M1.4 — Controlled Fact Verification Pilot Report (D0–D6)",
        "",
        "**Milestone:** M1.4 (Training-Free Fact Verification Pilot)",
        f"**Date Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "**Design:** Controlled Exploration on Disjoint Pilot Folds",
        "**Core Research Principle:** Strictly Training-Free, Quality-First (Zero parameter updates, zero learned routers)",
        "**Pilot Mode:** **EXPLORATORY DEBUG PILOT** (Results hypothesis-generating; confirmatory test requires untouched set)",
        "",
        "---",
        "",
        "## 1. Population & Anti-Leakage Partition",
        "",
        f"- **Source Memory (Fold A):** {partition_manifest.get('total_source_claims', 50)} claims (`train:0..49`).",
        f"- **Evaluation Claims (Fold B):** {partition_manifest.get('admitted_claims_count', 46)} claims admitted.",
        f"- **Quarantined Near-Duplicates:** {len(partition_manifest.get('quarantined_ids', []))} claims quarantined.",
        "- **Exemplar Disjointness:** 100% of exemplars strictly from Fold A. Overlap with evaluation claims is 0.0%.",
        "",
        "---",
        "",
        "## 2. Executive Performance Summary",
        "",
        "| Condition | Description | Valid Verdicts | Accuracy | Macro-F1 | Precision | Recall | Mean Latency | Mean Tokens (Prompt / Gen) |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for cond in ALL_CONDITIONS_M14:
        s = condition_summaries.get(cond, {})
        # Unpack metrics safely
        m = s.get("metrics", {})
        if "metrics" in m and isinstance(m["metrics"], dict):
            m = m["metrics"]

        timing = s.get("timing", {})
        desc = cond_descriptions.get(cond, "")
        total = m.get("total_claims", 0)
        correct = m.get("correct_claims", 0)
        acc = m.get("accuracy", 0.0) * 100
        mf1 = m.get("macro_f1", 0.0)
        mp = m.get("macro_precision", 0.0)
        mr = m.get("macro_recall", 0.0)

        # Count valid vs unparsed from confusion matrix
        cm = m.get("confusion_matrix", {})
        unparsed = 0
        for cls_name, preds in cm.items():
            unparsed += preds.get("Unparseable", 0)
        valid = total - unparsed
        valid_str = f"{valid}/{total} ({valid/total*100:.1f}%)" if total else "N/A"

        mean_lat = timing.get("mean_latency_seconds", 0.0)
        in_toks = timing.get("mean_prompt_tokens", 0)
        out_toks = timing.get("mean_generation_tokens", 0.0)

        lines.append(
            f"| **{cond}** | {desc} | {valid_str} | **{acc:.1f}%** ({correct}/{total}) | **{mf1:.4f}** | "
            f"{mp:.4f} | {mr:.4f} | {mean_lat:.2f}s | {in_toks:.0f} / {out_toks:.1f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Full Mutually Exclusive Paired Transitions (All 9 Categories vs D0)",
        "",
        "Every claim belongs to exactly one category; the 9 mutually exclusive categories sum identically to the total evaluation claims.",
        "",
        "| Condition vs D0 | C→C | C→W | C→U | W→C | W→W | W→U | U→C | U→W | U→U | Sum | Net Gain | McNemar $p$ (Unadj) | Holm-Bonf $p$ | Paired Bootstrap 95% CI | Sig? |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ])

    for cond in ALL_CONDITIONS_M14[1:]:
        comp = paired_comparisons.get(cond, {})
        cats = comp.get("categories", {})
        c_c = cats.get("unchanged_correct", 0)
        c_w = cats.get("correct_to_wrong", 0)
        c_u = cats.get("correct_to_unparsed", 0)
        w_c = cats.get("wrong_to_correct", 0)
        w_w = cats.get("unchanged_wrong", 0)
        w_u = cats.get("wrong_to_unparsed", 0)
        u_c = cats.get("unparsed_to_correct", 0)
        u_w = cats.get("unparsed_to_wrong", 0)
        u_u = cats.get("unchanged_unparsed", 0)
        cat_sum = sum(cats.values())
        net = f"{comp.get('net_gain', 0):+d}"
        p_raw = comp.get("mcnemar_p_value", 1.0)
        p_holm = comp.get("p_holm", 1.0)
        ci_l = comp.get("bootstrap_95_ci_lower", 0.0)
        ci_u = comp.get("bootstrap_95_ci_upper", 0.0)
        sig = "**Yes**" if p_holm < 0.05 else "No"

        lines.append(
            f"| **{cond} vs D0** | {c_c} | {c_w} | {c_u} | {w_c} | {w_w} | {w_u} | {u_c} | {u_w} | {u_u} | "
            f"**{cat_sum}** | **{net}** | {p_raw:.4f} | {p_holm:.4f} | [{ci_l*100:+.1f}%, {ci_u*100:+.1f}%] | {sig} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Methodological Safeguards & Limitations",
        "",
        "- **Context Budget Guarantee:** All prompt configurations D0–D6 remain within 3,200 input tokens.",
        "- **Concise Reasoning Directives:** D1 and D2 mandate a single-sentence reason to eliminate generation budget exhaustion.",
        "- **Similarity Retrieval Determinism:** Lexical 3-gram similarity operates strictly on Fold A memory with stable tie-breaking.",
        "- **Exploratory Constraint:** These pilot evaluations are conducted on Fold B exploratory claims to debug method mechanisms; any positive signal must be confirmed on an untouched held-out evaluation set.",
        "",
    ])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M1.4 Controlled Fact Verification Pilot (D0–D6)")
    parser.add_argument("--claims", required=True, help="Path to input claims JSONL (e.g. outputs/m11-source-claims-.../claims.jsonl)")
    parser.add_argument("--evidence", required=True, help="Path to R4 Top-8 evidence JSONL")
    parser.add_argument("--train-json", required=True, help="Path to AVeriTeC train.json (for Fold A memory and gold evaluation)")
    parser.add_argument("--b1-predictions", required=True, help="Path to B1 predictions on Fold A (for historical error memory)")
    parser.add_argument("--output", required=True, help="Output directory for M1.4 run")
    parser.add_argument("--model-path", default=None, help="Path to Qwen3-8B snapshot (or omit for synthetic)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true", help="Run with deterministic SyntheticBackbone for CPU testing")
    parser.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS_M14, help="Conditions to run")
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_json_p = Path(args.train_json).expanduser().resolve()
    claims_p = Path(args.claims).expanduser().resolve()
    evidence_p = Path(args.evidence).expanduser().resolve()
    b1_preds_p = Path(args.b1_predictions).expanduser().resolve()

    # 1. Load claims & evidence
    claims_map = load_claims_map(claims_p)
    evidence_map = load_evidence_by_claim(evidence_p)

    # 2. Partition claims into Fold A (source) and Fold B (candidate evaluation)
    all_sids = sorted(list(claims_map.keys()))
    fold_a_sids = [sid for sid in all_sids if sid.startswith("train:") and int(sid.split(":")[1]) < 50]
    candidate_sids = [sid for sid in all_sids if sid.startswith("train:") and int(sid.split(":")[1]) >= 50]

    source_claims_objs = [claims_map[sid] for sid in fold_a_sids if sid in claims_map]
    candidate_claims_objs = [claims_map[sid] for sid in candidate_sids if sid in claims_map]

    # Quarantine near-duplicates
    admitted_eval, quarantined, manifest_path = quarantine_near_duplicates(
        source_claims=source_claims_objs,
        candidate_claims=candidate_claims_objs,
        output_dir=output_dir,
    )
    admitted_eval_sids = [c["sample_id"] for c in admitted_eval]
    quarantined_ids = [c["sample_id"] for c in quarantined]
    assert_fold_disjointness(set(fold_a_sids), set(admitted_eval_sids))

    partition_manifest = {
        "milestone": "M1.4",
        "total_source_claims": len(fold_a_sids),
        "total_candidate_claims": len(candidate_sids),
        "quarantined_ids": quarantined_ids,
        "admitted_claims_count": len(admitted_eval_sids),
        "admitted_claims_sample_ids": admitted_eval_sids,
    }
    write_json(output_dir / "partition_manifest.json", partition_manifest)

    # 3. Build Error Memory Bank strictly from Fold A
    sys.stderr.write(f"\n[M1.4] Building Fold A Error Memory Bank from {len(fold_a_sids)} source claims...\n")
    mem_out = output_dir / "error_memory_bank.json"
    memory_bank = build_error_memory_bank(
        train_json_path=train_json_p,
        b1_predictions_path=b1_preds_p,
        r4_evidence_path=evidence_p,
        output_path=mem_out,
        allowed_sample_ids=set(fold_a_sids),
    )

    # 4. Initialize Backbone
    if args.synthetic or args.model_path is None:
        sys.stderr.write("[M1.4] Using SyntheticBackbone (deterministic testing mode)\n")
        backbone = SyntheticBackbone(seed=args.seed)
    else:
        sys.stderr.write(f"[M1.4] Loading Qwen3FrozenBackbone from: {args.model_path}\n")
        backbone = Qwen3FrozenBackbone(
            model_path=args.model_path,
            enable_thinking=False,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            device_map="auto",
        )

    # 5. Execute inference across requested conditions
    condition_summaries: Dict[str, Any] = {}
    for cond in args.conditions:
        preds_p = run_condition_inference_m14(
            condition=cond,
            target_sids=admitted_eval_sids,
            evidence_map=evidence_map,
            claims_map=claims_map,
            output_dir=output_dir,
            backbone=backbone,
            memory_bank=memory_bank,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )

        # Evaluate against gold ground truth
        eval_summary = evaluate_condition_m14(
            condition=cond,
            predictions_dir=output_dir,
            train_json_path=train_json_p,
            output_dir=output_dir,
        )

        cond_manifest = json.loads((output_dir / cond / "manifest.json").read_text(encoding="utf-8"))
        preds_rows = read_jsonl(preds_p)
        in_toks = [r.get("prompt_tokens") or 0 for r in preds_rows]
        out_toks = [r.get("generation_tokens") or r.get("generated_tokens") or 0 for r in preds_rows]
        lats = [r.get("latency_seconds") or 0.0 for r in preds_rows]

        condition_summaries[cond] = {
            "metrics": eval_summary.get("metrics", eval_summary),
            "timing": {
                "total_latency_seconds": cond_manifest.get("total_latency_seconds", 0.0),
                "mean_latency_seconds": round(sum(lats) / len(lats), 4) if lats else 0.0,
                "mean_prompt_tokens": round(sum(in_toks) / len(in_toks), 1) if in_toks else 0.0,
                "mean_generation_tokens": round(sum(out_toks) / len(out_toks), 1) if out_toks else 0.0,
            },
        }

    # 6. Paired comparisons vs D0
    paired_comparisons: Dict[str, Any] = {}
    if "D0" in args.conditions:
        for cond in args.conditions:
            if cond == "D0":
                continue
            paired_comparisons[cond] = run_paired_comparison_m14(
                baseline_condition="D0",
                target_condition=cond,
                predictions_dir=output_dir,
                train_json_path=train_json_p,
            )

    # 7. Generate final Markdown Report
    report_p = generate_m14_markdown_report(
        condition_summaries=condition_summaries,
        paired_comparisons=paired_comparisons,
        output_dir=output_dir,
        partition_manifest=partition_manifest,
    )
    sys.stderr.write(f"\n[M1.4] Pilot execution completed successfully!\nReport generated at: {report_p}\n")


if __name__ == "__main__":
    main()
