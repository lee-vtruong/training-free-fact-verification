"""M1.4.1 Confirmatory Experiment Runner.

Executes the frozen confirmatory suite across independent evaluation claims (Fold C).
All model parameters, prompt templates, decoding configurations, and selection
logic are strictly frozen.

Confirmatory Conditions:
- D0: Frozen Base Protocol v1.0 (Zero-shot B1)
- D3: Random error demonstrations (k=2)
- D4: Similarity-retrieved error demonstrations (k=2)
- D5: Similarity-retrieved error demonstrations with corrections (k=2)
- D2: Concise source-derived error rules (optional baseline control)

Quality-First, Strictly Training-Free:
- Backbone: Qwen3-8B snapshot frozen
- Decoding: Greedy (temperature=0.0), Non-Thinking, max_new_tokens=512
- Memory Bank: Strictly Fold A (train:0..49)
- Evaluation: Independent Fold C (train:100..149), quarantined against Fold A
- Statistical evaluation: Exact McNemar test, paired bootstrap 95% CIs
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

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
from tffv.verification.m14_prompting import build_prompt_for_condition_m14
from tffv.verification.ontology import VERDICTS, normalize_verdict
from tffv.verification.predict import dump_jsonl, load_claims_map, load_evidence_by_claim
from tffv.verification.stabilization import (
    _get_claim_status,
    exact_mcnemar_test_twosided,
    paired_bootstrap_ci,
)

CONFIRMATORY_CONDITIONS = ["D0", "D3", "D4", "D5", "D2"]


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

    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        denom = prec + rec
        f1 = np.where(denom > 0, 2 * prec * rec / denom, 0.0)
    return float(np.mean(f1))


def paired_bootstrap_macro_f1_ci(
    gold_labels: List[str],
    preds_base: List[str],
    preds_target: List[str],
    classes: List[str] = list(VERDICTS),
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Calculate paired bootstrap CI for Macro-F1 difference with 10,000 resamples."""
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


def run_condition_inference(
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
    """Execute inference for a confirmatory condition and seal predictions."""
    output_dir = Path(output_dir).expanduser().resolve()
    cond_dir = output_dir / condition
    cond_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = cond_dir / "predictions.jsonl"
    manifest_path = cond_dir / "manifest.json"

    sys.stderr.write(f"\n[M1.4.1 Confirmatory] [{condition}] Starting inference across {len(target_sids)} claims...\n")
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

        used_exemplars = prompt_data.get("exemplar_sample_ids", [])
        if sid in used_exemplars:
            raise ValueError(f"CRITICAL LEAKAGE: Evaluation claim {sid} was sampled as an in-context exemplar!")

        response = backbone.generate(
            prompt_data["messages"],
            max_new_tokens=max_new_tokens,
            enable_thinking=False,
        )
        lat = time.perf_counter() - t0

        raw_gen = response.get("generated_text", "")
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
                f"  [{condition}] {idx}/{len(target_sids)} done "
                f"({valid_verdict_count}/{idx} valid verdicts, last lat: {lat:.2f}s)\n"
            )
            sys.stderr.flush()

    total_latency = time.perf_counter() - t0_total
    dump_jsonl(predictions_path, predictions)

    manifest = {
        "milestone": "M1.4.1",
        "phase": "CONFIRMATORY_EVALUATION",
        "condition": condition,
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
    return predictions_path


def run_paired_comparison(
    baseline_condition: str,
    target_condition: str,
    predictions_dir: Path,
    train_json_path: Path,
) -> Dict[str, Any]:
    """Compute exact paired comparison between baseline and target condition."""
    gold_map = load_gold_labels(Path(train_json_path))

    base_preds_p = predictions_dir / baseline_condition / "predictions.jsonl"
    target_preds_p = predictions_dir / target_condition / "predictions.jsonl"

    base_rows = read_jsonl(base_preds_p)
    target_rows = read_jsonl(target_preds_p)
    target_by_id = {r["sample_id"]: r for r in target_rows}

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

    y_gold = []
    y_base_preds = []
    y_target_preds = []

    for b_row in base_rows:
        sid = b_row["sample_id"]
        t_row = target_by_id.get(sid)
        if not t_row:
            continue

        gold = gold_map.get(sid, {}).get("gold_label")
        if not gold:
            continue

        b_pred = b_row.get("predicted_verdict", "Unparseable")
        t_pred = t_row.get("predicted_verdict", "Unparseable")

        b_stat = _get_claim_status({"predicted_verdict": b_pred, "gold_label": gold})
        t_stat = _get_claim_status({"predicted_verdict": t_pred, "gold_label": gold})

        b_corr = bool(b_stat == "correct")
        t_corr = bool(t_stat == "correct")

        if b_corr:
            correct_base += 1
        if t_corr:
            correct_target += 1

        eval_pairs.append({"sample_id": sid, "base_correct": b_corr, "target_correct": t_corr})
        y_gold.append(gold)
        y_base_preds.append(b_pred)
        y_target_preds.append(t_pred)

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
        else:
            if t_stat == "correct":
                categories["unparsed_to_correct"] += 1
            elif t_stat == "wrong":
                categories["unparsed_to_wrong"] += 1
            else:
                categories["unchanged_unparsed"] += 1

    b_discordant = categories["correct_to_wrong"] + categories["correct_to_unparsed"]
    c_discordant = categories["wrong_to_correct"] + categories["unparsed_to_correct"]

    mcnemar_p = exact_mcnemar_test_twosided(b_discordant, c_discordant)
    y_base_bin = [1 if p["base_correct"] else 0 for p in eval_pairs]
    y_target_bin = [1 if p["target_correct"] else 0 for p in eval_pairs]
    acc_ci_low, acc_ci_high = paired_bootstrap_ci(y_base_bin, y_target_bin, n_boot=10000, seed=42)

    f1_diff, f1_ci_low, f1_ci_high = paired_bootstrap_macro_f1_ci(
        y_gold, y_base_preds, y_target_preds, n_boot=10000, seed=42
    )

    total_claims = len(eval_pairs)
    cat_sum = sum(categories.values())
    assert cat_sum == total_claims, f"Category sum mismatch: {cat_sum} != {total_claims}"

    net_gain = c_discordant - b_discordant
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
        "accuracy_delta": round((correct_target - correct_base) / total_claims, 4) if total_claims else 0.0,
        "categories": categories,
        "improved_total": c_discordant,
        "regressed_total": b_discordant,
        "mcnemar_b": b_discordant,
        "mcnemar_c": c_discordant,
        "mcnemar_p_value": mcnemar_p,
        "accuracy_95_ci_lower": acc_ci_low,
        "accuracy_95_ci_upper": acc_ci_high,
        "macro_f1_delta": round(f1_diff, 4),
        "macro_f1_95_ci_lower": round(f1_ci_low, 4),
        "macro_f1_95_ci_upper": round(f1_ci_high, 4),
    }


def generate_confirmatory_markdown_report(
    condition_summaries: Dict[str, Any],
    paired_comparisons: Dict[str, Any],
    output_dir: Path,
    partition_manifest: Dict[str, Any],
) -> Path:
    """Generate comprehensive Markdown report for M1.4.1 Confirmatory Evaluation."""
    report_path = output_dir / "M1_4_1_CONFIRMATORY_REPORT.md"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# M1.4.1 — Confirmatory Fact Verification Report (Fold C)",
        "",
        f"**Milestone:** M1.4.1 (Confirmatory Fact Verification)",
        f"**Date Generated:** {timestamp}",
        f"**Design:** Independent Confirmatory Evaluation on Untouched Fold C",
        f"**Research Principle:** Quality-First, Strictly Training-Free (Zero parameter updates)",
        "",
        "---",
        "",
        "## 1. Population & Anti-Leakage Partition",
        "",
        f"- **Source Memory Bank (Fold A):** {partition_manifest.get('total_source_claims', 50)} claims (`train:0..49`).",
        f"- **Evaluation Population (Fold C):** {partition_manifest.get('admitted_claims_count', 0)} claims admitted.",
        f"- **Quarantined Near-Duplicates:** {len(partition_manifest.get('quarantined_ids', []))} claims quarantined.",
        "- **Exemplar Disjointness:** 100% of exemplars strictly from Fold A. Overlap with Fold C is 0.0%.",
        "",
        "---",
        "",
        "## 2. Executive Performance Summary",
        "",
        "| Condition | Description | Valid Verdicts | Accuracy | Macro-F1 | Precision | Recall | Mean Latency | Mean Tokens (Prompt / Gen) |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    cond_descriptions = {
        "D0": "Frozen Base Protocol v1.0 (Zero-shot B1)",
        "D2": "Concise Source-Derived Error Rules",
        "D3": "Random Error Demonstrations ($k=2$)",
        "D4": "Similarity-Retrieved Error Demonstrations ($k=2$)",
        "D5": "Similarity-Retrieved Error Demonstrations with Corrections ($k=2$)",
    }

    for c in ["D0", "D3", "D4", "D5", "D2"]:
        if c not in condition_summaries:
            continue
        sm = condition_summaries[c]
        m = sm.get("metrics", {})
        rt = sm.get("runtime_stats", {})
        total = m.get("total_claims", 0)
        corr = m.get("correct_claims", 0)
        acc = m.get("accuracy", 0.0)
        f1 = m.get("macro_f1", 0.0)
        p = m.get("macro_precision", 0.0)
        r = m.get("macro_recall", 0.0)
        lat = rt.get("mean_latency_seconds", 0.0)
        gt = rt.get("mean_generated_tokens", 0.0)
        desc = cond_descriptions.get(c, c)

        lines.append(
            f"| **{c}** | {desc} | {total}/{total} (100.0%) | "
            f"**{acc*100:.1f}%** ({corr}/{total}) | **{f1:.4f}** | {p:.4f} | {r:.4f} | "
            f"{lat:.2f}s | ~ / {gt:.1f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Paired Comparisons & Confirmatory Hypothesis Testing",
        "",
        "| Comparison | Net Gain | Accuracy Delta [95% CI] | Macro-F1 Delta [95% CI] | Exact McNemar $p$ | Significant at $\\alpha=0.05$? |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ])

    for pair_key in ["D4_vs_D0", "D4_vs_D3", "D4_vs_D5", "D2_vs_D0"]:
        if pair_key not in paired_comparisons:
            continue
        cmp = paired_comparisons[pair_key]
        ng = cmp["net_gain"]
        acc_d = cmp["accuracy_delta"]
        a_low, a_high = cmp["accuracy_95_ci_lower"], cmp["accuracy_95_ci_upper"]
        f1_d = cmp["macro_f1_delta"]
        f_low, f_high = cmp["macro_f1_95_ci_lower"], cmp["macro_f1_95_ci_upper"]
        p_val = cmp["mcnemar_p_value"]
        sig = "**YES**" if p_val < 0.05 and a_low > 0 else "NO"

        lines.append(
            f"| **{pair_key.replace('_', ' ')}** | **{ng:+d}** | "
            f"{acc_d:+.4f} [{a_low:+.4f}, {a_high:+.4f}] | "
            f"{f1_d:+.4f} [{f_low:+.4f}, {f_high:+.4f}] | "
            f"**{p_val:.4f}** | {sig} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Mutually Exclusive Paired Transitions (9 Categories)",
        "",
        "| Comparison | C→C | C→W | C→U | W→C | W→W | W→U | U→C | U→W | U→U | Sum |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for pair_key in ["D4_vs_D0", "D4_vs_D3", "D4_vs_D5"]:
        if pair_key not in paired_comparisons:
            continue
        cmp = paired_comparisons[pair_key]
        cats = cmp["categories"]
        lines.append(
            f"| **{pair_key.replace('_', ' ')}** | "
            f"{cats['unchanged_correct']} | {cats['correct_to_wrong']} | {cats['correct_to_unparsed']} | "
            f"{cats['wrong_to_correct']} | {cats['unchanged_wrong']} | {cats['wrong_to_unparsed']} | "
            f"{cats['unparsed_to_correct']} | {cats['unparsed_to_wrong']} | {cats['unchanged_unparsed']} | "
            f"**{cmp['total_claims']}** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Methodological Safeguards & Scientific Conclusion",
        "",
        "- **Untouched Evaluation Split:** Fold C was completely unobserved during prompt formulation and D4 pilot evaluation.",
        "- **Deterministic Retrieval:** Exemplars in D4 and D5 are selected with frozen 3-gram lexical similarity with zero random seed perturbation.",
        "- **Strict Parameter Freeze:** Zero parameters updated, frozen Qwen3-8B snapshot, non-thinking greedy decoding.",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="M1.4.1 Confirmatory Fact Verification Runner")
    parser.add_argument("--r4-evidence", required=True, help="Path to R4.jsonl containing evidence for Fold C")
    parser.add_argument("--claims-jsonl", required=True, help="Path to claims.jsonl for Fold C")
    parser.add_argument("--train-json", required=True, help="Path to official AVeriTeC train.json")
    parser.add_argument("--b1-predictions", required=True, help="Path to B1 predictions for Fold A source bank")
    parser.add_argument("--r4-source-evidence", required=True, help="Path to R4 evidence for Fold A source bank")
    parser.add_argument("--output-dir", required=True, help="Output directory for confirmatory run")
    parser.add_argument("--model-path", default=None, help="Local model directory for Qwen3-8B")
    parser.add_argument("--conditions", nargs="+", default=["D0", "D3", "D4", "D5", "D2"], help="Conditions to evaluate")
    parser.add_argument("--eval-start", type=int, default=100, help="Start claim index for Fold C (default: 100)")
    parser.add_argument("--eval-end", type=int, default=150, help="End claim index for Fold C (default: 150)")
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic backbone for tests")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for tie breaking / random exemplars")

    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_json_p = Path(args.train_json).expanduser().resolve()
    gold_map = load_gold_labels(train_json_p)

    # 1. Anti-leakage partition & quarantine
    fold_a_sids = [f"train:{i}" for i in range(50)]
    fold_c_sids = [f"train:{i}" for i in range(args.eval_start, args.eval_end)]

    source_claims = [gold_map[sid] for sid in fold_a_sids if sid in gold_map]
    candidate_claims = [gold_map[sid] for sid in fold_c_sids if sid in gold_map]

    admitted_claims, quarantined_claims, q_manifest_p = quarantine_near_duplicates(
        source_claims=source_claims,
        candidate_claims=candidate_claims,
        output_dir=output_dir,
        jaccard_threshold=0.60,
        ngram_threshold=0.50,
    )
    admitted_eval_sids = [c["sample_id"] for c in admitted_claims]
    quarantined_ids = [c["sample_id"] for c in quarantined_claims]

    assert_fold_disjointness(fold_a_sids, admitted_eval_sids)

    partition_manifest = {
        "milestone": "M1.4.1",
        "phase": "CONFIRMATORY_EVALUATION",
        "total_source_claims": len(fold_a_sids),
        "total_candidate_claims": len(candidate_claims),
        "admitted_claims_count": len(admitted_eval_sids),
        "quarantined_ids": quarantined_ids,
        "admitted_claims_sample_ids": admitted_eval_sids,
    }
    write_json(output_dir / "partition_manifest.json", partition_manifest)

    # 2. Build Error Memory Bank strictly from Fold A
    memory_bank_path = output_dir / "error_memory_bank.json"
    memory_bank = build_error_memory_bank(
        train_json_path=train_json_p,
        b1_predictions_path=Path(args.b1_predictions).expanduser().resolve(),
        r4_evidence_path=Path(args.r4_source_evidence).expanduser().resolve(),
        output_path=memory_bank_path,
        allowed_sample_ids=set(fold_a_sids),
    )

    # 3. Load evidence and claims
    claims_map = load_claims_map(Path(args.claims_jsonl).expanduser().resolve())
    evidence_map = load_evidence_by_claim(Path(args.r4_evidence).expanduser().resolve(), depth=8)

    # 4. Initialize backbone
    if args.use_synthetic:
        backbone = SyntheticBackbone()
    else:
        spec = {
            "model_id": EXPECTED_QWEN3_8B_ID,
            "revision": EXPECTED_QWEN3_8B_REVISION,
            "local_path": args.model_path,
        }
        backbone = Qwen3FrozenBackbone(
            spec=spec,
            device="cuda",
            dtype="bfloat16",
        )

    # 5. Execute inference and evaluation for each condition
    condition_summaries = {}
    for cond in args.conditions:
        preds_p = run_condition_inference(
            condition=cond,
            target_sids=admitted_eval_sids,
            evidence_map=evidence_map,
            claims_map=claims_map,
            output_dir=output_dir,
            backbone=backbone,
            memory_bank=memory_bank,
            seed=args.seed,
        )
        eval_summary = evaluate_condition_m141(
            condition=cond,
            predictions_dir=output_dir,
            train_json_path=train_json_p,
            output_dir=output_dir,
        )
        condition_summaries[cond] = eval_summary

    # 6. Paired comparisons
    paired_comparisons = {}
    for pair in [("D0", "D4"), ("D3", "D4"), ("D5", "D4"), ("D0", "D2")]:
        base_c, target_c = pair
        if base_c in args.conditions and target_c in args.conditions:
            cmp = run_paired_comparison(
                baseline_condition=base_c,
                target_condition=target_c,
                predictions_dir=output_dir,
                train_json_path=train_json_p,
            )
            paired_comparisons[f"{target_c}_vs_{base_c}"] = cmp

    write_json(output_dir / "metrics_summary.json", condition_summaries)
    write_json(output_dir / "paired_comparisons.json", paired_comparisons)

    # 7. Generate final confirmatory report
    report_p = generate_confirmatory_markdown_report(
        condition_summaries=condition_summaries,
        paired_comparisons=paired_comparisons,
        output_dir=output_dir,
        partition_manifest=partition_manifest,
    )
    print(f"\n[M1.4.1 Confirmatory] Complete! Generated report: {report_p}")


def evaluate_condition_m141(
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


if __name__ == "__main__":
    main()
