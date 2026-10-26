"""M1.5 Runner: Controlled Verification Evaluation on Fold E (train:250..349).

Executes the frozen pre-registered M1.5 protocol across independent evaluation claims (Fold E).
All model parameters, prompt templates, decoding configurations, and selection logic are strictly frozen.

Pre-registered Conditions:
- M0: Matched generic control (single-pass, concise matched generic directives, zero error knowledge)
- M1: Generic claim-type clarification (single-pass, generic taxonomy definitions, zero error knowledge)
- M2: Source-derived mechanism guidance (conditional, triggered by label-free observable signals, fallback to M0)
- M3: Randomly assigned mechanism guidance (negative control, fixed random seed)
- M4: Permutation consistency baseline (training-free, zero error memory, 3-permutation majority vote)

Statistical Contrasts:
- Primary Pre-registered Hypothesis: M4 vs M0 (Permutation consistency vs single-pass control)
- Secondary Pre-registered Hypothesis: M1 vs M0 (Generic claim-type clarification vs single-pass control)
- Diagnostic Contrast: M2 vs M3 (Mechanism specificity on active claims)
- Multiplicity Control: Holm-Bonferroni FWER adjustment across contrast family.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys
import time
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
from tffv.verification.evaluate import compute_metrics, load_gold_labels
from tffv.verification.leakage import assert_fold_disjointness, quarantine_near_duplicates
from tffv.verification.ontology import VERDICTS, normalize_verdict
from tffv.verification.predict import dump_jsonl, load_claims_map, load_evidence_by_claim
from tffv.verification.prompting import (
    SYSTEM_PROMPT,
    VERDICT_DEFINITIONS,
    format_evidence_block,
)
from tffv.verification.stabilization import (
    _get_claim_status,
    exact_mcnemar_test_twosided,
    paired_bootstrap_ci,
)
from tffv.verification.taxonomy import (
    ALL_MECHANISMS,
    GUIDANCE_M0,
    GUIDANCE_M1,
    build_controlled_guidance,
    detect_active_mechanisms,
)

DEFAULT_CONDITIONS_M15 = ["M0", "M1", "M4"]
ALL_CONDITIONS_M15 = ["M0", "M1", "M2", "M3", "M4"]


def holm_bonferroni(p_values: Dict[str, float]) -> Dict[str, float]:
    """Compute Holm-Bonferroni step-down adjusted p-values."""
    sorted_items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(sorted_items)
    adjusted = {}
    running_max = 0.0

    for rank, (k, p) in enumerate(sorted_items):
        multiplier = m - rank
        adj_p = min(1.0, p * multiplier)
        adj_p = max(running_max, adj_p)
        running_max = adj_p
        adjusted[k] = round(adj_p, 6)

    return adjusted


def fast_macro_f1_from_indices(
    gold_idx: np.ndarray, pred_idx: np.ndarray, num_classes: int = 4
) -> float:
    """Vectorized calculation of macro-f1 from integer class indices."""
    valid_mask = (pred_idx >= 0) & (pred_idx < num_classes)
    flat_idx = np.where(valid_mask, gold_idx * num_classes + pred_idx, -1)
    valid_flat = flat_idx[flat_idx >= 0]
    counts = np.bincount(
        valid_flat, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)

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
    if n == 0:
        return 0.0, 0.0, 0.0

    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    gold_idx = np.array([class_to_idx[g] for g in gold_labels])
    base_idx = np.array([class_to_idx.get(p, -1) for p in preds_base])
    target_idx = np.array([class_to_idx.get(p, -1) for p in preds_target])

    point_base_f1 = fast_macro_f1_from_indices(gold_idx, base_idx, num_classes)
    point_target_f1 = fast_macro_f1_from_indices(gold_idx, target_idx, num_classes)
    point_diff = point_target_f1 - point_base_f1

    indices = np.arange(n)
    boot_indices = rng.choice(indices, size=(n_boot, n), replace=True)

    boot_gold = gold_idx[boot_indices]
    boot_base = base_idx[boot_indices]
    boot_target = target_idx[boot_indices]

    diffs = np.zeros(n_boot)
    for b in range(n_boot):
        f1_b = fast_macro_f1_from_indices(boot_gold[b], boot_base[b], num_classes)
        f1_t = fast_macro_f1_from_indices(boot_gold[b], boot_target[b], num_classes)
        diffs[b] = f1_t - f1_b

    ci_low = float(np.percentile(diffs, 100 * (alpha / 2)))
    ci_high = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return point_diff, ci_low, ci_high


def generate_permutations(evidence: List[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Generate 3 deterministic evidence permutations for M4.
    
    1. Order 1 (Canonical): Top-8 passages ranked by R4 score.
    2. Order 2 (Reverse): Top-8 passages in reverse order.
    3. Order 3 (Interleaved): Top-8 passages interleaved [1, 8, 2, 7, 3, 6, 4, 5].
    """
    n = len(evidence)
    if n <= 1:
        return [
            ("canonical", list(evidence)),
            ("reverse", list(evidence)),
            ("interleaved", list(evidence)),
        ]

    # Order 1: Canonical
    ord1 = list(evidence)

    # Order 2: Reverse
    ord2 = list(reversed(evidence))

    # Order 3: Interleaved
    ord3: List[Dict[str, Any]] = []
    left = 0
    right = n - 1
    toggle = True
    while left <= right:
        if toggle:
            ord3.append(evidence[left])
            left += 1
        else:
            ord3.append(evidence[right])
            right -= 1
        toggle = not toggle

    return [
        ("canonical", ord1),
        ("reverse", ord2),
        ("interleaved", ord3),
    ]


def aggregate_majority_vote(
    v1: Optional[str],
    v2: Optional[str],
    v3: Optional[str],
) -> Tuple[Optional[str], int, bool]:
    """Aggregate 3 verdicts via majority voting with canonical tie-breaker.
    
    Returns:
        (aggregated_verdict, agreement_count, tie_break_used)
    """
    valid_votes = [v for v in [v1, v2, v3] if v in VERDICTS]
    if not valid_votes:
        return None, 0, False

    counts = Counter(valid_votes)
    most_common = counts.most_common()
    top_verdict, top_count = most_common[0]

    # Majority agreement (2 or 3 votes)
    if top_count >= 2:
        return top_verdict, top_count, False

    # 3-way tie or single valid vote:
    # If canonical v1 is valid, Order 1 canonical verdict breaks the tie
    if v1 in VERDICTS:
        return v1, 1, True
    return top_verdict, 1, True


def build_m15_prompt(
    condition: str,
    claim: str,
    evidence: List[Dict[str, Any]],
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Construct structured verification prompt for M1.5 conditions."""
    guidance, meta = build_controlled_guidance(condition, claim, evidence, rng=rng)
    evidence_block = format_evidence_block(evidence)

    user_content = (
        f"Claim: \"{claim}\"\n\n"
        f"Retrieved Evidence:\n{evidence_block}\n\n"
        f"{guidance}\n\n"
        f"{VERDICT_DEFINITIONS}\n\n"
        "State your 1-sentence reasoning and output your verdict in the format:\n"
        "Verdict: <Supported|Refuted|Not Enough Evidence|Conflicting Evidence/Cherrypicking>"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    return {
        "messages": messages,
        "user": user_content,
        "condition": condition,
        "meta": meta,
        "prompt_sha256": digest(canonical(messages)),
    }


def execute_single_pass_condition(
    condition: str,
    target_sids: List[str],
    claims_map: Dict[str, Any],
    evidence_map: Dict[str, List[Dict[str, Any]]],
    backbone: Any,
    output_dir: Path,
    seed: int = 42,
    max_new_tokens: int = 64,
) -> Path:
    """Execute single-pass verification condition (M0, M1, M2, M3)."""
    cond_dir = output_dir / condition
    cond_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = cond_dir / "predictions.jsonl"
    manifest_path = cond_dir / "condition_manifest.json"

    records = []
    rng = random.Random(seed)
    valid_verdict_count = 0
    start_time = time.time()

    for sid in target_sids:
        claim_obj = claims_map.get(sid, {})
        claim_text = claim_obj.get("claim", "")
        ev_list = evidence_map.get(sid, [])[:8]

        prompt_obj = build_m15_prompt(condition, claim_text, ev_list, rng=rng)
        t0 = time.time()
        resp = backbone.generate(prompt_obj["messages"], max_new_tokens=max_new_tokens)
        lat = time.time() - t0

        raw_text = resp.get("raw_text") or resp.get("generated_text") or ""
        verdict = normalize_verdict(raw_text)
        is_valid = bool(verdict in VERDICTS)
        if is_valid:
            valid_verdict_count += 1

        rec = {
            "sample_id": sid,
            "claim": claim_text,
            "condition": condition,
            "meta_mechanism": prompt_obj.get("meta"),
            "raw_text": raw_text,
            "predicted_verdict": verdict or "Unparseable",
            "is_valid_verdict": is_valid,
            "latency_seconds": round(lat, 3),
            "input_tokens": resp.get("input_tokens") or resp.get("prompt_tokens") or 0,
            "output_tokens": resp.get("output_tokens") or resp.get("generated_tokens") or 0,
            "prompt_sha256": prompt_obj["prompt_sha256"],
        }
        records.append(rec)

    dump_jsonl(predictions_path, records)
    total_lat = time.time() - start_time

    manifest = {
        "condition": condition,
        "total_claims": len(target_sids),
        "valid_verdicts": valid_verdict_count,
        "valid_rate": round(valid_verdict_count / len(target_sids), 4) if target_sids else 0.0,
        "max_new_tokens": max_new_tokens,
        "total_latency_seconds": round(total_lat, 2),
        "mean_latency_seconds": round(total_lat / len(target_sids), 4) if target_sids else 0.0,
        "predictions_file": str(predictions_path),
        "predictions_sha256": file_hash(predictions_path),
        "seed": seed,
    }
    write_json(manifest_path, manifest)
    return predictions_path


def execute_permutation_condition_m4(
    target_sids: List[str],
    claims_map: Dict[str, Any],
    evidence_map: Dict[str, List[Dict[str, Any]]],
    backbone: Any,
    output_dir: Path,
    seed: int = 42,
    max_new_tokens: int = 64,
) -> Path:
    """Execute Condition M4: Permutation Consistency Baseline (3 passes, majority vote)."""
    cond_dir = output_dir / "M4"
    cond_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = cond_dir / "predictions.jsonl"
    details_path = cond_dir / "permutations_M4_detail.jsonl"
    manifest_path = cond_dir / "condition_manifest.json"

    records = []
    detail_records = []
    valid_verdict_count = 0
    tie_break_count = 0
    start_time = time.time()

    for sid in target_sids:
        claim_obj = claims_map.get(sid, {})
        claim_text = claim_obj.get("claim", "")
        canonical_ev = evidence_map.get(sid, [])[:8]

        permutations = generate_permutations(canonical_ev)
        perm_results = {}
        total_in_toks = 0
        total_out_toks = 0
        claim_lat = 0.0

        for perm_name, perm_ev in permutations:
            prompt_obj = build_m15_prompt("M0", claim_text, perm_ev)
            t0 = time.time()
            resp = backbone.generate(prompt_obj["messages"], max_new_tokens=max_new_tokens)
            dt = time.time() - t0
            claim_lat += dt
            total_in_toks += resp.get("input_tokens") or resp.get("prompt_tokens") or 0
            total_out_toks += resp.get("output_tokens") or resp.get("generated_tokens") or 0

            raw = resp.get("raw_text") or resp.get("generated_text") or ""
            verd = normalize_verdict(raw)
            perm_results[perm_name] = {
                "raw_text": raw,
                "verdict": verd,
                "prompt_sha256": prompt_obj["prompt_sha256"],
            }

        v1 = perm_results.get("canonical", {}).get("verdict")
        v2 = perm_results.get("reverse", {}).get("verdict")
        v3 = perm_results.get("interleaved", {}).get("verdict")

        agg_verdict, agreement, tie_used = aggregate_majority_vote(v1, v2, v3)
        if tie_used:
            tie_break_count += 1

        is_valid = bool(agg_verdict in VERDICTS)
        if is_valid:
            valid_verdict_count += 1

        # Canonical output record
        rec = {
            "sample_id": sid,
            "claim": claim_text,
            "condition": "M4",
            "predicted_verdict": agg_verdict or "Unparseable",
            "is_valid_verdict": is_valid,
            "agreement_count": agreement,
            "tie_break_used": tie_used,
            "total_latency_seconds": round(claim_lat, 3),
            "total_input_tokens": total_in_toks,
            "total_output_tokens": total_out_toks,
            "verdict_canonical": v1 or "Unparseable",
            "verdict_reverse": v2 or "Unparseable",
            "verdict_interleaved": v3 or "Unparseable",
        }
        records.append(rec)

        # Full detailed diagnostic record
        detail_rec = {
            "sample_id": sid,
            "claim": claim_text,
            "aggregated_verdict": agg_verdict or "Unparseable",
            "agreement_count": agreement,
            "tie_break_used": tie_used,
            "order_1_canonical": perm_results.get("canonical"),
            "order_2_reverse": perm_results.get("reverse"),
            "order_3_interleaved": perm_results.get("interleaved"),
        }
        detail_records.append(detail_rec)

    dump_jsonl(predictions_path, records)
    dump_jsonl(details_path, detail_records)
    total_lat = time.time() - start_time

    manifest = {
        "condition": "M4",
        "description": "Permutation Consistency Baseline (3 evidence orders, majority vote)",
        "total_claims": len(target_sids),
        "valid_verdicts": valid_verdict_count,
        "valid_rate": round(valid_verdict_count / len(target_sids), 4) if target_sids else 0.0,
        "tie_break_count": tie_break_count,
        "tie_break_rate": round(tie_break_count / len(target_sids), 4) if target_sids else 0.0,
        "forward_passes_per_claim": 3,
        "max_new_tokens": max_new_tokens,
        "total_latency_seconds": round(total_lat, 2),
        "mean_latency_seconds": round(total_lat / len(target_sids), 4) if target_sids else 0.0,
        "predictions_file": str(predictions_path),
        "predictions_sha256": file_hash(predictions_path),
        "details_file": str(details_path),
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

        eval_pairs.append({
            "sample_id": sid,
            "base_correct": b_corr,
            "target_correct": t_corr,
        })
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


def generate_m15_markdown_report(
    condition_summaries: Dict[str, Any],
    paired_comparisons: Dict[str, Any],
    adjusted_p_values: Dict[str, float],
    output_dir: Path,
) -> Path:
    """Generate structured markdown report for M1.5 Evaluation on Fold E."""
    report_path = output_dir / "M1_5_EVALUATION_REPORT.md"
    lines = [
        "# M1.5 — Fold E Controlled Verification Evaluation Report",
        "",
        "**Milestone:** M1.5 Controlled Verification Evaluation Protocol  ",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        "**Backbone:** Frozen Qwen3-8B snapshot (`b968826d9c46dd6066d109eabc6255188de91218`)  ",
        "**Target Population:** Fold E (`train:250..349`, 100 unobserved claims)  ",
        "**Primary Contrast:** **M4 vs M0** (Permutation Consistency Baseline vs Matched Generic Control)  ",
        "**Secondary Contrast:** **M1 vs M0** (Generic Claim-Type Clarification vs Matched Generic Control)  ",
        "",
        "---",
        "",
        "## 1. Executive Performance Summary",
        "",
        "| Condition | Role / Description | Accuracy | Macro-F1 | Macro-P | Macro-R | Valid Rate | Correct / Total |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for cond, sm in condition_summaries.items():
        role = "Primary Baseline" if cond == "M0" else ("Primary Candidate" if cond == "M4" else "Secondary Candidate")
        acc_pct = f"{sm['accuracy'] * 100:.1f}%"
        f1_str = f"{sm['macro_f1']:.4f}"
        p_str = f"{sm['macro_precision']:.4f}"
        r_str = f"{sm['macro_recall']:.4f}"
        val_str = f"{sm.get('valid_rate', 1.0) * 100:.1f}%"
        corr_str = f"{sm.get('correct_count', 0)} / {sm.get('total_evaluated', 0)}"
        lines.append(
            f"| **{cond}** | {role} | {acc_pct} | {f1_str} | {p_str} | {r_str} | {val_str} | {corr_str} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Hypothesis Testing & Paired Statistical Contrasts",
        "",
        "| Comparison | Role | Net Gain | Accuracy Delta [95% CI] | Macro-F1 Delta [95% CI] | Raw Exact McNemar $p$ | Holm Adjusted $p_{\\text{adj}}$ | Stat. Sig. ($\\alpha=0.05$) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for pair_key, cmp in paired_comparisons.items():
        role = "**PRIMARY**" if pair_key == "M4_vs_M0" else ("**SECONDARY**" if pair_key == "M1_vs_M0" else "Diagnostic")
        net = f"{cmp['net_gain']:+d}"
        acc_delta_str = f"{cmp['accuracy_delta']:+.4f} [{cmp['accuracy_95_ci_lower']:+.4f}, {cmp['accuracy_95_ci_upper']:+.4f}]"
        f1_delta_str = f"{cmp['macro_f1_delta']:+.4f} [{cmp['macro_f1_95_ci_lower']:+.4f}, {cmp['macro_f1_95_ci_upper']:+.4f}]"
        raw_p = f"{cmp['mcnemar_p_value']:.4f}"
        adj_p_val = adjusted_p_values.get(pair_key, cmp["mcnemar_p_value"])
        adj_p_str = f"{adj_p_val:.4f}"
        sig = "**YES**" if adj_p_val < 0.05 else "NO"
        lines.append(
            f"| **{pair_key.replace('_', ' ')}** | {role} | **{net}** | {acc_delta_str} | {f1_delta_str} | {raw_p} | {adj_p_str} | {sig} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Detailed Paired Transition Accounting",
        "",
        "| Comparison | C→C | C→W (Regr) | C→U | W→C (Corr) | W→W | W→U | U→C | U→W | U→U | Total Evaluated |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for pair_key, cmp in paired_comparisons.items():
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
        "## 4. Scientific Invariants & Protocol Safeguards",
        "",
        "- **Confound Isolation Notice:** Condition M4 uses zero historical error memory. Any observed improvement is strictly attributable to inference-time permutation consistency aggregation, and must NOT be interpreted as learning from historical errors.",
        "- **Pristine Split Isolation:** Fold E (`train:250..349`) was strictly held out and evaluated under pre-registered protocol contracts.",
        "- **Strict Quality-First, Training-Free Execution:** Zero parameters fitted, greedy non-thinking decoding (`max_new_tokens=64`), frozen Qwen3-8B.",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="M1.5 Controlled Verification Evaluation Runner (Fold E)"
    )
    parser.add_argument(
        "--r4-evidence",
        required=True,
        help="Path to R4.jsonl containing evidence for Fold E",
    )
    parser.add_argument(
        "--claims-jsonl", required=True, help="Path to claims.jsonl for Fold E"
    )
    parser.add_argument(
        "--train-json", required=True, help="Path to official AVeriTeC train.json"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for confirmatory run",
    )
    parser.add_argument(
        "--model-path", default=None, help="Local model directory for Qwen3-8B"
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS_M15,
        help="Conditions to evaluate (default: M0 M1 M4)",
    )
    parser.add_argument(
        "--eval-start",
        type=int,
        default=250,
        help="Start claim index for Fold E (default: 250)",
    )
    parser.add_argument(
        "--eval-end",
        type=int,
        default=350,
        help="End claim index for Fold E (default: 350)",
    )
    parser.add_argument(
        "--source-start",
        type=int,
        default=0,
        help="Start claim index for Fold A source bank (default: 0)",
    )
    parser.add_argument(
        "--source-end",
        type=int,
        default=50,
        help="End claim index for Fold A source bank (default: 50)",
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Use synthetic backbone for tests",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_json_p = Path(args.train_json).expanduser().resolve()
    gold_map = load_gold_labels(train_json_p)

    # 1. Anti-leakage partition & quarantine
    fold_a_sids = [f"train:{i}" for i in range(args.source_start, args.source_end)]
    fold_e_sids = [f"train:{i}" for i in range(args.eval_start, args.eval_end)]

    source_claims = [gold_map[sid] for sid in fold_a_sids if sid in gold_map]
    candidate_claims = [gold_map[sid] for sid in fold_e_sids if sid in gold_map]

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
        "milestone": "M1.5",
        "phase": "CONTROLLED_EVALUATION_FOLD_E",
        "total_source_claims": len(fold_a_sids),
        "total_candidate_claims": len(candidate_claims),
        "admitted_claims_count": len(admitted_eval_sids),
        "quarantined_ids": quarantined_ids,
        "admitted_claims_sample_ids": admitted_eval_sids,
    }
    write_json(output_dir / "partition_manifest.json", partition_manifest)

    # 2. Load evidence and claims
    claims_map = load_claims_map(Path(args.claims_jsonl).expanduser().resolve())
    evidence_map = load_evidence_by_claim(
        Path(args.r4_evidence).expanduser().resolve(), depth=8
    )

    # 3. Initialize backbone
    if args.use_synthetic:
        backbone = SyntheticBackbone()
    else:
        spec = {
            "model_id": EXPECTED_QWEN3_8B_ID,
            "revision": EXPECTED_QWEN3_8B_REVISION,
            "local_path": args.model_path or EXPECTED_QWEN3_8B_ID,
        }
        backbone = Qwen3FrozenBackbone(
            spec=spec,
            device="cuda",
            dtype="bfloat16",
        )

    # 4. Execute pre-registered conditions
    preds_map = {}
    for cond in args.conditions:
        c_upper = cond.upper()
        sys.stderr.write(f"\n>>> Running Condition: {c_upper} across {len(admitted_eval_sids)} claims...\n")
        sys.stderr.flush()

        if c_upper == "M4":
            pred_p = execute_permutation_condition_m4(
                target_sids=admitted_eval_sids,
                claims_map=claims_map,
                evidence_map=evidence_map,
                backbone=backbone,
                output_dir=output_dir,
                seed=args.seed,
                max_new_tokens=64,
            )
        else:
            pred_p = execute_single_pass_condition(
                condition=c_upper,
                target_sids=admitted_eval_sids,
                claims_map=claims_map,
                evidence_map=evidence_map,
                backbone=backbone,
                output_dir=output_dir,
                seed=args.seed,
                max_new_tokens=64,
            )
        preds_map[c_upper] = pred_p

    # 5. Compute standalone metrics for each condition
    gold_subset = [gold_map[sid]["gold_label"] for sid in admitted_eval_sids]
    condition_summaries = {}

    for cond, pred_path in preds_map.items():
        rows = read_jsonl(pred_path)
        row_map = {r["sample_id"]: r.get("predicted_verdict", "Unparseable") for r in rows}
        eval_rows = [
            {
                "sample_id": sid,
                "gold_label": gold_map.get(sid, {}).get("gold_label"),
                "predicted_verdict": row_map.get(sid, "Unparseable"),
                "is_correct": bool(
                    row_map.get(sid, "Unparseable")
                    == gold_map.get(sid, {}).get("gold_label")
                ),
            }
            for sid in admitted_eval_sids
        ]
        metrics = compute_metrics(eval_rows)
        correct_count = sum(1 for e in eval_rows if e["is_correct"])
        valid_count = sum(1 for e in eval_rows if e["predicted_verdict"] in VERDICTS)

        metrics["correct_count"] = correct_count
        metrics["total_evaluated"] = len(admitted_eval_sids)
        metrics["valid_rate"] = round(valid_count / len(admitted_eval_sids), 4) if admitted_eval_sids else 0.0
        condition_summaries[cond] = metrics

    # 6. Compute paired contrasts
    paired_comparisons = {}
    raw_p_values = {}

    # Define standard contrast pairs
    candidate_pairs = [
        ("M0", "M4"),  # Primary contrast
        ("M0", "M1"),  # Secondary contrast
    ]
    if "M2" in preds_map and "M3" in preds_map:
        candidate_pairs.append(("M3", "M2"))
    if "M2" in preds_map:
        candidate_pairs.append(("M0", "M2"))

    for base_c, target_c in candidate_pairs:
        if base_c in preds_map and target_c in preds_map:
            pair_key = f"{target_c}_vs_{base_c}"
            cmp = run_paired_comparison(
                baseline_condition=base_c,
                target_condition=target_c,
                predictions_dir=output_dir,
                train_json_path=train_json_p,
            )
            paired_comparisons[pair_key] = cmp
            raw_p_values[pair_key] = cmp["mcnemar_p_value"]

    # 7. Apply Holm-Bonferroni correction
    adjusted_p_values = holm_bonferroni(raw_p_values) if raw_p_values else {}

    # 8. Save summaries
    write_json(output_dir / "metrics_summary.json", condition_summaries)
    write_json(output_dir / "paired_comparisons.json", paired_comparisons)
    write_json(output_dir / "holm_adjusted_p_values.json", adjusted_p_values)

    # 9. Generate Markdown report
    report_p = generate_m15_markdown_report(
        condition_summaries=condition_summaries,
        paired_comparisons=paired_comparisons,
        adjusted_p_values=adjusted_p_values,
        output_dir=output_dir,
    )
    sys.stderr.write(f"\n>>> M1.5 Evaluation Complete! Report generated at: {report_p}\n")
    sys.stderr.flush()


if __name__ == "__main__":
    main()
