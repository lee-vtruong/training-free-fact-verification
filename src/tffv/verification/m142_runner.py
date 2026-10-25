"""M1.4.2 Mechanism Audit & Fold D Confirmatory Evaluation Runner.

Executes the frozen confirmatory suite across independent evaluation claims (Fold D: train:150..249).
All model parameters, prompt templates, decoding configurations, and selection
logic are strictly frozen.

Pre-registered Conditions:
- D0: Frozen Base Protocol v1.0 (Zero-shot B1)
- D1_matched: Matched generic concise rules (controlled format/length, zero error knowledge)
- D2: Concise source-derived error rules (full empirical rules: R1 + R2)
- D2_minus_r1: Leave-one-out ablation removing Rule 1 (Rule 2 only: missing context)
- D2_minus_r2: Leave-one-out ablation removing Rule 2 (Rule 1 only: media debate)
- D2_para1: Paraphrase robustness variant 1 (action/guideline style)
- D3: Random error demonstrations (k=2)
- D4: Similarity-retrieved error demonstrations (k=2)

Statistical Contrasts:
- Primary Pre-registered Hypothesis: D2 vs D1_matched
- Secondary Pre-registered Hypotheses:
    * D2 vs D0 (Replication of Fold C nominal signal on fresh N=100)
    * D2 vs D2_minus_r1 (Necessity of Rule 1)
    * D2 vs D2_minus_r2 (Necessity of Rule 2)
    * D2 vs D2_para1 (Syntactic paraphrase invariance)
    * D3 vs D0 & D4 vs D3 (Exemplar retrieval replication)
- Multiplicity Control: Holm-Bonferroni FWER adjustment across contrast family.
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
from tffv.verification.evaluate import compute_metrics, load_gold_labels
from tffv.verification.leakage import assert_fold_disjointness, quarantine_near_duplicates
from tffv.verification.m14_prompting import build_prompt_for_condition_m14
from tffv.verification.ontology import VERDICTS, normalize_verdict
from tffv.verification.predict import dump_jsonl, load_claims_map, load_evidence_by_claim
from tffv.verification.stabilization import (
    _get_claim_status,
    exact_mcnemar_test_twosided,
    paired_bootstrap_ci,
)

DEFAULT_CONDITIONS_M142 = [
    "D0",
    "D1_matched",
    "D2",
    "D2_minus_r1",
    "D2_minus_r2",
    "D2_para1",
    "D3",
    "D4",
]


def holm_bonferroni(p_values: Dict[str, float]) -> Dict[str, float]:
    """Compute Holm-Bonferroni step-down adjusted p-values."""
    sorted_pairs = sorted(p_values.items(), key=lambda x: x[1])
    m = len(sorted_pairs)
    adjusted = {}
    cum_max = 0.0
    for k, (name, p) in enumerate(sorted_pairs):
        adj_p = min(1.0, p * (m - k))
        cum_max = max(cum_max, adj_p)
        adjusted[name] = min(1.0, cum_max)
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
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    gold_idx = np.array([class_to_idx[g] for g in gold_labels])
    base_idx = np.array([class_to_idx.get(p, -1) for p in preds_base])
    target_idx = np.array([class_to_idx.get(p, -1) for p in preds_target])

    point_base_f1 = fast_macro_f1_from_indices(gold_idx, base_idx, num_classes)
    point_target_f1 = fast_macro_f1_from_indices(
        gold_idx, target_idx, num_classes
    )
    point_diff = point_target_f1 - point_base_f1

    boot_indices = rng.integers(0, n, size=(n_boot, n))
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample_idx = boot_indices[b]
        b_gold = gold_idx[sample_idx]
        b_base = base_idx[sample_idx]
        b_target = target_idx[sample_idx]
        f1_b = fast_macro_f1_from_indices(b_gold, b_base, num_classes)
        f1_t = fast_macro_f1_from_indices(b_gold, b_target, num_classes)
        diffs[b] = f1_t - f1_b

    low = float(np.percentile(diffs, 100 * (alpha / 2.0)))
    high = float(np.percentile(diffs, 100 * (1.0 - alpha / 2.0)))
    return point_diff, low, high


def execute_evaluation_condition(
    condition: str,
    claims_map: Dict[str, Dict[str, Any]],
    evidence_map: Dict[str, List[Dict[str, Any]]],
    target_sids: List[str],
    memory_bank: Optional[ErrorMemoryBank],
    backbone: Any,
    output_dir: Path,
    seed: int = 42,
    max_new_tokens: int = 512,
) -> Path:
    """Execute prediction pipeline for one verification condition."""
    cond_dir = output_dir / condition
    cond_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = cond_dir / "predictions.jsonl"
    prompts_path = cond_dir / "prompts.jsonl"
    manifest_path = cond_dir / "condition_manifest.json"

    predictions_records = []
    prompts_records = []
    valid_verdict_count = 0
    total_latency = 0.0

    model_name = getattr(
        backbone,
        "model_id",
        getattr(backbone, "spec", {}).get("model_id", "Qwen3FrozenBackbone"),
    )
    sys.stderr.write(
        f"\n[{condition}] Evaluating {len(target_sids)} claims with {model_name}...\n"
    )
    sys.stderr.flush()

    for idx, sid in enumerate(target_sids, 1):
        claim_data = claims_map.get(sid)
        if not claim_data:
            continue
        claim_text = claim_data.get("claim", "")
        evidence = evidence_map.get(sid, [])

        t0 = time.time()
        prompt_obj = build_prompt_for_condition_m14(
            condition=condition,
            claim=claim_text,
            evidence=evidence,
            memory_bank=memory_bank,
            seed=seed,
        )

        response = backbone.generate(
            messages=prompt_obj["messages"],
            max_new_tokens=max_new_tokens,
            enable_thinking=False,
        )
        latency = time.time() - t0
        total_latency += latency

        gen_text = (
            response.get("generated_text", "")
            if isinstance(response, dict)
            else str(response)
        )

        parsed_verdict = normalize_verdict(gen_text)
        is_valid = parsed_verdict in VERDICTS
        if is_valid:
            valid_verdict_count += 1

        pred_rec = {
            "sample_id": sid,
            "claim": claim_text,
            "condition": condition,
            "predicted_verdict": parsed_verdict,
            "is_valid_verdict": is_valid,
            "raw_generation": gen_text,
            "exemplar_sample_ids": prompt_obj.get("exemplar_sample_ids", []),
            "prompt_sha256": prompt_obj.get("prompt_sha256", ""),
            "latency_seconds": round(latency, 4),
        }
        predictions_records.append(pred_rec)
        prompts_records.append(
            {
                "sample_id": sid,
                "condition": condition,
                "prompt": prompt_obj["user"],
                "prompt_sha256": prompt_obj.get("prompt_sha256", ""),
            }
        )

        if idx % 20 == 0 or idx == len(target_sids):
            sys.stderr.write(
                f"  [{condition}] [{idx}/{len(target_sids)}] Valid so far: {valid_verdict_count}/{idx}\n"
            )
            sys.stderr.flush()

    dump_jsonl(predictions_path, predictions_records)
    dump_jsonl(prompts_path, prompts_records)

    manifest = {
        "milestone": "M1.4.2",
        "phase": "CONFIRMATORY_EVALUATION",
        "condition": condition,
        "total_claims": len(target_sids),
        "valid_verdicts": valid_verdict_count,
        "valid_rate": round(valid_verdict_count / len(target_sids), 4)
        if target_sids
        else 0.0,
        "max_new_tokens": max_new_tokens,
        "total_latency_seconds": round(total_latency, 2),
        "mean_latency_seconds": round(total_latency / len(target_sids), 4)
        if target_sids
        else 0.0,
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

        b_stat = _get_claim_status(
            {"predicted_verdict": b_pred, "gold_label": gold}
        )
        t_stat = _get_claim_status(
            {"predicted_verdict": t_pred, "gold_label": gold}
        )

        b_corr = bool(b_stat == "correct")
        t_corr = bool(t_stat == "correct")

        if b_corr:
            correct_base += 1
        if t_corr:
            correct_target += 1

        eval_pairs.append(
            {
                "sample_id": sid,
                "base_correct": b_corr,
                "target_correct": t_corr,
            }
        )
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

    b_discordant = (
        categories["correct_to_wrong"] + categories["correct_to_unparsed"]
    )
    c_discordant = (
        categories["wrong_to_correct"] + categories["unparsed_to_correct"]
    )

    mcnemar_p = exact_mcnemar_test_twosided(b_discordant, c_discordant)
    y_base_bin = [1 if p["base_correct"] else 0 for p in eval_pairs]
    y_target_bin = [1 if p["target_correct"] else 0 for p in eval_pairs]
    acc_ci_low, acc_ci_high = paired_bootstrap_ci(
        y_base_bin, y_target_bin, n_boot=10000, seed=42
    )

    f1_diff, f1_ci_low, f1_ci_high = paired_bootstrap_macro_f1_ci(
        y_gold, y_base_preds, y_target_preds, n_boot=10000, seed=42
    )

    total_claims = len(eval_pairs)
    cat_sum = sum(categories.values())
    assert (
        cat_sum == total_claims
    ), f"Category sum mismatch: {cat_sum} != {total_claims}"

    net_gain = c_discordant - b_discordant
    delta_corr = correct_target - correct_base
    assert (
        net_gain == delta_corr
    ), f"Mathematical identity failed: {net_gain} != {delta_corr}"

    return {
        "baseline_condition": baseline_condition,
        "target_condition": target_condition,
        "total_claims": total_claims,
        "correct_base": correct_base,
        "correct_target": correct_target,
        "net_gain": net_gain,
        "delta_correct": delta_corr,
        "accuracy_delta": round(
            (correct_target - correct_base) / total_claims, 4
        )
        if total_claims
        else 0.0,
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


def generate_m142_markdown_report(
    condition_summaries: Dict[str, Any],
    paired_comparisons: Dict[str, Any],
    adjusted_p_values: Dict[str, float],
    output_dir: Path,
) -> Path:
    """Generate structured markdown report for M1.4.2 Mechanism Audit & Confirmatory Evaluation."""
    report_path = output_dir / "M1_4_2_EVALUATION_REPORT.md"
    lines = [
        "# M1.4.2 — Mechanism Audit & Fold D Confirmatory Evaluation Report",
        "",
        "**Milestone:** M1.4.2 Controlled Verification Mechanism Analysis & Confirmatory Evaluation  ",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        "**Backbone:** Frozen Qwen3-8B snapshot (`b968826d9c46dd6066d109eabc6255188de91218`)  ",
        "**Target Population:** Fold D (`train:150..249`, 100 unobserved claims)  ",
        "**Primary Contrast:** **D2 vs D1_matched** (Empirical Error Rules vs Matched Generic Control)  ",
        "",
        "---",
        "",
        "## 1. Executive Performance Summary",
        "",
        "| Condition | Description | Accuracy | Macro-F1 | Macro-P | Macro-R | Correct / Total |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|",
    ]

    for cond, sm in condition_summaries.items():
        acc_pct = f"{sm['accuracy'] * 100:.1f}%"
        f1_str = f"{sm['macro_f1']:.4f}"
        p_str = f"{sm['macro_precision']:.4f}"
        r_str = f"{sm['macro_recall']:.4f}"
        corr_str = f"{sm.get('correct_count', 0)} / {sm.get('total_evaluated', 0)}"
        lines.append(
            f"| **{cond}** | {cond} | {acc_pct} | {f1_str} | {p_str} | {r_str} | {corr_str} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Hypothesis Testing & Paired Statistical Contrasts",
            "",
            "| Comparison | Role | Net Gain | Accuracy Delta [95% CI] | Macro-F1 Delta [95% CI] | Raw Exact McNemar $p$ | Holm Adjusted $p_{\\text{adj}}$ | Stat. Sig. ($\\alpha=0.05$) |",
            "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
    )

    for pair_key, cmp in paired_comparisons.items():
        role = (
            "**PRIMARY**" if pair_key == "D2_vs_D1_matched" else "Secondary"
        )
        acc_str = f"{cmp['accuracy_delta']:+.2%} `[{cmp['accuracy_95_ci_lower']:+.2f}, {cmp['accuracy_95_ci_upper']:+.2f}]`"
        f1_str = f"{cmp['macro_f1_delta']:+.4f} `[{cmp['macro_f1_95_ci_lower']:+.4f}, {cmp['macro_f1_95_ci_upper']:+.4f}]`"
        raw_p = cmp["mcnemar_p_value"]
        adj_p = adjusted_p_values.get(pair_key, raw_p)
        raw_p_str = f"{raw_p:.4f}" if raw_p is not None else "N/A"
        adj_p_str = f"{adj_p:.4f}" if adj_p is not None else "N/A"
        sig_str = (
            "**YES**" if adj_p is not None and adj_p < 0.05 else "**NO**"
        )
        net_str = f"{cmp['net_gain']:+d}"

        lines.append(
            f"| **{pair_key.replace('_', ' ')}** | {role} | {net_str} | {acc_str} | {f1_str} | {raw_p_str} | {adj_p_str} | {sig_str} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Transition Breakdown (9-Category Full Accounting)",
            "",
            "| Comparison | $C \\to C$ | $C \\to W$ | $C \\to U$ | $W \\to C$ | $W \\to W$ | $W \\to U$ | $U \\to C$ | $U \\to W$ | $U \\to U$ | Total |",
            "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
    )

    for pair_key, cmp in paired_comparisons.items():
        cats = cmp["categories"]
        lines.append(
            f"| **{pair_key.replace('_', ' ')}** | "
            f"{cats['unchanged_correct']} | {cats['correct_to_wrong']} | {cats['correct_to_unparsed']} | "
            f"{cats['wrong_to_correct']} | {cats['unchanged_wrong']} | {cats['wrong_to_unparsed']} | "
            f"{cats['unparsed_to_correct']} | {cats['unparsed_to_wrong']} | {cats['unchanged_unparsed']} | "
            f"**{cmp['total_claims']}** |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Scientific Conclusion & Protocol Safeguards",
            "",
            "- **Primary Contrast Interpretation:** If D2 vs D1_matched shows statistically significant lift ($p_{\\text{adj}} < 0.05$), empirical error rules are confirmed to contribute genuine semantic guidance beyond format brevity.",
            "- **Untouched Evaluation Population:** Fold D (`train:150..249`) was strictly isolated from prompt drafting and error synthesis.",
            "- **Strict Quality-First, Training-Free Execution:** Zero parameters fitted, greedy non-thinking decoding, frozen Qwen3-8B.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="M1.4.2 Mechanism Audit & Confirmatory Runner"
    )
    parser.add_argument(
        "--r4-evidence",
        required=True,
        help="Path to R4.jsonl containing evidence for Fold D",
    )
    parser.add_argument(
        "--claims-jsonl", required=True, help="Path to claims.jsonl for Fold D"
    )
    parser.add_argument(
        "--train-json", required=True, help="Path to official AVeriTeC train.json"
    )
    parser.add_argument(
        "--b1-predictions",
        required=True,
        help="Path to B1 predictions for Fold A source bank",
    )
    parser.add_argument(
        "--r4-source-evidence",
        required=True,
        help="Path to R4 evidence for Fold A source bank",
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
        default=DEFAULT_CONDITIONS_M142,
        help="Conditions to evaluate",
    )
    parser.add_argument(
        "--eval-start",
        type=int,
        default=150,
        help="Start claim index for Fold D (default: 150)",
    )
    parser.add_argument(
        "--eval-end",
        type=int,
        default=250,
        help="End claim index for Fold D (default: 250)",
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
        help="Random seed for tie breaking / random exemplars",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_json_p = Path(args.train_json).expanduser().resolve()
    gold_map = load_gold_labels(train_json_p)

    # 1. Anti-leakage partition & quarantine
    fold_a_sids = [f"train:{i}" for i in range(50)]
    fold_d_sids = [f"train:{i}" for i in range(args.eval_start, args.eval_end)]

    source_claims = [gold_map[sid] for sid in fold_a_sids if sid in gold_map]
    candidate_claims = [gold_map[sid] for sid in fold_d_sids if sid in gold_map]

    admitted_claims, quarantined_claims, q_manifest_p = (
        quarantine_near_duplicates(
            source_claims=source_claims,
            candidate_claims=candidate_claims,
            output_dir=output_dir,
            jaccard_threshold=0.60,
            ngram_threshold=0.50,
        )
    )
    admitted_eval_sids = [c["sample_id"] for c in admitted_claims]
    quarantined_ids = [c["sample_id"] for c in quarantined_claims]

    assert_fold_disjointness(fold_a_sids, admitted_eval_sids)

    partition_manifest = {
        "milestone": "M1.4.2",
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
    evidence_map = load_evidence_by_claim(
        Path(args.r4_evidence).expanduser().resolve(), depth=8
    )

    # 4. Initialize backbone
    if args.use_synthetic:
        backbone = SyntheticBackbone()
    else:
        model_spec = EXPECTED_QWEN3_8B_ID
        if args.model_path:
            p = Path(args.model_path).expanduser().resolve()
            if p.is_dir():
                model_spec = str(p)
        backbone = Qwen3FrozenBackbone(
            spec=model_spec,
            device="cuda",
            dtype="bfloat16",
        )

    # 5. Execute evaluation conditions
    condition_summaries = {}
    for cond in args.conditions:
        preds_p = execute_evaluation_condition(
            condition=cond,
            claims_map=claims_map,
            evidence_map=evidence_map,
            target_sids=admitted_eval_sids,
            memory_bank=memory_bank,
            backbone=backbone,
            output_dir=output_dir,
            seed=args.seed,
        )

        preds_rows = read_jsonl(preds_p)
        eval_rows = [
            {
                "sample_id": r["sample_id"],
                "gold_label": gold_map.get(r["sample_id"], {}).get(
                    "gold_label"
                ),
                "predicted_verdict": r["predicted_verdict"],
                "is_correct": bool(
                    r["predicted_verdict"]
                    == gold_map.get(r["sample_id"], {}).get("gold_label")
                ),
            }
            for r in preds_rows
        ]
        metrics = compute_metrics(eval_rows)
        metrics["correct_count"] = sum(1 for e in eval_rows if e["is_correct"])
        metrics["total_evaluated"] = len(eval_rows)
        condition_summaries[cond] = metrics

    # 6. Compute paired comparisons
    paired_comparisons = {}
    candidate_pairs = [
        ("D1_matched", "D2", "D2_vs_D1_matched"),  # PRIMARY
        ("D0", "D2", "D2_vs_D0"),
        ("D2_minus_r1", "D2", "D2_vs_D2_minus_r1"),
        ("D2_minus_r2", "D2", "D2_vs_D2_minus_r2"),
        ("D2_para1", "D2", "D2_vs_D2_para1"),
        ("D0", "D3", "D3_vs_D0"),
        ("D3", "D4", "D4_vs_D3"),
    ]

    raw_p_values = {}
    for base_c, target_c, pair_name in candidate_pairs:
        if base_c in args.conditions and target_c in args.conditions:
            res = run_paired_comparison(
                baseline_condition=base_c,
                target_condition=target_c,
                predictions_dir=output_dir,
                train_json_path=train_json_p,
            )
            paired_comparisons[pair_name] = res
            if res.get("mcnemar_p_value") is not None:
                raw_p_values[pair_name] = res["mcnemar_p_value"]

    # 7. Apply Holm-Bonferroni correction
    adjusted_p_values = holm_bonferroni(raw_p_values)
    for pair_name, adj_p in adjusted_p_values.items():
        if pair_name in paired_comparisons:
            paired_comparisons[pair_name]["mcnemar_p_adjusted"] = adj_p
            paired_comparisons[pair_name]["statistically_significant_05"] = (
                adj_p < 0.05
            )

    # 8. Dump summary artifacts and Markdown report
    write_json(output_dir / "metrics_summary.json", condition_summaries)
    write_json(output_dir / "paired_comparisons.json", paired_comparisons)

    report_p = generate_m142_markdown_report(
        condition_summaries=condition_summaries,
        paired_comparisons=paired_comparisons,
        adjusted_p_values=adjusted_p_values,
        output_dir=output_dir,
    )

    sys.stderr.write(
        f"\n[M1.4.2 Runner] Complete! Generated evaluation report at: {report_p}\n"
    )
    sys.stderr.flush()
    return output_dir


if __name__ == "__main__":
    main()
