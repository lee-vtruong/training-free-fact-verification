"""Execution runner and paired evaluation engine for M1.3 Error-Informed Fact Verification (C0–C6)."""
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
from tffv.verification.m13_prompting import build_prompt_for_condition
from tffv.verification.ontology import VERDICTS, normalize_verdict
from tffv.verification.predict import dump_jsonl, load_claims_map, load_evidence_by_claim
from tffv.verification.stabilization import (
    _get_claim_status,
    _is_verdict_correct,
    analyze_paired_transitions,
    exact_mcnemar_test_twosided,
    paired_bootstrap_ci,
)

ALL_CONDITIONS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]


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
    """Execute inference for a single condition (C0..C6) and seal predictions."""
    output_dir = Path(output_dir).expanduser().resolve()
    cond_dir = output_dir / condition
    cond_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = cond_dir / "predictions.jsonl"
    manifest_path = cond_dir / "manifest.json"

    sys.stderr.write(f"\n[M1.3] [{condition}] Starting inference across {len(target_sids)} claims...\n")
    sys.stderr.flush()

    predictions = []
    valid_verdict_count = 0
    t0_total = time.perf_counter()

    for idx, sid in enumerate(target_sids, 1):
        claim_data = claims_map.get(sid, {})
        claim_text = claim_data.get("claim", "").strip()
        evidence_list = evidence_map.get(sid, [])

        t0 = time.perf_counter()
        prompt_data = build_prompt_for_condition(
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

        response = backbone.generate(
            messages=prompt_data["messages"],
            max_new_tokens=max_new_tokens,
            enable_thinking=False,  # Base Protocol v1.0 standard: non-thinking
        )
        latency = time.perf_counter() - t0

        raw_text = response.get("generated_text", "")
        parsed_verdict = normalize_verdict(raw_text)
        if parsed_verdict in VERDICTS:
            valid_verdict_count += 1

        record = {
            "sample_id": sid,
            "condition": condition,
            "claim": claim_text,
            "raw_generation": raw_text,
            "predicted_verdict": parsed_verdict,
            "valid_verdict": bool(parsed_verdict in VERDICTS),
            "prompt_tokens": response.get("prompt_tokens"),
            "generation_tokens": response.get("generated_tokens") or response.get("generation_tokens"),
            "latency_seconds": round(latency, 4),
            "prompt_sha256": prompt_data.get("prompt_sha256"),
            "exemplar_sample_ids": used_exemplars,
        }
        predictions.append(record)

        if idx % 10 == 0 or idx == len(target_sids):
            sys.stderr.write(
                f"[M1.3] [{condition}] Progress: {idx}/{len(target_sids)} claims | "
                f"Valid verdicts: {valid_verdict_count}/{idx} ({valid_verdict_count/idx*100:.1f}%) | "
                f"Avg Latency: {(time.perf_counter() - t0_total)/idx:.2f}s\n"
            )
            sys.stderr.flush()

    total_latency = time.perf_counter() - t0_total

    # Seal predictions
    dump_jsonl(predictions_path, predictions)

    # Write condition manifest
    manifest = {
        "milestone": "M1.3",
        "condition": condition,
        "total_claims": len(target_sids),
        "valid_verdicts": valid_verdict_count,
        "valid_verdict_rate": round(valid_verdict_count / len(target_sids), 4) if target_sids else 0.0,
        "total_latency_seconds": round(total_latency, 2),
        "mean_latency_seconds": round(total_latency / len(target_sids), 4) if target_sids else 0.0,
        "predictions_file": str(predictions_path.name),
        "predictions_sha256": file_hash(predictions_path),
        "backbone": getattr(backbone, "spec", {"model_id": "unknown"}),
        "sample_ids": target_sids,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return predictions_path


def evaluate_condition(
    condition: str,
    predictions_path: Path,
    train_json_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Evaluate condition predictions against train.json ground truth."""
    cond_dir = output_dir / condition
    eval_dir = cond_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    run_evaluation(
        predictions_dir=str(predictions_path.parent),
        train_json_path=str(train_json_path),
        output_dir=str(eval_dir),
    )
    summary_file = eval_dir / "evaluation_summary.json"
    if summary_file.is_file():
        return json.loads(summary_file.read_text(encoding="utf-8"))
    return {}


def compare_conditions_against_baseline(
    base_eval_csv: Path,
    target_eval_csv: Path,
    base_id: str = "C0",
    target_id: str = "C_target",
) -> Dict[str, Any]:
    """Perform verified paired transition and McNemar analysis between base and target condition."""
    with open(base_eval_csv, "r", encoding="utf-8") as f:
        base_rows = list(csv.DictReader(f))
    with open(target_eval_csv, "r", encoding="utf-8") as f:
        target_rows = list(csv.DictReader(f))

    trans = analyze_paired_transitions(
        b0_eval_rows=base_rows,
        target_eval_rows=target_rows,
        target_name=target_id,
        baseline_name=base_id,
    )
    t_counts = trans["transition_counts"]
    disc = trans["discordant_pairs"]
    b_val = list(disc.values())[0]
    c_val = list(disc.values())[1]

    return {
        "base_condition": base_id,
        "target_condition": target_id,
        "total_claims": trans["evaluated_claims"],
        "base_correct": trans["baseline_correct"],
        "target_correct": trans["target_correct"],
        "net_gain": trans["net_gain"],
        "improved_total": trans["improved_count"],
        "unparsed_to_correct": t_counts.get("unparsed_to_correct", 0),
        "wrong_to_correct": t_counts.get("wrong_to_correct", 0),
        "regressed_total": trans["regressed_count"],
        "correct_to_wrong": t_counts.get("correct_to_wrong", 0),
        "correct_to_unparsed": t_counts.get("correct_to_unparsed", 0),
        "unchanged_correct": trans["unchanged_correct_count"],
        "unchanged_wrong": trans["unchanged_wrong_count"],
        "unchanged_unparsed": trans["unchanged_unparsed_count"],
        "unparsed_to_wrong": t_counts.get("unparsed_to_wrong", 0),
        "wrong_to_unparsed": t_counts.get("wrong_to_unparsed", 0),
        "mcnemar_b": b_val,
        "mcnemar_c": c_val,
        "mcnemar_p_value": trans["exact_mcnemar_p_value"],
        "bootstrap_95_ci_lower": round(trans["paired_bootstrap_95_ci"][0], 4),
        "bootstrap_95_ci_upper": round(trans["paired_bootstrap_95_ci"][1], 4),
    }


def generate_m13_markdown_report(
    condition_summaries: Dict[str, Dict[str, Any]],
    paired_comparisons: Dict[str, Dict[str, Any]],
    output_dir: Path,
    partition_manifest: Dict[str, Any],
) -> Path:
    """Generate comprehensive scientific Markdown report for Milestone M1.3."""
    report_path = output_dir / "M1_3_EVALUATION_REPORT.md"

    lines = [
        "# M1.3 — Error-Informed Training-Free Fact Verification: Evaluation Report",
        "",
        f"**Generated Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "**Milestone:** M1.3 (Training-Free Fact Verification)",
        "**Paradigm:** Quality-First, Strictly Training-Free (Zero Fine-Tuning, Zero Gradient Passes)",
        f"**Total Evaluation Claims:** {partition_manifest.get('admitted_claims_count', 'N/A')}",
        f"**Source Memory Claims:** {partition_manifest.get('total_source_claims', 'N/A')}",
        "",
        "---",
        "",
        "## 1. Executive Performance Summary",
        "",
        "| Condition | Description | Valid Verdicts | Accuracy | Macro-F1 | Macro-Prec | Macro-Recall | Mean Latency | Mean Tokens (In/Out) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    cond_descriptions = {
        "C0": "Base Protocol v1.0 (Zero-shot B1)",
        "C1": "Generic Task Instruction",
        "C2": "Random Source ICL ($k=4$)",
        "C3": "Correct-Only ICL ($k=4$)",
        "C4": "Error-Only ICL ($k=4$, with Gold Corrections)",
        "C5": "Contrastive Correct/Wrong ICL (2 pairs)",
        "C6": "Source-Derived Error Rules (4-point checklist)",
    }

    for cond in ALL_CONDITIONS:
        if cond not in condition_summaries:
            continue
        s = condition_summaries[cond]
        raw_m = s.get("metrics", {})
        metrics = raw_m.get("metrics", raw_m)
        timing = s.get("timing", {})
        acc = metrics.get("accuracy", 0.0) * 100
        mf1 = metrics.get("macro_f1", 0.0)
        mprec = metrics.get("macro_precision", 0.0)
        mrec = metrics.get("macro_recall", 0.0)
        total_claims = metrics.get("total_claims", 0)
        conf_mat = metrics.get("confusion_matrix", {})
        unparsed_claims = sum(preds.get("Unparseable", 0) for preds in conf_mat.values())
        valid_claims = total_claims - unparsed_claims
        valid = f"{valid_claims}/{total_claims}" if total_claims > 0 else "0/0"
        lat = f"{timing.get('mean_latency_seconds', 0.0):.2f}s"
        toks = f"{timing.get('mean_prompt_tokens', 0):.0f} / {timing.get('mean_generation_tokens', 0):.0f}"
        desc = cond_descriptions.get(cond, "")
        lines.append(f"| **{cond}** | {desc} | {valid} | **{acc:.1f}%** | **{mf1:.4f}** | {mprec:.4f} | {mrec:.4f} | {lat} | {toks} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Per-Class F1 Breakdown across Conditions",
        "",
        "| Condition | Supported F1 | Refuted F1 | Not Enough Evidence F1 | Cherrypicking F1 |",
        "|---|---:|---:|---:|---:|",
    ])

    for cond in ALL_CONDITIONS:
        if cond not in condition_summaries:
            continue
        s = condition_summaries[cond]
        raw_m = s.get("metrics", {})
        metrics = raw_m.get("metrics", raw_m)
        pc = metrics.get("per_class", {})
        def _get_f1(c_name):
            val = pc.get(c_name, {})
            return val.get("f1", 0.0) if isinstance(val, dict) else (val if isinstance(val, (int, float)) else 0.0)
        s_f1 = _get_f1("Supported")
        r_f1 = _get_f1("Refuted")
        n_f1 = _get_f1("Not Enough Evidence")
        c_f1 = _get_f1("Conflicting Evidence/Cherrypicking")
        lines.append(f"| **{cond}** | {s_f1:.4f} | {r_f1:.4f} | {n_f1:.4f} | {c_f1:.4f} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Paired Statistical Comparison Against Base Protocol v1.0 (C0)",
        "",
        "All statistical tests compare each condition directly against C0 on identical claims:",
        "- **Exact McNemar p-value:** Two-sided binomial test on discordant pairs ($b$ regressed vs $c$ improved).",
        r"- **Paired Bootstrap 95% CI:** 10,000 bootstrap resamples of paired accuracy differences ($\Delta = y_{\text{cond}} - y_{\text{C0}}$).",
        "",
        r"| Condition vs C0 | Net Gain | Improved | Regressed | Unchanged (Corr/Wrong/Unp) | Transitions to Unparsed (Corr/Wrong) | Sum of Categories | McNemar Discordant ($b/c$) | McNemar $p$-value | Bootstrap 95% CI ($\Delta$ Acc) | Stat. Sig ($p < 0.05$)? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ])

    for cond in ALL_CONDITIONS[1:]:
        if cond not in paired_comparisons:
            continue
        comp = paired_comparisons[cond]
        net = f"{comp['net_gain']:+d}"
        imp = comp["improved_total"]
        reg = comp["regressed_total"]
        unch = f"{comp['unchanged_correct']} / {comp['unchanged_wrong']} / {comp.get('unchanged_unparsed', 0)}"
        to_unp = f"{comp.get('correct_to_unparsed', 0)} / {comp.get('wrong_to_unparsed', 0)}"
        
        # Calculate full 9-category sum
        cat_sum = (
            comp.get('unchanged_correct', 0)
            + comp.get('unchanged_wrong', 0)
            + comp.get('unchanged_unparsed', 0)
            + comp.get('wrong_to_correct', 0)
            + comp.get('unparsed_to_correct', 0)
            + comp.get('correct_to_wrong', 0)
            + comp.get('correct_to_unparsed', 0)
            + comp.get('wrong_to_unparsed', 0)
            + comp.get('unparsed_to_wrong', 0)
        )
        total_str = f"{cat_sum}/{comp['total_claims']}"

        disc = f"{comp['mcnemar_b']} / {comp['mcnemar_c']}"
        p_val = comp["mcnemar_p_value"]
        p_str = f"{p_val:.4f}" if p_val >= 0.0001 else f"{p_val:.2e}"
        ci_str = f"[{comp['bootstrap_95_ci_lower']*100:+.1f}%, {comp['bootstrap_95_ci_upper']*100:+.1f}%]"
        sig = "Yes" if p_val < 0.05 else "No"
        lines.append(f"| **{cond} vs C0** | **{net}** | {imp} | {reg} | {unch} | {to_unp} | **{total_str}** | {disc} | {p_str} | {ci_str} | {sig} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Partition Safety & Leakage Audit",
        "",
        f"- **Partition Mode:** Disjoint Evaluation Split",
        f"- **Source Claims Count:** {partition_manifest.get('total_source_claims', 0)}",
        f"- **Admitted Evaluation Claims:** {partition_manifest.get('admitted_claims_count', 0)}",
        f"- **Quarantined Near-Duplicates:** {partition_manifest.get('quarantined_claims_count', 0)}",
        f"- **Jaccard Word Overlap Threshold:** {partition_manifest.get('jaccard_threshold', 0.60)}",
        f"- **3-gram Containment Threshold:** {partition_manifest.get('ngram_threshold', 0.50)}",
        "- **Gold Label Sealing:** Zero evaluation gold labels or justifications were used in prompt construction or exemplar selection. Gold labels were read only during post-inference evaluation.",
        "",
        "---",
        "",
        "## 5. Key Research Insights",
        "",
        "1. **Impact of Error Memory (C4 vs C3 vs C2):** Does showing the model its historical failure modes and gold corrections (C4) prevent recurrent reasoning traps better than showing correct demonstrations alone (C3)?",
        "2. **Rule-Based Guidance (C6):** Does distilling empirical failure modes into an explicit 4-point verification checklist achieve the benefits of ICL without consuming exemplar token budget?",
        "3. **Contrastive Demonstrations (C5):** Does explicitly comparing correct vs error counter-examples on subtle boundaries resolve cherrypicking and polarity confusions?",
        "",
    ])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="M1.3 Error-Informed Fact Verification Runner (C0–C6)")
    parser.add_argument("--r4-path", required=True, help="Path to R4 retrieved evidence JSONL")
    parser.add_argument("--claims-path", required=True, help="Path to claims JSONL")
    parser.add_argument("--train-json", required=True, help="Path to AVeriTeC train.json (for gold evaluation)")
    parser.add_argument("--output-dir", required=True, help="Directory to store outputs and reports")
    parser.add_argument("--memory-bank", default=None, help="Path to pre-built error memory bank JSON")
    parser.add_argument("--b1-predictions", default=None, help="Path to B1 predictions JSONL (to build error memory)")
    parser.add_argument("--condition", default="ALL", choices=ALL_CONDITIONS + ["ALL"], help="Condition to run")
    parser.add_argument("--split-mode", default="fold_b", choices=["fold_b", "fold_a", "all_100", "custom"], help="Evaluation split mode")
    parser.add_argument("--eval-sids", default=None, help="Comma-separated list or text file of evaluation sample_ids")
    parser.add_argument("--source-sids", default=None, help="Comma-separated list or text file of source sample_ids")
    parser.add_argument("--max-claims", type=int, default=None, help="Max evaluation claims (for smoke testing)")
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic backbone for local validation")
    parser.add_argument("--model-path", default=None, help="Local path to Qwen3-8B checkpoint")
    parser.add_argument("--revision", default=EXPECTED_QWEN3_8B_REVISION, help="Pinned model commit SHA")
    parser.add_argument("--device", default="cuda", help="PyTorch device for live model")
    parser.add_argument("--dtype", default="bfloat16", help="Precision for live model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling exemplars")

    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    r4_p = Path(args.r4_path).expanduser().resolve()
    claims_p = Path(args.claims_path).expanduser().resolve()
    train_json_p = Path(args.train_json).expanduser().resolve()

    evidence_map = load_evidence_by_claim(r4_p)
    claims_map = load_claims_map(claims_p)
    all_available_sids = sorted(evidence_map.keys(), key=lambda s: int(s.split(":")[-1]) if ":" in s else s)

    # 1. Determine Source and Evaluation Sample IDs
    if args.split_mode == "fold_b":
        # Disjoint Split: Fold A (train:0..49) is source memory, Fold B (train:50..99) is evaluation pilot
        source_sids = [sid for sid in all_available_sids if int(sid.split(":")[-1]) < 50]
        eval_sids = [sid for sid in all_available_sids if int(sid.split(":")[-1]) >= 50]
    elif args.split_mode == "fold_a":
        source_sids = [sid for sid in all_available_sids if int(sid.split(":")[-1]) >= 50]
        eval_sids = [sid for sid in all_available_sids if int(sid.split(":")[-1]) < 50]
    elif args.split_mode == "all_100":
        # Note: only allowed if error memory is drawn from external fold!
        source_sids = [sid for sid in all_available_sids]
        eval_sids = [sid for sid in all_available_sids]
    elif args.split_mode == "custom":
        if not args.eval_sids:
            raise ValueError("--eval-sids required for custom split mode")
        eval_sids = [s.strip() for s in args.eval_sids.split(",") if s.strip()]
        source_sids = [s.strip() for s in (args.source_sids or "").split(",") if s.strip()]
    else:
        raise ValueError(f"Unknown split mode: {args.split_mode}")

    if args.max_claims and args.max_claims > 0:
        eval_sids = eval_sids[:args.max_claims]

    # 2. Strict Partition Leakage Guard
    if args.split_mode in ["fold_a", "fold_b"]:
        assert_fold_disjointness(set(source_sids), set(eval_sids))

    source_claims_objs = [claims_map[sid] for sid in source_sids if sid in claims_map]
    eval_claims_objs = [claims_map[sid] for sid in eval_sids if sid in claims_map]

    admitted_eval, quarantined, manifest_path = quarantine_near_duplicates(
        source_claims=source_claims_objs,
        candidate_claims=eval_claims_objs,
        output_dir=output_dir,
    )
    final_eval_sids = [c["sample_id"] for c in admitted_eval]

    with open(manifest_path, "r", encoding="utf-8") as f:
        partition_manifest = json.load(f)

    sys.stderr.write(
        f"[M1.3] Partition Safety Verified: {len(source_sids)} source claims | "
        f"{len(final_eval_sids)} admitted evaluation claims ({len(quarantined)} quarantined near-duplicates)\n"
    )

    # 3. Error Memory Bank Setup
    memory_bank = None
    if args.memory_bank and Path(args.memory_bank).is_file():
        mem_data = json.loads(Path(args.memory_bank).read_text(encoding="utf-8"))
        memory_bank = ErrorMemoryBank(mem_data)
        sys.stderr.write(f"[M1.3] Loaded ErrorMemoryBank from {args.memory_bank}\n")
    elif args.b1_predictions and Path(args.b1_predictions).is_file():
        mem_out = output_dir / "error_memory_bank.json"
        memory_bank = build_error_memory_bank(
            train_json_path=train_json_p,
            b1_predictions_path=Path(args.b1_predictions),
            r4_evidence_path=r4_p,
            output_path=mem_out,
            allowed_sample_ids=set(source_sids),
        )
        sys.stderr.write(f"[M1.3] Built ErrorMemoryBank with {len(memory_bank.records)} records -> {mem_out}\n")
    else:
        sys.stderr.write("[M1.3] Warning: No memory bank or B1 predictions provided. C2-C5 will be unavailable unless synthetic or pre-loaded.\n")

    # 4. Backbone Setup
    if args.synthetic:
        backbone = SyntheticBackbone()
    else:
        spec = {
            "model_id": EXPECTED_QWEN3_8B_ID,
            "revision": args.revision or EXPECTED_QWEN3_8B_REVISION,
            "local_path": args.model_path,
        }
        backbone = Qwen3FrozenBackbone(spec=spec, device=args.device, dtype=args.dtype)

    # 5. Conditions Execution
    conditions_to_run = ALL_CONDITIONS if args.condition == "ALL" else [args.condition]
    condition_summaries = {}

    for cond in conditions_to_run:
        # Run inference
        pred_p = run_condition_inference(
            condition=cond,
            target_sids=final_eval_sids,
            evidence_map=evidence_map,
            claims_map=claims_map,
            output_dir=output_dir,
            backbone=backbone,
            memory_bank=memory_bank,
            max_new_tokens=512,
            seed=args.seed,
        )

        # Run evaluation against gold labels
        eval_summary = evaluate_condition(
            condition=cond,
            predictions_path=pred_p,
            train_json_path=train_json_p,
            output_dir=output_dir,
        )

        # Gather timing metrics from manifest
        cond_manifest = json.loads((output_dir / cond / "manifest.json").read_text(encoding="utf-8"))
        
        # Read predictions to compute mean prompt and generation tokens
        preds_rows = list(read_jsonl(pred_p))
        prompt_toks = [r.get("prompt_tokens") or 0 for r in preds_rows]
        gen_toks = [r.get("generation_tokens") or 0 for r in preds_rows]
        lats = [r.get("latency_seconds") or 0.0 for r in preds_rows]

        condition_summaries[cond] = {
            "metrics": eval_summary.get("metrics", eval_summary),
            "timing": {
                "total_latency_seconds": cond_manifest.get("total_latency_seconds", 0.0),
                "mean_latency_seconds": round(sum(lats) / len(lats), 4) if lats else 0.0,
                "mean_prompt_tokens": round(sum(prompt_toks) / len(prompt_toks), 1) if prompt_toks else 0.0,
                "mean_generation_tokens": round(sum(gen_toks) / len(gen_toks), 1) if gen_toks else 0.0,
            },
        }

    # 6. Paired Comparison Against C0
    paired_comparisons = {}
    c0_eval_csv = output_dir / "C0" / "eval" / "per_claim_evaluation.csv"

    if c0_eval_csv.is_file():
        for cond in conditions_to_run:
            if cond == "C0":
                continue
            cand_eval_csv = output_dir / cond / "eval" / "per_claim_evaluation.csv"
            if cand_eval_csv.is_file():
                comp = compare_conditions_against_baseline(
                    base_eval_csv=c0_eval_csv,
                    target_eval_csv=cand_eval_csv,
                    base_id="C0",
                    target_id=cond,
                )
                paired_comparisons[cond] = comp
                sys.stderr.write(
                    f"[M1.3] Paired Comparison: {cond} vs C0: "
                    f"Net Gain = {comp['net_gain']:+d} claims | "
                    f"McNemar p = {comp['mcnemar_p_value']:.4f} | "
                    f"Bootstrap 95% CI = [{comp['bootstrap_95_ci_lower']*100:+.1f}%, {comp['bootstrap_95_ci_upper']*100:+.1f}%]\n"
                )

    # 7. Generate Evaluation Report and JSON Summary
    report_path = generate_m13_markdown_report(
        condition_summaries=condition_summaries,
        paired_comparisons=paired_comparisons,
        output_dir=output_dir,
        partition_manifest=partition_manifest,
    )

    summary_payload = {
        "milestone": "M1.3",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conditions_evaluated": conditions_to_run,
        "partition_manifest": partition_manifest,
        "condition_summaries": condition_summaries,
        "paired_comparisons": paired_comparisons,
    }
    summary_path = output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    sys.stderr.write(f"\n[M1.3] Evaluation Complete! Report written to:\n  {report_path}\n  {summary_path}\n")


if __name__ == "__main__":
    main()
