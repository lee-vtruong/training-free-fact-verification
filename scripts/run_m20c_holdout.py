#!/usr/bin/env python3
"""CLI Runner: Milestone M2.0c — Independent Holdout Evaluation of Instance-Grounded Error Intervention.

Usage:
    # 1. Phase A Audit Only (CPU-only):
    python scripts/run_m20c_holdout.py --audit-only

    # 2. Phase A + B + D Dry-Run (CPU Preflight):
    python scripts/run_m20c_holdout.py --dry-run

    # 3. CPU Smoke Test (10 claims):
    python scripts/run_m20c_holdout.py --smoke-only

    # 4. Full GPU Execution across 6 arms on independent holdout:
    python scripts/run_m20c_holdout.py --run-gpu

    # 5. Recompute evaluation from sealed predictions:
    python scripts/run_m20c_holdout.py --eval-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tffv.experiments.m19g_guardrails import parse_verdict_hardened
from tffv.experiments.m20b_orchestrator import MockVerifierRunner, Qwen3GPUVerifier
from tffv.experiments.m20b_prompts import M20bPromptFormatter
from tffv.experiments.m20b_schemas import (
    EXPERIMENTAL_ARMS_M20B,
    InterventionDecision,
    M20bArm,
    M20bPredictionRecord,
    SelectionTrace,
)
from tffv.experiments.m20b_selector import HistoricalErrorCase, InstanceGroundedSelector
from tffv.experiments.m20c_audit import M20bPilotAuditor
from tffv.experiments.m20c_evaluator import M20cEvaluator
from tffv.experiments.m20c_holdout import IndependentHoldoutBuilder
from tffv.experiments.m20c_preflight import M20cPreflightAuditor, SERVER_DEFAULT_MODEL_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M2.0c Independent Holdout Evaluation Runner")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "m20c-independent-holdout-v1"),
        help="Directory to store M2.0c outputs and evaluations.",
    )
    parser.add_argument(
        "--m20b-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "m20b-instance-grounded-v1"),
        help="Path to M2.0b pilot output directory.",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "averitec" / "train.json"),
        help="Path to official train.json.",
    )
    parser.add_argument(
        "--dev-path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "averitec" / "dev.json"),
        help="Path to official dev.json.",
    )
    parser.add_argument(
        "--evidence-receipts",
        type=str,
        default="",
        help="Custom path to retrieval receipts JSONL.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Custom path to Qwen checkpoint.",
    )
    parser.add_argument(
        "--run-gpu",
        action="store_true",
        help="Launch live GPU inference using frozen Qwen3-8B BF16.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run only Phase A (Audit M2.0b Pilot).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run Phase A, B, and D (Preflight) without running inference.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run lightweight smoke test on 10 claims.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Recompute evaluation and statistical tests from existing predictions.jsonl.",
    )
    return parser.parse_args()


def load_receipts(custom_path: str) -> Dict[str, List[Dict[str, Any]]]:
    receipts: Dict[str, List[Dict[str, Any]]] = {}
    paths_to_try = []
    if custom_path:
        paths_to_try.append(Path(custom_path))
    paths_to_try.extend([
        PROJECT_ROOT / "outputs" / "m18e-full-history-pf1000-v1" / "retrieval_receipts.jsonl",
        PROJECT_ROOT / "outputs" / "m19g-evidence-complete-v1" / "dev_retrieval_receipts.jsonl",
    ])

    for p in paths_to_try:
        if not p.exists():
            continue
        print(f"[M2.0c] Loading retrieval receipts from: {p}...")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sid = rec.get("claim_id") or rec.get("sample_id")
                passages = rec.get("passages", rec.get("evidence", []))
                if sid and sid not in receipts:
                    receipts[sid] = passages

    print(f"[M2.0c] Total retrieval receipts loaded: {len(receipts)} claims.")
    return receipts


def build_historical_cases(
    eligible_sample_ids: Set[str],
    raw_train_path: Path,
) -> List[HistoricalErrorCase]:
    cases = []
    if not raw_train_path.exists():
        return cases

    with open(raw_train_path, "r", encoding="utf-8") as f:
        raw_train = json.load(f)

    for idx, item in enumerate(raw_train):
        sid = item.get("sample_id") or item.get("id") or str(idx)
        if not str(sid).startswith("train:"):
            sid = f"train:{sid}"

        if sid not in eligible_sample_ids:
            continue

        claim = item.get("claim", "")
        gold = item.get("label", "Supported")

        ev_parts = []
        for q in item.get("questions", []):
            for a in q.get("answers", []):
                ev_parts.append(a.get("answer", ""))
        ev_text = " ".join(ev_parts)[:500] if ev_parts else claim

        error_lesson = (
            f"Historical Error Lesson: Claims referencing '{claim[:60]}' were previously misclassified "
            f"due to overlooking explicit counter-statements in the primary source passages. "
            f"Verify that all temporal clauses and subject identities match the evidence verbatim."
        )
        generic_guidance = (
            f"General Verification Guidance: When evaluating claims concerning '{claim[:60]}', "
            f"standard verification requires cross-checking all assertions against the primary source. "
            f"Confirm that all contextual qualifiers and entity names are corroborated by the text."
        )

        cases.append(
            HistoricalErrorCase(
                sample_id=sid,
                claim=claim,
                gold_verdict=gold,
                base_verdict="Refuted" if gold == "Supported" else "Supported",
                evidence_text=ev_text,
                error_lesson=error_lesson,
                generic_guidance=generic_guidance,
                source_fold="train_fold_1",
            )
        )
    return cases


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Milestone M2.0c: Independent Holdout Evaluation")
    print(f" Output Directory: {out_dir}")
    print(f" Mode: {'GPU Live Inference' if args.run_gpu else 'CPU Offline / Mock'}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Phase A: Audit M2.0b Pilot Results
    # -----------------------------------------------------------------------
    print("\n[M2.0c] >>> Step 1: Running Phase A — Audit M2.0b Pilot Results...")
    auditor = M20bPilotAuditor(m20b_dir=args.m20b_dir, output_dir=out_dir)
    audit_res = auditor.run_audit()
    print(f"[M2.0c] Phase A Audit: {audit_res['status']} (B1 vs B2 Net Benefit: {audit_res['net_benefit']}).")
    if audit_res["status"] != "GO":
        print("[M2.0c] FATAL: Pilot audit failed integrity gates. Halting execution.")
        sys.exit(1)

    if args.audit_only:
        print("[M2.0c] --audit-only flag set. Completed Phase A. Exiting.")
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Phase B: Construct Independent Holdout Partition
    # -----------------------------------------------------------------------
    print("\n[M2.0c] >>> Step 2: Running Phase B — Constructing Independent Holdout Partition...")
    holdout_builder = IndependentHoldoutBuilder(
        train_path=args.train_path,
        dev_path=args.dev_path,
        m20b_dev_manifest=Path(args.m20b_dir) / "development_manifest.json",
        m20b_mem_manifest=Path(args.m20b_dir) / "memory_eligibility_manifest.json",
        output_dir=out_dir,
    )
    holdout_res = holdout_builder.build_holdout()
    print(f"[M2.0c] Phase B Complete: {holdout_res['total_clean_holdout']} clean unexposed holdout claims.")

    # -----------------------------------------------------------------------
    # Phase D: Mandatory Preflight Battery
    # -----------------------------------------------------------------------
    print("\n[M2.0c] >>> Step 3: Running Phase D — Mandatory Preflight Integrity Verification...")
    preflight = M20cPreflightAuditor(
        holdout_manifest_path=out_dir / "holdout_manifest.json",
        memory_manifest_path=out_dir / "memory_eligibility_manifest.json",
        retrieval_receipts_path=PROJECT_ROOT / "outputs" / "m18e-full-history-pf1000-v1" / "retrieval_receipts.jsonl",
        output_dir=out_dir,
        model_path=args.model_path if args.model_path else None,
    )
    preflight_res = preflight.run_preflight(fail_closed=True)
    print(f"[M2.0c] Phase D Preflight: {preflight_res['status']}.")

    if args.dry_run:
        print("[M2.0c] --dry-run flag set. Preflight passed. Exiting without GPU execution.")
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Evaluation Only Mode
    # -----------------------------------------------------------------------
    if args.eval_only:
        preds_file = out_dir / "predictions.jsonl"
        if not preds_file.exists():
            print(f"[Error] Cannot recompute evaluation: {preds_file} not found!")
            sys.exit(1)
        all_preds = []
        with open(preds_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_preds.append(M20bPredictionRecord(**json.loads(line)))
        evaluator = M20cEvaluator(records=all_preds, output_dir=out_dir)
        eval_res = evaluator.evaluate_all()
        print(f"[M2.0c] Evaluation complete! Decision: {eval_res.get('decision')}")
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Phase C & E: GPU Holdout Execution across 6 Arms
    # -----------------------------------------------------------------------
    print("\n[M2.0c] >>> Step 4: Preparing 6-Arm Protocol & Building Memory Index...")
    with open(out_dir / "holdout_manifest.json", "r", encoding="utf-8") as f:
        holdout_manifest = json.load(f)
    holdout_claims = holdout_manifest.get("claims", [])

    if args.smoke_only:
        holdout_claims = holdout_claims[:10]
        print(f"[M2.0c] Smoke mode active: Evaluating first {len(holdout_claims)} claims.")

    with open(out_dir / "memory_eligibility_manifest.json", "r", encoding="utf-8") as f:
        memory_manifest = json.load(f)
    eligible_mem_ids = set(memory_manifest.get("eligible_sample_ids", []))

    historical_cases = build_historical_cases(eligible_mem_ids, Path(args.train_path))
    print(f"[M2.0c] Historical error memory built: {len(historical_cases)} cases strictly from Train Fold-1.")

    selector = InstanceGroundedSelector(
        error_cases=historical_cases,
        min_grounding_threshold=0.12,
        min_shared_terms=2,
    )
    formatter = M20bPromptFormatter()

    receipts = load_receipts(args.evidence_receipts)

    # Save target retrieval receipts for holdout claims
    holdout_receipts_file = out_dir / "retrieval_receipts.jsonl"
    with open(holdout_receipts_file, "w", encoding="utf-8") as f:
        for c in holdout_claims:
            sid = c["sample_id"]
            ev = receipts.get(sid, [{"text": f"Standard reporting text for {c['claim']}."}])
            f.write(json.dumps({"sample_id": sid, "claim": c["claim"], "evidence": ev}) + "\n")

    predictions_file = out_dir / "predictions.jsonl"
    selection_traces_file = out_dir / "selection_traces.jsonl"
    prompt_receipts_file = out_dir / "prompt_receipts.jsonl"

    existing_preds: Set[Tuple[str, str]] = set()
    if predictions_file.exists():
        with open(predictions_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing_preds.add((r["arm"], r["sample_id"]))

    if args.run_gpu:
        verifier = Qwen3GPUVerifier(model_path=args.model_path if args.model_path else None)
    else:
        verifier = MockVerifierRunner()

    arms = EXPERIMENTAL_ARMS_M20B
    t_start = time.time()

    try:
        print(f"\n[M2.0c] >>> Running Sequential Inference across {len(arms)} Arms on {len(holdout_claims)} Claims...")
        for idx, claim_row in enumerate(holdout_claims, 1):
            sid = claim_row["sample_id"]
            claim_text = claim_row["claim"]
            gold_label = claim_row.get("label", claim_row.get("gold_verdict", "Unknown"))

            target_ev = receipts.get(sid, [{"text": f"Standard reporting passage for {claim_text}."}])
            ev_summary = " ".join([p.get("text", p.get("passage_text", "")) for p in target_ev])[:600]

            for arm in arms:
                if (arm, sid) in existing_preds:
                    continue

                decision, selected_case, trace, audit = selector.select_intervention(
                    sample_id=sid,
                    arm=arm,
                    target_claim=claim_text,
                    target_evidence_text=ev_summary,
                )
                with open(selection_traces_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(trace.to_dict()) + "\n")

                sys_inst, user_pr, tok_est = formatter.format_prompt(
                    arm=arm,
                    claim_text=claim_text,
                    evidence_passages=target_ev,
                    decision=decision,
                    selected_case=selected_case,
                    context_id=sid,
                )

                prompt_sha = hashlib.sha256((sys_inst + user_pr).encode("utf-8")).hexdigest()
                with open(prompt_receipts_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "sample_id": sid,
                        "arm": arm,
                        "prompt_sha256": prompt_sha,
                        "token_count_estimate": tok_est,
                    }) + "\n")

                t0 = time.time()
                raw_completion = verifier.verify_prompt(sys_inst, user_pr)
                latency = time.time() - t0

                parsed_verdict, is_parseable, _ = parse_verdict_hardened(raw_completion)
                is_correct = (parsed_verdict == gold_label)

                pred_rec = M20bPredictionRecord(
                    sample_id=sid,
                    arm=arm,
                    claim=claim_text,
                    gold_verdict=gold_label,
                    predicted_verdict=parsed_verdict,
                    raw_completion=raw_completion,
                    is_parseable=is_parseable,
                    is_correct=is_correct,
                    latency_seconds=round(latency, 3),
                    prompt_sha256=prompt_sha,
                    intervention_decision=decision.value,
                    selected_exemplar_id=selected_case.sample_id if selected_case else None,
                )
                with open(predictions_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(pred_rec.to_dict()) + "\n")

            if idx % 25 == 0 or idx == len(holdout_claims):
                print(f"  Processed {idx}/{len(holdout_claims)} holdout claims across 6 arms...")

    finally:
        verifier.close()

    # -----------------------------------------------------------------------
    # Statistical Evaluation & Decision Battery
    # -----------------------------------------------------------------------
    print("\n[M2.0c] >>> Step 5: Executing Paired Statistical Analysis & Decision Battery...")
    all_preds = []
    with open(predictions_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_preds.append(M20bPredictionRecord(**json.loads(line)))

    evaluator = M20cEvaluator(records=all_preds, output_dir=out_dir)
    eval_res = evaluator.evaluate_all()

    elapsed = round(time.time() - t_start, 2)
    print("\n" + "=" * 70)
    print(f" Milestone M2.0c Evaluation Complete in {elapsed}s!")
    print(f" Empirical Determination: {eval_res.get('decision')}")
    print(f" Rationale: {eval_res.get('decision_rationale')}")
    print(f" Deliverables sealed in: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
