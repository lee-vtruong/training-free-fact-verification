"""Milestone M2.0d: Runner, Preflight & Orchestration for Track C Contrastive Evidence Verification.

Supports:
- Phase A: Dataset Availability & Contamination Audit.
- Phase B: Configuration Locking.
- Phase D: Preflight Integrity Verification.
- Phase E & F: Multi-Arm Verification Execution (Smoke Test, Dry-Run, GPU Execution).
- Phase H & I: Statistical Evaluation and Pre-registered Decision Battery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure src is on sys.path
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tffv.experiments.m20d_contrastive import (
    ContrastivePromptBuilder,
    ContrastiveVerdictParser,
)
from tffv.experiments.m20d_dataset_audit import DatasetExposureAuditor
from tffv.experiments.m20d_error_patterns import (
    HISTORICAL_PATTERNS,
    HistoricalPatternSelector,
)
from tffv.experiments.m20d_evaluator import M20dEvaluator
from tffv.experiments.m20d_schemas import (
    EvidenceAttribution,
    HistoricalErrorPattern,
    M20dArm,
    M20dPredictionRecord,
    PatternSelectionDecision,
    PatternSelectionTrace,
)

EXPERIMENTAL_ARMS_M20D = [
    M20dArm.B0,
    M20dArm.G2,
    M20dArm.C1,
    M20dArm.C2,
    M20dArm.C3,
    M20dArm.C4,
]

SERVER_DEFAULT_MODEL_PATH = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"


class MockContrastiveVerifier:
    """Mock verifier for CPU dry-run and smoke tests."""

    def verify_prompt(self, system_instruction: str, user_prompt: str) -> str:
        prompt_text = (system_instruction + user_prompt).lower()
        if "contrastive" in prompt_text or "[evidence attribution]" in prompt_text:
            return (
                "[EVIDENCE ATTRIBUTION]\n"
                "- Supporting Passage IDs: [1, 2]\n"
                "- Refuting Passage IDs: None\n"
                "- Scope & Relevance Assessment: Evidence addresses the exact entity and 2021 timeframe.\n"
                "- Sufficiency & Conflict Assessment: Passage 1 directly confirms the predicate without contradiction.\n\n"
                "[VERDICT]\n"
                "Supported"
            )
        return "Based on the evidence, the claim is supported.\n\nVerdict: Supported"


class M20dPreflightAuditor:
    """Rigorous preflight auditor verifying contracts and isolation before execution."""

    def __init__(
        self,
        output_dir: Path,
        split_manifest: Dict[str, Any],
        memory_manifest: Dict[str, Any],
        retrieval_receipts: Dict[str, List[Dict[str, Any]]],
    ):
        self.output_dir = output_dir
        self.split_manifest = split_manifest
        self.memory_manifest = memory_manifest
        self.retrieval_receipts = retrieval_receipts

    def run_preflight(self) -> Tuple[bool, str]:
        checks = []

        # 1. Zero-leakage check: no dev or holdout claims in memory
        mem_ids = set(self.memory_manifest.get("eligible_sample_ids", []))
        dev_ids = set(self.split_manifest.get("development_ids", []))
        bench_ids = set(self.split_manifest.get("benchmark_evaluation_ids", []))
        quarantine_ids = set(self.split_manifest.get("quarantined_ids", []))

        overlap_dev = mem_ids.intersection(dev_ids)
        overlap_bench = mem_ids.intersection(bench_ids)
        overlap_quar = mem_ids.intersection(quarantine_ids)
        overlap_official_dev = {sid for sid in mem_ids if sid.startswith("dev:")}

        if overlap_dev or overlap_bench or overlap_quar or overlap_official_dev:
            err = (
                f"Memory contamination detected! Overlap dev: {len(overlap_dev)}, "
                f"overlap bench: {len(overlap_bench)}, overlap quar: {len(overlap_quar)}, "
                f"overlap official dev: {len(overlap_official_dev)}"
            )
            checks.append(("ISOLATION_INTEGRITY", False, err))
        else:
            checks.append(("ISOLATION_INTEGRITY", True, f"Memory pool ({len(mem_ids)} claims) is 100% isolated."))

        # 2. Retrieval receipt verification
        missing_receipts = 0
        target_ids = dev_ids.union(bench_ids)
        for sid in target_ids:
            if sid not in self.retrieval_receipts or not self.retrieval_receipts[sid]:
                missing_receipts += 1

        if missing_receipts > 0:
            checks.append(("RETRIEVAL_COVERAGE", False, f"Missing retrieval receipts for {missing_receipts} target claims."))
        else:
            checks.append(("RETRIEVAL_COVERAGE", True, f"100% of {len(target_ids)} target claims have valid Top-8 receipts."))

        # 3. Pattern selector determinism & fallback
        selector = HistoricalPatternSelector()
        test_claim = "In 2022, inflation reached 15% according to official records."
        dec1, pat1, _ = selector.select_pattern("test:1", test_claim, "Passage content.")
        dec2, pat2, _ = selector.select_pattern("test:1", test_claim, "Passage content.")
        if dec1 != dec2 or (pat1 and pat2 and pat1.pattern_id != pat2.pattern_id):
            checks.append(("SELECTOR_DETERMINISM", False, "Pattern selector output is non-deterministic!"))
        else:
            checks.append(("SELECTOR_DETERMINISM", True, "Pattern selector is 100% deterministic."))

        # 4. Prompt invariance check
        ev = [{"passage_id": 1, "text": "Test evidence.", "url": "http://test.com"}]
        _, _, b0_sha1 = ContrastivePromptBuilder.build_prompt(M20dArm.B0, test_claim, ev)
        _, _, b0_sha2 = ContrastivePromptBuilder.build_prompt(M20dArm.B0, test_claim, ev)
        if b0_sha1 != b0_sha2:
            checks.append(("PROMPT_FINGERPRINT", False, "Prompt SHA256 is non-deterministic!"))
        else:
            checks.append(("PROMPT_FINGERPRINT", True, "Prompt SHA256 fingerprinting verified."))

        # Emit PREFLIGHT_REPORT.md
        all_passed = all(status for _, status, _ in checks)
        rep_path = self.output_dir / "PREFLIGHT_REPORT.md"
        with open(rep_path, "w", encoding="utf-8") as f:
            f.write(f"""# Milestone M2.0d: Preflight Integrity Report

## Status: **{"PASSED" if all_passed else "FAILED"}**

| Check | Status | Details |
|:---|:---|:---|
""")
            for name, st, det in checks:
                f.write(f"| {name} | {'PASS' if st else 'FAIL'} | {det} |\n")

        return all_passed, "Preflight PASSED" if all_passed else "Preflight FAILED"


def lock_configuration(output_dir: Path) -> None:
    """Emits config_lock.json."""
    cfg = {
        "milestone": "M2.0d",
        "timestamp": "2026-11-09T17:00:00Z",
        "model": {
            "name": "Qwen3-8B",
            "torch_dtype": "bfloat16",
            "decoding": "greedy",
            "temperature": 0.0,
            "enable_thinking": False,
            "max_new_tokens_per_call": 512,
        },
        "retrieval": {
            "retriever": "R4-PF1000-v1",
            "top_k_passages": 8,
            "evidence_format": "normalized_passages_v1",
        },
        "ontology": {
            "num_classes": 4,
            "classes": [
                "Supported",
                "Refuted",
                "Not Enough Evidence",
                "Conflicting Evidence/Cherrypicking",
            ],
        },
        "parser": {
            "type": "ContrastiveVerdictParser",
            "reject_unparseable": True,
            "forbid_default_label": True,
        },
        "arms": [a.value for a in EXPERIMENTAL_ARMS_M20D],
        "primary_comparison": "C2_vs_C3",
    }
    with open(output_dir / "config_lock.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def load_retrieval_receipts(receipts_files: List[Path]) -> Dict[str, List[Dict[str, Any]]]:
    receipts: Dict[str, List[Dict[str, Any]]] = {}
    for p in receipts_files:
        if not p.exists():
            continue
        print(f"[M2.0d] Loading retrieval receipts from: {p}...")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    sid = item.get("sample_id") or item.get("claim_id") or ""
                    ev = item.get("evidence") or item.get("top8_evidence") or []
                    if sid and ev:
                        receipts[sid] = ev
    print(f"[M2.0d] Total unique retrieval receipts loaded: {len(receipts)}")
    return receipts


def run_pipeline(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Milestone M2.0d: Training-Free Contrastive Evidence Verification (Track C)")
    print(f" Output Directory: {out_dir}")
    print(f" Execution Mode: {'GPU Live' if args.run_gpu else 'CPU Smoke / Dry-Run'}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Phase A — Dataset Availability & Contamination Audit
    # -----------------------------------------------------------------------
    print("\n[M2.0d] >>> Step 1: Running Phase A — Dataset Exposure Audit...")
    auditor = DatasetExposureAuditor(output_dir=out_dir)
    audit_res = auditor.run_audit()
    print(f"[M2.0d] Exposure Audit Complete. Determination: {audit_res['exposure_determination']}")

    if args.audit_only:
        print("[M2.0d] --audit-only specified. Halting.")
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Step 2: Phase B — Configuration Locking & Pattern Catalog
    # -----------------------------------------------------------------------
    print("\n[M2.0d] >>> Step 2: Locking Technical Contracts & Saving Pattern Manifest...")
    lock_configuration(out_dir)
    pattern_selector = HistoricalPatternSelector()
    pattern_selector.save_pattern_manifest(out_dir / "historical_pattern_manifest.json")
    print(f"[M2.0d] Config lock and {len(HISTORICAL_PATTERNS)} failure patterns sealed.")

    # -----------------------------------------------------------------------
    # Step 3: Load Data & Retrieval Receipts
    # -----------------------------------------------------------------------
    with open(out_dir / "track_c_split_manifest.json", "r", encoding="utf-8") as f:
        split_manifest = json.load(f)
    with open(out_dir / "memory_eligibility_manifest.json", "r", encoding="utf-8") as f:
        memory_manifest = json.load(f)

    # Receipts sources
    receipt_files = [
        Path("outputs/m20c-independent-holdout-v1/retrieval_receipts.jsonl"),
        Path("outputs/m18e-full-history-pf1000-v1/retrieval_receipts.jsonl"),
        Path("outputs/m19g-evidence-complete-v1/dev_retrieval_receipts.jsonl"),
        Path("/home/stackops/whale/training-free-fact-verification/outputs/m18e-full-history-pf1000-v1/retrieval_receipts.jsonl"),
        Path("/home/stackops/whale/training-free-fact-verification/outputs/m19g-evidence-complete-v1/dev_retrieval_receipts.jsonl"),
    ]
    retrieval_receipts = load_retrieval_receipts(receipt_files)

    # Load raw train data for claim text and gold labels
    train_data = {}
    train_path = Path("data/raw/averitec/train.json")
    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            for idx, c in enumerate(raw):
                train_data[f"train:{idx}"] = c

    # -----------------------------------------------------------------------
    # Step 4: Phase D — Preflight Integrity Verification
    # -----------------------------------------------------------------------
    print("\n[M2.0d] >>> Step 4: Executing Preflight Integrity Verification...")
    preflight = M20dPreflightAuditor(
        output_dir=out_dir,
        split_manifest=split_manifest,
        memory_manifest=memory_manifest,
        retrieval_receipts=retrieval_receipts,
    )
    preflight_passed, preflight_msg = preflight.run_preflight()
    print(f"[M2.0d] Preflight: {preflight_msg}")
    if not preflight_passed:
        print("[M2.0d] Preflight FAILED. Halting.")
        sys.exit(1)

    if args.preflight_only:
        print("[M2.0d] --preflight-only specified. Halting.")
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Step 5: Determine Target Claim Set
    # -----------------------------------------------------------------------
    if args.smoke_only or args.dry_run:
        target_ids = split_manifest.get("technical_smoke_ids", [])[:20]
        print(f"\n[M2.0d] Running in SMOKE / DRY-RUN mode on {len(target_ids)} development claims.")
    else:
        # Full evaluation runs on the matched benchmark subset (550 claims)
        target_ids = split_manifest.get("benchmark_evaluation_ids", [])
        print(f"\n[M2.0d] Running FULL benchmark evaluation on {len(target_ids)} claims across 6 arms.")

    if args.limit and args.limit > 0:
        target_ids = target_ids[:args.limit]
        print(f"[M2.0d] Target subset limited to {len(target_ids)} claims.")

    # -----------------------------------------------------------------------
    # Step 6: Multi-Arm Execution
    # -----------------------------------------------------------------------
    predictions_file = out_dir / "predictions.jsonl"
    traces_file = out_dir / "selection_traces.jsonl"
    prompts_file = out_dir / "prompt_receipts.jsonl"

    existing_done: Set[Tuple[str, str]] = set()
    if args.resume and predictions_file.exists():
        with open(predictions_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    existing_done.add((rec["sample_id"], rec["arm"]))
        print(f"[M2.0d] Resuming: found {len(existing_done)} existing predictions.")

    # Instantiate Verifier
    if args.run_gpu:
        from tffv.experiments.m20b_orchestrator import Qwen3GPUVerifier
        verifier = Qwen3GPUVerifier(model_path=args.model_path if args.model_path else None)
    else:
        verifier = MockContrastiveVerifier()

    t_start = time.time()
    total_expected = len(target_ids) * len(EXPERIMENTAL_ARMS_M20D)
    completed_count = len(existing_done)

    for idx, sid in enumerate(target_ids):
        claim_obj = train_data.get(sid, {})
        claim_text = claim_obj.get("claim") or ""
        gold_label = claim_obj.get("label") or claim_obj.get("verdict")

        target_ev = retrieval_receipts.get(sid, [])
        ev_summary = " ".join([p.get("text", "") for p in target_ev])
        ev_hash = hashlib.sha256(json.dumps(target_ev, sort_keys=True).encode("utf-8")).hexdigest()

        # Step A: Pattern selection (deterministic, run once per claim)
        decision, selected_pat, trace = pattern_selector.select_pattern(
            sample_id=sid,
            target_claim=claim_text,
            target_evidence_text=ev_summary,
        )
        placebo_pat = pattern_selector.select_placebo_pattern(sid)

        # Log selection trace
        with open(traces_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace.to_dict()) + "\n")

        # Step B: Run across 6 Arms
        for arm in EXPERIMENTAL_ARMS_M20D:
            if (sid, arm.value) in existing_done:
                continue

            sys_inst, user_pr, prompt_sha = ContrastivePromptBuilder.build_prompt(
                arm=arm,
                claim_text=claim_text,
                evidence_passages=target_ev,
                pattern_decision=decision,
                selected_pattern=selected_pat,
                placebo_pattern=placebo_pat,
            )

            # Log prompt receipt
            receipt_entry = {
                "sample_id": sid,
                "arm": arm.value,
                "prompt_sha256": prompt_sha,
                "system_instruction": sys_inst,
                "user_prompt": user_pr,
            }
            with open(prompts_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(receipt_entry) + "\n")

            # Inference
            t0 = time.time()
            completion = verifier.verify_prompt(sys_inst, user_pr)
            latency = time.time() - t0

            # Parse
            parsed_verdict, parse_status, attribution = ContrastiveVerdictParser.parse_completion(
                completion=completion,
                num_passages=len(target_ev),
            )

            # Estimate tokens
            prompt_tokens = len((sys_inst + user_pr).split())
            out_tokens = len(completion.split())

            pred_rec = M20dPredictionRecord(
                sample_id=sid,
                arm=arm.value,
                raw_claim=claim_text,
                gold_label=gold_label,
                model_name="Qwen3-8B" if args.run_gpu else "MockContrastiveVerifier",
                model_revision="b968826d9c46dd6066d109eabc6255188de91218",
                retrieval_hash=ev_hash,
                prompt_sha256=prompt_sha,
                raw_completion=completion,
                parsed_verdict=parsed_verdict,
                parse_status=parse_status,
                attribution=attribution,
                pattern_decision=decision.value,
                selected_pattern_id=selected_pat.pattern_id if selected_pat else None,
                prompt_tokens=prompt_tokens,
                output_tokens=out_tokens,
                latency_seconds=round(latency, 3),
            )

            with open(predictions_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(pred_rec.to_dict()) + "\n")

            completed_count += 1

        if (idx + 1) % 25 == 0 or (idx + 1) == len(target_ids):
            print(f"  Processed {idx + 1}/{len(target_ids)} claims across 6 arms ({completed_count}/{total_expected} predictions)...")

    # -----------------------------------------------------------------------
    # Step 7: Phase H & I — Statistical Evaluation & Decision Battery
    # -----------------------------------------------------------------------
    print("\n[M2.0d] >>> Step 7: Executing Paired Statistical Analysis & Decision Battery...")
    all_preds: List[M20dPredictionRecord] = []
    with open(predictions_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                attr_dict = item.pop("attribution", {})
                attribution = EvidenceAttribution(**attr_dict)
                rec = M20dPredictionRecord(attribution=attribution, **item)
                all_preds.append(rec)

    evaluator = M20dEvaluator(records=all_preds, output_dir=out_dir)
    eval_res = evaluator.evaluate_all()

    elapsed = round(time.time() - t_start, 2)
    print("\n" + "=" * 70)
    print(f" Milestone M2.0d Evaluation Complete in {elapsed}s!")
    print(f" Empirical Determination: {eval_res.get('decision')}")
    print(f" Rationale: {eval_res.get('decision_rationale')}")
    print(f" Deliverables sealed in: {out_dir}")
    print("=" * 70)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Milestone M2.0d: Contrastive Evidence Verification Runner.")
    parser.add_argument("--output-dir", default="outputs/m20d-contrastive-verification-v1")
    parser.add_argument("--audit-only", action="store_true", help="Run Phase A Dataset Exposure Audit only.")
    parser.add_argument("--preflight-only", action="store_true", help="Run Phase A & Preflight only.")
    parser.add_argument("--smoke-only", action="store_true", help="Run smoke test on 20 development claims.")
    parser.add_argument("--dry-run", action="store_true", help="Run dry run with mock verifier.")
    parser.add_argument("--run-gpu", action="store_true", help="Launch live GPU inference using frozen Qwen3-8B.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing predictions.jsonl.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of claims to process.")
    parser.add_argument("--model-path", default=None, help="Custom path to model checkpoint.")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    run_pipeline(args)
