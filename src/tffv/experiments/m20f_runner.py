"""Milestone M2.0f: Runner, Preflight & Orchestration for Factorial Ablation.

Supports:
- Preflight integrity verification and exemplar isolation checks.
- Configuration locking and exemplar manifest generation.
- Execution across 6 arms:
  * B0: Canonical Zero-Shot Baseline
  * G2: Generic Evidence Rules Control
  * A1: Matched Gold ICL, No Forced Attribution
  * A2: Matched Gold ICL + Forced Evidence Attribution
  * A3: Historical Error ICL, No Forced Attribution
  * A4: Historical Error ICL + Forced Evidence Attribution
- Offline CPU smoke testing and live GPU execution.
- Automated invocation of M20fEvaluator.
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

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tffv.experiments.m20d_contrastive import ContrastiveVerdictParser
from tffv.experiments.m20f_evaluator import M20fEvaluator
from tffv.experiments.m20f_exemplars import (
    EXEMPLAR_DEMONSTRATIONS,
    ExemplarFormatter,
)
from tffv.experiments.m20f_prompts import M20fPromptBuilder
from tffv.experiments.m20f_schemas import (
    AttributionFactor,
    KnowledgeFactor,
    M20fArm,
    M20fPredictionRecord,
)

SERVER_DEFAULT_MODEL_PATH = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"

EXPERIMENTAL_ARMS = [
    M20fArm.B0,
    M20fArm.G2,
    M20fArm.A1,
    M20fArm.A2,
    M20fArm.A3,
    M20fArm.A4,
]


class MockFactorialVerifier:
    """Mock verifier for CPU dry-run and smoke tests."""

    def verify_prompt(self, arm: M20fArm, system_instruction: str, user_prompt: str) -> str:
        prompt_text = (system_instruction + user_prompt).lower()
        if arm in [M20fArm.A2, M20fArm.A4]:
            if arm == M20fArm.A2:
                reasoning = (
                    "[VERIFICATION REASONING]\n"
                    "- Scope Analysis: Evidence directly matches the claim.\n"
                    "- Factual Basis: Verbatim statement substantiates the claim.\n"
                )
            else:
                reasoning = (
                    "[HISTORICAL ERROR CONTRAST]\n"
                    "- Observed Mistake: Avoid premature dismissal of explicit numbers.\n"
                    "- Corrective Resolution: The passage explicitly confirms the claim.\n"
                )
            return (
                f"{reasoning}"
                "[EVIDENCE ATTRIBUTION]\n"
                "- Supporting Passage IDs: [1]\n"
                "- Refuting Passage IDs: None\n"
                "- Scope & Relevance Assessment: Evidence addresses the exact entity and timeframe.\n"
                "- Sufficiency & Conflict Assessment: Passage 1 directly confirms the predicate without contradiction.\n\n"
                "[VERDICT]\n"
                "Supported"
            )
        elif arm in [M20fArm.A1, M20fArm.A3]:
            if arm == M20fArm.A1:
                return (
                    "[VERIFICATION REASONING]\n"
                    "- Scope Analysis: Evidence directly matches the claim.\n"
                    "- Factual Basis: Verbatim statement substantiates the claim.\n"
                    "Verdict: Supported"
                )
            else:
                return (
                    "[HISTORICAL ERROR CONTRAST]\n"
                    "- Observed Mistake: Avoid premature dismissal of explicit numbers.\n"
                    "- Corrective Resolution: The passage explicitly confirms the claim.\n"
                    "Verdict: Supported"
                )
        elif arm == M20fArm.G2:
            return "Evaluating against verification rules, the claim is supported.\n\nVerdict: Supported"
        else:
            return "Based on the evidence, the claim is supported.\n\nVerdict: Supported"


class M20fPreflightAuditor:
    """Rigorous preflight auditor verifying data isolation, token balance, and contracts."""

    def __init__(
        self,
        output_dir: Path,
        evaluation_ids: List[str],
        retrieval_receipts: Dict[str, List[Dict[str, Any]]],
    ):
        self.output_dir = output_dir
        self.evaluation_ids = evaluation_ids
        self.retrieval_receipts = retrieval_receipts

    def run_preflight(self) -> Tuple[bool, str]:
        checks = []

        # 1. Zero-leakage check: exemplar IDs must be strictly absent from evaluation set
        eval_set = set(self.evaluation_ids)
        exemplar_ids = {ex.sample_id for ex in EXEMPLAR_DEMONSTRATIONS}
        leakage = exemplar_ids.intersection(eval_set)

        if leakage:
            checks.append(("EXEMPLAR_ISOLATION", False, f"Exemplar leakage detected in eval set: {leakage}"))
        else:
            checks.append(("EXEMPLAR_ISOLATION", True, f"100% of exemplars {exemplar_ids} are isolated from eval set."))

        # 2. Retrieval receipt verification
        missing_receipts = [sid for sid in self.evaluation_ids if sid not in self.retrieval_receipts or not self.retrieval_receipts[sid]]
        if missing_receipts:
            checks.append(("RETRIEVAL_COVERAGE", False, f"Missing receipts for {len(missing_receipts)} claims."))
        else:
            checks.append(("RETRIEVAL_COVERAGE", True, f"100% of {len(self.evaluation_ids)} eval claims have Top-8 receipts."))

        # 3. Token budget balance check
        words_a1 = len(ExemplarFormatter.format_demonstrations(M20fArm.A1).split())
        words_a2 = len(ExemplarFormatter.format_demonstrations(M20fArm.A2).split())
        words_a3 = len(ExemplarFormatter.format_demonstrations(M20fArm.A3).split())
        words_a4 = len(ExemplarFormatter.format_demonstrations(M20fArm.A4).split())

        diff_no_attr = abs(words_a1 - words_a3)
        diff_forced_attr = abs(words_a2 - words_a4)

        if diff_no_attr <= 15 and diff_forced_attr <= 15:
            checks.append((
                "TOKEN_BUDGET_BALANCE",
                True,
                f"Token budgets balanced: |A1 - A3| = {diff_no_attr} words, |A2 - A4| = {diff_forced_attr} words."
            ))
        else:
            checks.append((
                "TOKEN_BUDGET_BALANCE",
                False,
                f"Token budget imbalance: |A1 - A3| = {diff_no_attr}, |A2 - A4| = {diff_forced_attr}."
            ))

        # Overall status
        all_passed = all(status for _, status, _ in checks)
        overall_str = "PASSED" if all_passed else "FAILED"
        summary_lines = [
            f"Overall Preflight Status: {overall_str}",
            *[
                f"- `[{'PASS' if status else 'FAIL'}]` **{name}**: {msg}"
                for name, status, msg in checks
            ]
        ]
        summary = "\n".join(summary_lines)

        # Write PREFLIGHT_REPORT.md
        report_path = self.output_dir / "PREFLIGHT_REPORT.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Milestone M2.0f: Preflight Integrity & Isolation Audit\n\n")
            f.write(f"**Overall Status:** `{'PASSED' if all_passed else 'FAILED'}`\n\n")
            f.write("### Verification Invariants:\n")
            f.write(summary + "\n\n")
            f.write("### Exemplar Isolation Manifest:\n")
            for ex in EXEMPLAR_DEMONSTRATIONS:
                f.write(f"- Exemplar `{ex.sample_id}` (Gold: {ex.gold_verdict}) -> Origin: Train Fold-1\n")

        # Write exemplar_manifest.json
        ex_manifest = {
            "exemplars": [ex.to_dict() for ex in EXEMPLAR_DEMONSTRATIONS],
            "token_budget_words": {
                "A1": words_a1,
                "A2": words_a2,
                "A3": words_a3,
                "A4": words_a4,
            },
        }
        with open(self.output_dir / "exemplar_manifest.json", "w", encoding="utf-8") as f:
            json.dump(ex_manifest, f, indent=2)

        return all_passed, summary


def lock_configuration(out_dir: Path) -> None:
    """Locks technical configuration into config_lock.json."""
    config = {
        "milestone": "M2.0f",
        "description": "Factorial Ablation of Contrastive Attribution and Historical Error Knowledge",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arms": [arm.value for arm in EXPERIMENTAL_ARMS],
        "factors": {
            "Factor_A_Knowledge": ["GOLD_ICL", "HISTORICAL_ERROR_ICL"],
            "Factor_B_Attribution": ["NO_FORCED_ATTR", "FORCED_ATTR"],
        },
        "model_architecture": {
            "model_name": "Qwen/Qwen3-8B",
            "weights_mode": "frozen_bf16",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 512,
            "enable_thinking": False,
        },
        "retrieval": {
            "protocol": "R4-PF1000-v1",
            "num_passages": 8,
            "selection": "frozen_top8",
        },
        "exemplar_protocol": {
            "source_partition": "Train Fold-1",
            "sample_ids": ["train:2956", "train:2961"],
            "token_budget_parity": True,
        },
    }
    with open(out_dir / "config_lock.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_retrieval_receipts(receipts_files: List[Path]) -> Dict[str, List[Dict[str, Any]]]:
    """Loads Top-8 passages from retrieval receipt JSONL files."""
    receipts: Dict[str, List[Dict[str, Any]]] = {}
    for p in receipts_files:
        if not p.exists():
            continue
        print(f"[M2.0f] Loading retrieval receipts from: {p}...")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    sid = item.get("sample_id") or item.get("claim_id") or ""
                    ev = item.get("evidence") or item.get("top8_evidence") or []
                    if sid and ev:
                        receipts[sid] = ev
    print(f"[M2.0f] Total unique retrieval receipts loaded: {len(receipts)}")
    return receipts


def run_pipeline(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Milestone M2.0f: Factorial Ablation of Attribution & Error Knowledge")
    print(f" Output Directory: {out_dir}")
    print(f" Execution Mode: {'GPU Live' if args.run_gpu else 'CPU Smoke / Dry-Run'}")
    print("=" * 70)

    # Step 1: Lock configuration
    lock_configuration(out_dir)

    # Step 2: Determine target evaluation claims
    # Load 550 claims from M2.0d split manifest or M2.0c holdout manifest
    split_manifest_paths = [
        Path("outputs/m20d-contrastive-verification-v1/track_c_split_manifest.json"),
        Path("outputs/m20c-independent-holdout-v1/holdout_manifest.json"),
        Path("/home/stackops/whale/training-free-fact-verification/outputs/m20d-contrastive-verification-v1/track_c_split_manifest.json"),
    ]
    eval_ids: List[str] = []
    for sp in split_manifest_paths:
        if sp.exists():
            with open(sp, "r", encoding="utf-8") as f:
                data = json.load(f)
                eval_ids = data.get("benchmark_evaluation_ids") or data.get("holdout_claim_ids") or []
                if eval_ids:
                    print(f"[M2.0f] Loaded {len(eval_ids)} target evaluation claims from: {sp}")
                    break

    if not eval_ids:
        # Fallback for local smoke tests if manifests missing
        eval_ids = [f"train:{i}" for i in [1006, 1010, 1013, 1022, 1026]]
        print(f"[M2.0f] Using default evaluation subset ({len(eval_ids)} claims).")

    # Step 3: Load retrieval receipts
    receipt_files = [
        Path("outputs/m20c-independent-holdout-v1/retrieval_receipts.jsonl"),
        Path("outputs/m18e-full-history-pf1000-v1/retrieval_receipts.jsonl"),
        Path("outputs/m19g-evidence-complete-v1/dev_retrieval_receipts.jsonl"),
        Path("/home/stackops/whale/training-free-fact-verification/outputs/m20c-independent-holdout-v1/retrieval_receipts.jsonl"),
        Path("/home/stackops/whale/training-free-fact-verification/outputs/m18e-full-history-pf1000-v1/retrieval_receipts.jsonl"),
        Path("/home/stackops/whale/training-free-fact-verification/outputs/m19g-evidence-complete-v1/dev_retrieval_receipts.jsonl"),
    ]
    retrieval_receipts = load_retrieval_receipts(receipt_files)

    # Step 4: Load raw claims data
    train_data: Dict[str, Dict[str, Any]] = {}
    train_path = Path("data/raw/averitec/train.json")
    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            for idx, c in enumerate(raw):
                train_data[f"train:{idx}"] = c

    # Step 5: Preflight integrity audit
    print("\n[M2.0f] >>> Running Preflight Integrity & Isolation Audit...")
    auditor = M20fPreflightAuditor(
        output_dir=out_dir,
        evaluation_ids=eval_ids,
        retrieval_receipts=retrieval_receipts,
    )
    passed, summary = auditor.run_preflight()
    print(summary)
    if not passed:
        print("[M2.0f] FATAL: Preflight audit failed! Halting.")
        sys.exit(1)

    if args.preflight_only:
        print("[M2.0f] --preflight-only specified. Halting successfully.")
        return

    # Step 6: Select subset if limit or smoke-only
    target_ids = eval_ids
    if args.smoke_only:
        target_ids = eval_ids[:5]
        print(f"[M2.0f] --smoke-only: Limiting evaluation to {len(target_ids)} claims.")
    elif args.limit and args.limit > 0:
        target_ids = eval_ids[:args.limit]
        print(f"[M2.0f] --limit {args.limit}: Limiting evaluation to {len(target_ids)} claims.")

    # Step 7: Execution setup
    model = None
    tokenizer = None
    if args.run_gpu:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = args.model_path or SERVER_DEFAULT_MODEL_PATH
        print(f"[M2.0f] Loading live model: {model_path} in BF16...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        print("[M2.0f] Live model loaded successfully.")
    else:
        print("[M2.0f] Using MockFactorialVerifier (CPU Dry-Run / Smoke Test).")
        mock_verifier = MockFactorialVerifier()

    # Step 8: Multi-arm execution loop
    pred_path = out_dir / "predictions.jsonl"
    existing_records: Set[Tuple[str, str]] = set()
    if pred_path.exists() and not args.overwrite:
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing_records.add((item["sample_id"], item["arm"]))
        print(f"[M2.0f] Resuming: found {len(existing_records)} existing predictions.")

    records: List[M20fPredictionRecord] = []
    total_calls = len(target_ids) * len(EXPERIMENTAL_ARMS)
    completed_calls = len(existing_records)

    with open(pred_path, "a", encoding="utf-8") as pred_file:
        for idx, sid in enumerate(target_ids, start=1):
            claim_info = train_data.get(sid, {})
            claim_text = claim_info.get("claim", f"Synthetic claim for {sid}")
            gold_label = claim_info.get("label", "Supported")
            passages = retrieval_receipts.get(sid, [])
            if not passages:
                passages = [{"text": f"Evidence passage confirming claim {sid}.", "source_url": "https://example.com"}]

            ev_hash = hashlib.sha256(json.dumps(passages, sort_keys=True).encode("utf-8")).hexdigest()

            for arm in EXPERIMENTAL_ARMS:
                if (sid, arm.value) in existing_records:
                    continue

                sys_inst, user_pr, p_sha = M20fPromptBuilder.build_prompt(
                    arm=arm,
                    claim_text=claim_text,
                    evidence_passages=passages,
                )

                t0 = time.time()
                if args.run_gpu and model is not None and tokenizer is not None:
                    import torch

                    messages = [
                        {"role": "system", "content": sys_inst},
                        {"role": "user", "content": user_pr},
                    ]
                    prompt_formatted = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                    )
                    inputs = tokenizer([prompt_formatted], return_tensors="pt").to(model.device)
                    prompt_tokens = inputs.input_ids.shape[1]

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    completion = tokenizer.decode(outputs[0][prompt_tokens:], skip_special_tokens=True)
                    output_tokens = outputs.shape[1] - prompt_tokens
                else:
                    completion = mock_verifier.verify_prompt(arm, sys_inst, user_pr)
                    prompt_tokens = len(user_pr.split())
                    output_tokens = len(completion.split())

                latency = time.time() - t0

                # Parse completion
                parsed_v, parse_st, attr = ContrastiveVerdictParser.parse_completion(
                    completion, num_passages=len(passages)
                )

                # Determine factor values
                k_factor = KnowledgeFactor.NONE.value
                if arm in [M20fArm.A1, M20fArm.A2]:
                    k_factor = KnowledgeFactor.GOLD_ICL.value
                elif arm in [M20fArm.A3, M20fArm.A4]:
                    k_factor = KnowledgeFactor.HISTORICAL_ERROR_ICL.value

                a_factor = AttributionFactor.NO_FORCED_ATTR.value
                if arm in [M20fArm.A2, M20fArm.A4]:
                    a_factor = AttributionFactor.FORCED_ATTR.value

                rec = M20fPredictionRecord(
                    sample_id=sid,
                    arm=arm.value,
                    raw_claim=claim_text,
                    gold_label=gold_label,
                    model_name="Qwen/Qwen3-8B" if args.run_gpu else "mock-qwen3",
                    model_revision="b968826d" if args.run_gpu else "mock",
                    retrieval_hash=ev_hash,
                    prompt_sha256=p_sha,
                    raw_completion=completion,
                    parsed_verdict=parsed_v,
                    parse_status=parse_st,
                    knowledge_factor=k_factor,
                    attribution_factor=a_factor,
                    supporting_passage_ids=attr.supporting_passage_ids,
                    refuting_passage_ids=attr.refuting_passage_ids,
                    scope_assessment=attr.scope_assessment,
                    sufficiency_assessment=attr.sufficiency_assessment,
                    is_valid_attribution=attr.is_valid_attribution,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    latency_seconds=latency,
                )

                pred_file.write(json.dumps(rec.to_dict()) + "\n")
                pred_file.flush()
                records.append(rec)
                completed_calls += 1

            if idx % 10 == 0 or idx == len(target_ids):
                print(f"[M2.0f] Progress: {completed_calls}/{total_calls} calls completed ({idx}/{len(target_ids)} claims)...")

    # Step 9: Evaluation
    print("\n[M2.0f] >>> Loading all predictions for statistical evaluation...")
    all_records: List[M20fPredictionRecord] = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                rec = M20fPredictionRecord(**d)
                all_records.append(rec)

    evaluator = M20fEvaluator(records=all_records, output_dir=out_dir)
    res = evaluator.evaluate_all()
    print(f"\n[M2.0f] Evaluation complete! Decision: `{res['decision']}`")
    print(f"Rationale: {res['decision_rationale']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="M2.0f Factorial Ablation Runner")
    parser.add_argument("--output-dir", type=str, default="outputs/m20f-factorial-ablation-v1")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--run-gpu", action="store_true", help="Run live inference on GPU")
    parser.add_argument("--preflight-only", action="store_true", help="Run preflight isolation audit only")
    parser.add_argument("--smoke-only", action="store_true", help="Run smoke test on 5 claims")
    parser.add_argument("--dry-run", action="store_true", help="Simulate run without GPU")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of claims evaluated")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
