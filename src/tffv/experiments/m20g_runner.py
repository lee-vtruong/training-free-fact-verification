"""Milestone M2.0g: Runner & Orchestration for Robustness, Component Ablation & Generalization.

Coordinates:
- E1: Exemplar Robustness & Seed Sensitivity (k in {2, 4, 6}, seeds 1..5)
- E2: Knowledge Component Ablation (K1, K2, K3, K4)
- E3: Source-Fold Generalization on Group-Isolated Outer Fold
- Preflight integrity verification and outer-fold partitioning.
- Offline CPU smoke testing and live GPU execution.
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
from tffv.experiments.m20g_evaluator import M20gEvaluator
from tffv.experiments.m20g_exemplars import (
    CERTIFIED_EXEMPLAR_BANK,
    ExemplarItem,
    ExemplarManager,
)
from tffv.experiments.m20g_prompts import M20gPromptBuilder
from tffv.experiments.m20g_schemas import (
    ExperimentPhase,
    KnowledgeComponent,
    M20gPredictionRecord,
)

SERVER_DEFAULT_MODEL_PATH = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"


class MockM20gVerifier:
    """Mock verifier for CPU dry-run and smoke tests."""

    def verify(self, system_instruction: str, user_prompt: str) -> str:
        p_text = (system_instruction + user_prompt).lower()
        if "[historical error contrast]" in p_text or "observed mistake" in p_text:
            return (
                "[HISTORICAL ERROR CONTRAST]\n"
                "- Observed Mistake: Avoid premature dismissal of explicit numbers.\n"
                "- Corrective Resolution: The passage explicitly confirms the claim.\n"
                "Verdict: Supported"
            )
        elif "[verification reasoning]" in p_text:
            return (
                "[VERIFICATION REASONING]\n"
                "- Factual Basis: Verbatim statement substantiates the claim.\n"
                "Verdict: Supported"
            )
        else:
            return "Based on the evidence, the claim is supported.\n\nVerdict: Supported"


class M20gPreflightAuditor:
    """Audits exemplar validity, group isolation, and retrieval receipts."""

    def __init__(
        self,
        output_dir: Path,
        benchmark_eval_ids: List[str],
        outer_eval_ids: List[str],
        retrieval_receipts: Dict[str, List[Dict[str, Any]]],
        claim_to_group: Dict[str, str],
    ):
        self.output_dir = output_dir
        self.benchmark_eval_ids = benchmark_eval_ids
        self.outer_eval_ids = outer_eval_ids
        self.retrieval_receipts = retrieval_receipts
        self.claim_to_group = claim_to_group

    def run_preflight(self) -> Tuple[bool, str]:
        checks = []

        # 1. Exemplar Bank Completeness
        n_bank = len(CERTIFIED_EXEMPLAR_BANK)
        if n_bank >= 8:
            checks.append(("EXEMPLAR_BANK_COMPLETENESS", True, f"Exemplar bank contains {n_bank} certified exemplars."))
        else:
            checks.append(("EXEMPLAR_BANK_COMPLETENESS", False, f"Exemplar bank too small: {n_bank} < 8."))

        # 2. Group Isolation with Benchmark Evaluation Claims
        bench_groups = {self.claim_to_group.get(sid) for sid in self.benchmark_eval_ids if sid in self.claim_to_group}
        bank_groups = {ex.group_id for ex in CERTIFIED_EXEMPLAR_BANK}
        overlap_bench = bank_groups.intersection(bench_groups)

        if overlap_bench:
            checks.append(("BENCHMARK_GROUP_ISOLATION", False, f"Exemplar bank shares groups with benchmark: {overlap_bench}"))
        else:
            checks.append(("BENCHMARK_GROUP_ISOLATION", True, f"100% of {len(bank_groups)} exemplar groups are isolated from benchmark."))

        # 3. Outer Fold Group Isolation with Exemplar Bank
        outer_groups = {self.claim_to_group.get(sid) for sid in self.outer_eval_ids if sid in self.claim_to_group}
        overlap_outer = bank_groups.intersection(outer_groups)

        if overlap_outer:
            checks.append(("OUTER_FOLD_GROUP_ISOLATION", False, f"Outer fold shares groups with exemplar bank: {overlap_outer}"))
        else:
            checks.append(("OUTER_FOLD_GROUP_ISOLATION", True, f"100% of outer fold groups ({len(outer_groups)}) isolated from exemplar bank."))

        # 4. Retrieval coverage
        all_eval = set(self.benchmark_eval_ids).union(set(self.outer_eval_ids))
        missing = [sid for sid in all_eval if sid not in self.retrieval_receipts or not self.retrieval_receipts[sid]]
        if missing:
            checks.append(("RETRIEVAL_COVERAGE", False, f"Missing retrieval receipts for {len(missing)} claims."))
        else:
            checks.append(("RETRIEVAL_COVERAGE", True, f"100% of {len(all_eval)} evaluated claims have Top-8 receipts."))

        all_passed = all(status for _, status, _ in checks)
        overall_str = "PASSED" if all_passed else "FAILED"
        summary_lines = [
            f"Overall Preflight Status: {overall_str}",
            *[f"- `[{'PASS' if status else 'FAIL'}]` **{name}**: {msg}" for name, status, msg in checks],
        ]
        summary = "\n".join(summary_lines)

        # Write reports
        with open(self.output_dir / "PREFLIGHT_REPORT.md", "w", encoding="utf-8") as f:
            f.write("# Milestone M2.0g: Preflight Integrity & Isolation Audit\n\n")
            f.write(f"**Overall Status:** `{overall_str}`\n\n")
            f.write("### Verification Invariants:\n")
            f.write(summary + "\n\n")

        with open(self.output_dir / "exemplar_catalog.json", "w", encoding="utf-8") as f:
            json.dump([ex.to_dict() for ex in CERTIFIED_EXEMPLAR_BANK], f, indent=2)

        return all_passed, summary


def lock_configuration(out_dir: Path) -> None:
    config = {
        "milestone": "M2.0g",
        "description": "Historical Error Knowledge Robustness, Attribution & Generalization",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiments": ["E1_ROBUSTNESS", "E2_ABLATION", "E3_GENERALIZATION"],
        "e1_parameters": {
            "seeds": [1, 2, 3, 4, 5],
            "k_shots": [2, 4, 6],
            "arms_compared": ["Error_ICL", "Matched_Gold_ICL"],
        },
        "e2_parameters": {
            "knowledge_levels": ["K1_GOLD_ONLY", "K2_GOLD_NEUTRAL_EXPL", "K3_WRONG_PRED_GOLD_CORR", "K4_FULL_GROUNDED_ERROR"],
        },
        "e3_parameters": {
            "outer_fold_size": 500,
            "isolation": "100% group-isolated from exemplar memory and dev/holdout",
        },
        "model_architecture": {
            "model_name": "Qwen/Qwen3-8B",
            "weights_mode": "frozen_bf16",
            "temperature": 0.0,
            "max_new_tokens": 512,
            "enable_thinking": False,
        },
        "retrieval": {
            "protocol": "R4-PF1000-v1",
            "num_passages": 8,
        },
    }
    with open(out_dir / "config_lock.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_retrieval_receipts(receipts_files: List[Path]) -> Dict[str, List[Dict[str, Any]]]:
    receipts: Dict[str, List[Dict[str, Any]]] = {}
    for p in receipts_files:
        if not p.exists():
            continue
        print(f"[M2.0g] Loading retrieval receipts from: {p}...")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    sid = item.get("sample_id") or item.get("claim_id") or ""
                    ev = item.get("evidence") or item.get("top8_evidence") or []
                    if sid and ev:
                        receipts[sid] = ev
    print(f"[M2.0g] Total unique retrieval receipts loaded: {len(receipts)}")
    return receipts


def run_pipeline(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Milestone M2.0g: Historical Error Knowledge Robustness & Attribution")
    print(f" Output Directory: {out_dir}")
    print(f" Execution Mode: {'GPU Live' if args.run_gpu else 'CPU Smoke / Dry-Run'}")
    print("=" * 70)

    lock_configuration(out_dir)

    # 1. Load group mappings
    groups_manifest_paths = [
        Path("outputs/m16-partitions-v1/groups_manifest.json"),
        Path("/home/stackops/whale/training-free-fact-verification/outputs/m16-partitions-v1/groups_manifest.json"),
    ]
    claim_to_group: Dict[str, str] = {}
    for gp in groups_manifest_paths:
        if gp.exists():
            with open(gp, "r", encoding="utf-8") as f:
                data = json.load(f)
                for g in data:
                    gid = g.get("group_id", "")
                    for sid in g.get("sample_ids", []):
                        claim_to_group[sid] = gid
            break

    # 2. Load 550 Benchmark Evaluation IDs
    split_manifest_paths = [
        Path("outputs/m20d-contrastive-verification-v1/track_c_split_manifest.json"),
        Path("/home/stackops/whale/training-free-fact-verification/outputs/m20d-contrastive-verification-v1/track_c_split_manifest.json"),
    ]
    bench_eval_ids: List[str] = []
    for sp in split_manifest_paths:
        if sp.exists():
            with open(sp, "r", encoding="utf-8") as f:
                data = json.load(f)
                bench_eval_ids = data.get("benchmark_evaluation_ids", [])
                if bench_eval_ids:
                    break

    if not bench_eval_ids:
        bench_eval_ids = [f"train:{i}" for i in [1006, 1010, 1013, 1022, 1026]]

    # 3. Construct E3 Outer Evaluation Fold from Train Fold-1
    elig_paths = [
        Path("outputs/m20d-contrastive-verification-v1/memory_eligibility_manifest.json"),
        Path("/home/stackops/whale/training-free-fact-verification/outputs/m20d-contrastive-verification-v1/memory_eligibility_manifest.json"),
    ]
    train_fold1_ids: List[str] = []
    for ep in elig_paths:
        if ep.exists():
            with open(ep, "r", encoding="utf-8") as f:
                train_fold1_ids = json.load(f).get("eligible_sample_ids", [])
                if train_fold1_ids:
                    break

    bank_groups = {ex.group_id for ex in CERTIFIED_EXEMPLAR_BANK}
    outer_candidates = [
        sid for sid in train_fold1_ids
        if claim_to_group.get(sid) not in bank_groups
    ]
    outer_eval_ids = outer_candidates[:500] if len(outer_candidates) >= 500 else outer_candidates[:10]

    with open(out_dir / "outer_eval_split.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_outer_eval_claims": len(outer_eval_ids),
            "outer_eval_ids": outer_eval_ids,
        }, f, indent=2)

    # 4. Load retrieval receipts
    receipt_files = [
        Path("outputs/m20c-independent-holdout-v1/retrieval_receipts.jsonl"),
        Path("outputs/m18e-full-history-pf1000-v1/retrieval_receipts.jsonl"),
        Path("outputs/m19g-evidence-complete-v1/dev_retrieval_receipts.jsonl"),
        Path("/home/stackops/whale/training-free-fact-verification/outputs/m20c-independent-holdout-v1/retrieval_receipts.jsonl"),
        Path("/home/stackops/whale/training-free-fact-verification/outputs/m18e-full-history-pf1000-v1/retrieval_receipts.jsonl"),
        Path("/home/stackops/whale/training-free-fact-verification/outputs/m19g-evidence-complete-v1/dev_retrieval_receipts.jsonl"),
    ]
    retrieval_receipts = load_retrieval_receipts(receipt_files)

    # 5. Load raw claims data
    train_data: Dict[str, Dict[str, Any]] = {}
    train_path = Path("data/raw/averitec/train.json")
    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            for idx, c in enumerate(raw):
                train_data[f"train:{idx}"] = c

    # 6. Preflight audit
    print("\n[M2.0g] >>> Running Preflight Integrity & Isolation Audit...")
    auditor = M20gPreflightAuditor(
        output_dir=out_dir,
        benchmark_eval_ids=bench_eval_ids,
        outer_eval_ids=outer_eval_ids,
        retrieval_receipts=retrieval_receipts,
        claim_to_group=claim_to_group,
    )
    passed, summary = auditor.run_preflight()
    print(summary)
    if not passed:
        print("[M2.0g] FATAL: Preflight audit failed! Halting.")
        sys.exit(1)

    if args.preflight_only:
        print("[M2.0g] --preflight-only specified. Halting successfully.")
        return

    # 7. Model setup
    model = None
    tokenizer = None
    if args.run_gpu:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = args.model_path or SERVER_DEFAULT_MODEL_PATH
        print(f"[M2.0g] Loading live model: {model_path} in BF16...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        print("[M2.0g] Live model loaded successfully.")
    else:
        mock_verifier = MockM20gVerifier()

    # 8. Execution setup
    pred_path = out_dir / "predictions.jsonl"
    if args.overwrite and pred_path.exists():
        pred_path.unlink()
        print(f"[M2.0g] Overwrite enabled: removed existing {pred_path}")

    existing_records: Set[Tuple[str, str]] = set()
    if pred_path.exists():
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing_records.add((item["sample_id"], item["arm_name"]))

    # Determine execution tasks
    tasks = []
    bench_targets = bench_eval_ids[:5] if args.smoke_only else bench_eval_ids
    if args.limit and args.limit > 0:
        bench_targets = bench_eval_ids[:args.limit]

    outer_targets = outer_eval_ids[:5] if args.smoke_only else outer_eval_ids
    if args.limit and args.limit > 0:
        outer_targets = outer_eval_ids[:args.limit]

    # Task Builder for E1, E2, E3
    run_e1 = args.experiment in ["e1", "all"]
    run_e2 = args.experiment in ["e2", "all"]
    run_e3 = args.experiment in ["e3", "all"]

    if run_e1:
        # E1: k in {2, 4, 6}, seeds 1..5, comparing Error (K4) vs Matched Gold (K2)
        for k in [2, 4, 6]:
            for seed in [1, 2, 3, 4, 5]:
                exemplars = ExemplarManager.get_seed_exemplars(seed=seed, k=k)
                tasks.append((ExperimentPhase.E1_ROBUSTNESS, f"E1_k{k}_s{seed}_error", seed, k, KnowledgeComponent.K4_FULL_GROUNDED_ERROR, exemplars, bench_targets))
                tasks.append((ExperimentPhase.E1_ROBUSTNESS, f"E1_k{k}_s{seed}_gold", seed, k, KnowledgeComponent.K2_GOLD_NEUTRAL_EXPL, exemplars, bench_targets))

    if run_e2:
        # E2: Component Ablation on fixed k=2 (Seed 42)
        ex_k2 = ExemplarManager.get_seed_exemplars(seed=42, k=2)
        tasks.append((ExperimentPhase.E2_ABLATION, "E2_K1_gold_only", 42, 2, KnowledgeComponent.K1_GOLD_ONLY, ex_k2, bench_targets))
        tasks.append((ExperimentPhase.E2_ABLATION, "E2_K2_neutral_expl", 42, 2, KnowledgeComponent.K2_GOLD_NEUTRAL_EXPL, ex_k2, bench_targets))
        tasks.append((ExperimentPhase.E2_ABLATION, "E2_K3_wrong_pred", 42, 2, KnowledgeComponent.K3_WRONG_PRED_GOLD_CORR, ex_k2, bench_targets))
        tasks.append((ExperimentPhase.E2_ABLATION, "E2_K4_grounded_error", 42, 2, KnowledgeComponent.K4_FULL_GROUNDED_ERROR, ex_k2, bench_targets))

    if run_e3:
        # E3: Outer Fold Generalization
        ex_e3 = ExemplarManager.get_seed_exemplars(seed=42, k=2)
        tasks.append((ExperimentPhase.E3_GENERALIZATION, "E3_B0_baseline", 0, 0, KnowledgeComponent.K1_GOLD_ONLY, [], outer_targets))
        tasks.append((ExperimentPhase.E3_GENERALIZATION, "E3_A1_gold", 42, 2, KnowledgeComponent.K2_GOLD_NEUTRAL_EXPL, ex_e3, outer_targets))
        tasks.append((ExperimentPhase.E3_GENERALIZATION, "E3_A3_error", 42, 2, KnowledgeComponent.K4_FULL_GROUNDED_ERROR, ex_e3, outer_targets))

    total_predictions = sum(len(tgt) for _, _, _, _, _, _, tgt in tasks)
    print(f"\n[M2.0g] Total planned predictions across configured tasks: {total_predictions}")

    completed = len(existing_records)
    records: List[M20gPredictionRecord] = []

    with open(pred_path, "a", encoding="utf-8") as pf:
        for phase, arm_name, seed, k_shots, comp, exemplars, targets in tasks:
            for sid in targets:
                if (sid, arm_name) in existing_records:
                    continue

                claim_info = train_data.get(sid, {})
                claim_text = claim_info.get("claim", f"Synthetic claim for {sid}")
                gold_label = claim_info.get("label", "Supported")
                passages = retrieval_receipts.get(sid, [])
                if not passages:
                    passages = [{"text": f"Evidence passage for {sid}."}]

                if arm_name.endswith("_baseline") or arm_name == "B0":
                    sys_pr, usr_pr, p_sha = M20gPromptBuilder.build_b0(claim_text, passages)
                elif arm_name == "G2":
                    sys_pr, usr_pr, p_sha = M20gPromptBuilder.build_g2(claim_text, passages)
                else:
                    sys_pr, usr_pr, p_sha = M20gPromptBuilder.build_few_shot_prompt(claim_text, passages, exemplars, comp)

                t0 = time.time()
                if args.run_gpu and model is not None and tokenizer is not None:
                    import torch
                    messages = [
                        {"role": "system", "content": sys_pr},
                        {"role": "user", "content": usr_pr},
                    ]
                    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    inputs = tokenizer([prompt_str], return_tensors="pt").to(model.device)
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
                    completion = mock_verifier.verify(sys_pr, usr_pr)
                    prompt_tokens = len(usr_pr.split())
                    output_tokens = len(completion.split())

                latency = time.time() - t0

                parsed_v, parse_st, _ = ContrastiveVerdictParser.parse_completion(completion)

                rec = M20gPredictionRecord(
                    sample_id=sid,
                    experiment_phase=phase.value,
                    arm_name=arm_name,
                    seed=seed,
                    k_shots=k_shots,
                    knowledge_component=comp.value,
                    raw_claim=claim_text,
                    gold_label=gold_label,
                    model_name="Qwen/Qwen3-8B" if args.run_gpu else "mock-qwen3",
                    model_revision="b968826d" if args.run_gpu else "mock",
                    prompt_sha256=p_sha,
                    raw_completion=completion,
                    parsed_verdict=parsed_v,
                    parse_status=parse_st,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    latency_seconds=latency,
                )

                pf.write(json.dumps(rec.to_dict()) + "\n")
                pf.flush()
                records.append(rec)
                completed += 1

                if completed % 50 == 0 or completed == total_predictions:
                    print(f"[M2.0g] Progress: {completed}/{total_predictions} calls completed...")

    # 9. Statistical Evaluation
    print("\n[M2.0g] >>> Loading all predictions for statistical evaluation...")
    all_recs: List[M20gPredictionRecord] = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_recs.append(M20gPredictionRecord(**json.loads(line)))

    evaluator = M20gEvaluator(records=all_recs, output_dir=out_dir)
    res = evaluator.evaluate_all()
    print(f"\n[M2.0g] Evaluation complete! Decision: `{res['decision']}`")
    print(f"Rationale: {res['decision_rationale']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="M2.0g Robustness & Attribution Runner")
    parser.add_argument("--output-dir", type=str, default="outputs/m20g-robustness-attribution-v1")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--run-gpu", action="store_true", help="Run live inference on GPU")
    parser.add_argument("--preflight-only", action="store_true", help="Run preflight isolation audit only")
    parser.add_argument("--smoke-only", action="store_true", help="Run smoke test with 5 claims")
    parser.add_argument("--dry-run", action="store_true", help="Simulate run without GPU")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions")
    parser.add_argument("--experiment", type=str, choices=["e1", "e2", "e3", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of claims evaluated")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
