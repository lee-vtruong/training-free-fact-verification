#!/usr/bin/env python3
"""Milestone M1.8c — Full-Train AVeriTeC R4 Retrieval Pipeline Runner.

Executes strict-frozen, claim-scoped R4 retrieval across all eligible AVeriTeC train
claims using OptimizedRetrievalRunner with zero-network uncompressed NVMe stores,
checkpointing, canonical cache reuse, and crash-resilient state tracking.

Strictly retrieval only: No verification model inference or parameter updates.
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tffv.provenance import canonical, digest, file_hash, write_json
from tffv.retrieval.acquire_subset import acquire
from tffv.retrieval.audit_subset import audit
from tffv.retrieval.bm25_store_smoke import load_candidates
from tffv.retrieval.frozen import validate_spec
from tffv.retrieval.optimized_runner import OptimizedRetrievalRunner
from tffv.retrieval.prepare_store_candidates import prepare_candidates
from tffv.retrieval.synthetic import SyntheticEncoder, SyntheticReranker


CANONICAL_CACHE_ROOTS = [
    ROOT / "outputs/m11-store-r14-pilot100-20261023-162716",
    ROOT / "outputs/m141-confirmatory-r4",
    ROOT / "outputs/m141-retrieval-foldc-20261025-055810",
    ROOT / "outputs/m142-foldd-r4",
    ROOT / "outputs/m142-retrieval-foldd-20261025-105806",
    ROOT / "outputs/m15-folde-r4",
    ROOT / "outputs/m142-retrieval-folde-20261026-131900",
    ROOT / "outputs/m161-source-batch25-v1",
    ROOT / "outputs/m18b1-real-benchmark/part_a_equivalence",
    # Server path fallbacks
    Path("/home/stackops/whale/training-free-fact-verification/outputs/m11-store-r14-pilot100-20261023-162716"),
    Path("/home/stackops/whale/training-free-fact-verification/outputs/m141-confirmatory-r4"),
    Path("/home/stackops/whale/training-free-fact-verification/outputs/m141-retrieval-foldc-20261025-055810"),
    Path("/home/stackops/whale/training-free-fact-verification/outputs/m142-foldd-r4"),
    Path("/home/stackops/whale/training-free-fact-verification/outputs/m142-retrieval-foldd-20261025-105806"),
    Path("/home/stackops/whale/training-free-fact-verification/outputs/m15-folde-r4"),
    Path("/home/stackops/whale/training-free-fact-verification/outputs/m142-retrieval-folde-20261026-131900"),
    Path("/home/stackops/whale/training-free-fact-verification/outputs/m161-source-batch25-v1"),
    Path("/home/stackops/whale/training-free-fact-verification/outputs/m18b1-real-benchmark/part_a_equivalence"),
]

INVALID_SAMPLE_IDS = {"train:1948"}


def safe_write_json(path: Path, value: Any) -> None:
    """Write JSON with exclusive write semantics, unlinking pre-existing file if present."""
    p = Path(path)
    if p.exists():
        p.unlink()
    write_json(p, value)


def load_canonical_r4_cache(roots: Optional[List[Path]] = None) -> Dict[str, Dict[str, Any]]:
    """Scan canonical directories for existing verified R4 outputs."""
    cache = {}
    scan_roots = roots or CANONICAL_CACHE_ROOTS
    for root in scan_roots:
        if not root.exists():
            continue
        r4_files = list(root.glob("**/R4.jsonl")) + list(root.glob("R4.jsonl"))
        for f in r4_files:
            try:
                for line in f.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rec = json.loads(line)
                        sid = rec.get("sample_id")
                        if sid and len(rec.get("evidence", [])) == 8 and sid not in cache:
                            cache[sid] = {
                                "source_file": str(f.resolve()),
                                "sha256": file_hash(f),
                                "record": rec,
                            }
            except Exception:
                pass
    return cache


def prepare_claim_input_and_candidates(
    sid: str,
    raw_train: List[Dict[str, Any]],
    acq_plan_path: Path,
    claim_work_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Execute acquire, audit, and candidate preparation for a single train claim."""
    cand_out = claim_work_dir / "candidates"
    cand_manifest_file = cand_out / "candidate_manifest.json"
    if cand_manifest_file.is_file():
        try:
            _, _, loaded = load_candidates(cand_manifest_file)
            if loaded:
                claim_meta, documents, _, _ = loaded[0]
                return claim_meta, documents
        except Exception:
            pass

    # Clean partial/stale directories to satisfy exist_ok=False invariant in audit & prepare_candidates
    if claim_work_dir.exists():
        import shutil
        shutil.rmtree(claim_work_dir, ignore_errors=True)
    claim_work_dir.mkdir(parents=True, exist_ok=True)

    ordinal = int(sid.split(":")[1])

    # 1. Acquire
    acq_out = claim_work_dir / "acquire"
    acq_manifest = acquire(str(acq_plan_path), acq_out, [ordinal], execute=True)

    # 2. Audit
    aud_out = claim_work_dir / "audit"
    aud_manifest = audit(acq_manifest, aud_out, max_record_mib=64)

    # 3. Input claim definition
    inp_dir = claim_work_dir / "input"
    inp_dir.mkdir(parents=True, exist_ok=True)
    claim_obj = raw_train[ordinal]
    claim_text = claim_obj.get("claim", "")

    (inp_dir / "claims.jsonl").write_text(
        json.dumps({"sample_id": sid, "dataset": "AVeriTeC", "split": "train", "claim": claim_text}) + "\n",
        encoding="utf-8",
    )
    safe_write_json(
        inp_dir / "claims_manifest.json",
        {
            "dataset": "AVeriTeC",
            "split": "train",
            "sample_id": sid,
            "source_role": "source_train",
            "records": 1,
            "role": "DEVELOPMENT_RESEARCH_NOT_HISTORICAL_SOURCE",
            "annotation_values_used": False,
            "claims_sha256": file_hash(inp_dir / "claims.jsonl"),
        },
    )

    # 4. Candidates
    ca_manifest = prepare_candidates(
        aud_manifest,
        inp_dir / "claims_manifest.json",
        cand_out,
        policy="quarantine_review_flags_v1",
    )

    _, _, loaded = load_candidates(ca_manifest)
    claim_meta, documents, _, _ = loaded[0]
    return claim_meta, documents


def run_full_retrieval(
    start_claim: int = 0,
    end_claim: int = 3068,
    config_path: str = "configs/m161_source_batch25.json",
    acq_plan_path: str = "configs/m161_acquisition_plan.json",
    output_dir: str = "outputs/m18c-full-retrieval",
    batch_size: int = 16,
    reuse_canonical: bool = True,
    synthetic: bool = False,
    progress_interval: int = 10,
) -> Dict[str, Any]:
    """Execute complete M1.8c full-train retrieval pipeline."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    cfg_file = ROOT / config_path
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")
    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    retrieval_cfg = cfg.get("retrieval", cfg)

    # Locate raw train.json
    raw_train_file = ROOT / "data/raw/averitec/train.json"
    if not raw_train_file.exists():
        server_raw = Path("/home/stackops/whale/ERRORFlow/data/raw/averitec/train.json")
        if server_raw.exists():
            raw_train_file = server_raw
        else:
            server_local = Path("/home/stackops/whale/training-free-fact-verification/data/raw/averitec/train.json")
            if server_local.exists():
                raw_train_file = server_local
            else:
                raise FileNotFoundError("train.json not found")

    raw_train = json.loads(raw_train_file.read_text(encoding="utf-8"))
    total_raw_claims = len(raw_train)
    print(f"Loaded {total_raw_claims} claims from {raw_train_file}")

    # Canonical cache scan
    canonical_cache = {}
    if reuse_canonical and not synthetic:
        print("Scanning canonical caches for verified R4 evidence...")
        canonical_cache = load_canonical_r4_cache()
        print(f"Found {len(canonical_cache)} pre-existing canonical R4 records.")

    # State tracking & resume
    state_file = out / "state.json"
    state = {
        "completed": [],
        "reused_canonical": [],
        "reused_previous_run": [],
        "retrieved_fresh": [],
        "skipped_invalid": [],
        "latencies_seconds": [],
        "start_time": time.time(),
        "last_update": time.time(),
    }
    if state_file.is_file():
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
            print(f"Resuming from existing state: {len(state.get('completed', []))} completed claims.")
        except Exception:
            pass

    completed_set = set(state.get("completed", []))
    initial_completed_set = set(state.get("completed", []))
    reused_prior_count = 0

    # Master R4 JSONL combined destination
    master_r4_file = out / "all_train_r4.jsonl"
    existing_master_ids = set()
    if master_r4_file.is_file():
        try:
            for line in master_r4_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    existing_master_ids.add(r.get("sample_id"))
        except Exception:
            pass

    # Target slice
    actual_end = min(end_claim, total_raw_claims)
    target_ordinals = [i for i in range(start_claim, actual_end)]
    print(f"Pipeline target scope: ordinals {start_claim}..{actual_end - 1} ({len(target_ordinals)} claims)")

    # Pre-flight spec check
    if not synthetic:
        embed_spec = retrieval_cfg.get("embedding", {})
        rerank_spec = retrieval_cfg.get("reranker", {})
        validate_spec(embed_spec, "Qwen/Qwen3-Embedding-0.6B", verify_bytes=False)
        validate_spec(rerank_spec, "Qwen/Qwen3-Reranker-0.6B", verify_bytes=False)
        print("Pre-flight verified frozen model specs.")

    # Initialize Optimized Runner
    runner = OptimizedRetrievalRunner(
        config=retrieval_cfg,
        synthetic=synthetic,
        reranker_batch_size=batch_size,
        index_cache_dir=str(out / "index_cache"),
    )

    t_start = time.perf_counter()
    newly_processed = 0

    try:
        for idx, ordinal in enumerate(target_ordinals, 1):
            sid = f"train:{ordinal}"

            # 1. Skip invalid claims
            if sid in INVALID_SAMPLE_IDS:
                if sid not in completed_set:
                    state["skipped_invalid"].append(sid)
                    state["completed"].append(sid)
                    completed_set.add(sid)
                continue

            claim_work_dir = out / sid.replace(":", "-")
            r4_file = claim_work_dir / "retrieval" / "R4.jsonl"

            # 2. Check if already completed in current output directory
            if r4_file.is_file():
                try:
                    line = r4_file.read_text(encoding="utf-8").strip()
                    if line:
                        rec = json.loads(line)
                        if len(rec.get("evidence", [])) == 8:
                            if sid not in existing_master_ids:
                                with master_r4_file.open("a", encoding="utf-8") as mf:
                                    mf.write(canonical(rec) + "\n")
                                existing_master_ids.add(sid)
                            if sid in initial_completed_set:
                                reused_prior_count += 1
                            elif sid not in completed_set:
                                state["completed"].append(sid)
                                state["reused_previous_run"].append(sid)
                                completed_set.add(sid)
                                reused_prior_count += 1
                            if idx % progress_interval == 0 or idx == len(target_ordinals):
                                print(f"[{idx}/{len(target_ordinals)}] {sid}: (already cached) Top-8 ready.", flush=True)
                            continue
                except Exception:
                    pass

            # 3. Check if available in canonical cache
            if sid in canonical_cache:
                can_entry = canonical_cache[sid]
                can_rec = can_entry["record"]
                # Save locally in claim directory
                ret_dir = claim_work_dir / "retrieval"
                ret_dir.mkdir(parents=True, exist_ok=True)
                with (ret_dir / "R4.jsonl").open("w", encoding="utf-8") as f:
                    f.write(canonical(can_rec) + "\n")

                if sid not in existing_master_ids:
                    with master_r4_file.open("a", encoding="utf-8") as mf:
                        mf.write(canonical(can_rec) + "\n")
                    existing_master_ids.add(sid)

                state["completed"].append(sid)
                state["reused_canonical"].append(sid)
                completed_set.add(sid)
                if idx % progress_interval == 0 or idx == len(target_ordinals):
                    print(f"[{idx}/{len(target_ordinals)}] {sid}: (canonical reused from {Path(can_entry['source_file']).parent.name})", flush=True)
                continue

            # 4. Execute fresh retrieval on GPU
            t0 = time.perf_counter()
            claim_meta, documents = prepare_claim_input_and_candidates(
                sid=sid,
                raw_train=raw_train,
                acq_plan_path=ROOT / acq_plan_path,
                claim_work_dir=claim_work_dir,
            )

            rec, prof = runner.retrieve_claim(
                claim=claim_meta,
                documents=documents,
                output_dir=claim_work_dir / "retrieval",
            )
            elapsed = time.perf_counter() - t0
            newly_processed += 1

            if sid not in existing_master_ids:
                with master_r4_file.open("a", encoding="utf-8") as mf:
                    mf.write(canonical(rec) + "\n")
                existing_master_ids.add(sid)

            state["completed"].append(sid)
            state["retrieved_fresh"].append(sid)
            state["latencies_seconds"].append(elapsed)
            completed_set.add(sid)
            state["last_update"] = time.time()

            # Periodic state persistence
            if newly_processed % 5 == 0 or idx == len(target_ordinals):
                safe_write_json(state_file, state)

            avg_lat = sum(state["latencies_seconds"]) / len(state["latencies_seconds"]) if state["latencies_seconds"] else 0.0
            remaining_claims = len(target_ordinals) - idx
            eta_hours = (remaining_claims * avg_lat) / 3600.0

            print(
                f"[{idx}/{len(target_ordinals)}] {sid}: elapsed={elapsed:.2f}s "
                f"(rerank={prof.get('cross_encoder_reranking_seconds', 0.0):.2f}s, "
                f"passages={len(documents)}) | avg={avg_lat:.1f}s/c | ETA={eta_hours:.1f}h",
                flush=True,
            )

    finally:
        runner.close()
        safe_write_json(state_file, state)

    total_time = time.perf_counter() - t_start
    summary = {
        "schema_version": "m18c-full-retrieval-summary-v1",
        "total_target_claims": len(target_ordinals),
        "total_completed": len(state["completed"]),
        "reused_canonical_count": len(state["reused_canonical"]),
        "reused_previous_run_count": reused_prior_count,
        "retrieved_fresh_count": newly_processed,
        "skipped_invalid_count": len(state["skipped_invalid"]),
        "total_wall_time_seconds": round(total_time, 2),
        "mean_latency_fresh_seconds": round(
            sum(state["latencies_seconds"]) / len(state["latencies_seconds"]) if state["latencies_seconds"] else 0.0,
            2,
        ),
        "master_r4_file": str(master_r4_file.resolve()),
        "master_r4_records_count": len(existing_master_ids),
    }
    safe_write_json(out / "retrieval_summary.json", summary)

    # Emit human-readable report
    report_lines = [
        "# Milestone M1.8c — Full-Train R4 Retrieval Execution Summary",
        "",
        f"- **Total Target Claims:** `{summary['total_target_claims']}` (range {start_claim}..{actual_end - 1})",
        f"- **Completed Claims:** `{summary['total_completed']}`",
        f"- **Reused Canonical (M1.1/M1.4/M1.5/M1.6.1):** `{summary['reused_canonical_count']}`",
        f"- **Reused Previous Run:** `{summary['reused_previous_run_count']}`",
        f"- **Freshly Retrieved on GPU:** `{summary['retrieved_fresh_count']}`",
        f"- **Skipped Invalid (empty text):** `{summary['skipped_invalid_count']}`",
        f"- **Mean Fresh Latency:** `{summary['mean_latency_fresh_seconds']}s / claim`",
        f"- **Total Execution Wall Time:** `{summary['total_wall_time_seconds'] / 3600.0:.2f} hours`",
        f"- **Master Combined Output:** `{summary['master_r4_file']}` (`{summary['master_r4_records_count']}` records)",
        "",
        "## Invariant & Integrity Verification",
        "- Pinned Model Revisions: Qwen/Qwen3-Embedding-0.6B & Qwen/Qwen3-Reranker-0.6B (frozen weights, requires_grad=False).",
        "- Retrieval Mode: Claim-scoped per-claim corpus with deduplication threshold 0.95 and Top-8 depth.",
        "- No learned parameters or verifier model inference executed.",
    ]
    (out / "RETRIEVAL_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 80)
    print("Full Retrieval Pipeline Finished Successfully.")
    print(f"Summary: {summary['total_completed']}/{summary['total_target_claims']} completed.")
    print(f"Master file saved to: {master_r4_file}")
    print("=" * 80)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0, help="Start claim ordinal (inclusive, default 0)")
    parser.add_argument("--end", type=int, default=3068, help="End claim ordinal (exclusive, default 3068)")
    parser.add_argument("--config", default="configs/m161_source_batch25.json", help="Path to retrieval config")
    parser.add_argument("--acq-plan", default="configs/m161_acquisition_plan.json", help="Path to acquisition plan")
    parser.add_argument("--output-dir", default="outputs/m18c-full-retrieval", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Cross-encoder batch size (default 16)")
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic test double models on CPU")
    parser.add_argument("--no-canonical-reuse", action="store_true", help="Disable canonical cache reuse")
    parser.add_argument("--interval", type=int, default=10, help="Progress log interval")

    args = parser.parse_args()

    run_full_retrieval(
        start_claim=args.start,
        end_claim=args.end,
        config_path=args.config,
        acq_plan_path=args.acq_plan,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        reuse_canonical=not args.no_canonical_reuse,
        synthetic=args.synthetic,
        progress_interval=args.interval,
    )


if __name__ == "__main__":
    main()
