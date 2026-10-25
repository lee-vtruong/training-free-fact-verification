"""End-to-end pipeline for preparing arbitrary Fold (e.g. Fold D, train:150..249) R4 evidence.

Performs all M1.1 retrieval stages under strict Quality-First, training-free constraints:
1. Acquires candidate web stores from train_0_999.zip via bounded HTTP range reads.
2. Audits schemas and flags without label or verdict inspection.
3. Prepares candidate stores with quarantine_review_flags_v1 policy.
4. Executes frozen claim-scoped retrieval (R1 BM25, R2 Dense, R3 RRF, R4 Reranker).
5. Enforces verifier contract integrity: exact unique claims, aligned Top-8 evidence
   records, exact claim IDs, and seals evidence_manifest.json.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tffv.data import read_jsonl
from tffv.provenance import canonical, digest, file_hash, write_json
from tffv.retrieval.acquire_subset import acquire
from tffv.retrieval.audit_subset import audit
from tffv.retrieval.prepare_store_candidates import prepare_candidates
from tffv.retrieval.store_retrieval import checked_config, plan, run

DEFAULT_EMBED_MODEL = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
DEFAULT_EMBED_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
DEFAULT_EMBED_SHA256 = "667106c1965ebad2038f612af424b1990ab6e6dabcee5cf20f45ed5c4dee071a"

DEFAULT_RERANK_MODEL = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473"
DEFAULT_RERANK_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
DEFAULT_RERANK_SHA256 = "5219a20fa423e1054c34b93a5a8c0ffa640ca52320181603be32d7afe26de353"


def find_latest_file(pattern: str, base_dir: Path) -> Optional[Path]:
    """Find the most recently modified file matching pattern."""
    matches = sorted(
        base_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return matches[0] if matches else None


def resolve_plan_path(project_root: Path, candidate: Optional[str] = None) -> Path:
    """Resolve acquisition_plan.json path."""
    if candidate:
        p = Path(candidate).expanduser().resolve()
        if p.is_file():
            return p
    found = find_latest_file(
        "outputs/m11-corpus-plan-*/acquisition_plan.json", project_root
    )
    if found and found.is_file():
        return found
    raise FileNotFoundError(
        "Could not find acquisition_plan.json. Provide --plan path explicitly."
    )


def resolve_claims_manifest_path(
    project_root: Path, candidate: Optional[str] = None
) -> Path:
    """Resolve claims_manifest.json path."""
    if candidate:
        p = Path(candidate).expanduser().resolve()
        if p.is_file():
            return p
    found = find_latest_file(
        "outputs/m11-source-claims-*/claims_manifest.json", project_root
    )
    if found and found.is_file():
        return found
    raise FileNotFoundError(
        "Could not find claims_manifest.json. Provide --claims-manifest path explicitly."
    )


def verify_fold_evidence_contract(
    r4_path: Path,
    claims_path: Path,
    start_id: int,
    end_id: int,
) -> Dict[str, Any]:
    """Verify strict integrity of Fold R4 evidence against verifier contract."""
    r4_path = Path(r4_path).expanduser().resolve()
    claims_path = Path(claims_path).expanduser().resolve()

    if not r4_path.is_file():
        raise FileNotFoundError(f"R4 evidence file missing: {r4_path}")
    if not claims_path.is_file():
        raise FileNotFoundError(f"Claims file missing: {claims_path}")

    expected_sids = [f"train:{i}" for i in range(start_id, end_id)]
    expected_count = len(expected_sids)

    claims_rows = read_jsonl(claims_path)
    r4_rows = read_jsonl(r4_path)

    if len(claims_rows) != expected_count:
        raise ValueError(
            f"claims.jsonl record count mismatch: found {len(claims_rows)}, expected {expected_count}"
        )
    if len(r4_rows) != expected_count:
        raise ValueError(
            f"R4.jsonl record count mismatch: found {len(r4_rows)}, expected {expected_count}"
        )

    claims_by_id = {r["sample_id"]: r for r in claims_rows}
    r4_by_id = {r["sample_id"]: r for r in r4_rows}

    for sid in expected_sids:
        if sid not in claims_by_id:
            raise ValueError(f"Missing expected claim ID in claims.jsonl: {sid}")
        if sid not in r4_by_id:
            raise ValueError(f"Missing expected claim ID in R4.jsonl: {sid}")

        claim_rec = claims_by_id[sid]
        r4_rec = r4_by_id[sid]

        if not claim_rec.get("claim"):
            raise ValueError(f"Claim text empty for {sid}")

        evidence = r4_rec.get("evidence", [])
        if not (1 <= len(evidence) <= 8):
            raise ValueError(
                f"Invalid evidence depth for {sid}: {len(evidence)}"
            )

        for ev in evidence:
            for k in ("passage_id", "text", "source_url"):
                if not ev.get(k):
                    raise ValueError(f"Missing evidence field {k} in {sid}")

    return {
        "status": "verified_contract_pass",
        "expected_claims": expected_count,
        "verified_claims": len(claims_rows),
        "verified_r4_records": len(r4_rows),
        "start_claim_id": start_id,
        "end_claim_id": end_id,
        "claims_sha256": file_hash(claims_path),
        "r4_sha256": file_hash(r4_path),
    }


def prepare_fold_pipeline(
    project_root: Path,
    plan_path: Path,
    claims_manifest_path: Path,
    output_dir: Path,
    start_id: int = 150,
    end_id: int = 250,
    tag: str = "foldd",
    config_path: Optional[Path] = None,
    index_cache_dir: Optional[Path] = None,
    synthetic: bool = False,
    execute: bool = False,
) -> Path:
    """Execute end-to-end pipeline to acquire, audit, prepare, and retrieve Fold evidence."""
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    claim_ids = list(range(start_id, end_id))
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    sys.stderr.write(
        f"\n[Fold Evidence Preparation] Initializing tag '{tag}' for claims train:{start_id}..{end_id - 1} ({len(claim_ids)} claims)...\n"
    )
    sys.stderr.flush()

    # Step 1: Set Environment Variables for Qwen3 Adapters if not present
    os.environ.setdefault("TFFV_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    os.environ.setdefault("TFFV_EMBED_REVISION", DEFAULT_EMBED_REVISION)
    os.environ.setdefault("TFFV_EMBED_SHA256", DEFAULT_EMBED_SHA256)
    os.environ.setdefault("TFFV_RERANK_MODEL", DEFAULT_RERANK_MODEL)
    os.environ.setdefault("TFFV_RERANK_REVISION", DEFAULT_RERANK_REVISION)
    os.environ.setdefault("TFFV_RERANK_SHA256", DEFAULT_RERANK_SHA256)

    # Step 2: Stage 1 — Acquire Raw Stores from train_0_999.zip
    existing_store_dirs = sorted(
        [
            d
            for d in project_root.glob(f"outputs/*store-{tag}*")
            if d.is_dir() and any(d.glob("train-*.json"))
        ]
    )
    if existing_store_dirs:
        acq_dir = existing_store_dirs[-1]
        sys.stderr.write(
            f"[Fold Preparation] Reusing existing store directory to resume: {acq_dir}\n"
        )
    else:
        acq_dir = project_root / "outputs" / f"m142-store-{tag}-{timestamp}"
    sys.stderr.write(
        f"[Fold Preparation] [1/5] Acquiring raw stores into: {acq_dir}\n"
    )
    sys.stderr.flush()

    manifest_p = acq_dir / "subset_manifest.json"
    if manifest_p.is_file():
        try:
            m_data = json.loads(manifest_p.read_text(encoding="utf-8"))
            if m_data.get(
                "status"
            ) == "complete_raw_subset_not_admitted" and len(
                m_data.get("stores", [])
            ) == len(claim_ids):
                sys.stderr.write(
                    f"[Fold Preparation] Reusing sealed subset_manifest.json from: {acq_dir}\n"
                )
                acq_manifest = manifest_p
            else:
                acq_manifest = None
        except Exception:
            acq_manifest = None
    else:
        acq_manifest = None

    if acq_manifest is None:
        acq_manifest = acquire(
            plan_path=plan_path,
            destination=acq_dir,
            claim_ids=claim_ids,
            execute=execute,
        )
    if not execute:
        sys.stderr.write(
            f"[Fold Preparation] Preflight plan created: {acq_manifest}\n"
        )
        return acq_manifest

    # Step 3: Stage 2 — Audit Acquired Stores
    audit_dir = project_root / "outputs" / f"m142-audit-{tag}-{timestamp}"
    sys.stderr.write(
        f"[Fold Preparation] [2/5] Auditing raw stores into: {audit_dir}\n"
    )
    sys.stderr.flush()

    audit_summary_path = audit(
        manifest_path=acq_manifest,
        output=audit_dir,
        max_record_mib=64,
    )

    # Step 4: Stage 3 — Prepare Isolated Candidate Stores (Quarantine Review Flags)
    cand_dir = project_root / "outputs" / f"m142-candidates-{tag}-{timestamp}"
    sys.stderr.write(
        f"[Fold Preparation] [3/5] Preparing candidate stores into: {cand_dir}\n"
    )
    sys.stderr.flush()

    candidate_manifest_path = prepare_candidates(
        audit_summary=audit_summary_path,
        claims_manifest=claims_manifest_path,
        output=cand_dir,
        policy="quarantine_review_flags_v1",
    )

    # Step 5: Stage 4 — GPU Retrieval Benchmark (R1 BM25, R2 Dense, R3 RRF, R4 Reranker)
    retrieval_run_dir = (
        project_root / "outputs" / f"m142-retrieval-{tag}-{timestamp}"
    )
    sys.stderr.write(
        f"[Fold Preparation] [4/5] Executing R1-R4 retrieval into: {retrieval_run_dir}\n"
    )
    sys.stderr.flush()

    cfg = None
    if config_path and Path(config_path).is_file():
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))

    retrieval_out = run(
        manifest_path=candidate_manifest_path,
        output=retrieval_run_dir,
        methods=("R1", "R2", "R3", "R4"),
        config=cfg,
        synthetic=synthetic,
        index_cache=str(index_cache_dir) if index_cache_dir else None,
    )

    # Step 6: Stage 5 — Contract Verification and Delivery to Canonical Output Directory
    sys.stderr.write(
        f"[Fold Preparation] [5/5] Delivering and sealing evidence into: {output_dir}\n"
    )
    sys.stderr.flush()

    output_dir.mkdir(parents=True, exist_ok=True)
    files_to_deliver = [
        "R1.jsonl",
        "R2.jsonl",
        "R3.jsonl",
        "R4.jsonl",
        "claims.jsonl",
        "summary.json",
        "manifest.json",
        "diagnostics.jsonl",
        "manual_review_queue.jsonl",
    ]
    for fn in files_to_deliver:
        src_f = retrieval_run_dir / fn
        if src_f.is_file():
            shutil.copy2(src_f, output_dir / fn)

    target_r4 = output_dir / "R4.jsonl"
    target_claims = output_dir / "claims.jsonl"

    contract_report = verify_fold_evidence_contract(
        r4_path=target_r4,
        claims_path=target_claims,
        start_id=start_id,
        end_id=end_id,
    )

    evidence_manifest = {
        "milestone": "M1.4.2",
        "phase": "FOLD_EVIDENCE_PREPARATION",
        "tag": tag,
        "target_split": f"Fold {tag.upper()} (train:{start_id}..{end_id - 1})",
        "start_claim_id": start_id,
        "end_claim_id": end_id,
        "total_claims": len(claim_ids),
        "synthetic": synthetic,
        "contract_verification": contract_report,
        "provenance": {
            "acquisition_manifest": str(acq_manifest),
            "audit_summary": str(audit_summary_path),
            "candidate_manifest": str(candidate_manifest_path),
            "retrieval_run_dir": str(retrieval_run_dir),
            "embed_model": os.environ.get("TFFV_EMBED_MODEL"),
            "embed_revision": os.environ.get("TFFV_EMBED_REVISION"),
            "rerank_model": os.environ.get("TFFV_RERANK_MODEL"),
            "rerank_revision": os.environ.get("TFFV_RERANK_REVISION"),
            "r4_file": str(target_r4),
            "r4_sha256": file_hash(target_r4),
            "claims_file": str(target_claims),
            "claims_sha256": file_hash(target_claims),
        },
    }
    manifest_p = output_dir / "evidence_manifest.json"
    write_json(manifest_p, evidence_manifest)

    sys.stderr.write(
        f"\n[Fold Preparation] SUCCESS! Sealed Fold {tag.upper()} R4 evidence into: {output_dir}\n"
    )
    sys.stderr.write(
        f"  R4 Evidence: {target_r4} ({contract_report['r4_sha256'][:16]}...)\n"
    )
    sys.stderr.write(
        f"  Claims File: {target_claims} ({contract_report['claims_sha256'][:16]}...)\n"
    )
    sys.stderr.write(f"  Evidence Manifest: {manifest_p}\n\n")
    sys.stderr.flush()

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Fold Evidence Preparation Pipeline"
    )
    parser.add_argument("--plan", help="Path to acquisition_plan.json")
    parser.add_argument("--claims-manifest", help="Path to claims_manifest.json")
    parser.add_argument(
        "--start-claim",
        type=int,
        default=150,
        help="Start claim ordinal (default: 150)",
    )
    parser.add_argument(
        "--end-claim",
        type=int,
        default=250,
        help="End claim ordinal (default: 250)",
    )
    parser.add_argument(
        "--tag", default="foldd", help="Identifier tag for fold (default: foldd)"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/m142-foldd-r4",
        help="Delivery directory for R4.jsonl and claims.jsonl",
    )
    parser.add_argument(
        "--config",
        default="configs/store_retrieval_m11.template.json",
        help="Path to retrieval config template",
    )
    parser.add_argument(
        "--index-cache",
        default="outputs/index_cache",
        help="Directory for caching dense vectors",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic doubles for CPU testing",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute full acquisition and retrieval",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    plan_p = resolve_plan_path(project_root, args.plan)
    claims_m_p = resolve_claims_manifest_path(project_root, args.claims_manifest)

    config_p = project_root / args.config if args.config else None
    cache_dir = project_root / args.index_cache if args.index_cache else None

    out = prepare_fold_pipeline(
        project_root=project_root,
        plan_path=plan_p,
        claims_manifest_path=claims_m_p,
        output_dir=project_root / args.output_dir,
        start_id=args.start_claim,
        end_id=args.end_claim,
        tag=args.tag,
        config_path=config_p,
        index_cache_dir=cache_dir,
        synthetic=args.synthetic,
        execute=args.execute,
    )
    print(f"Fold Preparation complete: {out}")


if __name__ == "__main__":
    main()
