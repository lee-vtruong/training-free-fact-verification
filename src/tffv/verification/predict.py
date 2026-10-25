"""Label-free frozen base verification pipeline across evidence-grounded claims."""
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tffv.data import read_jsonl
from tffv.provenance import canonical, digest, file_hash, write_json
from tffv.verification.backbone import (
    EXPECTED_QWEN3_8B_ID,
    EXPECTED_QWEN3_8B_REVISION,
    Qwen3FrozenBackbone,
    SyntheticBackbone,
)
from tffv.verification.ontology import VERDICTS, normalize_verdict
from tffv.verification.prompting import build_verification_prompt


def dump_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_evidence_by_claim(r4_path: Path, depth: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Load Top-N evidence passages per claim from sealed R4 retrieval output."""
    r4_path = Path(r4_path).expanduser().resolve()
    if not r4_path.is_file():
        raise ValueError(f"R4 retrieval output not found: {r4_path}")

    evidence_map = {}
    for row in read_jsonl(r4_path):
        sid = row["sample_id"]
        ev = row.get("evidence", [])
        evidence_map[sid] = ev[:depth] if depth is not None else ev
    return evidence_map


def load_claims_map(claims_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load label-free claim texts from claims.jsonl."""
    claims_path = Path(claims_path).expanduser().resolve()
    if not claims_path.is_file():
        raise ValueError(f"Claims input file not found: {claims_path}")

    claims_map = {}
    for row in read_jsonl(claims_path):
        sid = row["sample_id"]
        claims_map[sid] = row
    return claims_map


def run_prediction(
    r4_path: str,
    claims_path: str,
    output_dir: str,
    spec: Optional[Dict[str, Any]] = None,
    synthetic: bool = False,
    device: str = "cuda",
    dtype: str = "bfloat16",
    max_claims: Optional[int] = None,
    max_new_tokens: int = 512,
    enable_thinking: Optional[bool] = None,
    experiment_id: str = "B0",
) -> Path:
    """Execute label-free frozen base verification and seal output predictions."""
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    r4_p = Path(r4_path).expanduser().resolve()
    claims_p = Path(claims_path).expanduser().resolve()

    evidence_map = load_evidence_by_claim(r4_p)
    claims_map = load_claims_map(claims_p)

    target_sids = sorted(evidence_map.keys(), key=lambda s: int(s.split(":")[-1]) if ":" in s else s)
    if max_claims and max_claims > 0:
        target_sids = target_sids[:max_claims]

    # Instantiate frozen backbone
    if synthetic:
        backbone = SyntheticBackbone(spec)
    else:
        if spec is None:
            raise ValueError("Backbone spec required for non-synthetic execution")
        backbone = Qwen3FrozenBackbone(spec, device=device, dtype=dtype)

    predictions = []
    started_total = time.perf_counter()

    import sys
    sys.stderr.write(
        f"[verification.predict] [{experiment_id}] Starting frozen verification: "
        f"claims={len(target_sids)}, max_new_tokens={max_new_tokens}, enable_thinking={enable_thinking}\n"
    )
    sys.stderr.flush()

    valid_verdict_count = 0

    for idx, sid in enumerate(target_sids, 1):
        claim_data = claims_map.get(sid)
        if not claim_data:
            raise ValueError(f"Claim text missing for sample_id {sid}")

        claim_text = claim_data["claim"]
        evidence = evidence_map.get(sid, [])

        prompt_info = build_verification_prompt(claim_text, evidence)
        
        t0 = time.perf_counter()
        backbone.reset_peak()
        gen_res = backbone.generate(
            prompt_info["messages"],
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
        )
        latency = time.perf_counter() - t0

        predicted_verdict = normalize_verdict(gen_res["generated_text"])
        is_valid = bool(predicted_verdict and predicted_verdict in VERDICTS)
        if is_valid:
            valid_verdict_count += 1

        peak_mem = backbone.peak_bytes()

        evidence_pids = [ev.get("passage_id") for ev in evidence if "passage_id" in ev]
        evidence_urls = [ev.get("source_url") for ev in evidence if "source_url" in ev]

        pred_record = {
            "sample_id": sid,
            "experiment_id": experiment_id,
            "claim_sha256": digest(claim_text),
            "predicted_verdict": predicted_verdict,
            "valid_verdict": is_valid,
            "raw_generation": gen_res["generated_text"],
            "prompt_tokens": gen_res["prompt_tokens"],
            "generated_tokens": gen_res["generated_tokens"],
            "max_new_tokens": max_new_tokens,
            "enable_thinking": enable_thinking,
            "latency_seconds": latency,
            "peak_gpu_memory_bytes": peak_mem,
            "evidence_count": len(evidence),
            "evidence_passage_ids": evidence_pids,
            "evidence_source_urls": evidence_urls,
            "prompt_sha256": prompt_info["prompt_sha256"],
            "gold_accessed": False,
        }
        predictions.append(pred_record)

        if idx % 10 == 0 or idx == len(target_sids):
            status_str = predicted_verdict if is_valid else "[UNPARSED/TRUNCATED]"
            sys.stderr.write(f"[verification.predict] [{idx}/{len(target_sids)}] {sid} -> {status_str} ({latency:.2f}s)\n")
            sys.stderr.flush()

    elapsed = time.perf_counter() - started_total
    backbone.close()

    # Persist predictions.jsonl
    predictions_file = destination / "predictions.jsonl"
    dump_jsonl(predictions_file, predictions)

    # Persist manifest.json
    manifest = {
        "milestone": "M1.2.1",
        "phase": "frozen_base_stabilization",
        "experiment_id": experiment_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "synthetic": synthetic,
        "n_claims": len(predictions),
        "valid_verdict_count": valid_verdict_count,
        "unparsed_count": len(predictions) - valid_verdict_count,
        "valid_verdict_rate": valid_verdict_count / len(predictions) if predictions else 0.0,
        "generation_config": {
            "max_new_tokens": max_new_tokens,
            "enable_thinking": enable_thinking,
            "temperature": 0.0,
            "do_sample": False,
            "device": device,
            "dtype": dtype,
        },
        "elapsed_seconds": elapsed,
        "r4_evidence_path": str(r4_p),
        "r4_evidence_sha256": file_hash(r4_p),
        "claims_path": str(claims_p),
        "claims_sha256": file_hash(claims_p),
        "predictions_sha256": file_hash(predictions_file),
        "model_spec": spec,
        "gold_labels_used": False,
        "ontology": list(VERDICTS),
    }
    write_json(destination / "manifest.json", manifest)

    sys.stderr.write(
        f"[verification.predict] Completed {len(predictions)} predictions ({valid_verdict_count} valid, "
        f"{len(predictions) - valid_verdict_count} unparsed) in {elapsed:.1f}s -> {destination}\n"
    )
    sys.stderr.flush()
    return destination


def main():
    parser = argparse.ArgumentParser(description="M1.2.1 Frozen Base Verification Predictor")
    parser.add_argument("--r4-path", required=True, help="Path to R4.jsonl retrieval output")
    parser.add_argument("--claims-path", required=True, help="Path to claims.jsonl")
    parser.add_argument("--output", required=True, help="Output directory for predictions")
    parser.add_argument("--model-path", default=None, help="Local path to Qwen3-8B checkpoint")
    parser.add_argument("--revision", default=EXPECTED_QWEN3_8B_REVISION, help="Pinned model commit SHA")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument("--dtype", default="bfloat16", help="Torch data type")
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic test double")
    parser.add_argument("--max-claims", type=int, default=None, help="Limit number of claims")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max generated tokens")
    parser.add_argument("--enable-thinking", dest="enable_thinking", action="store_true", default=None, help="Enable native reasoning")
    parser.add_argument("--no-thinking", dest="enable_thinking", action="store_false", help="Disable native reasoning")
    parser.add_argument("--experiment-id", default="B0", help="Identifier for experiment (B0, B1, B2)")
    parser.add_argument("--config", default=None, help="Optional JSON config file to load settings")
    args = parser.parse_args()

    # Load from config if provided
    max_new_tokens = args.max_new_tokens
    enable_thinking = args.enable_thinking
    experiment_id = args.experiment_id

    if args.config:
        cfg_p = Path(args.config).expanduser().resolve()
        if cfg_p.is_file():
            cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
            max_new_tokens = cfg.get("max_new_tokens", max_new_tokens)
            enable_thinking = cfg.get("enable_thinking", enable_thinking)
            experiment_id = cfg.get("experiment_id", experiment_id)

    spec = None
    if not args.synthetic:
        spec = {
            "model_id": EXPECTED_QWEN3_8B_ID,
            "local_path": args.model_path,
            "revision": args.revision,
        }

    run_prediction(
        r4_path=args.r4_path,
        claims_path=args.claims_path,
        output_dir=args.output,
        spec=spec,
        synthetic=args.synthetic,
        device=args.device,
        dtype=args.dtype,
        max_claims=args.max_claims,
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
        experiment_id=experiment_id,
    )


if __name__ == "__main__":
    main()
