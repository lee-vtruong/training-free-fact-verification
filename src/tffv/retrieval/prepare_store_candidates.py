"""Offline, claim-scoped candidate export; never emits benchmark admission."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

from tffv.provenance import canonical, digest, file_hash, write_json
from tffv.retrieval.audit_subset import MiB, audit_store
from tffv.retrieval.data import CLAIM_FIELDS, require_strings


POLICY = "quarantine_review_flags_v1"


def prepare_candidates(audit_summary, claims_manifest, output, policy):
    if policy != POLICY:
        raise ValueError("Explicit supported candidate policy required")
    audit_path = Path(audit_summary).resolve()
    claims_path = Path(claims_manifest).resolve()
    audit_hash, claims_manifest_hash = file_hash(audit_path), file_hash(claims_path)
    summary = json.loads(audit_path.read_text(encoding="utf-8"))
    cm = json.loads(claims_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete_audit_not_admitted" or summary.get("corpus_admitted") is not False:
        raise ValueError("Require completed unadmitted audit")
    if (cm.get("dataset") != "AVeriTeC" or cm.get("split") != "train"
            or cm.get("source_role") != "source_train" or cm.get("annotation_values_used") is not False):
        raise ValueError("Require claim-only source-train manifest")
    # Audit manifests pin the existing acquisition; no user-supplied raw paths.
    acquisition_path = Path(summary["source_manifest_path"])
    if file_hash(acquisition_path) != summary["source_manifest_sha256"]:
        raise ValueError("Acquisition manifest hash mismatch")
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    if acquisition.get("status") != "complete_raw_subset_not_admitted" or acquisition.get("labels_used") is not False:
        raise ValueError("Require complete label-free acquisition")
    claim_file = claims_path.parent / "claims.jsonl"
    if file_hash(claim_file) != cm["claims_sha256"]:
        raise ValueError("Claim export hash mismatch")
    claims = {}
    with claim_file.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            require_strings(row, CLAIM_FIELDS)
            if row["dataset"] != "AVeriTeC" or row["split"] != "train" or row["sample_id"] in claims:
                raise ValueError("Invalid/duplicate source claim")
            claims[row["sample_id"]] = row
    if len(claims) != cm["records"]:
        raise ValueError("Claim count mismatch")
    inventory = acquisition["stores"]
    if not 1 <= len(inventory) <= 3068 or len({r["source_ordinal"] for r in inventory}) != len(inventory):
        raise ValueError("Require unique bounded store inventory")
    profiles = {row["sample_id"]: row for row in summary["stores"]}
    if len(profiles) != len(inventory):
        raise ValueError("Audit inventory mismatch")
    checked = []
    for store in inventory:
        ordinal = store["source_ordinal"]
        if type(ordinal) is not int or not 0 <= ordinal <= 3067:
            raise ValueError("Invalid source ordinal")
        sid = f"train:{ordinal}"
        if sid not in claims or sid not in profiles:
            raise ValueError("Store lacks valid claim or full audit")
        name = store["file"]
        raw_path = acquisition_path.parent / name
        if Path(name).name != name or not raw_path.resolve().is_relative_to(acquisition_path.parent.resolve()):
            raise ValueError("Unsafe raw store path")
        if store.get("full_member_crc_verified") is not True or file_hash(raw_path) != store["raw_sha256"]:
            raise ValueError("Raw store integrity mismatch")
        # Recompute flags for every record; never depend on a truncated queue.
        profile, review = audit_store(raw_path, ordinal, 64 * MiB, review_limit=profiles[sid]["records"])
        if profile != profiles[sid]:
            # review_entries/truncation may differ when the old queue was capped.
            ignored = {"review_entries_exported", "review_queue_truncated"}
            if canonical({k: v for k, v in profile.items() if k not in ignored}) != canonical({k: v for k, v in profiles[sid].items() if k not in ignored}):
                raise ValueError("Full audit changed; regenerate/review audit")
        if profile["raw_sha256"] != store["raw_sha256"] or profile["raw_crc32"] != store["member_crc32"]:
            raise ValueError("Raw hash/CRC mismatch")
        if any(profile[k] for k in ("annotation_field_rows", "claim_id_mismatch_rows", "invalid_text_shape_rows")):
            raise ValueError("Source schema/identity failures require review")
        if profile["review_queue_truncated"]:
            raise ValueError("Incomplete review inventory")
        checked.append((sid, store, raw_path, {r["line_number"]: r for r in review}))
    destination = Path(output).resolve()
    allowed = (Path(__file__).resolve().parents[3] / "outputs").resolve()
    if destination == allowed or not destination.is_relative_to(allowed):
        raise ValueError("Output must be a fresh NEW project outputs/ subdirectory")
    destination.mkdir(parents=True, exist_ok=False)
    results = []
    try:
        for sid, store, raw_path, flagged in checked:
            directory = destination / sid.replace(":", "-")
            directory.mkdir()
            counts = Counter()
            with (directory / "corpus.jsonl").open("x", encoding="utf-8", newline="\n") as corpus, \
                    (directory / "record_ledger.jsonl").open("x", encoding="utf-8", newline="\n") as ledger, \
                    raw_path.open("rb") as raw:
                for line_number, line in enumerate(iter(lambda: raw.readline(64 * MiB + 1), b""), 1):
                    if len(line) > 64 * MiB:
                        raise ValueError("Record exceeds cap; no truncation")
                    if not line.strip(): continue
                    row = json.loads(line.decode("utf-8-sig"))
                    text = "\n".join(row["url2text"])
                    url = row.get("url")
                    try:
                        parsed = urlsplit(url) if isinstance(url, str) else None
                        valid_url = bool(parsed and parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password)
                    except ValueError:
                        valid_url = False
                    flags = flagged.get(line_number, {}).get("flags", [])
                    if not valid_url:
                        status = "quarantined_invalid_url" if text.strip() else "empty_text"
                    elif flags:
                        status = "quarantined_review_flag"
                    elif text.strip():
                        status = "retained"
                    else:
                        status = "empty_text"
                    counts[status] += 1
                    identity = dict(sample_id=sid, raw_sha256=store["raw_sha256"], line_number=line_number)
                    document_id = digest(identity)
                    ledger.write(canonical(dict(**identity, document_id=document_id, status=status,
                        flags=flags, source_url=row["url"], raw_record_sha256=hashlib.sha256(line).hexdigest())) + "\n")
                    if status == "retained":
                        corpus.write(canonical(dict(document_id=document_id, text=text, source_url=row["url"],
                            source_kind="web_document", retrieved_at="unknown",
                            acquisition_id=summary["source_manifest_sha256"])) + "\n")
            if file_hash(raw_path) != store["raw_sha256"]:
                raise ValueError("Raw file changed during conversion")
            if sum(counts.values()) != profiles[sid]["records"] or not counts["retained"]:
                raise ValueError("Record conservation/retained corpus failure")
            with (directory / "claims.jsonl").open("x", encoding="utf-8", newline="\n") as selected_claim:
                selected_claim.write(canonical(claims[sid]) + "\n")
            results.append(dict(sample_id=sid, directory=directory.name, source_records=sum(counts.values()),
                counts=dict(counts), claims_sha256=file_hash(directory / "claims.jsonl"),
                corpus_sha256=file_hash(directory / "corpus.jsonl"), ledger_sha256=file_hash(directory / "record_ledger.jsonl")))
        if file_hash(audit_path) != audit_hash or file_hash(claims_path) != claims_manifest_hash or file_hash(claim_file) != cm["claims_sha256"]:
            raise ValueError("Input manifest or claims changed during conversion")
        write_json(destination / "candidate_manifest.json", dict(status="complete_candidates_not_admitted",
            policy=policy, corpus_admitted=False, inference_executed=False, network_accessed=False,
            audit_summary_sha256=audit_hash, claims_manifest_sha256=claims_manifest_hash,
            implementation_sha256=file_hash(__file__), flag_policy_sha256=file_hash(Path(__file__).with_name("data.py")),
            audit_implementation_sha256=file_hash(Path(__file__).with_name("audit_subset.py")),
            stores=results, population="three_or_fewer_source_claims_operational_only",
            limitations=["Not a benchmark data/admission manifest; not complete official-split coverage.",
                "All heuristic-flagged records quarantined, including possible false positives; residual leakage unresolved.",
                "No truncation or sentence deletion. url2text strings joined with newline; provenance in per-record ledger.",
                "Retrieval timestamps unavailable and explicitly unknown. Query/type never used.",
                "Separate claim-scoped files; no global union and no models loaded."]))
    except Exception as exc:
        write_json(destination / "failure.json", dict(status="failed_unsealed", error_type=type(exc).__name__, message=str(exc)))
        raise
    return destination / "candidate_manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-summary", required=True)
    parser.add_argument("--claims-manifest", required=True)
    parser.add_argument("--policy", choices=[POLICY], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(prepare_candidates(args.audit_summary, args.claims_manifest, args.output, args.policy))


if __name__ == "__main__": main()
