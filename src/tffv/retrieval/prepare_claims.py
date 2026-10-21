"""Project a verified source-train JSON array into label-free retrieval claims."""
import argparse
import json
import re
from pathlib import Path
from tffv.provenance import digest, file_hash, write_json


def load_verified_source(train_json, expected_sha256, expected_records):
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or type(expected_records) is not int or expected_records < 1:
        raise ValueError("Verified SHA256 and positive source record count required")
    source = Path(train_json).expanduser().resolve()
    if file_hash(source) != expected_sha256:
        raise ValueError("Source-train content hash mismatch")
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list) or len(payload) != expected_records:
        raise ValueError("Source-train array/count mismatch")
    return source, payload


def invalid_claim_reason(row):
    if not isinstance(row, dict): return "row_not_object"
    if "claim" not in row: return "missing_claim"
    if not isinstance(row["claim"], str): return "claim_not_string"
    if not row["claim"].strip(): return "empty_claim"
    return None


def inspect_source_claims(train_json, expected_sha256, expected_records):
    """Report schema failures without reading annotation values or exporting rows."""
    source, payload = load_verified_source(train_json, expected_sha256, expected_records)
    invalid, counts = [], {}
    for index, row in enumerate(payload):
        reason = invalid_claim_reason(row)
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
            invalid.append(dict(sample_id=f"train:{index}", reason=reason,
                                claim_type=type(row.get("claim")).__name__ if isinstance(row, dict) else None))
    if file_hash(source) != expected_sha256:
        raise ValueError("Source changed during inspection")
    return dict(source_sha256=expected_sha256, records=len(payload), invalid_records=len(invalid),
                reason_counts=counts, invalid_rows=invalid, annotation_values_used=False,
                rows_skipped=False, export_executed=False,
                first_three_claims=[dict(sample_id=f"train:{i}",
                    claim=row["claim"] if invalid_claim_reason(row) is None else None)
                    for i, row in enumerate(payload[:3])])


def prepare_source_claims(train_json, output, expected_sha256, expected_records):
    source, payload = load_verified_source(train_json, expected_sha256, expected_records)
    claims, mapping = [], []
    for index, row in enumerate(payload):
        reason = invalid_claim_reason(row)
        if reason:
            raise ValueError(f"Invalid source claim train:{index}: {reason}; no export. Run --inspect for all affected IDs")
        # Access only claim text. Gold/QA/justification/annotation URLs are never
        # used to filter, order, classify, select sources, or populate output.
        sid = f"train:{index}"
        claims.append(dict(sample_id=sid, dataset="AVeriTeC", split="train", claim=row["claim"]))
        mapping.append(dict(sample_id=sid, source_array_index=index, claim_sha256=digest(row["claim"])))
    if file_hash(source) != expected_sha256:
        raise ValueError("Source changed during projection")
    destination = Path(output).expanduser().resolve()
    allowed = Path(__file__).resolve().parents[3] / "outputs"
    if destination == allowed.resolve() or not destination.is_relative_to(allowed.resolve()):
        raise ValueError("Use a fresh subdirectory of NEW project outputs/")
    destination.mkdir(parents=True, exist_ok=False)
    for name, rows in (("claims.jsonl", claims), ("id_mapping.jsonl", mapping)):
        with (destination / name).open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    manifest = dict(schema_version=1, source_role="source_train", dataset="AVeriTeC", split="train",
        source_path=str(source), source_sha256=expected_sha256, records=len(claims),
        id_protocol="train:<zero-based source-array ordinal>; exact source order preserved; not a native ID field",
        split_counts={"train": len(claims)}, population_status="previously_inspected_source",
        claims_sha256=file_hash(destination / "claims.jsonl"), id_mapping_sha256=file_hash(destination / "id_mapping.jsonl"),
        converter_sha256=file_hash(__file__), annotation_values_used=False,
        excluded_fields="all input fields except claim", corpus_provided=False,
        limitations=["Claim export only; not a complete retrieval data manifest.",
                     "Caller supplies verified source train file identity; no dev/test input is projected.",
                     "No deduplication, selection, ontology remapping, qrels or evidence construction."])
    write_json(destination / "claims_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-records", required=True, type=int)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output")
    action.add_argument("--inspect", action="store_true", help="Print claim-only diagnostics; no export, filtering or repairs")
    args = parser.parse_args()
    if args.inspect:
        print(json.dumps(inspect_source_claims(args.train_json, args.expected_sha256, args.expected_records), ensure_ascii=False, indent=2))
        return
    manifest = prepare_source_claims(args.train_json, args.output, args.expected_sha256, args.expected_records)
    print(json.dumps({key: manifest[key] for key in ("records", "source_sha256", "claims_sha256", "corpus_provided")}))


if __name__ == "__main__": main()
