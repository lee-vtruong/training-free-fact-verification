"""Offline full-record audit of already acquired source stores; no retrieval."""
import argparse
import binascii
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
from tffv.provenance import file_hash, write_json
from tffv.retrieval.data import VERDICT


MiB = 1024 * 1024
DOMAIN_MARKERS = ("politifact.com", "snopes.com", "fullfact.org", "factcheck.org", "africacheck.org", "leadstories.com")


def audit_store(path, identity, record_limit, review_limit=20):
    if not 1 <= record_limit <= 64 * MiB:
        raise ValueError("Audit record cap must be at most 64 MiB")
    counters, domains, fields = Counter(), Counter(), set()
    fingerprints, review, largest = set(), [], []
    raw_digest, raw_crc, raw_bytes = hashlib.sha256(), 0, 0
    passage_estimate = 0
    with Path(path).open("rb") as stream:
        line_number = 0
        while True:
            raw = stream.readline(record_limit + 1)
            if not raw: break
            line_number += 1
            if len(raw) > record_limit:
                raise ValueError("Record exceeds explicit audit cap; no truncation or row skipping")
            raw_digest.update(raw)
            raw_crc = binascii.crc32(raw, raw_crc)
            raw_bytes += len(raw)
            if not raw.strip(): continue
            row = json.loads(raw.decode("utf-8-sig"))
            if not isinstance(row, dict):
                raise ValueError("Expected JSONL object records")
            counters["records"] += 1
            fields.update(row)
            counters["claim_id_mismatch_rows"] += int(str(row.get("claim_id")) != str(identity))
            counters["annotation_field_rows"] += int(bool(set(row) & {"label", "gold_label", "justification", "questions", "answers"}))
            url = row.get("url")
            try:
                parsed = urlsplit(url) if isinstance(url, str) else None
                valid_url = bool(parsed and parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password)
            except ValueError:
                parsed, valid_url = None, False
            counters["invalid_source_url_rows"] += int(not valid_url)
            if valid_url: domains[parsed.hostname] += 1
            texts = row.get("url2text")
            if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
                counters["invalid_text_shape_rows"] += 1
                continue
            nonempty = any(text.strip() for text in texts)
            counters["empty_text_rows"] += int(not nonempty)
            counters["nonempty_text_rows"] += int(nonempty)
            words = sum(sum(1 for _ in re.finditer(r"\S+", text)) for text in texts)
            if words:
                passage_estimate += 1 + max(0, math.ceil((words - 180) / 150))
            text_bytes = sum(len(text.encode("utf-8")) for text in texts)
            counters["text_utf8_bytes"] += text_bytes
            largest.append(dict(line_number=line_number, raw_record_bytes=len(raw), text_utf8_bytes=text_bytes, words=words))
            largest = sorted(largest, key=lambda r: (-r["raw_record_bytes"], r["line_number"]))[:5]
            # Streaming digest avoids building another full copy of long documents.
            text_digest = hashlib.sha256()
            for text in texts:
                text_digest.update(text.encode("utf-8"))
                text_digest.update(b"\n")
            if nonempty:
                counters["exact_text_duplicate_rows"] += int(text_digest.hexdigest() in fingerprints)
                fingerprints.add(text_digest.hexdigest())
            match = next(((index, found) for index, text in enumerate(texts) if (found := VERDICT.search(text))), None)
            source_flag = isinstance(url, str) and any(marker in url.casefold() for marker in DOMAIN_MARKERS)
            counters["verdict_pattern_rows"] += int(match is not None)
            counters["fact_check_source_url_rows"] += int(source_flag)
            flags = (["verdict_pattern"] if match else []) + (["fact_check_source_url"] if source_flag else [])
            counters["review_flagged_rows"] += int(bool(flags))
            if flags and len(review) < review_limit:
                if match:
                    index, found = match
                    snippet = texts[index][max(0, found.start() - 100):found.end() + 150]
                else:
                    snippet = next((text[:250] for text in texts if text.strip()), "")
                review.append(dict(sample_id=f"train:{identity}", line_number=line_number,
                    raw_record_sha256=hashlib.sha256(raw).hexdigest(), source_url=url if valid_url else None,
                    flags=flags, bounded_source_excerpt=snippet,
                    assessment="Observed heuristic flag; human review needed, not confirmed answer leakage"))
    keys = ("records", "claim_id_mismatch_rows", "annotation_field_rows", "invalid_source_url_rows", "invalid_text_shape_rows",
            "empty_text_rows", "nonempty_text_rows", "text_utf8_bytes", "exact_text_duplicate_rows", "verdict_pattern_rows",
            "fact_check_source_url_rows", "review_flagged_rows")
    result = {key: counters[key] for key in keys}
    result.update(sample_id=f"train:{identity}", raw_bytes=raw_bytes, raw_sha256=raw_digest.hexdigest(),
        raw_crc32=f"{raw_crc & 0xffffffff:08x}", field_names=sorted(fields), largest_records=largest,
        estimated_passages_180_words_overlap30_before_filtering=passage_estimate,
        unique_domains=len(domains), top_domains=domains.most_common(10),
        review_entries_exported=len(review), review_queue_truncated=counters["review_flagged_rows"] > len(review),
        corpus_admitted=False, full_record_audit=True)
    return result, review


def audit(manifest_path, output, max_record_mib=64):
    if type(max_record_mib) is not int or not 1 <= max_record_mib <= 64:
        raise ValueError("Explicit record limit must be 1-64 MiB")
    manifest_path = Path(manifest_path).expanduser().resolve()
    source_hash = file_hash(manifest_path)
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete_raw_subset_not_admitted" or metadata.get("labels_used") is not False:
        raise ValueError("Require a complete label-free source-subset acquisition")
    stores = metadata.get("stores", [])
    if not 1 <= len(stores) <= 3068 or len({r["source_ordinal"] for r in stores}) != len(stores):
        raise ValueError("Invalid source-store inventory")
    for row in stores:
        name = row["file"]
        path = manifest_path.parent / name
        if Path(name).name != name or not path.resolve().is_relative_to(manifest_path.parent) or row.get("full_member_crc_verified") is not True:
            raise ValueError("Invalid source path or incomplete CRC provenance")
        if not 0 <= row["source_ordinal"] <= 3067 or path.stat().st_size != row["raw_bytes"] or file_hash(path) != row["raw_sha256"]:
            raise ValueError("Raw store identity/hash mismatch")
    destination = Path(output).expanduser().resolve()
    allowed = (Path(__file__).resolve().parents[3] / "outputs").resolve()
    if destination == allowed or not destination.is_relative_to(allowed):
        raise ValueError("Output must be a fresh NEW project outputs/ subdirectory")
    destination.mkdir(parents=True, exist_ok=False)
    results, review = [], []
    started = time.perf_counter()
    try:
        for row in stores:
            result, queue = audit_store(manifest_path.parent / row["file"], row["source_ordinal"], max_record_mib * MiB)
            if result["raw_sha256"] != row["raw_sha256"] or result["raw_crc32"] != row["member_crc32"]:
                raise ValueError("Source hash/CRC changed during audit")
            results.append(result)
            review.extend(queue)
        if file_hash(manifest_path) != source_hash:
            raise ValueError("Acquisition manifest changed during audit")
        write_json(destination / "review_queue.json", review)
        write_json(destination / "audit_summary.json", dict(status="complete_audit_not_admitted", stores=results,
            source_manifest_sha256=source_hash, source_manifest_path=str(manifest_path),
            audit_implementation_sha256=file_hash(__file__), flag_policy_sha256=file_hash(Path(__file__).with_name("data.py")),
            configuration=dict(max_record_mib=max_record_mib, review_entries_per_store=20,
                               text_selection="none", record_truncation=False),
            elapsed_seconds=time.perf_counter() - started,
            corpus_admitted=False, network_accessed=False, inference_executed=False,
            review_queue_sha256=file_hash(destination / "review_queue.json"),
            limitations=["Encoded record cap is not an RSS memory measurement.",
                         "Flags/snippets need source review; no verdict or gold label inferred.",
                         "Passage count is a workload estimate before filtering/deduplication.",
                         "Audit changes neither raw data nor model tokenizer/context settings."]))
    except Exception as exc:
        write_json(destination / "failure.json", dict(status="failed_unsealed", error_type=type(exc).__name__, message=str(exc)))
        raise
    return destination / "audit_summary.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-record-mib", required=True, type=int)
    args = parser.parse_args()
    print(audit(args.subset_manifest, args.output, args.max_record_mib))


if __name__ == "__main__": main()
