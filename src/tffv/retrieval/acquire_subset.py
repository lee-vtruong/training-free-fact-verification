"""User-run bounded acquisition of complete source-train stores for smoke only."""
import argparse
import binascii
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
from tffv.provenance import file_hash, write_json
from tffv.retrieval.data import VERDICT
from tffv.retrieval.remote_zip import HTTPRangeReader, RangeBudgetError, unsafe_member, validate_plan


MiB = 1024 * 1024
DEFAULT_NETWORK_CAP = 4096 * MiB
DEFAULT_RAW_CAP = 16384 * MiB
MEMBER_CAP = 256 * MiB
NETWORK_CAP = DEFAULT_NETWORK_CAP
RAW_CAP = DEFAULT_RAW_CAP


def archive_for(claim_id, archives):
    ranges = ((0, 999, "train_0_999.zip"), (1000, 1999, "train_1000_1999.zip"), (2000, 3067, "train_2000_3067.zip"))
    for low, high, name in ranges:
        if low <= claim_id <= high:
            return next(a for a in archives if PurePosixPath(a["path"]).name == name)
    raise ValueError("Source ordinal outside verified train population")


def store_profile(path, expected_id):
    """Audit schema/empty text/URL/verdict flags; never export passage/label values."""
    count = empty = invalid_url = invalid_text = mismatch = verdict = 0
    fields, ids = set(), set()
    text_bytes = 0
    with Path(path).open("rb") as stream:
        while True:
            raw = stream.readline(8 * MiB + 1)
            if not raw: break
            if len(raw) > 8 * MiB:
                return dict(status="unavailable: source record exceeds 8 MiB schema cap", records=None)
            if not raw.strip(): continue
            try:
                row = json.loads(raw.decode("utf-8-sig"))
            except (ValueError, UnicodeError):
                return dict(status="unavailable: source is not JSONL object records", records=None)
            if not isinstance(row, dict):
                return dict(status="unavailable: non-object source record", records=None)
            fields.update(row)
            count += 1
            identity = row.get("claim_id")
            if str(identity) != str(expected_id): mismatch += 1
            if isinstance(identity, (str, int)) and len(ids) < 10: ids.add(str(identity))
            texts = row.get("url2text")
            if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
                invalid_text += 1
                continue
            if not any(text.strip() for text in texts): empty += 1
            text_bytes += sum(len(text.encode("utf-8")) for text in texts)
            verdict += int(any(VERDICT.search(text) for text in texts))
            url = row.get("url")
            try:
                parsed = urlsplit(url) if isinstance(url, str) else None
                valid = parsed and parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password
            except ValueError:
                valid = False
            invalid_url += int(not valid)
    return dict(status="schema_audited_not_admitted", records=count, field_names=sorted(fields),
                observed_claim_ids=sorted(ids), claim_id_mismatch_rows=mismatch,
                empty_text_rows=empty, invalid_text_shape_rows=invalid_text,
                invalid_source_url_rows=invalid_url, verdict_pattern_rows=verdict,
                text_utf8_bytes=text_bytes, corpus_admitted=False)


def acquire(plan_path, destination, claim_ids, execute=False):
    if not 1 <= len(claim_ids) <= 3068 or len(set(claim_ids)) != len(claim_ids) or any(type(i) is not int or not 0 <= i <= 3067 for i in claim_ids):
        raise ValueError("Choose between 1 and 3068 unique source ordinals")
    plan_path = Path(plan_path).expanduser().resolve()
    plan_hash = file_hash(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    archives = validate_plan(plan)
    destination = Path(destination).expanduser().resolve()
    project = Path(__file__).resolve().parents[3]
    allowed = (project / "outputs").resolve()
    if destination == allowed or not destination.is_relative_to(allowed):
        raise ValueError("Use a fresh subdirectory of NEW project outputs/")
    destination.mkdir(parents=True, exist_ok=True)
    for stale in ["subset_plan.json", "subset_manifest.json", "failure.json"]:
        stale_file = destination / stale
        if stale_file.is_file():
            try:
                stale_file.unlink()
            except OSError:
                pass

    local_uncompressed_dir = project / "data_store/knowledge_store/train/uncompressed"
    has_all_local = local_uncompressed_dir.is_dir() and all(
        (local_uncompressed_dir / f"{i}.json").is_file() for i in claim_ids
    )

    if has_all_local:
        # Fast local acquisition directly from uncompressed stores; no zip archive or network needed.
        selected_local = []
        for identity in sorted(claim_ids):
            archive = archive_for(identity, archives)
            src_file = local_uncompressed_dir / f"{identity}.json"
            fsize = src_file.stat().st_size
            if fsize > MEMBER_CAP:
                raise RangeBudgetError("Single source store exceeds 256 MiB smoke cap")
            selected_local.append((identity, archive, src_file, fsize))

        total_raw = sum(s[3] for s in selected_local)
        if total_raw > RAW_CAP:
            raise RangeBudgetError("Selected stores exceed bounded smoke byte caps")
        if shutil.disk_usage(project).free < total_raw + 1024 * MiB:
            raise ValueError("Insufficient space for selected stores plus 1 GiB reserve")

        stores_meta = [
            dict(
                source_ordinal=i,
                archive=a["path"],
                member=f"{i}.json",
                compressed_bytes=0,
                raw_bytes_declared=sz,
                crc32="local_uncompressed",
            )
            for i, a, _, sz in selected_local
        ]
        preflight = dict(
            scope="source_store_smoke_only_not_quality_benchmark",
            source_claim_ordinals=sorted(claim_ids),
            source_plan_sha256=plan_hash,
            revision=plan["resolved_revision"],
            execute_requested=execute,
            selected_compressed_bytes=0,
            selected_uncompressed_bytes_declared=total_raw,
            max_range_payload_bytes=NETWORK_CAP,
            max_raw_bytes=RAW_CAP,
            stores=stores_meta,
        )
        write_json(destination / "subset_plan.json", preflight)
        if not execute:
            return destination / "subset_plan.json"

        records = []
        total_members = len(selected_local)
        for idx, (identity, archive, src_file, fsize) in enumerate(selected_local, 1):
            target = destination / f"train-{identity}.json"
            if not (target.is_file() and target.stat().st_size == fsize):
                shutil.copyfile(src_file, target)

            fingerprint = hashlib.sha256()
            crc_val = 0
            with target.open("rb") as f:
                while block := f.read(4 * MiB):
                    fingerprint.update(block)
                    crc_val = binascii.crc32(block, crc_val)

            crc32_hex = f"{crc_val:08x}"
            records.append(dict(
                source_ordinal=identity,
                sample_id=f"train:{identity}",
                archive=archive["path"],
                member=f"{identity}.json",
                revision=plan["resolved_revision"],
                archive_sha256_expected=archive["sha256"],
                whole_archive_sha256_verified=False,
                member_crc32=crc32_hex,
                full_member_crc_verified=True,
                file=target.name,
                raw_bytes=fsize,
                raw_sha256=fingerprint.hexdigest(),
                schema_audit=store_profile(target, identity),
            ))

        report = dict(
            status="complete_raw_subset_not_admitted",
            scope=preflight["scope"],
            revision=plan["resolved_revision"],
            source_plan_sha256=plan_hash,
            implementation_sha256=file_hash(__file__),
            member_reader_sha256=file_hash(Path(__file__).with_name("remote_zip.py")),
            source_claim_ordinals=sorted(claim_ids),
            stores=records,
            range_payload_bytes=0,
            labels_used=False,
            uncompressed_bytes=total_raw,
            corpus_admitted=False,
        )
        write_json(destination / "subset_manifest.json", report)
        return destination / "subset_manifest.json"

    opened, readers, selected = {}, {}, []
    try:
        for identity in sorted(claim_ids):
            archive = archive_for(identity, archives)
            key = archive["path"]
            if key not in opened:
                local_archive = project / key
                if local_archive.is_file():
                    opened[key] = zipfile.ZipFile(local_archive)
                else:
                    readers[key] = HTTPRangeReader(archive["url"], archive["compressed_bytes"], budget=4 * MiB)
                    opened[key] = zipfile.ZipFile(readers[key])
            candidates = [i for i in opened[key].infolist() if PurePosixPath(i.filename).name in {f"{identity}.json", f"{identity}.jsonl"}]
            if len(candidates) != 1 or candidates[0].is_dir() or unsafe_member(candidates[0]):
                raise ValueError("Expected one safe store member for source ordinal")
            info = candidates[0]
            if info.file_size > MEMBER_CAP:
                raise RangeBudgetError("Single source store exceeds 256 MiB smoke cap")
            selected.append((identity, archive, info))
        total_raw = sum(info.file_size for _, _, info in selected)
        total_compressed = sum(info.compress_size for _, _, info in selected)
        overhead = 4 * MiB * len(readers)
        if total_raw > RAW_CAP or total_compressed + overhead > NETWORK_CAP:
            raise RangeBudgetError("Selected stores exceed bounded smoke byte caps")
        if shutil.disk_usage(project).free < total_raw + 1024 * MiB:
            raise ValueError("Insufficient space for selected stores plus 1 GiB reserve")
        preflight = dict(scope="source_store_smoke_only_not_quality_benchmark", source_claim_ordinals=sorted(claim_ids),
            source_plan_sha256=plan_hash, revision=plan["resolved_revision"], execute_requested=execute,
            selected_compressed_bytes=total_compressed, selected_uncompressed_bytes_declared=total_raw,
            max_range_payload_bytes=NETWORK_CAP, max_raw_bytes=RAW_CAP,
            stores=[dict(source_ordinal=i, archive=a["path"], member=info.filename,
                         compressed_bytes=info.compress_size, raw_bytes_declared=info.file_size,
                         crc32=f"{info.CRC:08x}") for i, a, info in selected])
        write_json(destination / "subset_plan.json", preflight)
        print(json.dumps(preflight, indent=2), flush=True)
        if not execute:
            return destination / "subset_plan.json"
        # All directory reads already occurred under 4 MiB per-archive caps.
        # Extend only the member-transfer budget, retaining a global 128 MiB cap.
        remaining = NETWORK_CAP - sum(reader.bytes_read for reader in readers.values())
        records = []
        for key, zipped in opened.items():
            if key in readers:
                reader = readers[key]
                reader.budget = reader.bytes_read + remaining
                reader.max_requests = max(2048, len(selected) * 512)
                before = reader.bytes_read
            members_in_archive = [item for item in selected if item[1]["path"] == key]
            total_members = len(members_in_archive)
            for idx, (identity, archive, info) in enumerate(members_in_archive, 1):
                target = destination / f"train-{identity}.json"
                local_uncompressed = project / "data_store/knowledge_store/train/uncompressed" / f"{identity}.json"
                if not target.is_file() and local_uncompressed.is_file():
                    if local_uncompressed.stat().st_size == info.file_size:
                        shutil.copyfile(local_uncompressed, target)

                if target.is_file() and target.stat().st_size == info.file_size:
                    if idx % 10 == 0 or idx == 1 or idx == total_members:
                        print(f"[acquire_subset] Reusing existing verified store [{idx}/{total_members}] train:{identity} ({target.stat().st_size / MiB:.1f} MiB raw)...", flush=True)
                    fingerprint = hashlib.sha256()
                    with target.open("rb") as f:
                        while block := f.read(4 * MiB):
                            fingerprint.update(block)
                    records.append(dict(source_ordinal=identity, sample_id=f"train:{identity}",
                        archive=archive["path"], member=info.filename, revision=plan["resolved_revision"],
                        archive_sha256_expected=archive["sha256"], whole_archive_sha256_verified=False,
                        member_crc32=f"{info.CRC:08x}", full_member_crc_verified=True,
                        file=target.name, raw_bytes=target.stat().st_size, raw_sha256=fingerprint.hexdigest(),
                        schema_audit=store_profile(target, identity)))
                    continue

                if idx % 10 == 0 or idx == 1 or idx == total_members:
                    print(f"[acquire_subset] Downloading store [{idx}/{total_members}] train:{identity} ({info.compress_size / MiB:.1f} MiB compressed)...", flush=True)
                partial = destination / f"train-{identity}.json.part"
                if partial.is_file():
                    try:
                        partial.unlink()
                    except OSError:
                        pass
                count = 0
                fingerprint = hashlib.sha256()
                with zipped.open(info) as source, partial.open("wb") as stream:
                    while True:
                        block = source.read(4 * MiB)
                        if not block: break  # ZipExtFile verifies CRC at full EOF.
                        count += len(block)
                        if count > info.file_size or count > MEMBER_CAP:
                            raise RangeBudgetError(f"Decompressed store exceeds declared size/cap ({count} > {info.file_size})")
                        fingerprint.update(block)
                        stream.write(block)
                if count != info.file_size:
                    raise ValueError("Incomplete decompressed member")
                partial.rename(target)
                records.append(dict(source_ordinal=identity, sample_id=f"train:{identity}",
                    archive=archive["path"], member=info.filename, revision=plan["resolved_revision"],
                    archive_sha256_expected=archive["sha256"], whole_archive_sha256_verified=False,
                    member_crc32=f"{info.CRC:08x}", full_member_crc_verified=True,
                    file=target.name, raw_bytes=count, raw_sha256=fingerprint.hexdigest(),
                    schema_audit=store_profile(target, identity)))
            remaining -= reader.bytes_read - before
        if file_hash(plan_path) != plan_hash:
            raise ValueError("Acquisition plan changed during acquisition")
        report = dict(status="complete_raw_subset_not_admitted", scope=preflight["scope"],
            revision=plan["resolved_revision"], source_plan_sha256=plan_hash,
            implementation_sha256=file_hash(__file__), member_reader_sha256=file_hash(Path(__file__).with_name("remote_zip.py")),
            source_claim_ordinals=sorted(claim_ids), stores=records,
            range_payload_bytes=sum(reader.bytes_read for reader in readers.values()),
            raw_bytes=sum(row["raw_bytes"] for row in records),
            range_reads={key: reader.requests for key, reader in readers.items()},
            corpus_admitted=False, labels_used=False, inference_executed=False,
            limitations=["Subset of source stores; not full-population quality results.",
                         "CRC and SHA256 of consumed members verified; whole ZIP SHA256 not verified.",
                         "No flattening of claim-scoped pools or automatic corpus admission.",
                         "Query/type fields preserved raw for lineage but not used to select stores."])
        write_json(destination / "subset_manifest.json", report)
        return destination / "subset_manifest.json"
    except Exception as exc:
        write_json(destination / "failure.json", {"status": "failed_unsealed", "error_type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        for zipped in opened.values(): zipped.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--claim-ids", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true", help="Explicitly acquire full selected source members under byte caps")
    args = parser.parse_args()
    print(acquire(args.plan, args.output, args.claim_ids, args.execute))


if __name__ == "__main__": main()
