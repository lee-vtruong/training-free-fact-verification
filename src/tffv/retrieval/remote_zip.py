"""Bounded HTTP Range inspection of pinned train archives, without extraction."""
import argparse
import hashlib
import io
import json
import re
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
from tffv.provenance import file_hash, write_json


class RangeBudgetError(ValueError):
    pass


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url, size, budget=4 * 1024 * 1024, max_requests=64):
        if type(size) is not int or size < 1 or type(budget) is not int or budget < 1:
            raise ValueError("Positive archive size and range budget required")
        self.url, self.size, self.budget = url, size, budget
        self.max_requests, self.position = max_requests, 0
        self.bytes_read, self.requests = 0, []
        self.cache = {}

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.position, 2: self.size}.get(whence)
        if base is None or not 0 <= base + offset <= self.size:
            raise ValueError("ZIP seek outside archive")
        self.position = base + offset
        return self.position

    def read(self, size=-1):
        length = self.size - self.position if size is None or size < 0 else min(size, self.size - self.position)
        if not length: return b""
        key = (self.position, length)
        if key not in self.cache:
            if length > self.budget - self.bytes_read:
                raise RangeBudgetError(
                    f"Archive inspection byte cap reached (needed {length} bytes, "
                    f"remaining {self.budget - self.bytes_read} of {self.budget} bytes); refusing full fetch"
                )
            if len(self.requests) >= self.max_requests:
                raise RangeBudgetError(
                    f"Archive inspection request cap reached ({len(self.requests)} >= {self.max_requests}); "
                    "refusing full fetch"
                )
            start, end = self.position, self.position + length - 1
            request = Request(self.url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
                                                 "User-Agent": "tffv-zip-range-inspection/0.1"})
            with urlopen(request, timeout=30) as response:
                # Check headers BEFORE reading. A server ignoring Range must not
                # turn this metadata operation into a multi-GB file download.
                if response.status != 206:
                    raise ValueError("Server did not honor Range (206 required); no full-download fallback")
                expected = f"bytes {start}-{end}/{self.size}"
                if response.headers.get("Content-Range") != expected:
                    raise ValueError("Unexpected Content-Range/archive size")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise ValueError("Encoded range responses are unsupported")
                declared = response.headers.get("Content-Length")
                if declared is not None and int(declared) != length:
                    raise ValueError("Unexpected range length")
                data = response.read(length + 1)
            if len(data) != length:
                raise ValueError("Truncated/oversized range payload")
            self.cache[key] = data
            self.bytes_read += length
            self.requests.append(dict(start=start, bytes=length, sha256=hashlib.sha256(data).hexdigest()))
        result = self.cache[key]
        self.position += length
        return result


def shape(value, depth=3):
    if isinstance(value, dict) and depth:
        return {key: shape(item, depth - 1) for key, item in sorted(value.items())}
    if isinstance(value, list) and depth:
        return dict(type="list", first_item_shape=shape(value[0], depth - 1) if value else None)
    return type(value).__name__


def unsafe_member(info):
    name = info.orig_filename.replace("\\", "/")
    return ("\0" in name or name.startswith("/") or ".." in PurePosixPath(name).parts
            or bool(re.match(r"^[A-Za-z]:", name)) or stat.S_ISLNK(info.external_attr >> 16)
            or bool(info.flag_bits & 1))


def inspect_archive(archive, budget=4 * 1024 * 1024, sample=False):
    reader = HTTPRangeReader(archive["url"], archive["compressed_bytes"], budget)
    with zipfile.ZipFile(reader) as opened:
        infos = opened.infolist()
        if len(infos) > 20000:
            raise RangeBudgetError("Too many ZIP members for bounded inspection")
        members = [dict(name=i.filename, bytes=i.file_size, compressed_bytes=i.compress_size,
                        crc32=f"{i.CRC:08x}", compression_method=i.compress_type,
                        is_directory=i.is_dir(), suspicious_path_or_link_or_encryption=bool(unsafe_member(i))) for i in infos]
        sample_result = None
        if sample:
            candidates = [i for i in infos if not i.is_dir() and not unsafe_member(i)
                          and PurePosixPath(i.filename).suffix.lower() in {".json", ".jsonl"}]
            if candidates:
                target = sorted(candidates, key=lambda i: i.filename)[0]
                sample_result = dict(member=target.filename, full_member_crc_verified=False, content_admitted=False)
                try:
                    with opened.open(target) as stream:
                        line = stream.readline(256 * 1024)
                    if len(line) >= 256 * 1024:
                        sample_result["status"] = "record prefix exceeds schema cap; not parsed"
                    else:
                        value = json.loads(line.decode("utf-8-sig"))
                        sample_result.update(status="first_line_shape_only", record_shape=shape(value),
                            first_line_sha256=hashlib.sha256(line).hexdigest(), first_line_bytes=len(line))
                except RangeBudgetError:
                    sample_result["status"] = "range cap reached while sampling; directory remains available"
                except (ValueError, UnicodeError, NotImplementedError, zipfile.BadZipFile, RuntimeError):
                    sample_result["status"] = "first-line schema unavailable; no primitive values exported"
            else:
                sample_result = {"status": "no eligible JSON/JSONL member"}
    files = [i for i in members if not i["is_directory"]]
    return dict(path=archive["path"], expected_archive_sha256=archive["sha256"],
        whole_archive_sha256_verified=False, metadata_source="pinned remote ZIP central directory",
        member_count=len(members), file_count=len(files),
        uncompressed_bytes_declared=sum(i["bytes"] for i in files),
        compressed_member_bytes_declared=sum(i["compressed_bytes"] for i in files),
        suspicious_member_count=sum(i["suspicious_path_or_link_or_encryption"] for i in members),
        members=members, schema_sample=sample_result, range_payload_bytes=reader.bytes_read,
        range_reads=reader.requests, archive_extracted=False, corpus_admitted=False)


def validate_plan(plan):
    revision = plan.get("resolved_revision", "")
    if plan.get("repo_id") != "chenxwh/AVeriTeC" or plan.get("split") != "train" or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Expected pinned AVeriTeC train acquisition plan")
    expected = {"train_0_999.zip", "train_1000_1999.zip", "train_2000_3067.zip"}
    archives = plan.get("archives", [])
    if len(archives) != 3 or {PurePosixPath(a["path"]).name for a in archives} != expected:
        raise ValueError("Unexpected train archive set")
    for archive in archives:
        prefix = "data_store/knowledge_store/train/"
        url = f"https://huggingface.co/chenxwh/AVeriTeC/resolve/{revision}/{archive['path']}"
        if archive["path"] != prefix + PurePosixPath(archive["path"]).name or archive.get("url") != url:
            raise ValueError("Archive path/URL must match pinned official repository")
        if not re.fullmatch(r"[0-9a-f]{64}", archive.get("sha256", "")) or type(archive.get("compressed_bytes")) is not int or archive["compressed_bytes"] < 1:
            raise ValueError("Invalid archive size/hash")
    return archives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-first-record", action="store_true", help="Read at most one 256 KiB source record prefix per archive")
    args = parser.parse_args()
    source = Path(args.plan).expanduser().resolve()
    input_hash = file_hash(source)
    plan = json.loads(source.read_text(encoding="utf-8"))
    archives = validate_plan(plan)
    project = Path(__file__).resolve().parents[3]
    destination = Path(args.output).expanduser().resolve()
    allowed = (project / "outputs").resolve()
    if destination == allowed or not destination.is_relative_to(allowed):
        parser.error("Use a fresh subdirectory of NEW project outputs/")
    destination.mkdir(parents=True, exist_ok=False)
    summaries = []
    try:
        for archive in archives:
            report = inspect_archive(archive, sample=args.sample_first_record)
            name = PurePosixPath(archive["path"]).name + ".inventory.json"
            write_json(destination / name, report)
            summaries.append({k: v for k, v in report.items() if k not in {"members", "range_reads"}})
        if file_hash(source) != input_hash:
            raise ValueError("Acquisition plan changed during inspection")
        declared = sum(r["uncompressed_bytes_declared"] for r in summaries)
        compressed = sum(a["compressed_bytes"] for a in archives)
        free = shutil.disk_usage(project).free
        write_json(destination / "zip_summary.json", dict(status="complete", plan_sha256=input_hash,
            implementation_sha256=file_hash(__file__), archives=summaries,
            total_compressed_bytes=compressed, total_uncompressed_bytes_declared=declared,
            total_uncompressed_GiB_declared=declared / 2**30,
            compressed_plus_uncompressed_declared_bytes=compressed + declared,
            filesystem_free_bytes=free,
            archives_plus_declared_extraction_fit_without_headroom=compressed + declared <= free,
            range_payload_bytes=sum(r["range_payload_bytes"] for r in summaries),
            max_range_payload_bytes=3 * 4 * 1024 * 1024,
            vector_cache_bytes=None, extraction_executed=False, ready_for_bulk_download=False,
            limitations=["Declared ZIP sizes are metadata, not allocated disk usage.",
                         "Partial inspection does not verify whole-archive SHA256 or full-member CRC.",
                         "Schema samples contain shapes only; corpus lineage/leakage review remains pending.",
                         "HTTP headers/TLS/redirect overhead are excluded from payload-byte accounting."]))
    except Exception as exc:
        write_json(destination / "failure.json", {"status": "failed_unsealed", "error_type": type(exc).__name__, "message": str(exc)})
        raise
    print(destination / "zip_summary.json")


if __name__ == "__main__": main()
