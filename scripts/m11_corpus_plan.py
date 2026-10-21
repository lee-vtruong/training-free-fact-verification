"""Fetch a small public metadata JSON; NEVER download or extract corpus archives."""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


REPOSITORY = "chenxwh/AVeriTeC"
ARCHIVES = (
    "data_store/knowledge_store/train/train_0_999.zip",
    "data_store/knowledge_store/train/train_1000_1999.zip",
    "data_store/knowledge_store/train/train_2000_3067.zip",
)
MAX_METADATA_BYTES = 4 * 1024 * 1024


def fetch_metadata(revision="main"):
    if revision != "main" and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use main for metadata resolution or a full commit SHA")
    url = f"https://huggingface.co/api/models/{REPOSITORY}/revision/{quote(revision, safe='')}?blobs=true"
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "tffv-m11-metadata-only/0.1"})
    with urlopen(request, timeout=30) as response:
        if response.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            raise ValueError("Expected metadata JSON; refusing any archive/file response")
        data = response.read(MAX_METADATA_BYTES + 1)
    if len(data) > MAX_METADATA_BYTES:
        raise ValueError("Metadata response exceeds 4 MiB cap")
    return json.loads(data.decode("utf-8")), data, url


def build_plan(metadata, free_bytes, requested_revision="main"):
    revision = metadata.get("sha", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Metadata does not provide a full immutable repository revision")
    if metadata.get("id") != REPOSITORY:
        raise ValueError("Unexpected repository identity")
    if requested_revision != "main" and revision != requested_revision:
        raise ValueError("Requested/resolved revision mismatch")
    by_path = {}
    for row in metadata.get("siblings", []):
        name = row.get("rfilename")
        if name not in ARCHIVES:
            continue
        if name in by_path:
            raise ValueError("Duplicate archive metadata")
        lfs = row.get("lfs") or {}
        checksum = lfs.get("sha256", "")
        size = row.get("size", lfs.get("size"))
        if not re.fullmatch(r"[0-9a-f]{64}", checksum) or type(size) is not int or size < 1:
            raise ValueError("Exact archive size and LFS SHA256 required; Git blob IDs are insufficient")
        if lfs.get("size") != size:
            raise ValueError("Archive size metadata mismatch")
        by_path[name] = dict(path=name, compressed_bytes=size, sha256=checksum,
            url=f"https://huggingface.co/{REPOSITORY}/resolve/{revision}/{name}",
            uncompressed_bytes=None, member_count=None)
    if set(by_path) != set(ARCHIVES):
        raise ValueError("Expected train archive set changed or metadata is incomplete")
    archives = [by_path[name] for name in ARCHIVES]
    total = sum(item["compressed_bytes"] for item in archives)
    return dict(schema_version=1, repo_id=REPOSITORY, repo_type="model", split="train",
        requested_revision=requested_revision, resolved_revision=revision, archives=archives,
        total_compressed_bytes=total, total_compressed_GiB=total / 2**30,
        destination_filesystem_free_bytes=free_bytes,
        compressed_only_fits=free_bytes >= total,
        archive_extraction_bytes=None, passage_cache_bytes=None, vector_cache_bytes=None,
        ready_for_bulk_download=False, download_executed=False,
        corpus_admitted=False,
        blockers=["ZIP member sizes and schema not inspected; extracted footprint unknown.",
                  "Local independent-corpus acquisition, filtering and claim-store mapping pending.",
                  "Need headroom for passage/vector artifacts; compressed-only fit is insufficient."],
        next_action="Review this metadata plan, then design bounded archive inspection/acquisition; no automatic downloads.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--metadata-json", help="Optional previously saved public API response; offline mode")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    destination = Path(args.output).expanduser().resolve()
    allowed = (project / "outputs").resolve()
    if destination == allowed or not destination.is_relative_to(allowed):
        parser.error("Use a fresh subdirectory of NEW project outputs/")
    if destination.exists():
        raise FileExistsError("Output already exists; use a fresh directory")
    if args.metadata_json:
        path = Path(args.metadata_json)
        if path.stat().st_size > MAX_METADATA_BYTES:
            raise ValueError("Metadata exceeds 4 MiB cap")
        raw = path.read_bytes()
        metadata, endpoint = json.loads(raw.decode("utf-8-sig")), None
    else:
        metadata, raw, endpoint = fetch_metadata(args.revision)
    plan = build_plan(metadata, shutil.disk_usage(project).free, args.revision)
    plan.update(metadata_sha256=hashlib.sha256(raw).hexdigest(), metadata_endpoint=endpoint,
                planner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "hub_metadata.json").write_bytes(raw)
    (destination / "acquisition_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(destination / "acquisition_plan.json")


if __name__ == "__main__": main()
