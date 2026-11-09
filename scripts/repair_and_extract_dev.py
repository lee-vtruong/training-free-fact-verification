import os
import sys
import time
import zipfile
import struct
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEST_DIR = Path('/home/stackops/whale/training-free-fact-verification/data_store/knowledge_store/dev')
ZIP_FILE = DEST_DIR / 'dev_knowledge_store.zip'
UNCOMPRESSED_DIR = DEST_DIR / 'uncompressed'
URL = 'https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data_store/knowledge_store/dev_knowledge_store.zip'

def fetch_and_repair_member(m, info):
    offset = info.header_offset
    # Read local header to determine extra_len
    req_hdr = urllib.request.Request(URL, headers={'Range': f'bytes={offset}-{offset + 100}'})
    with urllib.request.urlopen(req_hdr, timeout=60) as resp:
        hdr = resp.read()
    sig, ver, flag, method, mtime, mdate, crc, csize, usize, fn_len, extra_len = struct.unpack('<IHHHHHIIIHH', hdr[:30])
    total_entry_len = 30 + fn_len + extra_len + info.compress_size
    
    # Fetch exact full entry
    req_full = urllib.request.Request(URL, headers={'Range': f'bytes={offset}-{offset + total_entry_len - 1}'})
    with urllib.request.urlopen(req_full, timeout=120) as resp:
        entry_bytes = resp.read()
    
    with open(ZIP_FILE, 'r+b') as fp:
        fp.seek(offset)
        fp.write(entry_bytes)
    return m, len(entry_bytes)

def main():
    UNCOMPRESSED_DIR.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(ZIP_FILE, 'r')
    members = [m for m in zf.namelist() if m.endswith('.json')]
    print(f'[M1.9g] Total dev members in zip: {len(members)}')

    to_repair = []
    t0 = time.time()
    
    # Pass 1: Extract already valid files
    for m in members:
        basename = Path(m).name
        target_path = UNCOMPRESSED_DIR / basename
        if target_path.exists() and target_path.stat().st_size > 1000:
            continue
        try:
            data = zf.read(m)
            target_path.write_bytes(data)
        except Exception:
            info = zf.getinfo(m)
            to_repair.append((m, info))

    existing_count = len(list(UNCOMPRESSED_DIR.glob('*.json')))
    print(f'[M1.9g] Files ready: {existing_count}/500 | Need range repair: {len(to_repair)}')

    if to_repair:
        print(f'[M1.9g] Starting parallel range repair for {len(to_repair)} members (16 workers)...')
        repaired_count = 0
        with ThreadPoolExecutor(max_workers=16) as executor:
            future_to_member = {executor.submit(fetch_and_repair_member, m, info): m for m, info in to_repair}
            for fut in as_completed(future_to_member):
                m, nbytes = fut.result()
                repaired_count += 1
                if repaired_count % 20 == 0 or repaired_count == len(to_repair):
                    print(f'[M1.9g] Repaired [{repaired_count}/{len(to_repair)}] members via HTTP range ({time.time() - t0:.1f}s)...', flush=True)

        # Pass 2: Extract repaired members
        print('[M1.9g] Extracting repaired members into uncompressed directory...')
        zf_reloaded = zipfile.ZipFile(ZIP_FILE, 'r')
        for m, _ in to_repair:
            data = zf_reloaded.read(m)
            basename = Path(m).name
            (UNCOMPRESSED_DIR / basename).write_bytes(data)

    final_count = len(list(UNCOMPRESSED_DIR.glob('*.json')))
    print(f'[M1.9g] Total uncompressed dev files: {final_count}/500 in {time.time() - t0:.1f}s.')
    assert final_count == 500, f'Expected 500 JSON files, got {final_count}!'
    print('[M1.9g] SUCCESS: All 500 Official Dev knowledge store files verified and extracted!')

if __name__ == '__main__':
    main()
