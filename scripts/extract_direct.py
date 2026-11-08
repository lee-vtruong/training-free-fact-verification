import os
import sys
import zlib
import zipfile
import struct
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEST_DIR = Path('/home/stackops/whale/training-free-fact-verification/data_store/knowledge_store/dev')
ZIP_FILE = DEST_DIR / 'dev_knowledge_store.zip'
UNCOMPRESSED_DIR = DEST_DIR / 'uncompressed'
URL = 'https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data_store/knowledge_store/dev_knowledge_store.zip'

def download_and_extract_member(info):
    offset = info.header_offset
    # Step 1: Read local header to get filename and extra length
    req_hdr = urllib.request.Request(URL, headers={'Range': f'bytes={offset}-{offset + 100}'})
    with urllib.request.urlopen(req_hdr, timeout=60) as resp:
        hdr = resp.read()
    sig, ver, flag, method, mtime, mdate, crc, csize, usize, fn_len, extra_len = struct.unpack('<IHHHHHIIIHH', hdr[:30])
    
    data_offset = offset + 30 + fn_len + extra_len
    data_end = data_offset + info.compress_size - 1
    
    # Step 2: Download the compressed data directly
    req_data = urllib.request.Request(URL, headers={'Range': f'bytes={data_offset}-{data_end}'})
    with urllib.request.urlopen(req_data, timeout=120) as resp:
        compressed_bytes = resp.read()
        
    assert len(compressed_bytes) == info.compress_size, f"Size mismatch: expected {info.compress_size}, got {len(compressed_bytes)}"
    
    # Step 3: Decompress
    if method == 8: # Deflated
        decompressed_data = zlib.decompress(compressed_bytes, -15)
    elif method == 0: # Stored
        decompressed_data = compressed_bytes
    else:
        raise ValueError(f"Unsupported compression method: {method}")
        
    # Step 4: Verify CRC
    computed_crc = zlib.crc32(decompressed_data) & 0xffffffff
    assert computed_crc == info.CRC, f"CRC mismatch: expected {info.CRC}, got {computed_crc}"
    
    # Step 5: Save directly to destination
    basename = Path(info.filename).name
    out_path = UNCOMPRESSED_DIR / basename
    out_path.write_bytes(decompressed_data)
    return basename, len(decompressed_data)

def main():
    UNCOMPRESSED_DIR.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(ZIP_FILE, 'r')
    members = {Path(m).name: zf.getinfo(m) for m in zf.namelist() if m.endswith('.json')}
    print(f"Total members in zip: {len(members)}")
    
    # Check what is currently missing or corrupted (< 100 bytes)
    missing = []
    for i in range(500):
        name = f"{i}.json"
        p = UNCOMPRESSED_DIR / name
        if not p.exists() or p.stat().st_size < 100:
            if name in members:
                missing.append(members[name])
            else:
                print(f"WARNING: {name} not found in zip catalog!")
                
    print(f"Already extracted and valid: {500 - len(missing)}/500")
    print(f"Needs download and extraction: {len(missing)}")
    
    if not missing:
        print("All 500 files are present and valid!")
        return
        
    print(f"Starting direct range download & extraction with 16 workers...")
    done = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_and_extract_member, info): info.filename for info in missing}
        for fut in as_completed(futures):
            fn = futures[fut]
            try:
                bname, sz = fut.result()
                done += 1
                if done % 10 == 0 or done == len(missing):
                    print(f"Extracted [{done}/{len(missing)}] {bname} ({sz:,} bytes)...", flush=True)
            except Exception as e:
                print(f"FAILED on {fn}: {e}", flush=True)
                raise
                
    print("Done! Validating all 500 files...")
    count = len(list(UNCOMPRESSED_DIR.glob('*.json')))
    print(f"Total files in uncompressed: {count}/500")
    assert count == 500, f"Expected 500, found {count}"

if __name__ == '__main__':
    main()
