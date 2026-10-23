import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/m11-audit-pilot100-20261023-160109/audit_summary.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    keys = ["annotation_field_rows", "claim_id_mismatch_rows", "invalid_source_url_rows", "invalid_text_shape_rows"]
    found = 0
    raw_dir = Path("outputs/m11-store-pilot100-20261021-142310")
    for s in data.get("stores", []):
        bad = {k: s.get(k, 0) for k in keys if s.get(k, 0) > 0}
        if bad:
            found += 1
            sid = s.get('sample_id')
            print(f"Store {sid}: {bad}")
            ordinal = s.get("source_ordinal")
            store_file = raw_dir / f"train-{ordinal}.json"
            if store_file.exists():
                with store_file.open("rb") as sf:
                    for line_no, raw in enumerate(sf, 1):
                        row = json.loads(raw)
                        url = row.get("url")
                        try:
                            parsed = urlsplit(url) if isinstance(url, str) else None
                            valid = bool(parsed and parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password)
                        except ValueError:
                            valid = False
                        if not valid:
                            print(f"  Line {line_no}: url={repr(url)}, text_len={sum(len(t) for t in row.get('url2text', []))}")
    print(f"Total problematic stores: {found}")

if __name__ == "__main__":
    main()
