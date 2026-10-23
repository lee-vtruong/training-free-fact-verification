import json
import sys

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/m11-audit-pilot100-20261023-160109/audit_summary.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    keys = ["annotation_field_rows", "claim_id_mismatch_rows", "invalid_source_url_rows", "invalid_text_shape_rows"]
    found = 0
    for s in data.get("stores", []):
        bad = {k: s.get(k, 0) for k in keys if s.get(k, 0) > 0}
        if bad:
            found += 1
            print(f"Store {s.get('sample_id')}: {bad}")
    print(f"Total problematic stores: {found}")

if __name__ == "__main__":
    main()
