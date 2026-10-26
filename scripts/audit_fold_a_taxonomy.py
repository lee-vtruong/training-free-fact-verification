"""Source-Only Failure Mechanism Audit on Fold A (train:0..49)."""
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

root = Path(__file__).resolve().parent.parent
mem_bank_p = root / "outputs" / "m142-confirmatory-20261025-233407" / "error_memory_bank.json"
data = json.loads(mem_bank_p.read_text(encoding="utf-8"))

records = data["records"]
correct_sids = set(data["correct_sids"])
wrong_sids = set(data["wrong_sids"])

print(f"Total source claims in Fold A: {len(records)}")
print(f"Correct: {len(correct_sids)}, Wrong: {len(wrong_sids)}")

# Let's inspect each wrong claim in Fold A
print("\n" + "="*80)
print("ALL 25 WRONG CLAIMS IN FOLD A WITH GOLD JUSTIFICATIONS")
print("="*80)

for sid in sorted(wrong_sids, key=lambda s: int(s.split(":")[1])):
    r = records[sid]
    g = r["gold_label"]
    b = r["base_predicted_verdict"]
    c = r["claim"]
    j = r.get("gold_justification", "")
    print(f"[{sid}] {g} -> Pred: {b}")
    print(f"  Claim: \"{c}\"")
    print(f"  Justification: {j}")
    top_ev = r.get("compact_evidence", [{}])[0]
    print(f"  Top-1 Ev ({top_ev.get('source_url', 'url')}): {top_ev.get('text', '')[:100]}...")
    print()
