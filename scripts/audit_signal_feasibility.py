"""Audit Signal Feasibility and Mechanism Coverage on Fold A (train:0..49)."""
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))

from tffv.verification.taxonomy import (
    ALL_MECHANISMS,
    TAXONOMY_PROFILES,
    detect_active_mechanisms,
    extract_attribution_signal,
    extract_evidence_sufficiency_signal,
    extract_negation_signal,
    extract_partial_truth_signal,
    extract_scope_quantifier_signal,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

root = Path(__file__).resolve().parent.parent
mem_bank_p = root / "outputs" / "m142-confirmatory-20261025-233407" / "error_memory_bank.json"
data = json.loads(mem_bank_p.read_text(encoding="utf-8"))

records = data["records"]
correct_sids = set(data["correct_sids"])
wrong_sids = set(data["wrong_sids"])
all_sids = sorted(records.keys(), key=lambda s: int(s.split(":")[1]))

print(f"Total source claims in Fold A: {len(all_sids)}")
print(f"Base Correct: {len(correct_sids)}, Base Wrong: {len(wrong_sids)}")

signals = [
    ("Scope/Quantifier/Denominator", extract_scope_quantifier_signal),
    ("Partial Truth/Conjunction", extract_partial_truth_signal),
    ("Evidence Sufficiency (Sparse)", extract_evidence_sufficiency_signal),
    ("Attribution/Quote", extract_attribution_signal),
    ("Negation/Polarity", extract_negation_signal),
]

print("\n" + "="*90)
print("OBSERVABLE SIGNAL FEASIBILITY TABLE ON FOLD A (N=50)")
print("="*90)
print(f"{'Signal Name':<32} | {'Fired':<8} | {'Correct Fired':<14} | {'Wrong Fired':<12} | {'Signal Prec':<12} | {'Risk of Regress'}")
print("-"*90)

for name, fn in signals:
    fired_correct = [s for s in correct_sids if fn(records[s]["claim"], records[s].get("compact_evidence"))]
    fired_wrong = [s for s in wrong_sids if fn(records[s]["claim"], records[s].get("compact_evidence"))]
    n_total = len(fired_correct) + len(fired_wrong)
    prec = len(fired_wrong) / n_total if n_total > 0 else 0.0
    risk = len(fired_correct) / len(correct_sids) if correct_sids else 0.0
    print(f"{name:<32} | {n_total:<8} | {len(fired_correct):<14} | {len(fired_wrong):<12} | {prec:<12.1%} | {risk:.1%}")

print("\n" + "="*90)
print("MECHANISM AUDIT: SUPPORT COUNTS AND FIDELITY IN FOLD A (N=50)")
print("="*90)
for m_name in ALL_MECHANISMS:
    p = TAXONOMY_PROFILES[m_name]
    print(f"Mechanism: {p.name}")
    print(f"  - Support Count in Fold A: {p.fold_a_support_count} / 25 errors ({p.fold_a_support_count/25:.1%})")
    print(f"  - Adequately Supported: {'YES' if p.is_adequately_supported else 'NO (SAMPLE INSUFFICIENT)'}")
    print(f"  - Example SIDs: {', '.join(p.fold_a_example_ids)}")
    print()
