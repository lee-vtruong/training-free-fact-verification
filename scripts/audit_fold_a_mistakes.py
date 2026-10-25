"""Audit failure modes in Fold A that generated the 4 empirical rules."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
from collections import Counter

root = Path(__file__).resolve().parent.parent
mem_bank_p = root / "outputs" / "m141-confirmatory-20261025-101555" / "error_memory_bank.json"
data = json.loads(mem_bank_p.read_text(encoding="utf-8"))

records = data["records"]
correct_sids = set(data["correct_sids"])
wrong_sids = set(data["wrong_sids"])

print(f"Total Fold A claims: {len(records)}")
print(f"Correct: {len(correct_sids)} ({len(correct_sids)/len(records)*100:.1f}%)")
print(f"Wrong: {len(wrong_sids)} ({len(wrong_sids)/len(records)*100:.1f}%)")
print(f"\nConfusion pairs in Fold A errors:")
for conf, sids in sorted(data.get("by_confusion", {}).items()):
    print(f"  {conf:<50}: {len(sids)} claims ({', '.join(sids)})")

print("\n" + "="*80)
print("ANALYSIS OF THE 4 EMPIRICAL RULES & MATCHING FOLD A MISTAKES")
print("="*80)

# The 4 empirical rules from M1.3 / error_memory.py:
rules = [
    ("Rule 1: Media Debate vs Factual Contradiction",
     "If retrieved articles quote political opponents arguing about a claim, verify whether the underlying factual statement itself is substantiated. Do NOT classify political disagreement as Conflicting Evidence if official records confirm the fact."),
    ("Rule 2: Evidence Sufficiency Threshold",
     "If evidence directly confirms or contradicts the core proposition, do not default to Not Enough Evidence simply because minor peripheral details are unmentioned."),
    ("Rule 3: Cherrypicking Precision",
     "Choose 'Conflicting Evidence/Cherrypicking' only when the claim quotes true facts out of context to convey a demonstrably false implication. Outright false claims are Refuted."),
    ("Rule 4: Polarity Check",
     "Carefully evaluate conditional denials and double negatives to ensure the direction of entailment matches the claim.")
]

for title, text in rules:
    print(f"\n### {title}")
    print(f"Text: \"{text}\"")

print("\n" + "="*80)
print("DETAILS OF ALL WRONG CLAIMS IN FOLD A:")
print("="*80)

for sid in sorted(wrong_sids, key=lambda s: int(s.split(":")[1])):
    r = records[sid]
    g = r.get("gold_label")
    b = r.get("base_predicted_verdict")
    claim = r.get("claim", "")
    print(f"\n[{sid}] Gold: {g} | Base Pred: {b}")
    print(f"  Claim: \"{claim}\"")
