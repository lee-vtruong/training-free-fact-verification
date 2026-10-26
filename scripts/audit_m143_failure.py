"""Audit Fold D predictions, transitions, discordant pairs, and rule mechanisms for M1.4.3."""
import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))

import json
from collections import Counter

from tffv.verification.evaluate import load_gold_labels, compute_metrics
from tffv.verification.ontology import VERDICTS
train_json = root / "data" / "raw" / "averitec" / "train.json"
gold_map = load_gold_labels(train_json)

out_dir = root / "outputs" / "m142-confirmatory-20261025-233407"
conditions = ["D0", "D1_matched", "D2", "D2_minus_r1", "D2_minus_r2", "D2_para1", "D3", "D4"]

cond_preds = {}
for c in conditions:
    p_file = out_dir / c / "predictions.jsonl"
    rows = [json.loads(l) for l in p_file.read_text(encoding="utf-8").splitlines()]
    cond_preds[c] = {r["sample_id"]: r for r in rows}

sids = sorted(list(cond_preds["D0"].keys()), key=lambda s: int(s.split(":")[1]))
print(f"Total evaluated claims: {len(sids)}")

# 1. Inspect D2 vs D1_matched discordant claims
print("\n" + "="*80)
print("DISCORDANT CLAIMS: D2 vs D1_matched")
print("="*80)
d1_corr_d2_wrong = []
d2_corr_d1_wrong = []

for sid in sids:
    g = gold_map[sid]["gold_label"]
    p_d1 = cond_preds["D1_matched"][sid]["predicted_verdict"]
    p_d2 = cond_preds["D2"][sid]["predicted_verdict"]
    claim = cond_preds["D1_matched"][sid]["claim"]

    c_d1 = (p_d1 == g)
    c_d2 = (p_d2 == g)

    if c_d1 and not c_d2:
        d1_corr_d2_wrong.append((sid, g, p_d1, p_d2, claim))
    elif c_d2 and not c_d1:
        d2_corr_d1_wrong.append((sid, g, p_d1, p_d2, claim))

print(f"D1_matched CORRECT, D2 WRONG ({len(d1_corr_d2_wrong)} claims):")
for sid, g, p_d1, p_d2, claim in d1_corr_d2_wrong:
    print(f"  [{sid}] Gold: {g} | D1: {p_d1} | D2: {p_d2}")
    print(f"    Claim: \"{claim}\"")
    raw_d1 = cond_preds["D1_matched"][sid]["raw_generation"].replace("\n", " ")[:100]
    raw_d2 = cond_preds["D2"][sid]["raw_generation"].replace("\n", " ")[:100]
    print(f"    D1 gen: {raw_d1}")
    print(f"    D2 gen: {raw_d2}")

print(f"\nD2 CORRECT, D1_matched WRONG ({len(d2_corr_d1_wrong)} claims):")
for sid, g, p_d1, p_d2, claim in d2_corr_d1_wrong:
    print(f"  [{sid}] Gold: {g} | D1: {p_d1} | D2: {p_d2}")
    print(f"    Claim: \"{claim}\"")

# 2. Inspect D2 vs D2_minus_r2 (Effect of Rule 2: "Missing context is not Cherrypicking")
print("\n" + "="*80)
print("DISCORDANT CLAIMS: D2 vs D2_minus_r2 (Effect of Rule 2)")
print("="*80)
minus_r2_corr_d2_wrong = []
d2_corr_minus_r2_wrong = []

for sid in sids:
    g = gold_map[sid]["gold_label"]
    p_r2 = cond_preds["D2_minus_r2"][sid]["predicted_verdict"]
    p_d2 = cond_preds["D2"][sid]["predicted_verdict"]
    claim = cond_preds["D2"][sid]["claim"]

    c_r2 = (p_r2 == g)
    c_d2 = (p_d2 == g)

    if c_r2 and not c_d2:
        minus_r2_corr_d2_wrong.append((sid, g, p_r2, p_d2, claim))
    elif c_d2 and not c_r2:
        d2_corr_minus_r2_wrong.append((sid, g, p_r2, p_d2, claim))

print(f"D2_minus_r2 CORRECT, D2 WRONG ({len(minus_r2_corr_d2_wrong)} claims) -> Claims harmed by Rule 2:")
for sid, g, p_r2, p_d2, claim in minus_r2_corr_d2_wrong:
    print(f"  [{sid}] Gold: {g} | D2_minus_r2: {p_r2} | D2: {p_d2}")
    print(f"    Claim: \"{claim}\"")
    raw_r2 = cond_preds["D2_minus_r2"][sid]["raw_generation"].replace("\n", " ")[:100]
    raw_d2 = cond_preds["D2"][sid]["raw_generation"].replace("\n", " ")[:100]
    print(f"    D2_minus_r2 gen: {raw_r2}")
    print(f"    D2 gen:          {raw_d2}")

print(f"\nD2 CORRECT, D2_minus_r2 WRONG ({len(d2_corr_minus_r2_wrong)} claims) -> Claims helped by Rule 2:")
for sid, g, p_r2, p_d2, claim in d2_corr_minus_r2_wrong:
    print(f"  [{sid}] Gold: {g} | D2_minus_r2: {p_r2} | D2: {p_d2}")
    print(f"    Claim: \"{claim}\"")
