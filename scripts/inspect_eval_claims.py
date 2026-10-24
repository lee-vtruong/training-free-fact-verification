import json
from pathlib import Path

claims_p = Path("outputs/m11-source-claims-20261021-055413/claims.jsonl")
claims = [json.loads(l) for l in claims_p.read_text(encoding="utf-8").splitlines() if l.strip()]

print(f"Total claims in dataset: {len(claims)}")
source_claims = claims[:100]
eval_candidates = claims[100:150]

print(f"Source claims: {source_claims[0]['sample_id']} to {source_claims[-1]['sample_id']}")
print(f"Eval candidates (50 claims): {eval_candidates[0]['sample_id']} to {eval_candidates[-1]['sample_id']}")

# Check near-duplicates between source and eval candidates
def get_words(text):
    return set(text.lower().replace(".", "").replace(",", "").split())

max_sim = 0.0
leaks = []
for ev in eval_candidates:
    ev_words = get_words(ev["claim"])
    for sc in source_claims:
        sc_words = get_words(sc["claim"])
        jaccard = len(ev_words & sc_words) / len(ev_words | sc_words) if (ev_words | sc_words) else 0.0
        if jaccard > 0.6:
            leaks.append((ev["sample_id"], sc["sample_id"], jaccard, ev["claim"], sc["claim"]))
        if jaccard > max_sim:
            max_sim = jaccard

print(f"Maximum pairwise Jaccard similarity: {max_sim:.3f}")
print(f"High-similarity pairs (> 0.6): {len(leaks)}")
for l in leaks:
    print(f"  Leakage alert: {l[0]} vs {l[1]} (sim={l[2]:.2f}):\n    Eval: {l[3]}\n    Source: {l[4]}")
