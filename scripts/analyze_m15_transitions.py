"""Analyze transitions between M0, M1, and M4 on Fold E."""
import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent
    eval_dir = root / "outputs" / "m15-evaluation-20261026-151554"
    train_json = root / "data" / "raw" / "averitec" / "train.json"

    with open(train_json, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
    gold = {f"train:{i}": row["label"] for i, row in enumerate(gold_data)}

    def load_preds(p):
        preds = {}
        claims = {}
        with open(p, "r", encoding="utf-8") as f:
            for l in f:
                r = json.loads(l)
                preds[r["sample_id"]] = r["predicted_verdict"]
                claims[r["sample_id"]] = r["claim"]
        return preds, claims

    m0_preds, claims = load_preds(eval_dir / "M0" / "predictions.jsonl")
    m1_preds, _ = load_preds(eval_dir / "M1" / "predictions.jsonl")
    m4_preds, _ = load_preds(eval_dir / "M4" / "predictions.jsonl")

    print("=== M1 vs M0 Improvements (W -> C) ===")
    for sid in m0_preds:
        g = gold[sid]
        if m0_preds[sid] != g and m1_preds[sid] == g:
            print(f"{sid}: Gold={g} | M0={m0_preds[sid]} -> M1={m1_preds[sid]}")
            print(f"   Claim: {claims[sid]}")

    print("\n=== M1 vs M0 Regressions (C -> W) ===")
    for sid in m0_preds:
        g = gold[sid]
        if m0_preds[sid] == g and m1_preds[sid] != g:
            print(f"{sid}: Gold={g} | M0={m0_preds[sid]} -> M1={m1_preds[sid]}")
            print(f"   Claim: {claims[sid]}")

    print("\n=== M4 vs M0 Changes ===")
    for sid in m0_preds:
        g = gold[sid]
        if m0_preds[sid] != m4_preds[sid]:
            status = "CORRECTION (W -> C)" if m4_preds[sid] == g else "REGRESSION (C -> W)"
            print(f"{sid} [{status}]: Gold={g} | M0={m0_preds[sid]} -> M4={m4_preds[sid]}")
            print(f"   Claim: {claims[sid]}")

if __name__ == "__main__":
    main()
