"""CPU-only, read-only cross-milestone audit; writes only the requested new report.

Gold is read exclusively for offline scoring, never to build memory or prompts.
No model, retrieval, or inference code is invoked.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tffv.verification.ontology import VERDICTS, normalize_verdict


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def score(gold, predictions):
    assert gold and len(gold) == len(predictions)
    per_class = {}
    for label in VERDICTS:
        tp = sum(g == p == label for g, p in zip(gold, predictions))
        support = gold.count(label)
        predicted = predictions.count(label)
        per_class[label] = {"support": support, "tp": tp, "predicted": predicted,
                            "precision": tp / predicted if predicted else 0.0,
                            "recall": tp / support if support else 0.0,
                            "f1": 2 * tp / (support + predicted) if support + predicted else 0.0}
    correct = sum(g == p for g, p in zip(gold, predictions))
    return {"n": len(gold), "correct": correct, "accuracy": correct / len(gold),
            "macro_f1": sum(v["f1"] for v in per_class.values()) / 4,
            "valid": sum(p in VERDICTS for p in predictions), "per_class": per_class}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    gold_data = json.loads(args.gold.read_text(encoding="utf-8-sig"))
    gold = {f"train:{i}": r["label"] for i, r in enumerate(gold_data)}
    runs = {"B_m13_C0": ("m13-error-informed-20261025-000523", "C0"),
            "B_m14_D0": ("m14-pilot-20261025-014111", "D0"),
            "D_D0": ("m142-confirmatory-20261025-233407", "D0"),
            "D_D1_matched": ("m142-confirmatory-20261025-233407", "D1_matched"),
            "D_D2": ("m142-confirmatory-20261025-233407", "D2"),
            "D_D3": ("m142-confirmatory-20261025-233407", "D3"),
            "D_D4": ("m142-confirmatory-20261025-233407", "D4"),
            **{f"E_{c}": ("m15-evaluation-20261026-151554", c) for c in ("M0", "M1", "M4")}}
    result = {"audit_version": 1, "gold_sha256": sha(args.gold), "runs": {}, "paired": {}}
    loaded = {}
    for name, (run, condition) in runs.items():
        path = root / "outputs" / run / condition / "predictions.jsonl"
        data = rows(path)
        ids = [r["sample_id"] for r in data]
        assert len(ids) == len(set(ids)), f"duplicate IDs: {path}"
        preds = [r.get("predicted_verdict") for r in data]
        raw = [r.get("raw_text", r.get("raw_generation", r.get("generated_text", ""))) for r in data]
        parsed = [normalize_verdict(t) for t in raw]
        manifests = list(path.parent.glob("*manifest.json"))
        manifest = json.loads(manifests[0].read_text()) if manifests else {}
        entry = score([gold[i] for i in ids], preds)
        entry.update({"path": path.relative_to(root).as_posix(), "sha256": sha(path),
                      "manifest_hash_match": manifest.get("predictions_sha256") == sha(path),
                      "max_new_tokens": manifest.get("max_new_tokens"),
                      "raw_nonempty": sum(bool(t.strip()) for t in raw),
                      "raw_parsed_valid": sum(p in VERDICTS for p in parsed),
                      "raw_vs_stored_mismatches": [i for i, a, b in zip(ids, parsed, preds) if a != b]})
        for token_key in ("input_tokens", "output_tokens"):
            keys = (token_key, "total_" + token_key)
            vals = [next((r[k] for k in keys if k in r), None) for r in data]
            entry["total_" + token_key] = sum(vals) if all(v is not None for v in vals) else None
        latencies = [r.get("latency_seconds") for r in data]
        entry["mean_latency_seconds"] = sum(latencies) / len(data) if all(v is not None for v in latencies) else None
        if condition == "M4":
            detail = rows(path.parent / "permutations_M4_detail.jsonl")
            assert {r["sample_id"] for r in detail} == set(ids)
            votes = [v for r in detail for k, v in r.items() if k.startswith("order_")]
            entry["permutation_raw_count"] = len(votes)
            entry["permutation_raw_valid"] = sum(normalize_verdict(v["raw_text"]) in VERDICTS for v in votes)
            assert all(normalize_verdict(v["raw_text"]) == v["verdict"] for v in votes)
            entry["raw_parsed_valid"] = None
            entry["raw_vs_stored_mismatches"] = None
            entry["raw_nonempty"] = None
            entry["raw_note"] = "Aggregate record has no generation; inspect 297 permutation raw generations."
            aggregate = {r["sample_id"]: r["aggregated_verdict"] for r in detail}
            assert aggregate == dict(zip(ids, preds))
            for r in detail:
                vv = [v["verdict"] for k, v in r.items() if k.startswith("order_")]
                expected = max(set(vv), key=vv.count) if max(vv.count(v) for v in vv) >= 2 else r["order_1_canonical"]["verdict"]
                assert expected == r["aggregated_verdict"]
        result["runs"][name] = entry
        if name == "D_D2":
            prompts = rows(path.parent / "prompts.jsonl")
            assert {p["sample_id"] for p in prompts} == set(ids)
            entry["saved_prompts_with_two_exact_directives"] = sum(
                "1. Media debate is not factual contradiction:" in p["prompt"] and
                "2. Missing context is not Cherrypicking:" in p["prompt"] for p in prompts)
        loaded[name] = dict(zip(ids, preds))
    for base, target in [("B_m13_C0", "B_m14_D0"), ("D_D0", "D_D1_matched"), ("D_D1_matched", "D_D2"), ("E_M0", "E_M1"), ("E_M0", "E_M4")]:
        a, b = loaded[base], loaded[target]
        assert set(a) == set(b), f"population mismatch: {base}, {target}"
        transitions = {k: 0 for k in ("CC", "CW", "WC", "WW")}
        changed = []
        for i in sorted(a):
            transitions[("C" if a[i] == gold[i] else "W") + ("C" if b[i] == gold[i] else "W")] += 1
            if a[i] != b[i]:
                changed.append(i)
        result["paired"][f"{target}_vs_{base}"] = {**transitions, "changed_ids": changed}
    canonical = root / "outputs/m13-repaired-audit/canonical_per_claim_predictions.csv"
    with canonical.open(encoding="utf-8-sig", newline="") as f:
        c0 = [r for r in csv.DictReader(f) if r["condition"] == "C0"]
    result["m13_canonical_C0"] = score([r["gold_label"] for r in c0], [r["predicted_label"] for r in c0])
    assert {r["claim_id"]: r["predicted_label"] for r in c0} == loaded["B_m13_C0"]
    frozen_dirs = [root / "outputs" / x for x in (
        "m13-error-informed-20261025-000523", "m13-repaired-audit", "m14-pilot-20261025-014111",
        "m141-confirmatory-20261025-101555", "m142-confirmatory-20261025-233407",
        "m15-evaluation-20261026-142010", "m15-evaluation-20261026-151554", "m15-folde-r4")]
    inventory = {p.relative_to(root).as_posix(): sha(p) for d in frozen_dirs for p in sorted(d.rglob("*")) if p.is_file()}
    result["frozen_inventory"] = inventory
    result["frozen_inventory_sha256"] = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
    assert not any(args.output.resolve().is_relative_to(d.resolve()) for d in frozen_dirs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "files_hashed": len(inventory), "runs": {k: {m: v[m] for m in ("n", "correct", "macro_f1", "valid", "raw_parsed_valid", "manifest_hash_match")} for k, v in result["runs"].items()}}, indent=2))


if __name__ == "__main__":
    main()
