"""Real archived-artifact checks for the M1.6 CPU auditor; no inference."""
import csv
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m16_auditor", ROOT / "scripts/audit_m16_integrity.py")
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


@pytest.mark.skipif(
    not (ROOT / "outputs/m13-repaired-audit/canonical_per_claim_predictions.csv").exists()
    or not (ROOT / "outputs/m14-pilot-20261025-014111/D0/predictions.jsonl").exists(),
    reason="Archived experiment outputs are external, ignored artifacts; fetch originals to run this real-data check.",
)
def test_real_fold_b_canonical_score_and_reexecution_agree():
    with (ROOT / "outputs/m13-repaired-audit/canonical_per_claim_predictions.csv").open(encoding="utf-8-sig", newline="") as f:
        records = [r for r in csv.DictReader(f) if r["condition"] == "C0"]
    metric = auditor.score([r["gold_label"] for r in records], [r["predicted_label"] for r in records])
    assert metric["n"] == 46
    assert metric["correct"] == 28
    assert abs(metric["macro_f1"] - 0.4882901013714967) < 1e-12
    raw = auditor.rows(ROOT / "outputs/m14-pilot-20261025-014111/D0/predictions.jsonl")
    assert {r["claim_id"]: r["predicted_label"] for r in records} == {r["sample_id"]: r["predicted_verdict"] for r in raw}


def test_sealed_server_audit_checks_raw_votes_without_inventing_aggregate_text():
    result = json.loads((ROOT / "docs/audits/M1_6_INTEGRITY_EVIDENCE.json").read_text(encoding="utf-8-sig"))
    assert all(r["manifest_hash_match"] for r in result["runs"].values())
    d0 = result["runs"]["D_D0"]
    assert (d0["n"], d0["correct"], d0["max_new_tokens"]) == (91, 49, 512)
    assert result["runs"]["D_D2"]["saved_prompts_with_two_exact_directives"] == 91
    m4 = result["runs"]["E_M4"]
    assert (m4["valid"], m4["permutation_raw_count"], m4["permutation_raw_valid"]) == (99, 297, 297)
    assert m4["raw_parsed_valid"] is None
    assert m4["total_input_tokens"] == 782913


def test_scoring_retains_invalids_and_all_four_classes():
    # Evaluator contract only: this is not model-quality evidence.
    labels = list(auditor.VERDICTS)
    metric = auditor.score(labels, [labels[0], labels[1], "Unparseable", labels[3]])
    assert metric["accuracy"] == 0.75
    assert metric["macro_f1"] == 0.75
    assert metric["valid"] == 3
    assert metric["per_class"][labels[2]]["support"] == 1
