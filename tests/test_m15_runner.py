"""Unit tests for M1.5 Runner: Permutation Consistency, Majority Voting, and End-to-End Pipeline."""
import json
from pathlib import Path
import tempfile
import pytest

from tffv.verification.m15_runner import (
    aggregate_majority_vote,
    generate_permutations,
    holm_bonferroni,
    main as runner_main,
)
from tffv.verification.ontology import VERDICTS


def test_generate_permutations():
    """Verify evidence ordering permutations for Condition M4."""
    dummy_ev = [{"id": i, "text": f"Passage {i}"} for i in range(8)]
    perms = generate_permutations(dummy_ev)

    assert len(perms) == 3
    names = [p[0] for p in perms]
    assert names == ["canonical", "reverse", "interleaved"]

    ord1 = [p["id"] for p in perms[0][1]]
    ord2 = [p["id"] for p in perms[1][1]]
    ord3 = [p["id"] for p in perms[2][1]]

    assert ord1 == [0, 1, 2, 3, 4, 5, 6, 7]
    assert ord2 == [7, 6, 5, 4, 3, 2, 1, 0]
    assert ord3 == [0, 7, 1, 6, 2, 5, 3, 4]


def test_aggregate_majority_vote():
    """Verify majority voting logic and canonical tie-breaker."""
    # 1. Unanimous (3 votes)
    v, count, tie = aggregate_majority_vote("Supported", "Supported", "Supported")
    assert v == "Supported"
    assert count == 3
    assert tie is False

    # 2. Majority (2 vs 1)
    v, count, tie = aggregate_majority_vote("Supported", "Supported", "Refuted")
    assert v == "Supported"
    assert count == 2
    assert tie is False

    v, count, tie = aggregate_majority_vote("Refuted", "Supported", "Supported")
    assert v == "Supported"
    assert count == 2
    assert tie is False

    # 3. 3-way split: Canonical Order 1 breaks tie
    v, count, tie = aggregate_majority_vote("Supported", "Refuted", "Not Enough Evidence")
    assert v == "Supported"
    assert count == 1
    assert tie is True

    # 4. Handling unparsed
    v, count, tie = aggregate_majority_vote("Refuted", "Unparseable", "Refuted")
    assert v == "Refuted"
    assert count == 2
    assert tie is False

    v, count, tie = aggregate_majority_vote("Unparseable", "Unparseable", "Unparseable")
    assert v is None
    assert count == 0


def test_holm_bonferroni_correction():
    """Verify step-down Holm-Bonferroni p-value adjustment."""
    raw_p = {
        "contrast_A": 0.01,
        "contrast_B": 0.04,
        "contrast_C": 0.03,
    }
    adj = holm_bonferroni(raw_p)
    # Order: A (0.01*3 = 0.03), C (0.03*2 = 0.06), B (0.04*1 = 0.06)
    assert adj["contrast_A"] == 0.03
    assert adj["contrast_C"] == 0.06
    assert adj["contrast_B"] == 0.06


def test_m15_runner_synthetic_end_to_end(monkeypatch):
    """Verify full M1.5 runner execution with SyntheticBackbone on temporary directory."""
    root = Path(__file__).resolve().parent.parent
    train_json_p = root / "data" / "raw" / "averitec" / "train.json"

    if not train_json_p.is_file():
        pytest.skip("AVeriTeC train.json not present locally.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)

        # Create dummy claims.jsonl and R4.jsonl for 3 claims (train:250, train:251, train:252)
        claims = [
            {"sample_id": "train:250", "claim": "Claim 250 test.", "speaker": "A"},
            {"sample_id": "train:251", "claim": "Counties in Kenya are striving to reach the 10% budget allocation to agriculture recommended by the Malabo declaration.", "speaker": "B"},
            {"sample_id": "train:252", "claim": "Claim 252 test empirical proposition.", "speaker": "C"},
        ]
        claims_file = tmp_p / "claims.jsonl"
        with open(claims_file, "w", encoding="utf-8") as f:
            for c in claims:
                f.write(json.dumps(c) + "\n")

        evidence_rows = [
            {
                "claim_id": "train:250",
                "sample_id": "train:250",
                "predicted_evidence": [
                    {"text": "Evidence for 250.", "source_url": "http://test.com/1", "score": 0.9}
                ],
            },
            {
                "claim_id": "train:251",
                "sample_id": "train:251",
                "predicted_evidence": [
                    {"text": "Evidence for 251.", "source_url": "http://test.com/2", "score": 0.8}
                ],
            },
            {
                "claim_id": "train:252",
                "sample_id": "train:252",
                "predicted_evidence": [
                    {"text": "Evidence for 252.", "source_url": "http://test.com/3", "score": 0.85}
                ],
            },
        ]
        r4_file = tmp_p / "R4.jsonl"
        with open(r4_file, "w", encoding="utf-8") as f:
            for e in evidence_rows:
                f.write(json.dumps(e) + "\n")

        out_run = tmp_p / "m15_run_output"

        # Invoke runner CLI with synthetic backbone
        test_args = [
            "m15_runner.py",
            "--r4-evidence", str(r4_file),
            "--claims-jsonl", str(claims_file),
            "--train-json", str(train_json_p),
            "--output-dir", str(out_run),
            "--conditions", "M0", "M1", "M4",
            "--eval-start", "250",
            "--eval-end", "253",
            "--source-start", "0",
            "--source-end", "50",
            "--use-synthetic",
            "--seed", "42",
        ]
        monkeypatch.setattr("sys.argv", test_args)

        runner_main()

        # Check deliverables exist
        assert (out_run / "partition_manifest.json").is_file()
        assert (out_run / "quarantine_manifest.json").is_file()
        assert (out_run / "M0" / "predictions.jsonl").is_file()
        assert (out_run / "M1" / "predictions.jsonl").is_file()
        assert (out_run / "M4" / "predictions.jsonl").is_file()
        assert (out_run / "M4" / "permutations_M4_detail.jsonl").is_file()
        assert (out_run / "metrics_summary.json").is_file()
        assert (out_run / "paired_comparisons.json").is_file()
        assert (out_run / "M1_5_EVALUATION_REPORT.md").is_file()

        # Check quarantine caught train:251
        q_manifest = json.loads((out_run / "quarantine_manifest.json").read_text(encoding="utf-8"))
        assert q_manifest["quarantined_claims_count"] == 1
        assert "train:251" in q_manifest["quarantined_sample_ids"]

        # Check admitted claims in predictions
        part_manifest = json.loads((out_run / "partition_manifest.json").read_text(encoding="utf-8"))
        assert part_manifest["admitted_claims_count"] == 2
        assert set(part_manifest["admitted_claims_sample_ids"]) == {"train:250", "train:252"}
