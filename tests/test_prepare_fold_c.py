"""Unit tests for Fold C R4 evidence preparation and verifier contract verification."""
import json
import shutil
import tempfile
from pathlib import Path
import pytest

from tffv.retrieval.prepare_fold_c import (
    prepare_fold_c_pipeline,
    resolve_claims_manifest_path,
    resolve_plan_path,
    verify_fold_c_evidence_contract,
)
from tffv.verification.predict import dump_jsonl


def make_mock_evidence_pair(directory: Path, count: int = 50, start: int = 100):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    claims_rows = [
        {"sample_id": f"train:{i}", "claim": f"Mock claim {i}", "dataset": "AVeriTeC", "split": "train"}
        for i in range(start, start + count)
    ]
    r4_rows = [
        {
            "sample_id": f"train:{i}",
            "dataset": "AVeriTeC",
            "evidence": [
                {
                    "passage_id": f"p_{i}_{j}",
                    "text": f"Mock passage text {j} for claim {i}",
                    "source_url": f"https://example.com/source/{i}/{j}",
                    "score": 0.95 - j * 0.05,
                }
                for j in range(8)
            ],
            "evaluation_ranking_top10": [{"passage_id": f"p_{i}_{j}", "rank": j + 1} for j in range(10)],
        }
        for i in range(start, start + count)
    ]
    claims_p = directory / "claims.jsonl"
    r4_p = directory / "R4.jsonl"
    dump_jsonl(claims_p, claims_rows)
    dump_jsonl(r4_p, r4_rows)
    return r4_p, claims_p


def test_verify_fold_c_evidence_contract_valid():
    """Verify clean 50-claim evidence passes contract verification."""
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        r4_p, claims_p = make_mock_evidence_pair(tmp_dir, count=50, start=100)
        report = verify_fold_c_evidence_contract(r4_p, claims_p, start_id=100, end_id=150)
        assert report["status"] == "verified_verifier_contract_clean"
        assert report["claims_count"] == 50
        assert report["r4_count"] == 50
        assert report["sample_ids_exact_match"] is True
        assert report["mean_evidence_depth"] == 8.0
    finally:
        shutil.rmtree(tmp_dir)


def test_verify_fold_c_evidence_contract_mismatched_count():
    """Verify error raised when claim counts are fewer than 50."""
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        r4_p, claims_p = make_mock_evidence_pair(tmp_dir, count=49, start=100)
        with pytest.raises(ValueError, match="record count mismatch"):
            verify_fold_c_evidence_contract(r4_p, claims_p, start_id=100, end_id=150)
    finally:
        shutil.rmtree(tmp_dir)


def test_verify_fold_c_evidence_contract_mismatched_ids():
    """Verify error raised when claim IDs do not align with train:100..149."""
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        # Starts at 101 instead of 100
        r4_p, claims_p = make_mock_evidence_pair(tmp_dir, count=50, start=101)
        with pytest.raises(ValueError, match="sample_ids do not match expected range"):
            verify_fold_c_evidence_contract(r4_p, claims_p, start_id=100, end_id=150)
    finally:
        shutil.rmtree(tmp_dir)


def test_verify_fold_c_evidence_contract_empty_evidence():
    """Verify error raised when evidence list is empty."""
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        r4_p, claims_p = make_mock_evidence_pair(tmp_dir, count=50, start=100)
        # Corrupt one record
        rows = [json.loads(line) for line in r4_p.read_text(encoding="utf-8").splitlines()]
        rows[10]["evidence"] = []
        dump_jsonl(r4_p, rows)

        with pytest.raises(ValueError, match="empty evidence list"):
            verify_fold_c_evidence_contract(r4_p, claims_p, start_id=100, end_id=150)
    finally:
        shutil.rmtree(tmp_dir)


def test_resolve_plan_and_claims_paths():
    """Verify path resolvers find existing project manifests."""
    root = Path(__file__).resolve().parent.parent
    plan_p = resolve_plan_path(root)
    assert plan_p.is_file()
    assert plan_p.name == "acquisition_plan.json"

    claims_m_p = resolve_claims_manifest_path(root)
    assert claims_m_p.is_file()
    assert claims_m_p.name == "claims_manifest.json"
