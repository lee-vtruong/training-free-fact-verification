import json
from pathlib import Path
import shutil
import uuid

import pytest
from tffv.retrieval.leakage_review import (
    DEFAULT_POLICY,
    inspect_evidence_item,
    load_policy,
    review_evidence,
)


@pytest.fixture
def review_env(tmp_path):
    root = Path(__file__).resolve().parents[1] / "outputs" / ("pytest-leakage-" + uuid.uuid4().hex)
    root.mkdir(parents=True)
    try:
        yield root, tmp_path
    finally:
        shutil.rmtree(root)


def test_inspect_verdict_and_factcheck_url_detection():
    policy = load_policy()
    # Verdict phrase match
    item1 = {
        "sample_id": "train:0",
        "claim": "The moon is made of green cheese",
        "passage_id": "p1",
        "text": "Earlier today, BBC reported. Verdict: the claim is false and unsupported.",
        "source_url": "https://example.com/news",
    }
    res1 = inspect_evidence_item(item1, policy)
    assert "verdict_pattern" in res1["flags"]
    assert res1["disposition"] == "flagged_for_review"

    # Fact-check domain match
    item2 = {
        "sample_id": "train:1",
        "claim": "Tax cuts were the largest ever",
        "passage_id": "p2",
        "text": "The economy saw various changes over the decade.",
        "source_url": "https://www.politifact.com/factchecks/2020/tax-cuts/",
    }
    res2 = inspect_evidence_item(item2, policy)
    assert "fact_check_domain" in res2["flags"]
    assert "fact_check_url_path" in res2["flags"]

    # URL path match on non-standard domain (e.g., BBC Reality Check)
    item3 = {
        "sample_id": "train:1",
        "claim": "Trump claimed taxes fell by half",
        "passage_id": "p3",
        "text": "Official Treasury data showed different numbers.",
        "source_url": "https://www.bbc.com/news/reality-check-12345",
    }
    res3 = inspect_evidence_item(item3, policy)
    assert "fact_check_url_path" in res3["flags"]


def test_boilerplate_and_zero_overlap_detection():
    policy = load_policy()
    item_bp = {
        "sample_id": "train:2",
        "claim": "Unemployment rate dropped to 3 percent",
        "passage_id": "p_bp",
        "text": "Please enable javascript to view this website. Cookie policy and terms of use apply.",
        "source_url": "https://example.com/denied",
    }
    res_bp = inspect_evidence_item(item_bp, policy)
    assert "boilerplate_noise" in res_bp["flags"]

    item_zero = {
        "sample_id": "train:3",
        "claim": "Astronauts landed on Mars in 1969",
        "passage_id": "p_zero",
        "text": "The recipe calls for two cups of flour and one teaspoon of vanilla extract.",
        "source_url": "https://recipes.example.com/cake",
    }
    res_zero = inspect_evidence_item(item_zero, policy)
    assert "zero_claim_token_overlap" in res_zero["flags"]


def test_claim_verdict_coupling():
    policy = load_policy()
    item = {
        "sample_id": "train:4",
        "claim": "Senator Smith voted against the highway bill in 2021",
        "passage_id": "p_couple",
        "text": "Regarding the allegation that Senator Smith voted against the highway bill, independent analysts found this claim was debunked yesterday.",
        "source_url": "https://news.example.com/article",
    }
    res = inspect_evidence_item(item, policy)
    assert "claim_verdict_coupling" in res["flags"]


def test_reject_forbidden_gold_fields():
    policy = load_policy()
    bad_item = {
        "sample_id": "train:5",
        "claim": "Test claim",
        "text": "Evidence text",
        "gold_label": "REFUTED",
    }
    with pytest.raises(ValueError, match="Forbidden evaluation gold fields"):
        inspect_evidence_item(bad_item, policy)


def test_custom_policy_hashing_and_quarantine_action(review_env):
    root, tmp = review_env
    custom_policy = dict(DEFAULT_POLICY)
    custom_policy["policy_id"] = "custom_test_policy_v1"
    custom_policy["version"] = "1.0.0"
    custom_policy["action"] = "quarantine"

    p_loaded = load_policy(custom_policy)
    assert p_loaded["policy_id"] == "custom_test_policy_v1"
    assert p_loaded["policy_sha256"] is not None
    assert p_loaded["action"] == "quarantine"

    # Test full review run with queue file
    queue_file = tmp / "manual_review_queue.jsonl"
    rows = [
        {
            "sample_id": "train:0",
            "claim": "Clean statement text",
            "evidence": {
                "passage_id": "p_clean",
                "text": "Statement text discussed by analysts during conference.",
                "source_url": "https://example.com/clean",
            },
        },
        {
            "sample_id": "train:1",
            "claim": "Leaky statement text",
            "evidence": {
                "passage_id": "p_leaky",
                "text": "Verdict: false statement text.",
                "source_url": "https://factcheck.org/check",
            },
        },
    ]
    queue_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    out_dir = root / "review-output"
    report_json = review_evidence(queue_file, out_dir, policy_spec=custom_policy)
    assert report_json.is_file()

    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["total_records_reviewed"] == 2
    assert report["flagged_records_count"] == 1
    assert report["quarantined_records_count"] == 1
    assert "verdict_pattern" in report["flag_breakdown"]

    # Verify artifacts
    assert (out_dir / "review_report.md").is_file()
    assert (out_dir / "flagged_evidence.jsonl").is_file()
    assert (out_dir / "quarantined_ledger.jsonl").is_file()

    flagged_rows = [json.loads(line) for line in (out_dir / "flagged_evidence.jsonl").read_text().splitlines()]
    assert len(flagged_rows) == 1
    assert flagged_rows[0]["passage_id"] == "p_leaky"
    assert flagged_rows[0]["disposition"] == "quarantined"
