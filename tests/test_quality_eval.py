import json
import shutil
import uuid
from pathlib import Path
import pytest

from tffv.provenance import file_hash
from tffv.retrieval.quality_eval import (
    extract_gold_urls_from_source,
    evaluate_method_claim,
    run_evaluation,
    verify_sealed_run,
)
from tffv.retrieval.data import Passage


@pytest.fixture
def eval_fixture(tmp_path):
    root = Path(__file__).resolve().parents[1] / "outputs" / ("pytest-qualdemo-" + uuid.uuid4().hex)
    root.mkdir(parents=True)

    # 1. Mock train.json
    train_data = [
        {
            "claim": "Claim zero",
            "label": "Supported",
            "fact_checking_article": "https://snopes.com/fact-check/zero",
            "questions": [
                {
                    "question": "What happened?",
                    "answers": [
                        {"answer": "Yes", "source_url": "https://www.bbc.com/news/article-1/"},
                        {"answer": "Indeed", "source_url": "http://nytimes.com/world/article-2"}
                    ]
                }
            ]
        },
        {
            "claim": "Claim one",
            "label": "Refuted",
            "fact_checking_article": "https://politifact.com/factchecks/one",
            "questions": [
                {
                    "question": "Is it true?",
                    "answers": [
                        {"answer": "No", "source_url": "https://reuters.com/news/story-3"}
                    ]
                }
            ]
        }
    ]
    train_json = tmp_path / "train.json"
    train_json.write_text(json.dumps(train_data), encoding="utf-8")

    # 2. Mock sealed run dir
    run_dir = root / "mock_run"
    run_dir.mkdir(parents=True)

    # Store passages
    p0 = [
        Passage("p0_1", "doc1", "Text 1", "https://bbc.com/news/article-1", "web_document", "2026-10-21", "acq1", 0, 10, "hash1"),
        Passage("p0_2", "doc2", "Text 2", "https://other.com/news", "web_document", "2026-10-21", "acq1", 0, 10, "hash2"),
        Passage("p0_3", "doc3", "Text 3", "https://nytimes.com/world/article-2", "web_document", "2026-10-21", "acq1", 0, 10, "hash3"),
    ]
    p1 = [
        Passage("p1_1", "doc4", "Text 4", "https://other.com/a", "web_document", "2026-10-21", "acq1", 0, 10, "hash4"),
        Passage("p1_2", "doc5", "Text 5", "https://reuters.com/news/story-3", "web_document", "2026-10-21", "acq1", 0, 10, "hash5"),
    ]

    (run_dir / "train-0").mkdir()
    (run_dir / "train-1").mkdir()
    (run_dir / "train-0" / "passages.jsonl").write_text("\n".join(json.dumps(p.to_dict()) for p in p0) + "\n", encoding="utf-8")
    (run_dir / "train-1" / "passages.jsonl").write_text("\n".join(json.dumps(p.to_dict()) for p in p1) + "\n", encoding="utf-8")

    # Method R1: found p0_1 at rank 1, p0_3 at rank 3
    r1_records = [
        {
            "sample_id": "train:0",
            "method": "R1",
            "evidence": [{"source_url": "https://bbc.com/news/article-1"}],
            "ranked_candidates": [
                {"passage_id": "p0_1", "rank": 1, "score": 10.0},
                {"passage_id": "p0_2", "rank": 2, "score": 8.0},
                {"passage_id": "p0_3", "rank": 3, "score": 5.0},
            ]
        },
        {
            "sample_id": "train:1",
            "method": "R1",
            "evidence": [],
            "ranked_candidates": [
                {"passage_id": "p1_1", "rank": 1, "score": 6.0},
                {"passage_id": "p1_2", "rank": 15, "score": 2.0},  # Rank 15 -> Hit@100 but Miss@10!
            ]
        }
    ]
    (run_dir / "R1.jsonl").write_text("\n".join(json.dumps(r) for r in r1_records) + "\n", encoding="utf-8")

    # Method R3: rank 15 for p1_2
    r3_records = [
        {
            "sample_id": "train:0",
            "method": "R3",
            "evidence": [{"source_url": "https://bbc.com/news/article-1"}],
            "ranked_candidates": [
                {"passage_id": "p0_1", "rank": 2, "score": 0.05},
                {"passage_id": "p0_3", "rank": 4, "score": 0.03},
            ]
        },
        {
            "sample_id": "train:1",
            "method": "R3",
            "evidence": [],
            "ranked_candidates": [
                {"passage_id": "p1_1", "rank": 1, "score": 0.04},
                {"passage_id": "p1_2", "rank": 15, "score": 0.02},
            ]
        }
    ]
    (run_dir / "R3.jsonl").write_text("\n".join(json.dumps(r) for r in r3_records) + "\n", encoding="utf-8")

    # Method R4: promoted p1_2 from rank 15 to rank 1!
    r4_records = [
        {
            "sample_id": "train:0",
            "method": "R4",
            "evidence": [{"source_url": "https://bbc.com/news/article-1"}],
            "ranked_candidates": [
                {"passage_id": "p0_1", "rank": 1, "score": 0.99},
                {"passage_id": "p0_3", "rank": 2, "score": 0.85},
            ]
        },
        {
            "sample_id": "train:1",
            "method": "R4",
            "evidence": [{"source_url": "https://reuters.com/news/story-3"}],
            "ranked_candidates": [
                {"passage_id": "p1_2", "rank": 1, "score": 0.98},  # Promoted to rank 1!
                {"passage_id": "p1_1", "rank": 2, "score": 0.10},
            ]
        }
    ]
    (run_dir / "R4.jsonl").write_text("\n".join(json.dumps(r) for r in r4_records) + "\n", encoding="utf-8")

    # manifest.json
    manifest = {
        "status": "complete_operational_smoke_not_admitted",
        "corpus_admitted": False,
        "gold_labels_used": False,
        "config": {"methods": ["R1", "R3", "R4"]},
        "outputs_sha256": {
            "R1.jsonl": file_hash(run_dir / "R1.jsonl"),
            "R3.jsonl": file_hash(run_dir / "R3.jsonl"),
            "R4.jsonl": file_hash(run_dir / "R4.jsonl"),
        }
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    try:
        yield train_json, run_dir, root
    finally:
        shutil.rmtree(root)


def test_extract_gold_urls_from_source(eval_fixture):
    train_json, _, _ = eval_fixture
    extracted = extract_gold_urls_from_source(train_json)
    assert len(extracted) == 2
    assert extracted[0]["sample_id"] == "train:0"
    assert extracted[0]["gold_evidence_urls"] == [
        "https://bbc.com/news/article-1",
        "https://nytimes.com/world/article-2",
    ]
    assert extracted[0]["fact_checking_article_url"] == "https://snopes.com/fact-check/zero"
    assert extracted[1]["gold_evidence_urls"] == ["https://reuters.com/news/story-3"]


def test_run_evaluation_metrics_and_reranker_comparison(eval_fixture):
    train_json, run_dir, root = eval_fixture
    eval_out = root / "eval_results"

    res_dir = run_evaluation(str(run_dir), str(train_json), str(eval_out))
    assert res_dir == eval_out

    # Check generated files
    assert (eval_out / "metrics_table.csv").is_file()
    assert (eval_out / "per_claim_metrics.csv").is_file()
    assert (eval_out / "evaluation_summary.json").is_file()
    assert (eval_out / "evaluation_report.md").is_file()
    assert (eval_out / "manifest.json").is_file()

    summary = json.loads((eval_out / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert summary["corpus_admitted"] is False
    assert "R1" in summary["methods"]
    assert "R3" in summary["methods"]
    assert "R4" in summary["methods"]

    # R1 check: train:0 found 2/2 gold URLs in Top-5; train:1 found 1/1 at rank 15
    r1_means = summary["methods"]["R1"]["means"]
    assert r1_means["hit@5"] == 0.5  # 1/2 claims had a gold in Top-5
    assert r1_means["hit@10"] == 0.5
    assert r1_means["hit@20"] == 1.0  # both claims had gold in Top-20
    assert r1_means["hit@100"] == 1.0
    assert summary["methods"]["R1"]["hit100_miss10_count"] == 1
    assert summary["methods"]["R1"]["hit100_miss10_sample_ids"] == ["train:1"]

    # R4 promoted train:1 from rank 15 to rank 1!
    r3_r4 = summary["r3_r4_reranker_comparison"]
    assert r3_r4["promoted_to_top10_count"] == 1
    assert r3_r4["improved_rank_count"] >= 1
    assert r3_r4["net_rank_shift_sum"] > 0


def test_refuse_unsealed_or_tampered_run(eval_fixture):
    _, run_dir, root = eval_fixture
    bad_run = root / "bad_run"
    bad_run.mkdir()
    # Missing manifest
    with pytest.raises(ValueError, match="missing manifest.json"):
        verify_sealed_run(bad_run)

    # Tampered file hash
    (bad_run / "manifest.json").write_text(json.dumps({
        "status": "complete", "gold_labels_used": False,
        "outputs_sha256": {"file.txt": "wrong_hash"}
    }), encoding="utf-8")
    (bad_run / "file.txt").write_text("content", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_sealed_run(bad_run)
