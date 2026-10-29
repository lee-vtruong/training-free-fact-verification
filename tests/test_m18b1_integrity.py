"""Unit tests for M1.8b.1 Real-GPU Benchmark Suite, Equivalence Comparator, and Decision Logic."""
import json
from pathlib import Path
import pytest
import numpy as np

from tffv.retrieval.profiler import RetrievalProfiler
from scripts.benchmark_m18b1_real_gpu import (
    compare_r4_equivalence,
    generate_decision_markdown_report,
    percentile,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "m18b1_benchmark_manifest.json"


def test_manifest_schema_and_separation():
    """Verify deterministic manifest structure, categories, and split disjointness."""
    assert MANIFEST_PATH.exists()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "m18b1-benchmark-manifest-v1"

    # Part A verification
    part_a = manifest["part_a_canonical_equivalence"]
    assert part_a["count"] == 25
    assert len(part_a["claims"]) == 25
    categories = part_a["categories"]
    required_cats = {"near_ties", "long_passages", "duplicate_urls", "diverse_corpus_sizes", "dedup_suppressed"}
    assert set(categories.keys()) == required_cats
    for cat, sids in categories.items():
        assert len(sids) > 0

    # Part B verification
    part_b = manifest["part_b_uncached_throughput"]
    assert part_b["count"] == 25
    assert len(part_b["claims"]) == 25
    assert 64 in part_b["batch_sizes"]

    # Mutual exclusion between Part A and Part B
    sids_a = {c["sample_id"] for c in part_a["claims"]}
    sids_b = {c["sample_id"] for c in part_b["claims"]}
    assert sids_a.isdisjoint(sids_b)

    # Invariant: No invalid or quarantined claims in Part B
    quarantined = {"train:406", "train:1514", "train:1796", "train:2593"}
    invalid = {"train:1948"}
    assert sids_b.isdisjoint(quarantined)
    assert sids_b.isdisjoint(invalid)


def test_compare_r4_equivalence_logic():
    """Verify strict 5-point equivalence comparator logic under matching and drifting inputs."""
    rec1 = {
        "sample_id": "train:100",
        "evidence": [
            {
                "passage_id": "doc1_p0",
                "text": "First passage content here.",
                "scores": {"reranker_yes_probability": 0.85},
            },
            {
                "passage_id": "doc1_p1",
                "text": "Second passage content here.",
                "scores": {"reranker_yes_probability": 0.75},
            },
        ],
        "dedup_suppressed": [],
    }

    # 1. Exact match
    res_exact = compare_r4_equivalence(rec1, rec1, score_tolerance=1e-4)
    assert res_exact["status"] == "MATCH"
    assert res_exact["mismatches"] == []
    assert res_exact["max_score_diff"] == 0.0

    # 2. Score within tolerance
    rec_close = {
        "sample_id": "train:100",
        "evidence": [
            {
                "passage_id": "doc1_p0",
                "text": "First passage content here.",
                "scores": {"reranker_yes_probability": 0.85005},
            },
            {
                "passage_id": "doc1_p1",
                "text": "Second passage content here.",
                "scores": {"reranker_yes_probability": 0.74998},
            },
        ],
        "dedup_suppressed": [],
    }
    res_close = compare_r4_equivalence(rec_close, rec1, score_tolerance=1e-4)
    assert res_close["status"] == "MATCH"

    # 3. Score exceeds tolerance
    rec_drift = {
        "sample_id": "train:100",
        "evidence": [
            {
                "passage_id": "doc1_p0",
                "text": "First passage content here.",
                "scores": {"reranker_yes_probability": 0.86},
            },
            {
                "passage_id": "doc1_p1",
                "text": "Second passage content here.",
                "scores": {"reranker_yes_probability": 0.75},
            },
        ],
        "dedup_suppressed": [],
    }
    res_drift = compare_r4_equivalence(rec_drift, rec1, score_tolerance=1e-4)
    assert res_drift["status"] == "MISMATCH"
    assert any("exceeds tolerance" in m for m in res_drift["mismatches"])

    # 4. Passage ordering flip
    rec_flipped = {
        "sample_id": "train:100",
        "evidence": [
            {
                "passage_id": "doc1_p1",
                "text": "Second passage content here.",
                "scores": {"reranker_yes_probability": 0.75},
            },
            {
                "passage_id": "doc1_p0",
                "text": "First passage content here.",
                "scores": {"reranker_yes_probability": 0.85},
            },
        ],
        "dedup_suppressed": [],
    }
    res_flipped = compare_r4_equivalence(rec_flipped, rec1, score_tolerance=1e-4)
    assert res_flipped["status"] == "MISMATCH"
    assert any("ordering mismatch" in m for m in res_flipped["mismatches"])

    # 5. Text span mismatch
    rec_text_diff = {
        "sample_id": "train:100",
        "evidence": [
            {
                "passage_id": "doc1_p0",
                "text": "Modified passage content.",
                "scores": {"reranker_yes_probability": 0.85},
            },
            {
                "passage_id": "doc1_p1",
                "text": "Second passage content here.",
                "scores": {"reranker_yes_probability": 0.75},
            },
        ],
        "dedup_suppressed": [],
    }
    res_text_diff = compare_r4_equivalence(rec_text_diff, rec1, score_tolerance=1e-4)
    assert res_text_diff["status"] == "MISMATCH"
    assert any("Text span byte mismatch" in m for m in res_text_diff["mismatches"])

    # 6. Dedup suppression mismatch
    rec_dedup_diff = {
        "sample_id": "train:100",
        "evidence": rec1["evidence"],
        "dedup_suppressed": [{"passage_id": "suppressed_p3"}],
    }
    res_dedup_diff = compare_r4_equivalence(rec_dedup_diff, rec1, score_tolerance=1e-4)
    assert res_dedup_diff["status"] == "MISMATCH"
    assert any("Dedup suppression mismatch" in m for m in res_dedup_diff["mismatches"])


def test_profiler_gpu_metrics_schema():
    """Verify that profiler reports all required peak allocated, reserved, and system VRAM metrics."""
    prof = RetrievalProfiler(device="cpu")
    prof.start_claim()
    with prof.measure("dense_retrieval"):
        pass
    with prof.measure("cross_encoder_reranking"):
        pass
    rep = prof.end_claim()

    # Schema completeness
    assert "peak_vram_bytes" in rep
    assert "peak_vram_allocated_bytes" in rep
    assert "peak_vram_reserved_bytes" in rep
    assert "total_system_vram_bytes" in rep
    assert "wall_time_seconds" in rep
    assert "cpu_time_seconds" in rep
    assert "gpu_time_seconds" in rep
    assert "stages" in rep
    assert "cross_encoder_reranking_seconds" in rep


def test_decision_report_markdown_generation():
    """Verify Markdown decision report formatting, tables, and invariant checks."""
    dummy_summary = {
        "schema_version": "m18b1-decision-report-v1",
        "decision": "GO",
        "decision_rationale": "All 25 claims matched with 100% equivalence.",
        "equivalence_audit": {
            "total_claims": 25,
            "matched_claims": 25,
            "mismatched_claims": 0,
            "match_rate": 1.0,
            "max_score_diff_overall": 1.2e-6,
            "results": [
                {
                    "sample_id": "train:12",
                    "category": "near_ties",
                    "status": "MATCH",
                    "max_score_diff": 1.2e-6,
                }
            ],
        },
        "performance_profile": {
            "batch_size_64": {
                "cold_latency_seconds": 18.5,
                "warm_mean_seconds": 47.8,
                "warm_p50_seconds": 46.2,
                "warm_p95_seconds": 58.1,
                "throughput_claims_per_hour": 75.3,
                "substages_mean_seconds": {
                    "bm25": 1.8,
                    "dense_retrieval": 4.5,
                    "cross_encoder_reranking": 41.2,
                    "serialization": 0.005,
                },
                "peak_vram_allocated_bytes": 5 * 1024**3,
                "peak_vram_reserved_bytes": 6 * 1024**3,
                "total_system_vram_bytes": 32 * 1024**3,
            },
            "batch_size_128": {
                "evaluated": True,
                "warm_mean_seconds": 38.4,
            },
            "projection_2618_claims": {
                "mean_hours": 34.8,
                "mean_days": 1.45,
                "p95_hours": 42.2,
            },
        },
        "cache_integrity": {
            "embedding_revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
            "reranker_revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
        },
    }

    md = generate_decision_markdown_report(dummy_summary)
    assert "# M1.8b.1 Real-GPU R4 Equivalence & Throughput Benchmark Decision Report" in md
    assert "Final Verdict: 🟢 **GO" in md
    assert "25/25" in md
    assert "75.3 claims/hr" in md
    assert "34.8 hrs (1.45 days)" in md
    assert "R4-Adaptive-50" in md
    assert "NOT activated" in md


def test_percentile_computation():
    """Verify statistical percentile helper."""
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(vals, 50) == 30.0
    assert percentile(vals, 0) == 10.0
    assert percentile(vals, 100) == 50.0
    assert percentile([], 50) == 0.0


def test_optimized_runner_equivalence_with_comparator(tmp_path):
    """End-to-end integration check: run OptimizedRetrievalRunner and verify self-equivalence."""
    from tffv.retrieval.optimized_runner import OptimizedRetrievalRunner

    claim = {
        "sample_id": "train:8888",
        "dataset": "AVeriTeC",
        "split": "train",
        "claim": "Renewable energy adoption reached record levels globally in 2023.",
    }
    documents = [
        {
            "document_id": f"doc_{i}",
            "source_url": f"https://example.org/doc_{i}",
            "source_kind": "web_document",
            "retrieved_at": "2024-01-01T00:00:00Z",
            "acquisition_id": "acq_001",
            "text": f"Document content for index {i} discussing solar and wind capacity expansion in 2023.",
        }
        for i in range(15)
    ]

    cfg = {
        "device": "cpu",
        "dtype": "float32",
        "max_length": 1024,
        "batch_size": 16,
        "candidate_depth": 10,
        "fusion_depth": 10,
        "rerank_depth": 10,
        "evidence_depth": 8,
        "rrf_constant": 60,
        "dedup_threshold": 0.95,
        "segmentation": {"words": 180, "overlap": 30},
        "bm25": {"k1": 1.2, "b": 0.75},
    }

    runner1 = OptimizedRetrievalRunner(config=cfg, synthetic=True, reranker_batch_size=4)
    rec1, prof1 = runner1.retrieve_claim(claim, documents, output_dir=tmp_path / "run1")
    runner1.close()

    runner2 = OptimizedRetrievalRunner(config=cfg, synthetic=True, reranker_batch_size=4)
    rec2, prof2 = runner2.retrieve_claim(claim, documents, output_dir=tmp_path / "run2")
    runner2.close()

    comp = compare_r4_equivalence(rec1, rec2, score_tolerance=1e-5)
    assert comp["status"] == "MATCH"
    assert comp["mismatches"] == []
    assert comp["max_score_diff"] < 1e-5

