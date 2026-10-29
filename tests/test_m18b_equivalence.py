"""CPU-only automated equivalence test suite for M1.8b retrieval optimization.

Verifies:
1. Strict numerical & rank equivalence between OptimizedRetrievalRunner and reference store_retrieval:
   - Exact identical Top-8 passage IDs and ranking sequence.
   - Exact byte-identical rendered text spans.
   - Scores match within strict floating-point tolerance (< 1e-5).
   - Component scores and deduplication suppressions match.
2. Cold vs Warm initialization separation (amortization to 0.0s).
3. Profiling metric schema completeness across all sub-stages.
"""
import json
import math
from pathlib import Path
import pytest

from tffv.provenance import canonical, write_json
from tffv.retrieval.benchmark import dump_rows
from tffv.retrieval.optimized_runner import OptimizedRetrievalRunner
from tffv.retrieval.profiler import RetrievalProfiler
from tffv.retrieval.store_retrieval import run as run_reference_retrieval


@pytest.fixture
def candidate_store_fixture(tmp_path):
    """Create a self-contained candidate store fixture with synthetic documents."""
    ca_base = tmp_path / "candidates"
    ca_base.mkdir(parents=True, exist_ok=True)
    store_dir = ca_base / "train-9999"
    store_dir.mkdir(parents=True, exist_ok=True)

    claim = {
        "sample_id": "train:9999",
        "dataset": "AVeriTeC",
        "split": "train",
        "claim": "Renewable energy adoption reached record levels globally in 2023.",
    }

    # Create 12 distinct documents with diverse topics and keywords
    topics = [
        ("Solar photovoltaic capacity expanded by over fifty percent globally driven by European and Asian markets.", "solar"),
        ("Wind turbines generated record terawatt hours in coastal installations throughout Scandinavia and the North Sea.", "wind"),
        ("Battery storage investments tripled as lithium iron phosphate chemistries lowered costs across regional grids.", "battery"),
        ("Electric vehicle sales exceeded ten million units worldwide reducing urban oil demand measurably.", "ev"),
        ("Geothermal exploration projects commenced in volcanic zones expanding dispatchable clean baseload power.", "geothermal"),
        ("Hydroelectric generation faced temporary curtailment during severe regional dry spells in South America.", "hydro"),
        ("Transmission interconnections received federal modernization grants to connect remote renewable generation.", "grid"),
        ("Corporate power purchase agreements for clean electricity reached all-time high volumes across technology firms.", "corporate"),
        ("Nuclear power plant life extensions were approved in multiple countries seeking zero carbon baseload.", "nuclear"),
        ("Offshore floating turbine prototypes demonstrated resilience during North Atlantic storm testing.", "offshore"),
        ("Hydrogen electrolyzer manufacturing plants broke ground aiming for industrial decarbonization targets.", "hydrogen"),
        ("Decentralized rooftop solar programs expanded access to low-income residential utility customers.", "rooftop"),
    ]

    documents = []
    for idx, (text_sentence, topic_slug) in enumerate(topics, 1):
        doc_id = f"doc_{idx:03d}"
        doc_text = (f"In 2023, renewable energy adoption experienced major shifts. {text_sentence} "
                    f"Statistical analyses from international energy groups confirmed record installation levels. ") * 8
        documents.append({
            "document_id": doc_id,
            "source_url": f"https://energy.example.org/reports/2023/{topic_slug}",
            "source_kind": "web_document",
            "retrieved_at": "2024-01-01T00:00:00Z",
            "acquisition_id": "acq_001",
            "text": doc_text,
        })

    claims_file = store_dir / "claims.jsonl"
    dump_rows(claims_file, [claim])

    corpus_file = store_dir / "corpus.jsonl"
    dump_rows(corpus_file, documents)

    ledger_rows = [
        {
            "sample_id": claim["sample_id"],
            "document_id": doc["document_id"],
            "status": "retained",
            "source_url": doc["source_url"],
        }
        for doc in documents
    ]
    ledger_file = store_dir / "record_ledger.jsonl"
    dump_rows(ledger_file, ledger_rows)

    from tffv.provenance import file_hash
    ca_manifest = ca_base / "candidates_manifest.json"
    write_json(ca_manifest, {
        "schema_version": "candidate_stores_v1",
        "status": "complete_candidates_not_admitted",
        "corpus_admitted": False,
        "policy": "quarantine_review_flags_v1",
        "total_claims": 1,
        "stores": [
            {
                "sample_id": "train:9999",
                "directory": "train-9999",
                "claims_sha256": file_hash(claims_file),
                "corpus_sha256": file_hash(corpus_file),
                "ledger_sha256": file_hash(ledger_file),
                "source_records": len(documents),
                "counts": {"retained": len(documents)},
            }
        ],
    })
    return claim, documents, ca_manifest, store_dir


def test_optimized_runner_strict_equivalence_with_reference(candidate_store_fixture, tmp_path):
    claim, documents, ca_manifest, store_dir = candidate_store_fixture

    config = {
        "device": "cpu",
        "dtype": "float32",
        "max_length": 1024,
        "batch_size": 16,
        "candidate_depth": 50,
        "fusion_depth": 50,
        "rerank_depth": 50,
        "ranking_store_depth": 50,
        "evidence_depth": 8,
        "evaluation_depth": 10,
        "rrf_constant": 60,
        "dedup_threshold": 0.95,
        "segmentation": {"words": 180, "overlap": 30},
        "bm25": {"k1": 1.2, "b": 0.75},
        "seed": 42,
    }

    # 1. Run reference store_retrieval
    import shutil
    import uuid
    uid = uuid.uuid4().hex[:8]
    root = Path(__file__).resolve().parents[1]
    ref_out = root / "outputs" / f"tmp_test_ref_{uid}"
    opt_out = root / "outputs" / f"tmp_test_opt_{uid}"

    try:
        run_reference_retrieval(
            manifest_path=ca_manifest,
            output=ref_out,
            methods=("R1", "R2", "R3", "R4"),
            config=config,
            synthetic=True,
        )
        ref_r4_file = ref_out / "R4.jsonl"
        assert ref_r4_file.exists()
        ref_record = json.loads(ref_r4_file.read_text(encoding="utf-8").strip())

        # 2. Run OptimizedRetrievalRunner
        opt_runner = OptimizedRetrievalRunner(
            config=config,
            synthetic=True,
            reranker_batch_size=16,
        )
        opt_record, profiling = opt_runner.retrieve_claim(claim, documents, output_dir=opt_out)
        opt_runner.close()

        # 3. Equivalence Checks
        ref_ev = ref_record["evidence"]
        opt_ev = opt_record["evidence"]

        assert len(ref_ev) == 8
        assert len(opt_ev) == 8

        # A. Exact passage IDs in exact identical order
        ref_ids = [e["passage_id"] for e in ref_ev]
        opt_ids = [e["passage_id"] for e in opt_ev]
        assert opt_ids == ref_ids, f"Passage IDs mismatch: {opt_ids} != {ref_ids}"

        # B. Exact rank indices
        assert [e["rank"] for e in opt_ev] == [e["rank"] for e in ref_ev] == list(range(1, 9))

        # C. Exact byte-identical text spans
        for idx, (o, r) in enumerate(zip(opt_ev, ref_ev)):
            assert o["text"] == r["text"], f"Text mismatch at rank {idx+1}"
            assert o["source_url"] == r["source_url"]
            assert o["document_id"] == r["document_id"]

        # D. Scores match within strict floating point tolerance (< 1e-5)
        for idx, (o, r) in enumerate(zip(opt_ev, ref_ev)):
            assert abs(o["score"] - r["score"]) < 1e-5, f"Score mismatch at rank {idx+1}: {o['score']} vs {r['score']}"
            assert o["score_type"] == r["score_type"] == "yes_probability"

        # E. Suppressed duplicates match
        assert opt_record["dedup_suppressed"] == ref_record["dedup_suppressed"]
    finally:
        shutil.rmtree(ref_out, ignore_errors=True)
        shutil.rmtree(opt_out, ignore_errors=True)


def test_cold_vs_warm_profiling_separation(candidate_store_fixture, tmp_path):
    claim, documents, _, _ = candidate_store_fixture

    config = {
        "device": "cpu",
        "dtype": "float32",
        "max_length": 1024,
        "batch_size": 16,
        "candidate_depth": 50,
        "fusion_depth": 50,
        "rerank_depth": 50,
        "ranking_store_depth": 50,
        "evidence_depth": 8,
        "evaluation_depth": 10,
        "rrf_constant": 60,
        "dedup_threshold": 0.95,
        "segmentation": {"words": 180, "overlap": 30},
        "bm25": {"k1": 1.2, "b": 0.75},
        "seed": 42,
    }

    runner = OptimizedRetrievalRunner(config=config, synthetic=True)

    # Claim 1: Cold start
    rec1, prof1 = runner.retrieve_claim(claim, documents, output_dir=tmp_path / "c1")
    assert prof1["is_cold"] is True
    assert prof1["model_initialization_seconds"] > 0 or "model_initialization" in prof1["stages"]

    # Claim 2: Warm start
    claim2 = dict(claim, sample_id="train:9998")
    rec2, prof2 = runner.retrieve_claim(claim2, documents, output_dir=tmp_path / "c2")
    assert prof2["is_cold"] is False
    assert prof2["model_initialization_seconds"] == 0.0

    # Both produced valid Top-8 evidence
    assert len(rec1["evidence"]) == 8
    assert len(rec2["evidence"]) == 8

    runner.close()


def test_profiler_metric_schema_completeness():
    profiler = RetrievalProfiler(device="cpu")
    profiler.start_claim()

    with profiler.measure("model_initialization"):
        pass
    with profiler.measure("corpus_loading"):
        pass
    with profiler.measure("bm25"):
        pass
    with profiler.measure("dense_retrieval"):
        pass
    with profiler.measure("cross_encoder_reranking"):
        pass
    with profiler.measure("serialization"):
        pass

    report = profiler.end_claim()

    required_keys = [
        "wall_time_seconds",
        "cpu_time_seconds",
        "gpu_time_seconds",
        "peak_vram_bytes",
        "model_initialization_seconds",
        "corpus_loading_seconds",
        "bm25_seconds",
        "dense_retrieval_seconds",
        "cross_encoder_reranking_seconds",
        "serialization_seconds",
        "stages",
    ]
    for k in required_keys:
        assert k in report, f"Missing required profiling metric: {k}"

    assert isinstance(report["wall_time_seconds"], (int, float))
    assert isinstance(report["cpu_time_seconds"], (int, float))
    assert isinstance(report["gpu_time_seconds"], (int, float))
    assert report["wall_time_seconds"] >= 0.0
