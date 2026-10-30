"""Unified Retrieval Arms Engine for M1.8d Automated Benchmark.

Implements all 9 comparison arms (R0..R8) under an identical interface:
- R0: Canonical R4 (reference)
- R1: Qwen3 Dense Optimized (Resident GPU worker + SDPA)
- R2: E5-base-v2 Hybrid (intfloat/e5-base-v2 Dense + BM25 + Qwen3 Reranker)
- R3: BGE-base-en-v1.5 Hybrid (BAAI/bge-base-en-v1.5 Dense + BM25 + Qwen3 Reranker)
- R4: BM25-PF1000 + Qwen3 Dense
- R5: BM25-PF2000 + Qwen3 Dense
- R6: BM25-PF5000 + Qwen3 Dense
- R7: BM25-PF10000 + Qwen3 Dense
- R8: Learned Sparse Hybrid (SPLADE / BM25 fallback marked with reason)
"""
import copy
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tffv.provenance import canonical, digest, file_hash, write_json
from tffv.retrieval.data import segment
from tffv.retrieval.frozen import QwenFrozen
from tffv.retrieval.optimized_runner import OptimizedRetrievalRunner
from tffv.retrieval.ranking import BM25, Hit, dense_search, ordered, rrf
from tffv.retrieval.store_retrieval import result_record

E5_MODEL_ID = "intfloat/e5-base-v2"
BGE_MODEL_ID = "BAAI/bge-base-en-v1.5"

AVAILABLE_ARMS = [
    "R0_CANONICAL_R4",
    "R1_QWEN3_DENSE_OPT",
    "R2_E5_HYBRID",
    "R3_BGE_HYBRID",
    "R4_BM25_PF1000",
    "R5_BM25_PF2000",
    "R6_BM25_PF5000",
    "R7_BM25_PF10000",
    "R8_SPLADE_HYBRID",
]


class ArmSkippedError(Exception):
    """Raised when an arm is skipped due to missing optional weights/libraries."""
    pass


class DenseGenericEncoder:
    """Wrapper for standard sentence embedding models (E5, BGE) with model-specific prefixes."""

    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "float16", batch_size: int = 64):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        self._init_model()

    def _init_model(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch_dtype = getattr(torch, self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id, torch_dtype=torch_dtype).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def encode(self, texts: List[str], query: bool = False) -> np.ndarray:
        import torch

        # Model-specific formatting
        if "e5" in self.model_id.lower():
            prefix = "query: " if query else "passage: "
            formatted = [f"{prefix}{t}" for t in texts]
        elif "bge" in self.model_id.lower():
            if query:
                formatted = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
            else:
                formatted = texts
        else:
            formatted = texts

        vectors = []
        for start in range(0, len(formatted), self.batch_size):
            batch = formatted[start:start + self.batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                outputs = self.model(**inputs)
                # Mean pooling
                mask = inputs["attention_mask"].unsqueeze(-1)
                emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                normed = torch.nn.functional.normalize(emb.float(), p=2, dim=1).cpu().numpy()
                vectors.append(normed)
            del inputs, outputs

        return np.concatenate(vectors) if vectors else np.empty((0, 768), dtype=np.float32)

    def close(self):
        del self.model
        del self.tokenizer
        gc.collect()


class ArmRunner:
    """Unified runner for executing specific retrieval arms."""

    def __init__(
        self,
        arm_id: str,
        retrieval_cfg: Dict[str, Any],
        reranker: Any,
        qwen_encoder: Optional[Any] = None,
        canonical_cache: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        synthetic: bool = False,
    ):
        self.arm_id = arm_id
        self.config = retrieval_cfg
        self.reranker = reranker
        self.qwen_encoder = qwen_encoder
        self.canonical_cache = canonical_cache or {}
        self.device = device
        self.synthetic = synthetic
        self.external_encoder = None
        self.available = True
        self.skip_reason = None
        self._init_encoder()

    def _init_encoder(self):
        if self.synthetic:
            return
        if self.arm_id == "R2_E5_HYBRID":
            try:
                self.external_encoder = DenseGenericEncoder(E5_MODEL_ID, device=self.device)
            except Exception as e:
                self.available = False
                self.skip_reason = f"E5 initialization failed: {e}; SKIPPED_WITH_REASON"
        elif self.arm_id == "R3_BGE_HYBRID":
            try:
                self.external_encoder = DenseGenericEncoder(BGE_MODEL_ID, device=self.device)
            except Exception as e:
                self.available = False
                self.skip_reason = f"BGE initialization failed: {e}; SKIPPED_WITH_REASON"
        elif self.arm_id == "R8_SPLADE_HYBRID":
            self.available = False
            self.skip_reason = "SPLADE model weights not available offline; SKIPPED_WITH_REASON"

    def retrieve(self, claim: Dict[str, Any], documents: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Execute single-claim retrieval under this arm's contract."""
        if not self.available:
            raise ArmSkippedError(self.skip_reason)

        sid = claim.get("sample_id", "")
        # Check canonical cache for R0 reference arm
        if self.arm_id == "R0_CANONICAL_R4" and sid in self.canonical_cache:
            cached_rec = self.canonical_cache[sid]["record"]
            cached_timing = {
                "total_wall": cached_rec.get("latency_seconds", 0.0),
                "is_cached": True,
            }
            return cached_rec, cached_timing

        t0 = time.perf_counter()
        timings = {}

        # 1. Corpus segmentation
        t_seg = time.perf_counter()
        passages = segment(documents, **self.config.get("segmentation", {"words": 180, "overlap": 30}))
        passage_map = {p.passage_id: p for p in passages}
        timings["segmentation"] = time.perf_counter() - t_seg

        # 2. BM25 Search
        t_bm25 = time.perf_counter()
        bm25 = BM25(passages, **self.config.get("bm25", {"k1": 1.2, "b": 0.75}))
        b_hits = bm25.search(claim["claim"], self.config.get("candidate_depth", 200))
        timings["bm25"] = time.perf_counter() - t_bm25

        # Determine target passages for Dense search
        prefilter_k = None
        if self.arm_id == "R4_BM25_PF1000":
            prefilter_k = 1000
        elif self.arm_id == "R5_BM25_PF2000":
            prefilter_k = 2000
        elif self.arm_id == "R6_BM25_PF5000":
            prefilter_k = 5000
        elif self.arm_id == "R7_BM25_PF10000":
            prefilter_k = 10000

        if prefilter_k and len(passages) > prefilter_k:
            prefilter_hits = bm25.search(claim["claim"], prefilter_k)
            dense_target_passages = [passage_map[h.passage_id] for h in prefilter_hits]
        else:
            dense_target_passages = passages

        # 3. Dense Retrieval
        t_dense = time.perf_counter()
        doc_texts = [p.text for p in dense_target_passages]
        p_ids = [p.passage_id for p in dense_target_passages]

        if self.synthetic:
            # Synthetic score simulation
            d_hits = [Hit(p.passage_id, 0.5, {"dense_cosine": 0.5}) for p in dense_target_passages[:200]]
        else:
            if self.arm_id in ("R2_E5_HYBRID", "R3_BGE_HYBRID"):
                encoder = self.external_encoder
                corpus_matrix = encoder.encode(doc_texts, query=False)
                query_vec = encoder.encode([claim["claim"]], query=True)[0]
            else:
                encoder = self.qwen_encoder
                corpus_matrix = encoder.encode(doc_texts)
                query_vec = encoder.encode([claim["claim"]], query=True)[0]

            d_hits = dense_search(query_vec, corpus_matrix, p_ids, self.config.get("candidate_depth", 200))
        timings["dense"] = time.perf_counter() - t_dense

        # 4. RRF Fusion
        t_rrf = time.perf_counter()
        fusion_depth = self.config.get("fusion_depth", 200)
        fused_hits = rrf({"bm25": b_hits, "dense": d_hits}, fusion_depth, self.config.get("rrf_constant", 60))
        timings["rrf"] = time.perf_counter() - t_rrf

        # 5. Cross-Encoder Reranking
        t_rerank = time.perf_counter()
        candidate_texts = [passage_map[h.passage_id].text for h in fused_hits]

        if self.synthetic:
            scores = [0.8 - idx * 0.01 for idx in range(len(fused_hits))]
        else:
            scores = self.reranker.score(claim["claim"], candidate_texts)

        rerank_depth = self.config.get("rerank_depth", len(fused_hits))
        r4_hits = ordered(
            [Hit(h.passage_id, float(s), dict(h.component_scores, reranker_yes_probability=float(s)))
             for h, s in zip(fused_hits, scores)],
            rerank_depth,
        )
        timings["rerank"] = time.perf_counter() - t_rerank

        # Format final R4 record
        rec_config = dict(self.config)
        if "dedup_threshold" not in rec_config:
            rec_config["dedup_threshold"] = 0.95

        claim_dict = dict(claim)
        if "dataset" not in claim_dict:
            claim_dict["dataset"] = "AVeriTeC"
        if "split" not in claim_dict:
            claim_dict["split"] = "train"

        rec = result_record(
            claim=claim_dict,
            method="R4",
            hits=r4_hits,
            passages=passage_map,
            config=rec_config,
            timing=timings["rerank"],
        )
        rec["arm_id"] = self.arm_id
        timings["total_wall"] = time.perf_counter() - t0
        rec["latency_seconds"] = timings["total_wall"]
        rec["substage_timings"] = timings
        rec["passages_count"] = len(passages)
        rec["dense_evaluated_passages"] = len(dense_target_passages)

        return rec, timings

    def close(self):
        if self.external_encoder:
            self.external_encoder.close()
            self.external_encoder = None
