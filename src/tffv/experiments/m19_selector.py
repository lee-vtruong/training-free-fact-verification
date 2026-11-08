"""
Milestone M1.9: Historical Memory Retrieval Selector & Arm Prompt Builder.

Implements deterministic claim similarity indexing, strict leakage filtering,
exemplar pool restriction, and matched prompt formatting across all six experimental arms:
- B0: Zero-Shot Base
- B1: Gold-Only ICL (Matched Wrong Examples for H2)
- H1: Correct-History ICL
- H2: Error-History ICL (Primary Method)
- H3: Mixed Historical Outcome ICL
- B2: Gold-Only ICL (Matched Mixed Examples for H3)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tffv.experiments.m19_outcome_store import HistoricalRecord, norm_claim_text


SYSTEM_INSTRUCTION_BASE = (
    "You are an expert fact-checking verifier. You are given a claim and retrieved evidence passages. "
    "Your task is to verify whether the claim is supported or refuted by the evidence, or if the evidence is insufficient or conflicting.\n\n"
    "You must choose exactly one verdict from the following four classes:\n"
    "- Supported: The evidence confirms the claim is factually true.\n"
    "- Refuted: The evidence directly contradicts or disproves the claim.\n"
    "- Not Enough Evidence: The evidence lacks sufficient factual information to prove or disprove the claim.\n"
    "- Conflicting Evidence/Cherrypicking: Multiple credible sources present contradictory findings, or evidence is selectively framed.\n\n"
    "Respond in the following format:\n"
    "Explanation: <brief factual reasoning based strictly on the provided evidence>\n"
    "Verdict: <Supported|Refuted|Not Enough Evidence|Conflicting Evidence/Cherrypicking>"
)


def tokenize_text(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for TF-IDF / BM25 ngram indexing."""
    t = text.lower()
    return re.findall(r"\b[a-z0-9]+\b", t)


class ClaimSimilarityIndex:
    """
    Deterministic TF-IDF ngram similarity index over historical claims.
    Provides fast, dependency-free, 100% deterministic claim similarity retrieval.
    """

    def __init__(self, records: List[HistoricalRecord]):
        self.records = records
        self.doc_freq: Counter[str] = Counter()
        self.doc_vectors: Dict[str, Dict[str, float]] = {}
        self.doc_norms: Dict[str, float] = {}
        self.record_by_id: Dict[str, HistoricalRecord] = {r.sample_id: r for r in records}
        self.total_docs = len(records)

        # Build vocabulary and doc frequency
        for r in records:
            tokens = set(tokenize_text(r.claim))
            for tok in tokens:
                self.doc_freq[tok] += 1

        # Build TF-IDF vectors
        for r in records:
            tokens = tokenize_text(r.claim)
            tf = Counter(tokens)
            vec: Dict[str, float] = {}
            norm_sq = 0.0
            for tok, count in tf.items():
                df = self.doc_freq.get(tok, 1)
                idf = math.log((self.total_docs + 1) / (df + 1)) + 1.0
                val = (1.0 + math.log(count)) * idf
                vec[tok] = val
                norm_sq += val * val
            self.doc_vectors[r.sample_id] = vec
            self.doc_norms[r.sample_id] = math.sqrt(norm_sq) if norm_sq > 0 else 1.0

    def query_similarity(
        self,
        query_text: str,
        candidate_ids: List[str],
        top_k: int = 2,
    ) -> List[Tuple[str, float]]:
        """
        Rank candidate claims by cosine similarity with deterministic tie-breaking.
        Tie-breaking rule: (-score, integer sample ordinal).
        """
        tokens = tokenize_text(query_text)
        q_tf = Counter(tokens)
        q_vec: Dict[str, float] = {}
        q_norm_sq = 0.0

        for tok, count in q_tf.items():
            if tok in self.doc_freq:
                df = self.doc_freq[tok]
                idf = math.log((self.total_docs + 1) / (df + 1)) + 1.0
                val = (1.0 + math.log(count)) * idf
                q_vec[tok] = val
                q_norm_sq += val * val

        q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 1.0

        scores: List[Tuple[str, float, int]] = []
        for sid in candidate_ids:
            vec = self.doc_vectors.get(sid, {})
            d_norm = self.doc_norms.get(sid, 1.0)
            dot = sum(val * vec.get(tok, 0.0) for tok, val in q_vec.items())
            sim = dot / (q_norm * d_norm) if (q_norm * d_norm) > 0 else 0.0

            # Ordinal for stable tie-breaking
            try:
                ord_val = int(sid.split(":")[1])
            except Exception:
                ord_val = 999999
            scores.append((sid, sim, ord_val))

        # Sort: descending by score, ascending by ordinal
        scores.sort(key=lambda x: (-x[1], x[2]))
        return [(s[0], round(s[1], 4)) for s in scores[:top_k]]


@dataclass
class SelectionResult:
    target_sample_id: str
    arm: str
    chosen_exemplar_ids: List[str]
    similarity_scores: List[float]
    exemplar_outcomes: List[str]
    exemplar_gold_labels: List[str]
    formatted_prompt: str
    prompt_token_count_estimate: int


class HistoricalMemorySelector:
    """
    Selects historical exemplars and formats prompts across all experimental arms.
    Enforces strict leakage isolation, token limits, and matched control identity.
    """

    def __init__(
        self,
        records: List[HistoricalRecord],
        k: int = 2,
        max_prompt_tokens: int = 3072,
    ):
        self.records = records
        self.k = k
        self.max_prompt_tokens = max_prompt_tokens
        self.record_by_id = {r.sample_id: r for r in records}

        # Filter selectable pools
        self.correct_pool = [r.sample_id for r in records if r.is_selectable_correct]
        self.wrong_pool = [r.sample_id for r in records if r.is_selectable_wrong]

        # Leakage index: map normalized claim text hash to sample_ids
        self.leakage_map: Dict[str, Set[str]] = {}
        for r in records:
            if r.leakage_group not in self.leakage_map:
                self.leakage_map[r.leakage_group] = set()
            self.leakage_map[r.leakage_group].add(r.sample_id)

        # Build similarity index
        self.index = ClaimSimilarityIndex(records)

    def _filter_candidates(
        self,
        target_sample_id: str,
        target_claim: str,
        pool: List[str],
    ) -> List[str]:
        """Exclude self, near-duplicate leakage group, and known overlaps."""
        target_leakage_group = hashlib.sha256(norm_claim_text(target_claim).encode("utf-8")).hexdigest()[:16]
        leaking_ids = self.leakage_map.get(target_leakage_group, set())

        valid_cands = []
        for sid in pool:
            if sid == target_sample_id:
                continue
            if sid in leaking_ids:
                continue
            valid_cands.append(sid)
        return valid_cands

    def select_exemplars_for_arm(
        self,
        arm: str,
        target_sample_id: str,
        target_claim: str,
        target_evidence: List[Dict[str, Any]],
        matched_exemplars: Optional[List[str]] = None,
    ) -> SelectionResult:
        """
        Select exemplars and format prompt for the specified arm.
        Arms:
        - B0: Zero-Shot Base
        - B1: Gold-Only ICL (Matched Wrong Examples for H2)
        - H1: Correct-History ICL
        - H2: Error-History ICL
        - H3: Mixed Historical Outcome ICL
        - B2: Gold-Only ICL (Matched Mixed Examples for H3)
        """
        chosen_ids: List[str] = []
        sim_scores: List[float] = []

        if arm == "B0":
            # Zero-shot: no exemplars
            chosen_ids = []
            sim_scores = []

        elif arm == "H1":
            # Select k from correct pool
            cands = self._filter_candidates(target_sample_id, target_claim, self.correct_pool)
            ranked = self.index.query_similarity(target_claim, cands, top_k=self.k)
            chosen_ids = [r[0] for r in ranked]
            sim_scores = [r[1] for r in ranked]

        elif arm == "H2":
            # Select k from wrong pool
            cands = self._filter_candidates(target_sample_id, target_claim, self.wrong_pool)
            ranked = self.index.query_similarity(target_claim, cands, top_k=self.k)
            chosen_ids = [r[0] for r in ranked]
            sim_scores = [r[1] for r in ranked]

        elif arm == "B1":
            # Matched Gold-Only ICL for H2: MUST use exact same exemplar IDs as H2!
            if matched_exemplars is not None:
                chosen_ids = list(matched_exemplars)
            else:
                # If matched exemplars not passed in, compute identical H2 selection
                cands = self._filter_candidates(target_sample_id, target_claim, self.wrong_pool)
                ranked = self.index.query_similarity(target_claim, cands, top_k=self.k)
                chosen_ids = [r[0] for r in ranked]
            sim_scores = [0.0] * len(chosen_ids)

        elif arm == "H3":
            # Mixed history: select 1 correct and 1 wrong (for k=2)
            k_correct = max(1, self.k // 2)
            k_wrong = max(1, self.k - k_correct)

            cands_c = self._filter_candidates(target_sample_id, target_claim, self.correct_pool)
            ranked_c = self.index.query_similarity(target_claim, cands_c, top_k=k_correct)

            cands_w = self._filter_candidates(target_sample_id, target_claim, self.wrong_pool)
            ranked_w = self.index.query_similarity(target_claim, cands_w, top_k=k_wrong)

            chosen_ids = [r[0] for r in ranked_c] + [r[0] for r in ranked_w]
            sim_scores = [r[1] for r in ranked_c] + [r[1] for r in ranked_w]

        elif arm == "B2":
            # Matched Gold-Only ICL for H3: MUST use exact same exemplar IDs as H3!
            if matched_exemplars is not None:
                chosen_ids = list(matched_exemplars)
            else:
                k_correct = max(1, self.k // 2)
                k_wrong = max(1, self.k - k_correct)
                cands_c = self._filter_candidates(target_sample_id, target_claim, self.correct_pool)
                ranked_c = self.index.query_similarity(target_claim, cands_c, top_k=k_correct)
                cands_w = self._filter_candidates(target_sample_id, target_claim, self.wrong_pool)
                ranked_w = self.index.query_similarity(target_claim, cands_w, top_k=k_wrong)
                chosen_ids = [r[0] for r in ranked_c] + [r[0] for r in ranked_w]
            sim_scores = [0.0] * len(chosen_ids)

        else:
            raise ValueError(f"Unknown experimental arm: {arm}")

        # Gather exemplar metadata
        outcomes = [self.record_by_id[sid].correctness for sid in chosen_ids if sid in self.record_by_id]
        gold_labels = [self.record_by_id[sid].gold_verdict for sid in chosen_ids if sid in self.record_by_id]

        # Format prompt
        formatted_prompt = self._format_prompt(
            arm=arm,
            target_claim=target_claim,
            target_evidence=target_evidence,
            exemplar_ids=chosen_ids,
        )
        token_estimate = len(formatted_prompt.split()) * 4 // 3

        return SelectionResult(
            target_sample_id=target_sample_id,
            arm=arm,
            chosen_exemplar_ids=chosen_ids,
            similarity_scores=sim_scores,
            exemplar_outcomes=outcomes,
            exemplar_gold_labels=gold_labels,
            formatted_prompt=formatted_prompt,
            prompt_token_count_estimate=token_estimate,
        )

    def _format_evidence_block(self, evidence: List[Dict[str, Any]], max_chars_per_passage: int = 350) -> str:
        blocks = []
        for i, e in enumerate(evidence[:8], 1):
            txt = (e.get("rendered_text") or e.get("quote") or "").strip()
            if len(txt) > max_chars_per_passage:
                txt = txt[:max_chars_per_passage] + "..."
            blocks.append(f"[{i}] {txt}")
        return "\n".join(blocks) if blocks else "[No evidence passages retrieved]"

    def _format_prompt(
        self,
        arm: str,
        target_claim: str,
        target_evidence: List[Dict[str, Any]],
        exemplar_ids: List[str],
    ) -> str:
        prompt_parts = [SYSTEM_INSTRUCTION_BASE, "\n---\n"]

        # Format In-Context Exemplars
        if exemplar_ids and arm != "B0":
            prompt_parts.append("### Relevant Historical Verification Examples:\n")

            for idx, eid in enumerate(exemplar_ids, 1):
                rec = self.record_by_id.get(eid)
                if not rec:
                    continue
                ev_str = self._format_evidence_block(rec.evidence_references, max_chars_per_passage=250)

                prompt_parts.append(f"Example {idx}:")
                prompt_parts.append(f"Claim: \"{rec.claim}\"")
                prompt_parts.append(f"Evidence Excerpts:\n{ev_str}")

                if arm in {"B1", "B2"}:
                    # Gold-Only ICL: Hide historical base prediction and correctness
                    prompt_parts.append(f"Verdict: {rec.gold_verdict}\n")

                elif arm == "H1":
                    # Correct-History ICL
                    prompt_parts.append(f"Model Prediction: {rec.base_verdict} (CORRECT)")
                    prompt_parts.append(f"Verdict: {rec.gold_verdict}\n")

                elif arm == "H2":
                    # Error-History ICL: Explicit negative outcome instruction
                    prompt_parts.append(f"Previous Model Prediction: {rec.base_verdict} (INCORRECT)")
                    prompt_parts.append(f"Correct Gold Verdict: {rec.gold_verdict}")
                    prompt_parts.append(
                        f"Analysis: The previous model incorrectly predicted '{rec.base_verdict}'. "
                        f"Do NOT repeat this error; based on the provided evidence, the correct verdict is '{rec.gold_verdict}'.\n"
                    )

                elif arm == "H3":
                    # Mixed Historical Outcome ICL
                    status_tag = "CORRECT" if rec.correctness == "CORRECT" else "INCORRECT"
                    prompt_parts.append(f"Previous Model Prediction: {rec.base_verdict} ({status_tag})")
                    prompt_parts.append(f"Correct Verdict: {rec.gold_verdict}\n")

            prompt_parts.append("---\n")

        # Target Claim Block
        target_ev_str = self._format_evidence_block(target_evidence, max_chars_per_passage=400)
        prompt_parts.append("### Target Claim for Verification:")
        prompt_parts.append(f"Claim: \"{target_claim}\"")
        prompt_parts.append(f"Retrieved Evidence Passages:\n{target_ev_str}\n")
        prompt_parts.append("Provide your factual reasoning and final verdict for the Target Claim:")

        return "\n".join(prompt_parts)
