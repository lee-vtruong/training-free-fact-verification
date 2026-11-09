"""Milestone M1.9f: Deterministic In-Context Exemplar Selector & Formatter.

Enforces:
  1. Exact matched exemplar identities (B1 == H2, B2 == H3).
  2. Nonempty evidence passage formatting via centralized normalizer.
  3. Absolute gold hiding from verifier prompt and selection.
  4. Deterministic TF-IDF ngram tie-breaking (-score, ordinal).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from tffv.experiments.m19f_evidence import NormalizedPassage, format_evidence_block
from tffv.experiments.m19f_store import HistoricalRecord, norm_claim_text

EXPERIMENTAL_ARMS = ["B0", "B1", "H1", "H2", "H3", "B2"]

SYSTEM_INSTRUCTION_BASE = """You are a rigorous, objective fact-checking assistant.
Your task is to verify whether the target claim is supported, refuted, or unresolvable based strictly on the provided evidence passages.

Output Schema:
Explanation: <Concise step-by-step reasoning citing specific passages>
Verdict: <Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking>"""


class ClaimSimilarityIndex:
    def __init__(self, records: List[HistoricalRecord], ngram_range: Tuple[int, int] = (1, 2)):
        self.ngram_range = ngram_range
        self.doc_ids: List[str] = []
        self.doc_tokens: Dict[str, List[str]] = {}
        self.df: Counter = Counter()
        self.num_docs = 0

        for r in records:
            if not r.claim:
                continue
            tokens = self._tokenize(r.claim)
            self.doc_ids.append(r.sample_id)
            self.doc_tokens[r.sample_id] = tokens
            unique_terms = set(tokens)
            for term in unique_terms:
                self.df[term] += 1
            self.num_docs += 1

        self.idf: Dict[str, float] = {}
        for term, count in self.df.items():
            self.idf[term] = math.log((self.num_docs + 1) / (count + 1)) + 1.0

        self.doc_vectors: Dict[str, Dict[str, float]] = {}
        self.doc_norms: Dict[str, float] = {}
        for sid, tokens in self.doc_tokens.items():
            tf = Counter(tokens)
            vec: Dict[str, float] = {}
            for term, freq in tf.items():
                w = (1.0 + math.log(freq)) * self.idf.get(term, 1.0)
                vec[term] = w
            norm = math.sqrt(sum(v * v for v in vec.values()))
            self.doc_vectors[sid] = vec
            self.doc_norms[sid] = norm if norm > 0 else 1.0

    def _tokenize(self, text: str) -> List[str]:
        words = norm_claim_text(text).split()
        tokens = []
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            for i in range(len(words) - n + 1):
                tokens.append("_".join(words[i : i + n]))
        return tokens

    def query_similarity(
        self,
        query_text: str,
        candidate_ids: List[str],
        top_k: int = 2,
    ) -> List[Tuple[str, float]]:
        q_tokens = self._tokenize(query_text)
        q_tf = Counter(q_tokens)
        q_vec: Dict[str, float] = {}
        for term, freq in q_tf.items():
            if term in self.idf:
                q_vec[term] = (1.0 + math.log(freq)) * self.idf[term]

        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if q_norm == 0:
            return [(cid, 0.0) for cid in sorted(candidate_ids, key=lambda x: int(x.split(":")[1]) if ":" in x else 0)[:top_k]]

        scores = []
        for sid in candidate_ids:
            if sid not in self.doc_vectors:
                continue
            d_vec = self.doc_vectors[sid]
            dot = sum(q_vec[t] * d_vec[t] for t in q_vec if t in d_vec)
            sim = dot / (q_norm * self.doc_norms[sid])
            ord_val = int(sid.split(":")[1]) if ":" in sid else 0
            scores.append((sid, sim, ord_val))

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

        self.correct_pool = [r.sample_id for r in records if r.is_selectable_correct]
        self.wrong_pool = [r.sample_id for r in records if r.is_selectable_wrong]

        self.leakage_map: Dict[str, Set[str]] = {}
        for r in records:
            if r.leakage_group not in self.leakage_map:
                self.leakage_map[r.leakage_group] = set()
            self.leakage_map[r.leakage_group].add(r.sample_id)

        self.index = ClaimSimilarityIndex(records)

    def _filter_candidates(
        self,
        target_sample_id: str,
        target_claim: str,
        pool: List[str],
    ) -> List[str]:
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
        target_evidence: List[NormalizedPassage | Dict[str, Any]],
        matched_exemplar_ids: Optional[List[str]] = None,
    ) -> SelectionResult:
        if arm == "B0":
            chosen_ids = []
            sim_scores = []
        elif arm in ("B1", "B2"):
            if matched_exemplar_ids is None:
                raise ValueError(f"Arm {arm} requires matched_exemplar_ids from corresponding H arm!")
            chosen_ids = matched_exemplar_ids
            sim_scores = [0.0] * len(chosen_ids)
        elif arm == "H1":
            cands = self._filter_candidates(target_sample_id, target_claim, self.correct_pool)
            ranked = self.index.query_similarity(target_claim, cands, top_k=self.k)
            chosen_ids = [r[0] for r in ranked]
            sim_scores = [r[1] for r in ranked]
        elif arm == "H2":
            cands = self._filter_candidates(target_sample_id, target_claim, self.wrong_pool)
            ranked = self.index.query_similarity(target_claim, cands, top_k=self.k)
            chosen_ids = [r[0] for r in ranked]
            sim_scores = [r[1] for r in ranked]
        elif arm == "H3":
            if self.k < 2:
                cands = self._filter_candidates(target_sample_id, target_claim, self.wrong_pool)
                ranked = self.index.query_similarity(target_claim, cands, top_k=1)
                chosen_ids = [r[0] for r in ranked]
            else:
                k_correct = self.k // 2
                k_wrong = max(1, self.k - k_correct)
                cands_c = self._filter_candidates(target_sample_id, target_claim, self.correct_pool)
                ranked_c = self.index.query_similarity(target_claim, cands_c, top_k=k_correct)
                cands_w = self._filter_candidates(target_sample_id, target_claim, self.wrong_pool)
                ranked_w = self.index.query_similarity(target_claim, cands_w, top_k=k_wrong)
                chosen_ids = [r[0] for r in ranked_c] + [r[0] for r in ranked_w]
            sim_scores = [0.0] * len(chosen_ids)
        else:
            raise ValueError(f"Unknown experimental arm: {arm}")

        outcomes = [self.record_by_id[sid].correctness for sid in chosen_ids if sid in self.record_by_id]
        gold_labels = [self.record_by_id[sid].gold_verdict for sid in chosen_ids if sid in self.record_by_id]

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

    def _format_prompt(
        self,
        arm: str,
        target_claim: str,
        target_evidence: List[NormalizedPassage | Dict[str, Any]],
        exemplar_ids: List[str],
    ) -> str:
        prompt_parts = [SYSTEM_INSTRUCTION_BASE, "\n---\n"]

        if exemplar_ids and arm != "B0":
            prompt_parts.append("### Relevant Historical Verification Examples:\n")

            for idx, eid in enumerate(exemplar_ids, 1):
                rec = self.record_by_id.get(eid)
                if not rec:
                    continue

                ev_block = format_evidence_block(rec.evidence_references)
                prompt_parts.append(f"#### Example {idx}:\n")
                prompt_parts.append(f"Claim: {rec.claim}\n")
                prompt_parts.append(f"Evidence:\n{ev_block}\n")

                if arm in ("B1", "B2"):
                    prompt_parts.append(f"Verdict: {rec.gold_verdict}\n\n")
                elif arm == "H1":
                    prompt_parts.append(
                        f"Previous Model Verdict: {rec.base_verdict} (CORRECT)\n"
                        f"Verdict: {rec.gold_verdict}\n\n"
                    )
                elif arm == "H2":
                    prompt_parts.append(
                        f"Previous Model Verdict: {rec.base_verdict} (INCORRECT)\n"
                        f"Instruction: Do NOT repeat this error; based on the evidence, the correct verdict is {rec.gold_verdict}.\n"
                        f"Verdict: {rec.gold_verdict}\n\n"
                    )
                elif arm == "H3":
                    if rec.correctness == "CORRECT":
                        prompt_parts.append(
                            f"Previous Model Verdict: {rec.base_verdict} (CORRECT)\n"
                            f"Verdict: {rec.gold_verdict}\n\n"
                        )
                    else:
                        prompt_parts.append(
                            f"Previous Model Verdict: {rec.base_verdict} (INCORRECT)\n"
                            f"Instruction: Do NOT repeat this error; based on the evidence, the correct verdict is {rec.gold_verdict}.\n"
                            f"Verdict: {rec.gold_verdict}\n\n"
                        )

            prompt_parts.append("---\n")

        # Format Target Claim & Evidence
        prompt_parts.append("### Target Verification Task:\n")
        prompt_parts.append(f"Claim: {target_claim}\n")
        target_ev_block = format_evidence_block(target_evidence)
        prompt_parts.append(f"Evidence:\n{target_ev_block}\n\n")
        prompt_parts.append("Provide your reasoning and final verdict following the schema above:")

        return "".join(prompt_parts)
