"""Non-parametric BM25, rank fusion and transparent candidate deduplication."""
import math
import re
from difflib import SequenceMatcher
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from tffv.retrieval.data import normalize, tokens


@dataclass(frozen=True)
class Hit:
    passage_id: str
    score: float
    component_scores: dict = field(default_factory=dict)


def ordered(hits, k):
    if type(k) is not int or k <= 0:
        raise ValueError("Top K must be a positive integer")
    if any(not math.isfinite(h.score) for h in hits):
        raise ValueError("Nonfinite retrieval score")
    if len({h.passage_id for h in hits}) != len(hits):
        raise ValueError("Duplicate IDs in ranking")
    return sorted(hits, key=lambda h: (-h.score, h.passage_id))[:k]


class BM25:
    def __init__(self, passages, k1=1.2, b=.75):
        if not passages or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 parameters/corpus")
        self.k1, self.b = k1, b
        self.lengths, self.postings = {}, defaultdict(dict)
        for passage in passages:
            counts = Counter(tokens(passage.text))
            self.lengths[passage.passage_id] = sum(counts.values())
            for term, count in counts.items():
                self.postings[term][passage.passage_id] = count
        if len(self.lengths) != len(passages):
            raise ValueError("Duplicate passage IDs")
        self.n = len(passages)
        self.average = sum(self.lengths.values()) / self.n or 1

    def search(self, query, k=50):
        scores = defaultdict(float)
        for term in sorted(set(tokens(query))):
            posting = self.postings.get(term, {})
            df = len(posting)
            idf = math.log(1 + (self.n - df + .5) / (df + .5))
            for pid, tf in posting.items():
                denominator = tf + self.k1 * (1 - self.b + self.b * self.lengths[pid] / self.average)
                scores[pid] += idf * tf * (self.k1 + 1) / denominator
        return ordered([Hit(pid, score, {"bm25": score}) for pid, score in scores.items()], k)


def rrf(rankings, k=40, constant=60):
    if constant <= 0:
        raise ValueError("RRF constant must be positive")
    total, components = defaultdict(float), defaultdict(dict)
    for name, hits in sorted(rankings.items()):
        if len({h.passage_id for h in hits}) != len(hits):
            raise ValueError("Duplicate IDs in RRF input")
        for rank, hit in enumerate(hits, 1):
            total[hit.passage_id] += 1 / (constant + rank)
            components[hit.passage_id].update(hit.component_scores)
            components[hit.passage_id][name + "_rank"] = rank
    return ordered([Hit(pid, value, dict(components[pid], rrf=value)) for pid, value in total.items()], k)


def near_duplicate(left, right, threshold=.95):
    if normalize(left) == normalize(right):
        return True
    # Do not collapse two passages solely because they differ in a number/date.
    if Counter(re.findall(r"\d+(?:[.,]\d+)*", left)) != Counter(re.findall(r"\d+(?:[.,]\d+)*", right)):
        return False
    protected = {"no", "not", "never", "without", "before", "after", "increase", "decrease", "more", "less"}
    left_tokens, right_tokens = tokens(left), tokens(right)
    if Counter(t for t in left_tokens if t in protected) != Counter(t for t in right_tokens if t in protected):
        return False
    a, b = set(left_tokens), set(right_tokens)
    return (bool(a and b) and len(a & b) / len(a | b) >= threshold
            and SequenceMatcher(None, normalize(left), normalize(right), autojunk=False).ratio() >= threshold)


def select_evidence(hits, passages, k=8, threshold=.95):
    if k <= 0 or not 0 < threshold <= 1:
        raise ValueError("Invalid evidence selection settings")
    selected, suppressed = [], []
    for hit in hits:
        if hit.passage_id not in passages:
            raise ValueError("Unknown passage ID")
        duplicate = next((h for h in selected if near_duplicate(passages[h.passage_id].text,
                                                               passages[hit.passage_id].text, threshold)), None)
        if duplicate:
            suppressed.append({"passage_id": hit.passage_id, "duplicate_of": duplicate.passage_id})
        else:
            selected.append(hit)
            if len(selected) == k:
                break
    return selected, suppressed


def dense_search(query_vector, matrix, passage_ids, k=50, block_size=32768):
    import numpy as np
    vector = np.asarray(query_vector, dtype=np.float32)
    if matrix.ndim != 2 or vector.shape != (matrix.shape[1],) or len(passage_ids) != len(matrix):
        raise ValueError("Embedding shape/ID alignment mismatch")
    if not np.isfinite(vector).all() or abs(float(np.linalg.norm(vector)) - 1) > 1e-4:
        raise ValueError("Query must be finite and L2 normalized")
    # Exact cosine search, bounded temporary memory, stable lexical ID ties.
    best = []
    for start in range(0, len(matrix), block_size):
        scores = np.asarray(matrix[start:start + block_size] @ vector)
        if not np.isfinite(scores).all():
            raise ValueError("Invalid corpus embeddings")
        ids = passage_ids[start:start + block_size]
        top = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), ids[i]))[:k]
        best = ordered(best + [Hit(ids[i], float(scores[i]), {"dense_cosine": float(scores[i])}) for i in top], k)
    return best
