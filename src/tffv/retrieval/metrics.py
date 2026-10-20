"""Relevance metrics are evaluation-only; coverage/diversity are label-free proxies."""
import math
from collections import Counter
from itertools import combinations
from urllib.parse import urlsplit
from tffv.retrieval.data import normalize, tokens


def relevance_metrics(ranked_ids, judgments):
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("Duplicate ranked passage IDs")
    if any(type(g) not in (int, float) or not math.isfinite(g) or not 0 <= g <= 4 for g in judgments.values()):
        raise ValueError("Relevance grades must be finite in [0,4]")
    relevant = {pid for pid, grade in judgments.items() if grade > 0}
    if not relevant:
        return {"recall@5": None, "recall@8": None, "recall@10": None, "mrr@10": None, "ndcg@10": None}
    result = {f"recall@{k}": len(set(ranked_ids[:k]) & relevant) / len(relevant) for k in (5, 8, 10)}
    result["mrr@10"] = next((1 / rank for rank, pid in enumerate(ranked_ids[:10], 1) if pid in relevant), 0.)
    dcg = sum((2 ** judgments.get(pid, 0) - 1) / math.log2(rank + 1) for rank, pid in enumerate(ranked_ids[:10], 1))
    ideal = sum((2 ** grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(sorted(judgments.values(), reverse=True)[:10], 1))
    result["ndcg@10"] = dcg / ideal if ideal else None
    return result


def proxies(evidence, requested=8):
    texts = [r["text"] for r in evidence]
    domains = [urlsplit(r["source_url"]).hostname for r in evidence]
    similarities = []
    for a, b in combinations(texts, 2):
        left, right = set(tokens(a)), set(tokens(b))
        similarities.append(len(left & right) / len(left | right) if left | right else 0.)
    counts = Counter(domains)
    n = len(texts)
    return {
        "evidence_count": n,
        "slot_coverage": n / requested,
        "has_evidence": bool(n),
        "exact_text_redundancy": 1 - len({normalize(t) for t in texts}) / n if n else None,
        "mean_pairwise_token_jaccard": sum(similarities) / len(similarities) if similarities else None,
        "unique_documents": len({r["document_id"] for r in evidence}),
        "unique_urls": len({r["source_url"] for r in evidence}),
        "unique_domains": len(counts),
        "domain_entropy": -sum((v / n) * math.log(v / n) for v in counts.values()) if n else None,
        "semantic_evidence_coverage": None,
    }


def summarize(rows):
    if not rows:
        raise ValueError("No evaluation rows")
    fields = sorted({k for r in rows for k in r})
    means, counts = {}, {}
    for key in fields:
        values = [r[key] for r in rows if r.get(key) is not None and type(r[key]) in (int, float, bool)]
        means[key] = sum(values) / len(values) if values else None
        counts[key] = len(values)
    return {"n_claims": len(rows), "means": means, "measured_counts": counts}
