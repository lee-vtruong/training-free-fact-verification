from collections import Counter, defaultdict
from tffv.evaluation import classification_metrics
from tffv.memory import normalized_claim


def summarize(records, labels):
    result = classification_metrics([r["gold_label"] for r in records], [r["prediction"] for r in records], labels)
    distributions = {"evidence_words": defaultdict(Counter), "confidence": defaultdict(Counter)}
    for r in records:
        n = r["evidence_words"]
        length = "0" if n == 0 else "1-100" if n <= 100 else "101-500" if n <= 500 else "501+"
        c = r["confidence"]
        confidence = "missing" if c is None else "[0,.5)" if c < .5 else "[.5,.8)" if c < .8 else "[.8,1]"
        for field, bucket in (("evidence_words", length), ("confidence", confidence)):
            distributions[field][bucket]["n"] += 1
            distributions[field][bucket]["errors"] += int(not r["correct"])
    result["distributions"] = {key: {bucket: dict(count, error_rate=count["errors"] / count["n"])
                                    for bucket, count in sorted(value.items())} for key, value in distributions.items()}
    result["error_pairs"] = [dict(gold=g, predicted=p, count=count) for (g, p), count in
                             sorted(Counter((r["gold_label"], r["prediction"]) for r in records if not r["correct"]).items())]
    duplicate = defaultdict(list)
    for r in records:
        duplicate[normalized_claim(r["claim"])].append(r)
    clusters = [rs for rs in duplicate.values() if len(rs) > 1]
    result["duplicates"] = {
        "normalized_claim_clusters": len(clusters),
        "excess_records": sum(len(rs) - 1 for rs in clusters),
        "conflicting_gold_clusters": sum(len({r["gold_label"] for r in rs}) > 1 for rs in clusters),
        "split_group_clusters": sum(len({r["group_id"] for r in rs}) > 1 for rs in clusters),
        "near_duplicate_status": "supplied groups; semantic completeness requires independent review",
    }
    result["examples"] = {name: [r["sample_id"] for r in records if r["correct"] == correct][:5]
                          for name, correct in (("correct", True), ("incorrect", False))}
    result["root_causes"] = "Not inferred. Confusion pairs are observations; causal categories need evidence review."
    return result
