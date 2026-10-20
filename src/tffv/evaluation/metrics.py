"""Fixed-ontology metrics; absent-class F1 is zero, never silently omitted."""


def classification_metrics(gold, predicted, labels):
    if not gold or len(gold) != len(predicted):
        raise ValueError("Non-empty aligned predictions required")
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("Unique fixed ontology required")
    if not set(gold + predicted) <= set(labels):
        raise ValueError("Unknown label")
    matrix = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold, predicted):
        matrix[g][p] += 1
    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted_count = sum(matrix[g][label] for g in labels)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted_count) if support + predicted_count else 0.0
        per_class[label] = dict(precision=precision, recall=recall, f1=f1, support=support)
    return dict(n=len(gold), accuracy=sum(g == p for g, p in zip(gold, predicted)) / len(gold),
                macro_f1=sum(v["f1"] for v in per_class.values()) / len(labels),
                per_class=per_class, confusion_matrix=matrix, matrix_orientation="rows=gold, columns=predicted",
                zero_division=0, labels=labels)
