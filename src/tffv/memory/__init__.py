"""Group-safe source partitions; no learned index or router."""
import re
from tffv.provenance import digest


def normalized_claim(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def grouped_folds(records, n_folds=5, seed=2026):
    if n_folds < 2:
        raise ValueError("At least two folds required")
    groups = {}
    ids = set()
    for row in records:
        if row["sample_id"] in ids:
            raise ValueError("Duplicate sample_id")
        ids.add(row["sample_id"])
        groups.setdefault(row["group_id"], []).append(row["sample_id"])
    if len(groups) < n_folds:
        raise ValueError("Too few groups for requested folds")
    sizes, assignment = [0] * n_folds, {}
    for group in sorted(groups, key=lambda g: (-len(groups[g]), digest([seed, g]))):
        fold = min(range(n_folds), key=lambda i: (sizes[i], i))
        for sid in groups[group]:
            assignment[sid] = fold
        sizes[fold] += len(groups[group])
    return assignment


def assert_isolated(memory, evaluation):
    for field in ("sample_id", "group_id"):
        if {r[field] for r in memory} & {r[field] for r in evaluation}:
            raise ValueError(f"Contamination via {field}")
    if {normalized_claim(r["claim"]) for r in memory} & {normalized_claim(r["claim"]) for r in evaluation}:
        raise ValueError("Normalized claim overlap despite supplied groups")


def source_partition(records, assignment, evaluation_fold):
    if set(assignment) != {r["sample_id"] for r in records} or evaluation_fold not in set(assignment.values()):
        raise ValueError("Incomplete assignment or absent evaluation fold")
    memory = [r for r in records if assignment[r["sample_id"]] != evaluation_fold]
    evaluation = [r for r in records if assignment[r["sample_id"]] == evaluation_fold]
    assert_isolated(memory, evaluation)
    return memory, evaluation
