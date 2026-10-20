# Base error profile

Status as of 2026-10-20: **real Base audit blocked by unavailable server data**.
The local ERRORFlow checkout contains no processed research data and only a
three-record synthetic demo prediction file. That fixture is excluded.

| Requested result | Verified research result |
|---|---|
| Source records admitted | 0 (no real input cache available for admission) |
| Accuracy / Macro-F1 | Unavailable, not zero |
| Per-class precision / recall / F1 | Unavailable |
| Confusion matrix and error pairs | Unavailable |
| Evidence-length error distribution | Unavailable |
| Confidence error distribution | Unavailable |
| Duplicate / contamination counts | Unavailable |
| Correct / incorrect real examples | Unavailable |
| Evidence-reviewed root causes | None established |

Historical report numbers and expected dataset sizes have not been replayed.
Synthetic tests check hand-computable metrics and full audit replay; their
scores are software validation and must never be presented as Base results.

## Ready-to-run analysis

The source audit emits per-example claim, dataset, evidence protocol/reference,
gold label, prediction, correctness, optional scores/confidence, confusion pair,
group, exact checkpoint/prompt and evidence word count. It computes fixed-label
Accuracy and Macro-F1, per-class metrics, gold-row confusion matrix, error-pair
counts, evidence-word and confidence bucket counts/rates, normalized duplicates,
conflicting duplicate labels and example IDs. Full examples remain in audit JSONL.

Potential categories for later evidence review: negation, temporal/entity
alignment, numeric/scope mismatch, partial support, absence versus contradiction,
and mixed evidence. These are hypotheses, not explanations of observed errors.

## Base compatibility limitation

Local `v3_prompts.py` shows `conflict_audit_v1` emphasizes credible support and
contradiction and cherry-picking. It is an audit-style instruction, not the
ordinary four-label `standard_verdict_v1` prompt. The old standard-Base protocol
document describes proposed cache paths but does not prove those files exist.
Both should retain separate identifiers. Never claim a historical audit-style
cache establishes performance of a conventional Base verifier.

The old action runner's `model_hash` fingerprints the model path string. This
cannot certify checkpoint contents. Content provenance must be recovered from
historical model manifests/snapshot revisions, or that cache is not admissible
for a strict checkpoint-matched experiment.

## Minimum additional work / inference

First restore read-only server connectivity and inventory actual artifacts.
Recover and inspect source rows, predictions, sidecars, groups and exact scoring
inputs; build a reviewed canonical conversion in the NEW project. No model calls
are necessary if cache identity and completeness can be established.

If only some cache rows are missing and compatible historical provenance is
verified, run only those missing source rows. If provenance cannot be recovered,
regenerate one frozen Base cache on the declared source population. Determine
the actual N first: label-scoring workload is N times L label forward passes
(L=4 for a four-label protocol; L=3 for a native three-label protocol). Report
token lengths, truncation, precision, estimated runtime/storage and GPU capacity
before dispatch. Do not rerun actions, routers, target evaluations or every
baseline. No inference has been authorized or launched by this milestone.
