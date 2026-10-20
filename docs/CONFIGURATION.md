# Configuration and input contracts

Paths belong in environment variables from `.env.example` or explicit CLI
flags. No personal path is embedded in library code. The audit command accepts
one homogeneous dataset, split, evidence protocol, checkpoint, prompt and
scoring protocol per run. Keep separate protocol populations in separate runs.

## Canonical source JSONL

Every object requires string fields `sample_id`, `dataset`, `split`, `claim`,
`evidence`, `evidence_reference`, `gold_label`, `group_id`, `evidence_protocol`.
`evidence` is the exact text used for inference, including its order and
preprocessing. Empty evidence is permitted; missing evidence is rejected.
IDs must be unique. Dataset-specific label mappings require human review;
there is no implicit normalization. Preserve original IDs in sidecar provenance.

## Canonical prediction JSONL

Required: `sample_id`, `prediction`, `input_sha256`, `checkpoint_sha256`,
`prompt_sha256`, `scoring_protocol`, `gold_accessed: false`.
Optional: `label_scores` (finite values for every ontology label) and
`confidence` (finite [0,1] proxy with a manifest definition).
The input hash is `tffv.provenance.digest` of exactly
`{"claim": ..., "evidence": ..., "evidence_protocol": ...}`.
Confidence is never synthesized or treated as calibrated by default.
Prediction and source IDs must match exactly; partial caches are rejected.
Scores need not be probabilities; declare log-score aggregation and label
verbalizers. The audit does not invent scores from predictions.

## Manifest

Start from `configs/audit_manifest.template.json`; it deliberately fails until
verified values replace placeholders. Declare ontology order, population
status, expected record count, content hashes, near-duplicate grouping
provenance, exact checkpoint and prompt, evidence and scoring protocols.
The checkpoint fingerprint must cover snapshot content using a documented
algorithm, including tokenizer/config/weight files. A hash of a filesystem path
is insufficient. Historical caches with identifier-only hashes stay quarantined
until content identity can be established; do not assert a current snapshot was
used historically without evidence.

Record prompt chat template, tokenizer revision, label verbalizers, precision,
truncation/max length, generation/scoring settings, evidence construction,
original manifest and input hashes in additional manifest fields. They are
preserved in output. A conversion must record its script hash, raw input hashes,
field mapping and join proof. It cannot retrospectively repair unknown inputs.
No ERRORFlow adapter is imported or run automatically.

Grouping must be computed label-blind across the declared population, with
transitive components for near duplicates. Save algorithm/version, threshold,
normalization, input hash and group artifact hash in `grouping_provenance`.
The supplied grouping is checked for normalized exact-claim splits but semantic
near-duplicate completeness still needs a separate review. Folds are balanced
deterministically by whole supplied group, without label stratification.

## Replay and artifacts

Audit produces `audit_records.jsonl`, `metrics.json`, `folds.json`, `report.md`,
and `manifest.json` in a fresh subdirectory of this checkout's `outputs/`.
The manifest stores input and implementation hashes, full input metadata,
seed, fold count, Python version and Git revision when available.
Accuracy is correct/n. Macro-F1 averages over the full declared ontology,
including absent classes with zero F1. Confusion rows are gold, columns predicted.
Evidence bins count whitespace-delimited words, not model tokens. Missing
confidence gets an explicit bucket. Bucket rates are descriptive, not causal.

Prediction keys bind the inference-visible claim/evidence, baseline spec and
knowledge content hash. `PredictionCache` writes exclusively; key collisions
never replace a record. Frozen-verifier outputs reserve optional input/output
tokens, forward passes, peak GPU bytes and runtime; unknown costs remain null.
No workload is dispatched by this milestone.

For external evaluation, seal a label-free input manifest, source artifact and
full inference configuration before generating predictions; save and hash the
complete predictions before separately joining labels. The source audit CLI
rejects target-role manifests. A dedicated target runner and seal verification
are future work, not an implemented protection claimed here.
