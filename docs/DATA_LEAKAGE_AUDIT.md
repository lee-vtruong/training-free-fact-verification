# Data leakage audit

Status: infrastructure checked with synthetic inputs; real-data contamination
audit pending. No scientific population has been certified clean.

## Historical exposure

Local reports/configurations show AVeriTeC train was used for method development,
an AVeriTeC dev holdout had been opened, and CLIMATE-FEVER/SciFact had been used
for external evaluations. The reproducibility config explicitly calls AVeriTeC
results VERIFIED_BUT_TRAIN_SELECTED. Treat these populations as previously
inspected/exploratory in the new project, not untouched confirmatory holdouts.
The new audit has read specifications, not actual target labels or predictions.
Individual exposure histories and experiment chronology still need server review.

## Source out-of-fold policy

1. Build label-blind transitive near-duplicate groups before fold assignment;
   save input hash, normalization, similarity threshold and group artifact hash.
2. Assign whole groups to folds, with a fixed seed. Check both IDs and groups,
   plus normalized exact-claim overlap. `source_partition` enforces exclusion.
3. Supply only the other folds to memory/rule/demo/taxonomy generators. Do not
   generate a global error taxonomy and then filter its examples after the fact.
4. Version each artifact with allowed source IDs, source manifest, generator
   configuration and content hash. `knowledge_manifest` provides the envelope;
   actual generation is deferred. An envelope alone cannot prove generator behavior.
5. If selecting prompts/hyperparameters, use inner group-safe folds inside the
   outer source partition; evaluate the outer fold once. Existing action choices
   selected on all source data are exploratory, even if memory is fold-excluded.

The global Base profile is descriptive only. Never feed it, its examples,
confusion frequencies, or hypothesized causes into an OOF method. Recompute all
knowledge within each allowed partition. The current CLI runs Base replay only,
not OOF inference or source prompt selection.

## Evidence and cross-population contamination

Audit exact evidence text/order and whether retrieval saw gold rationale,
cited-document IDs, gold labels, verdict-bearing text or answer justifications.
AVeriTeC annotation Q/A and SciFact gold-rationale oracle evidence are distinct
from deployment-style retrieval. Do not pool those conditions. Inspect old
adapter metadata (including justification) before defining inference-visible
fields. Our verifier interface contains claim/evidence, without a gold field.

Check raw and normalized claim overlap across datasets/splits, semantic near
duplicates and reused evidence/documents. Persist overlap lists and grouping
method. Document-level leakage may require stricter grouping than claim-only
similarity. Current exact normalization cannot certify semantic completeness.
No real overlap counts are available yet. A trusted `grouping_verified` flag is
an explicit input attestation, not an automatic duplicate detector.

## External evaluation

Freeze source artifacts, prompts, model/tokenizer, scoring, ontology,
hyperparameters and label-free target inputs first. Produce and hash complete
predictions before loading target labels in a separate evaluation step.
Track exposure status and all target accesses. No target tuning or explanation
generation using target gold. Previously opened populations remain exploratory.
A dedicated external runner/seal checker is deferred; the implemented audit
rejects target-role manifests and does not claim to enforce future workflows.

## Implemented rejection checks and limitations

Tests cover whole-group and normalized-claim isolation, missing/extra/duplicate
prediction IDs, changed inputs, evidence hashes, unknown labels, prompt and
checkpoint mismatches, gold-access declarations, malformed scores/confidence,
identifier-only model hashes and immutable cache writes.
Exact claim groups from the old `v3_folds.py` are not accepted as proof of
near-duplicate grouping. Historical model-identifier hashes do not prove weights.
Unknown gold access or preprocessing provenance causes rejection, not guessing.

Remaining blockers: source files, Base predictions, original manifests,
checkpoint fingerprints, evidence lineage, near-duplicate group files and
cross-population overlap review are unavailable. No cache has passed admission.
