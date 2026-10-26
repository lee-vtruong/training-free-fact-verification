# M1.6.0 — Source / Development / Test Partition Plan

Date: 2026-10-26. Status: prospective plan. New partition assignments, usable group counts, correction counts, Macro-F1 power and final confirmatory N are **INCOMPLETE / not measured**. No claim IDs in this document are certified untouched merely because their index is high.

## Exposure ledger and official splits

| Existing range | Exposure for M1.6 | Permitted role |
|---|---|---|
| Fold A `train:0..49` | Historical source, inspected errors and labels | Legacy SOURCE after provenance/group audit |
| Fold B `train:50..99` | M1.3/1.4 results and examples inspected | Legacy exploratory development; never untouched test |
| Fold C `train:100..149` | M1.4.1 and subsequent analyses inspected | Legacy exploratory development |
| Fold D `train:150..249` | M1.4.2/3, rules and failure analysis inspected | Legacy exploratory development |
| Fold E `train:250..349` | M1.5 results and per-claim transitions inspected | Legacy exploratory development |
| Remaining official train | Exposure outside these runs not yet fully audited | Candidate pool only, not automatically clean |
| Official dev/test | Exposure/access status and benchmark rules require audit | Separate external evaluation only if sealed and allowed |

Claims near-duplicate to any exposed A–E claim are exposed too. Build connected groups first and propagate the most restrictive exposure status over the whole component. Do not recycle B/C/D/E into the new historical error bank in the main design: their errors already influenced the method. Keep them for named diagnostic replications. Fold A is eligible only on the SOURCE side. Its duplicate/topic components cannot spill into development/test.

Official train remains train. New SOURCE, DEVELOPMENT, TEST are internal research partitions of unexposed official train groups. Name the held-out research TEST clearly; do not call it official AVeriTeC test. Official dev/test never supply historical memory, rules, matching labels, correction explanations or development selection. Their raw content is not inspected in this phase. If any official split is used for external confirmation later, retain its identity and document access history independently.

## Group before assigning

1. Inventory dataset release/file hash, ID convention, source URLs, claim text and allowed metadata. Inspect prior run manifests/scripts/predictions, retrieval claim coverage and human access history. A missing output file does not prove a claim was unseen. Publish exposure flags and unknowns; unknown exposure excludes a group from a defensible untouched test.
2. Use **label-free** normalized claim matching: Unicode normalization/case/whitespace, exact normalized equality, token-Jaccard ≥0.60 or token 3-gram containment ≥0.50 (existing thresholds retained as candidate edges). Add canonical fact-check URL identity and event/entity/date overlap signals; same topic alone is not automatically a duplicate. Save edge method, score and rationale.
3. Form connected components, including transitive duplicates across A–E and candidate pool. Audit borderline edges using claim/source metadata only, blind to outcomes/labels; store reviewer decisions. A large component must stay together or be excluded, not split for desired N. Hash the entire edge/component map before assignment.
4. Build topic/event clusters from claim/evidence metadata with a predeclared deterministic taxonomy and URL/event/entity/date rules. No trained classifier, fitted clustering model or target-label optimization. Assign broad topics (politics, economy, health, science, other) and specific event families. Exact topic leakage cannot be ruled out by lexical deduplication alone.
5. Source and test must not share near-duplicate or specific event-family groups. Broader topic distribution overlap is allowed and reported for the main in-distribution test. Reserve a named topic-disjoint stress slice where all its specific topic families are absent from source/development; it is a separate estimand, not pooled silently with the primary test. Use frozen similarity diagnostics to report residual source↔test topic similarity without test labels or outcome-based thresholds.

Existing source-only quarantine is insufficient: the new checks cover SOURCE↔DEVELOPMENT↔TEST and within-test clustering. Avoid removing difficult target items after seeing outcomes. If a missed duplicate is discovered after scoring, report the original primary result plus a labeled sensitivity analysis; do not silently redefine the registered population.

## Proposed assignment and sealing procedure

Use stable salted SHA-256 group ordering, seed string `HEAL-M16-partition-v1-20261026`, committed before assignment. A dedicated CPU partitioning process reads allowed text/metadata, never labels or predictions; the inference process later receives only its assigned claims/evidence. Label custody is separate even when one operator runs both processes.

Provisional resource allocation (not observed counts): retain eligible legacy A SOURCE groups, add roughly **400 newly assigned SOURCE claims**, allocate **200 new DEVELOPMENT claims**, and reserve approximately **1,400 eligible TEST claims** for a +5-point Accuracy scenario with conservative secondary power. Assign whole event/duplicate groups in stable hash order to fill these targets; overshoot by a whole group is recorded. Remaining groups become an unused reserve, not a source of post-test replacements. Actual available counts may make this allocation impossible. No arbitrary “Fold F=350..449 is untouched” declaration is permitted.

The final N must follow power, class-support and group-count checks rather than this provisional quota. If too few eligible claims/groups exist, reduce the research scope or use a separately governed external holdout. Do not repurpose official dev as source or label an underpowered 100-claim slice confirmatory. Insufficient data is a NO-GO for confirmation, not permission to lower the target effect or rename explored data.

Freeze a manifest containing ID→official split→exposure→duplicate group→event family→partition, exclusion reasons, seed, dataset hash, grouping code/config hashes and all group-level assignment decisions. Freeze SOURCE, DEVELOPMENT, TEST, reserve and stress-slice IDs together. A label custodian may check prespecified per-class minimum counts after label-free assignment and before inference; no outcome is available then. If class support is inadequate, do not repeatedly search split seeds for favorable composition: stop/revise before experiments and document the revision. Test labels never become available to the memory or inference process.

Seal target claim/evidence files without `label`, `questions`, `answers`, `justification` or gold retrieval fields. Keep gold in a separate evaluator-only path. Record access/creation commands, inference allowlist and sealing hashes; use separate Unix permissions or a container input mount to enforce separation where possible. Existing runners that load one combined gold map before inference are references, not an acceptable isolation design for new HEAL execution. The present CPU audit is explicitly offline scoring and does not perform inference.

## Source memory expansion

- Expand only from the already assigned SOURCE plus its preallocated source reserve; all groups must pass provenance and exposure checks first. No target/error-triggered expansion, no test neighbors imported into memory.
- Source size schedule: first 100 new source claims, then batches of 100 up to the provisional 400 cap. The deterministic order is independent of labels. Run frozen D0, seal outputs, then source-only gold join. Record every batch, including correct cases, invalid generations and excluded explanations.
- Stop according to prespecified memory coverage/cost rules in the method proposal, not observed development/test improvement. If 400 claims cannot provide enough verified mechanisms/matched correct controls, report that limitation and revise with source-only evidence before another exploratory phase.
- Only source evidence may ground corrections. Verify original retrieval/store/cache/model/config hashes before reuse. ERRORFlow data/environments/caches remain read-only references; all new indices, memory records and review artifacts belong under this project's new M1.6 paths.
- Freeze the bank before development scoring. If development feedback causes schema/rule/selection changes, issue a new exploratory version and disclose all trials; keep the test sealed. Never append development/test gold examples to the bank.

## Prospective power and multiplicity

[CPU planning script](../scripts/plan_m16_power.py) and [scenario output](audits/M1_6_POWER_SCENARIOS.json) use paired Accuracy difference `d`, discordant-pair probability `q` and normal approximation:

`N ≈ (z_(1−α/2) + z_0.90)^2 × (q − d²) / d²`.

The 90% marginal power planning target is conservative preparation for at least 80% power to pass **both** co-primary endpoints by a union bound, only if Macro-F1 independently achieves ≥90% power too. That condition is not established by this Accuracy formula. Values below are hypothetical sensitivity scenarios, not measured model power or recommended final N. The 1.2 design effect is assumed, not estimated.

| Assumed Accuracy gain | Assumed discordance | Primary Holm bound α=.025: independent N | With hypothetical group factor 1.2 | Secondary Holm bound α=.0125: independent N | With factor 1.2 |
|---|---:|---:|---:|---:|---:|
| +0.03 | .20 | 2,746 | 3,296 | 3,160 | 3,792 |
| +0.05 | .20 | 981 | 1,178 | 1,129 | 1,355 |
| +0.08 | .20 | 376 | 452 | 433 | 520 |

The output also varies q=.10/.30. A 50–100-claim pilot is not enough to establish a few-point gain. Even the +3-point practical target may require more clean data than available. Power at a larger assumed effect is not evidence of adequate power at the minimum effect of interest.

Before the final confirmatory lock:

1. From sealed **development** runs only, estimate the full joint gold/base/H/N0 prediction distribution, discordance by group/topic, class prevalence, validity and uncertainty. Do not infer Macro-F1 power from Accuracy power; rare NEI/Conflicting classes may dominate uncertainty.
2. CPU simulate ≥10,000 prospective experiments per scenario, preserving paired four-class confusion structure and group-size distribution. Use independent simulation seeds, implement the exact planned cluster/singleton tests and Holm serial gate, and vary uncertainty in effect sizes/class prevalences rather than plugging in only the selected winner's optimistic estimate. Include null scenarios to check empirical type-I error and zero-division/missing-class behavior.
3. Evaluate effect grid Accuracy +.03/+.05/+.08 and Macro-F1 +.02/+.03/+.05; report probability of both primary rejections, secondary-family rejection probability, CI width/coverage, class counts and Monte Carlo standard error. Include paired degeneracies and invalid rates. All simulated values must be labeled prospective assumptions, not real experimental gains.
4. Choose N from the stricter of joint power ≥80%, per-class support (provisional ≥30 each; this is not by itself enough to power class-specific effects), available independent groups and GPU budget. Seek secondary power sufficient for the error-specific claim; otherwise explicitly limit that claim. Make a conservative power choice for the minimum relevant effects, not only +8 points.
5. If these requirements cannot be met, **NO-GO for confirmatory evaluation**. Finish exploratory reporting and leave the confirmatory milestone INCOMPLETE. No optional stopping, test-outcome-driven sample enlargement or repeated unsealing is permitted. Infrastructure failure recovery follows predeclared rules without looking at gold.

Primary family: H vs D1_matched, Accuracy/Macro-F1 (2 Holm tests). Gated secondary: H vs D0 and H vs matched non-error N0, both metrics (4 Holm tests). Additional formal comparisons change multiplicity and require a new pre-unsealing power calculation. [Ablation design](M1_6_ABLATION_DESIGN.md) specifies tests and family gates.

## Completion checklist for the later execution manifest

Required before confirmation: exposure audit signed; all group/event leakage checks passed; official split identities preserved; exact IDs and exclusions hashed; source bank/reviews frozen; one H selected on development; N0 matching/reporting frozen; target fields scrubbed and permissions verified; exact prompts/effective inputs/decoder/output budgets locked; analysis/null-calibration/power code and outputs committed; final N and stop rule feasible; no GPU job queued by M1.6.0. None of the unexecuted items above is represented as completed empirical work.
