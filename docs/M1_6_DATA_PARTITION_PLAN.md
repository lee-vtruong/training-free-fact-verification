# M1.6.0 — Source / Development / Test Partition Plan

Updated 2026-10-27 after conditional approval. CPU inventory, grouping and versioned ID manifests are produced. **Clean confirmatory holdout readiness remains INCOMPLETE: current train has zero certified clean test claims/groups.** No new retrieval, memory construction or inference has run. The former 400/200/1,400 allocation is withdrawn.

## Actual audit and external exposure finding

The official train file has **3,068 rows / 3,067 nonempty claims**, SHA-256 `ae5eda7c42ddf1695ef185a7ba1bc716928f5adf57103e4f78aae5f9afe00f9c`. Ordinals are preserved, including the empty claim. A M1.1 label-free export already accessed all valid claim texts. New-project outcome artifacts cover only train:0..349, but this does not establish external non-exposure.

A read-only ERRORFlow audit found `outputs/v3/averitec/b1_train_all.jsonl` and `outputs/v2/averitec/oracle_train_all_s0.jsonl`, **3,068 rows each**, with explicit train:0..3067 IDs and exact text matches covering all 3,067 nonempty current claims. The v3 manifest reports status complete and 3,068 requested/completed rows. Prediction-file hashes are `b252223d1eed377d3607968375aeec8b7844b5dd7c9260eb63188ee31f107ab4` and `b43e91ceb813cb769c18bd06e8e9007721aed6dff08efa0ae796118d17f20160`. [External ID/provenance ledger](audits/M1_6_EXTERNAL_EXPOSURE.json) contains no verdict/gold/rationale values. Nothing from ERRORFlow predictions or memory is reused as HEAL content.

The user reports no personal method-development use of train:350+, but collaborator history is unverified. **Prediction execution is KNOWN; human inspection/method use beyond that report remains UNKNOWN.** The [custody ledger](audits/M1_6_EXPOSURE_CUSTODY.json) records both separately. This is not evidence that anyone inspected every outcome or trained these model weights on the claims. Under the approved cautious exposure policy, no current train claim is certified untouched for confirmation. Official dev/test are not assumed clean and were not read/reassigned here.

Label-free grouping found **2,647 lexical components / 1,952 final leakage blocks**; largest block 102 claims. Edges: 868 lexical, 1,067 canonical fact-check article, 396 original-claim URL, 104 speaker/month/event-anchor. All transitive bridges remain. These are operational separation blocks, **not proven statistically independent groups** or complete semantic/topic isolation.

| Locked v2 role | Claims | Leakage blocks | Use |
|---|---:|---:|---|
| SOURCE | 1,013 | 550 | 973 memory-eligible, including A's 50 |
| DEVELOPMENT | 541 | 364 | Exploratory; external prediction exposure disclosed |
| LEGACY_EXPLORATORY | 283 | 202 | Other A?E-related diagnostics |
| TEST_QUARANTINE_EXPOSED | 1,230 | 835 | Former candidates; prohibited as clean test; not recycled |
| QUARANTINE_INVALID | 1 | 1 | Empty claim, ordinal retained |
| TEST_SEALED | **0** | **0** | Explicit empty clean-test manifest |
| Total | 3,068 | 1,952 | Full ID inventory |

The 973 eligible source records are 50 A records in 21 blocks plus **923 new records in 529 blocks**. Forty other A-linked component members remain source-side to keep groups intact but cannot enter memory. B/C/D/E labels never become new source demonstrations.

[Manifest index](partitions/m16-v2/README.md), [v2 seal](partitions/m16-v2/seal.json), [all IDs/exposure flags](partitions/m16-v2/claims_manifest.json), [groups](partitions/m16-v2/groups_manifest.json), [edges](partitions/m16-v2/group_edges.json), [source](partitions/m16-v2/source_manifest.json), [development](partitions/m16-v2/development_manifest.json), [empty test](partitions/m16-v2/test_sealed_manifest.json) and [test quarantine](partitions/m16-v2/test_quarantine_exposed_manifest.json) are locked by exact byte hashes. The initial project-only v1 candidate allocation is invalidated before any retrieval/inference; [its seal](partitions/m16-v1-invalidated/seal.json) and original server files remain preserved. The v2 exposure update does not change the original group seed or recycle excluded test candidates.

[Input inventory](audits/M1_6_DATA_INVENTORY.json) catalogs 631 ID-bearing files and 144 metadata files. Eight project output logs contain no explicit train:350+ references; absence does not prove no external use. The metadata projector exports no labels/QA/justifications or annotation-derived claim types/strategies. Fact-check URLs serve only as equality keys for grouping; they must not enter future model inputs as gold information.

Checks: all original IDs unique/covered; zero cross-partition edges/groups; zero development/clean-test links to A?E; zero memory eligibility outside SOURCE; no external prediction-exposed clean test; all sealed file hashes match. Run `python scripts/m16_validate_seal.py docs/partitions/m16-v2`. Hashes preserve bytes across Windows/Linux through .gitattributes. Filesystem write protection is an additional guard, not a claim of untouched data.

The requested resource/stopping plan is in [M1_6_RESOURCE_AND_STAGE_PLAN.md](M1_6_RESOURCE_AND_STAGE_PLAN.md); the [fixed source/development schedule](partitions/m16-execution-plan-v1/phase_schedule.json) is sealed separately. Actual new HEAL quality, correction-review counts and joint power remain null/not measured.

## Adopted stage order

1. **M1.6.0 Data & Integrity Audit:** current CPU work; inventory/manifests/cost plan delivered, clean-test readiness INCOMPLETE.
2. **M1.6.1 Historical Memory Construction:** future source-only D0 and reviewed wrong/gold/correction bank, with coverage/resource stopping. No fixed 400 quota.
3. **M1.6.2 Controlled HEAL Pilot:** future development-only selection of one H; D0/D1 and same-ID gold-only vs wrong+gold controls retained.
4. **M1.6.3 Confirmatory Evaluation:** only after both-metric pilot benefit, clean provenance, joint power and feasible GPU budget. **NO-GO on current train.** No 1,400-claim retrieval/inference is scheduled.

A future distinct holdout requires explicit dataset/custody authorization and its own exposure audit. Otherwise train-based work must remain exploratory. No automatic dataset substitution/download occurs here.

## Exposure ledger and official splits

| Existing range | Exposure for M1.6 | Permitted role |
|---|---|---|
| Fold A `train:0..49` | Historical source, inspected errors and labels | Legacy SOURCE after provenance/group audit |
| Fold B `train:50..99` | M1.3/1.4 results and examples inspected | Legacy exploratory development; never untouched test |
| Fold C `train:100..149` | M1.4.1 and subsequent analyses inspected | Legacy exploratory development |
| Fold D `train:150..249` | M1.4.2/3, rules and failure analysis inspected | Legacy exploratory development |
| Fold E `train:250..349` | M1.5 results and per-claim transitions inspected | Legacy exploratory development |
| Remaining official train | All have ERRORFlow prediction artifacts; human method-use history partly UNKNOWN | Exploratory source/development or exposed-test quarantine only |
| Official dev/test | Exposure/access status and benchmark rules require audit | Separate external evaluation only if sealed and allowed |

Claims near-duplicate to any exposed A–E claim are exposed too. Build connected groups first and propagate the most restrictive exposure status over the whole component. Do not recycle B/C/D/E into the new historical error bank in the main design: their errors already influenced the method. Keep them for named diagnostic replications. Fold A is eligible only on the SOURCE side. Its duplicate/topic components cannot spill into development/test.

Official train remains train. New SOURCE, DEVELOPMENT, TEST are internal research partitions of unexposed official train groups. Name the held-out research TEST clearly; do not call it official AVeriTeC test. Official dev/test never supply historical memory, rules, matching labels, correction explanations or development selection. Their raw content is not inspected in this phase. If any official split is used for external confirmation later, retain its identity and document access history independently.

## Group before assigning

1. Inventory dataset release/file hash, ID convention, source URLs, claim text and allowed metadata. Inspect prior run manifests/scripts/predictions, retrieval claim coverage and human access history. A missing output file does not prove a claim was unseen. Publish exposure flags and unknowns; unknown exposure excludes a group from a defensible untouched test.
2. Use **label-free** normalized claim matching: Unicode normalization/case/whitespace, exact normalized equality, token-Jaccard ≥0.60 or token 3-gram containment ≥0.50 (existing thresholds retained as candidate edges). Add canonical fact-check URL identity and event/entity/date overlap signals; same topic alone is not automatically a duplicate. Save edge method, score and rationale.
3. Form connected components, including transitive duplicates across A–E and candidate pool. Audit borderline edges using claim/source metadata only, blind to outcomes/labels; store reviewer decisions. A large component must stay together or be excluded, not split for desired N. Hash the entire edge/component map before assignment.
4. Applied event blocks use canonical URL equality and fixed speaker/month/anchor rules, with day-month-year date parsing and year-month-day fallback. Further topic-disjoint stress-slice construction remains prospective. Build any later topic/event diagnostics from label-free metadata with a predeclared deterministic taxonomy. No trained classifier, fitted clustering model or target-label optimization. Assign broad topics (politics, economy, health, science, other) and specific event families. Exact topic leakage cannot be ruled out by lexical deduplication alone.
5. Source and test must not share near-duplicate or specific event-family groups. Broader topic distribution overlap is allowed and reported for the main in-distribution test. Reserve a named topic-disjoint stress slice where all its specific topic families are absent from source/development; it is a separate estimand, not pooled silently with the primary test. Use frozen similarity diagnostics to report residual source↔test topic similarity without test labels or outcome-based thresholds.

Existing source-only quarantine is insufficient: the new checks cover SOURCE↔DEVELOPMENT↔TEST and within-test clustering. Avoid removing difficult target items after seeing outcomes. If a missed duplicate is discovered after scoring, report the original primary result plus a labeled sensitivity analysis; do not silently redefine the registered population.

## Proposed assignment and sealing procedure

Use stable salted SHA-256 group ordering, seed string `HEAL-M16-partition-v1-20261026`, committed before assignment. A dedicated CPU partitioning process reads allowed text/metadata, never labels or predictions; the inference process later receives only its assigned claims/evidence. Label custody is separate even when one operator runs both processes.

Applied policy: group-hash intervals .30 SOURCE / .20 DEVELOPMENT / .50 candidate test, with the previously committed seed. A?E propagation overrides these roles. The external audit then excludes all would-be clean-test groups into TEST_QUARANTINE_EXPOSED. v2 actual counts above replace the former provisional quotas. A sealed candidate reservoir never authorizes confirmation or fixes its sample size before development.

The final N must follow power, class-support and group-count checks rather than this provisional quota. If too few eligible claims/groups exist, reduce the research scope or use a separately governed external holdout. Do not repurpose official dev as source or label an underpowered 100-claim slice confirmatory. Insufficient data is a NO-GO for confirmation, not permission to lower the target effect or rename explored data.

Freeze a manifest containing ID→official split→exposure→duplicate group→event family→partition, exclusion reasons, seed, dataset hash, grouping code/config hashes and all group-level assignment decisions. Freeze SOURCE, DEVELOPMENT, TEST, reserve and stress-slice IDs together. A label custodian may check prespecified per-class minimum counts after label-free assignment and before inference; no outcome is available then. If class support is inadequate, do not repeatedly search split seeds for favorable composition: stop/revise before experiments and document the revision. Test labels never become available to the memory or inference process.

Seal target claim/evidence files without `label`, `questions`, `answers`, `justification` or gold retrieval fields. Keep gold in a separate evaluator-only path. Record access/creation commands, inference allowlist and sealing hashes; use separate Unix permissions or a container input mount to enforce separation where possible. Existing runners that load one combined gold map before inference are references, not an acceptable isolation design for new HEAL execution. The present CPU audit is explicitly offline scoring and does not perform inference.

## Source memory expansion

- Expand only from the already assigned SOURCE plus its preallocated source reserve; all groups must pass provenance and exposure checks first. No target/error-triggered expansion, no test neighbors imported into memory.
- Source schedule: fixed round-robin block order, 25-claim batches, proposed 12-hour GPU session envelope; stop by source-only coverage or resource/reviewer bounds. The pool is 923 new eligible records, not a quota. Run frozen D0 and seal outputs before source-only gold join. Record correct cases, invalid generations and rejected explanations.
- Stop at the first checkpoint with at least 30 verified error records, three prespecified mechanisms with at least five verified records each, and 80% source-only correct-control matching coverage; otherwise stop before the next batch exceeds time/disk/reviewer limits or the pool is exhausted. No development/test-performance stopping.
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
