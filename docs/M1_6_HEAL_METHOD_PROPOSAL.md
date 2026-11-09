# M1.6.0 — HEAL Method Proposal

Date: 2026-10-26. Status: research design specified; M1.6 experiments, new partitions and new memory construction **INCOMPLETE / NOT RUN**. This document supersedes the M1.5 synthesis's proposed M1.6 architecture for this task. No paper or novelty claim is intended.

Update 2026-10-27: [actual data audit](M1_6_DATA_PARTITION_PLAN.md) now supplies sealed ID manifests and discovers external prediction exposure across all current train rows. There are **zero certified clean confirmatory train claims**. Adopt the user's stage order: M1.6.0 CPU Data & Integrity Audit; M1.6.1 source memory construction; M1.6.2 controlled development pilot; M1.6.3 conditional confirmation. The [resource/stopping plan](M1_6_RESOURCE_AND_STAGE_PLAN.md) replaces the former 400-source quota and 1,400-test planning assumption. None of these later GPU stages has started.

## Objective and evidence boundary

Historical Error-Augmented Fact Verification (HEAL) tests whether frozen Base wrong predictions, source gold corrections and independently verified corrective explanations improve frozen Qwen3-8B. D0 remains the original anchor. D1_matched is the strong generic primary comparator. Accuracy and four-class Macro-F1 are co-primary; success on one cannot compensate for harm on the other.

The [integrity audit](M1_6_INTEGRITY_AUDIT.md) establishes the corrected prior results. Fold D D1_matched improves Accuracy but reduces Macro-F1 relative to D0. Fold E M1 offers a small, nonsignificant signal. These findings motivate controlled experiments, not a claim that HEAL works.

The explicit M1.6 user instruction permits this design/CPU audit beyond AGENTS.md's stale M1.1 scope line. All other work rules apply: no fitting, optimizers, trained features, neural routers, learned rerankers, evaluation-label access at inference, or writes to ERRORFlow. No GPU experiments are authorized in M1.6.0.

## Proposed pipeline

1. Seal source/development/test groups before collecting new source errors. Keep official AVeriTeC splits intact; [partition plan](M1_6_DATA_PARTITION_PLAN.md) defines exposure accounting.
2. Run the frozen **D0 protocol** only on admitted source claims in a later execution phase, with the historical 512-token non-thinking configuration. Store predictions before joining source gold. Existing Fold A errors are reusable only after source/prediction/evidence/group provenance checks. This phase creates no new source memory.
3. Join source-only gold in a separate offline builder, mark actual Base errors, and assign descriptive confusion types. Reject missing predictions or missing evidence; an invalid generation is an engineering error, not a semantic confusion exemplar.
4. Verify an explanation against the exact evidence shown with the exemplar. A source gold verdict alone never establishes that a rationale is evidence-grounded. Withhold unsupported explanations; disagreement between gold and retrieved evidence is an explicit exclusion reason for the explanation-bearing common bank.
5. Retrieve a fixed exemplar slate using only target claim and target retrieved evidence, with a deterministic source-only index. Initial pilot: k=1 and claim-text BM25 with frozen tokenizer/parameters and ID tie-break. A predeclared source-only random slate is a diagnostic, not a per-target best-of search. Dense retrieval can be a separately named later exploratory variant, not an unrecorded replacement.
6. Render the same selected IDs/evidence in E0/E1/E2, varying only visible fields. Add optional Rule 1 or response structure in E3/E4/E5. Use one frozen verifier call, no learned selection and no verification-vote ensemble.
7. Seal raw responses, prompts, effective inputs, evidence and prediction hashes before the evaluator accesses target gold. Score the full admitted population, counting invalid outputs as wrong.

Target labels, gold QA, dataset justifications, target correctness and post-hoc gains never enter retrieval, exemplar selection, correction creation, retry decisions, or stopping. Development labels may score sealed runs and choose one discrete method; they cannot select an exemplar for an individual target. Confirmatory labels remain inaccessible until all predictions are sealed.

## Historical memory schema (design only)

| Field | Required content and validation |
|---|---|
| `record_id`, `schema_version` | Content-addressed record identity; versioned serialization |
| `source_claim_id`, `source_claim` | Official split-qualified ID and verbatim text; raw-file SHA-256 and original index |
| `source_group_id`, `topic_cluster_id`, `partition_id` | Immutable group/partition manifest references; only SOURCE is eligible |
| `evidence[]` | Passage ID, text, document hash, URL, store path/hash, passage offsets, retrieval rank/score, retriever model/config/revision; rendered snippet offsets/hash |
| `base_prediction` | Frozen D0 verdict, validity, complete raw output/hash, prompt/effective-input hashes, run ID, model/tokenizer revisions, decoder/cap and code provenance |
| `gold_verdict` | One of the exact four official labels; source label file hash and join key |
| `is_base_error`, `confusion_type` | Derived source-only wrong→gold pair; optional mechanism tags and annotation method/version/confidence |
| `correction` | Nullable evidence-grounded explanation, cited passage spans, mistaken inference, corrected evidence relation, limitations; no unsupported external facts |
| `correction_verification` | `unreviewed/verified/rejected`, reviewer IDs and time, span checks, consistency checks, rejection reason, explanation hash |
| `provenance` | Parent record hashes, builder commit/dirty diff, raw/evidence/index/manifest SHA-256, creation time, license/access provenance |
| `eligibility` | Error/non-error bank, exclusion reasons, duplicate/topic checks, rendered token length and common-bank membership |

An absent correction is null, never “verified” because a label is gold. Historical correct controls use the same schema with `is_base_error=false`; no invented wrong prediction is attached. Avoid copying old `gold_justification` fields into HEAL corrections. Evidence and provenance must remain adequate after compact rendering; if the decisive span cannot fit, exclude that record for all matched E0/E1/E2 conditions before target scoring.

Two reviewers should independently check source evidence→correction entailment and whether the wrong-to-gold contrast is supported; resolve disagreements without seeing target outcomes. If reviewers are unavailable, mark verification incomplete and do not run E2–E5 as “Verified Correction.” Automated span/hash checks help but do not substitute for this review. Separate correction quality from retrieval relevance.

## Component identification

The [ablation design](M1_6_ABLATION_DESIGN.md) fixes B0–B4 and E0–E5. E1−E0 isolates showing the historical wrong verdict; E2−E1 isolates the verified explanation; gold masking controls test gold-label content. E2 versus matched source-correct controls tests the contribution of source errors beyond ordinary demonstrations. Selection from error versus correct pools still changes examples, so report residual matching imbalance and avoid claiming perfect causal identification.

“D1” throughout the M1.6 matrix means the **D1_matched prompt**, never legacy D1. Corrective Memory means the E2 payload (Wrong + Gold + Verified Correction). M4 permutation voting is excluded from core HEAL. It may appear as a separately costed auxiliary baseline after the single-call design is resolved, with no attribution of its effect to historical memory.

## Resource-conscious pilot and decisions

Run the following only in a later GPU-authorized phase, with outputs under new M1.6 paths:

1. CPU: finalize group inventory, source-only provenance, reviewer eligibility, fixed slates, tokenization, prompt/effective-input snapshots, evaluator and leakage assertions. No model-quality claims from synthetic tests.
2. M1.6.1 future source stage: reuse audited own-project Fold A data first; use the sealed 923-record expansion pool in 25-claim batches under a proposed 12-hour session envelope. Stop by source-only verified coverage or time/disk/reviewer limits, not a fixed 400 quota. No ERRORFlow predictions or memory content is reused; no target performance guides expansion.
3. Small paired development pilot (first 50 claims by fixed group order): B0, B1, B2, B3, B4, E0, E1, E2 and matched non-error N0. Cache B0/B1 predictions by exact identity; never reuse predictions from a different cap/template/population. These nine arms cost 450 verifier calls plus source calls. Native/length-control companions add explicitly budgeted calls.
4. If E2 has no sign of joint benefit or fails correction-quality gates, stop expensive explanation/reasoning expansions. Otherwise run E3 then E4 on the same development slice, followed by the full planned development population for viable arms. E5 is last: long rationale generation can add cost without adding information. Keep the same hard cap in its budget-controlled comparison; a larger-cap variant is separate.
5. Choose exactly one eligible HEAL variant on development, publish all development arms and failures, then freeze the confirmatory manifest/power analysis. Do not run M4 to rescue a failed HEAL result. Run the selected method plus D0, D1_matched and matched non-error control once on the sealed test.

Source coverage gate: at least 30 evidence-verified errors overall, at least three prespecified mechanisms with at least five verified errors each, and at least 80% source-only matched-correct-control coverage. Stop at the first eligible 25-claim checkpoint; otherwise stop at resource/pool limits. These are thresholds, not achieved counts. Unsupported mechanisms stay descriptive; no target-label balancing or target-performance stopping is allowed.

Exploratory GO to full development: validity ≥99%, no more than 0.5 percentage points below B1, positive changes on both metrics, and an estimated quality/cost tradeoff that can fit the later power/budget gate. Advancement to confirmation additionally requires development Accuracy gain ≥2 points and Macro-F1 gain ≥0.02, stable direction across predefined topic slices, no observed per-class recall loss >0.05 in classes with ≥10 development examples, and mean latency ≤2× B1 plus prompt-token overhead ≤768 at k=1. Sparse-class safety remains unknown. Thresholds select candidates; pilot significance is not confirmation.

NO-GO/MODIFY: evidence-grounding failure, leakage/provenance failure, insufficient eligible controls, invalidity above limits, gain only against D0, harm to either primary metric, or inadequate attainable joint power. In that case report the milestone experiment INCOMPLETE or negative as appropriate; preserve all results and redesign using development only. Do not weaken a registered threshold after test inspection.

Confirmatory success requires positive adjusted evidence on **both** primary metrics versus D1_matched, with Accuracy improvement ≥3 points and Macro-F1 ≥0.02 as practical criteria. Incremental error-specific benefit additionally needs the prespecified matched non-error comparison. A package benefit alone does not prove a historical-error mechanism.

## Authorized work boundary and readiness

Current authorization covers CPU audits/tests, design documents, server inspection/new CPU artifacts, and safe Git commit/push. After design, CPU implementation of validators, partition manifests and prompt snapshots fits that scope. The source D0 inference, expanded retrieval requiring GPU, B/E/N pilots, E5 and confirmatory runs are proposed **future GPU experiments**, not commands launched or queued now. A later explicit GPU instruction must precede them. No background GPU job has been installed. Final sample size, partition IDs, correction-review completion and observed Macro-F1/joint power remain execution prerequisites, not completed experimental evidence.
