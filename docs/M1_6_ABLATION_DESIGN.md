# M1.6.0 — Controlled Ablation Design

Date: 2026-10-26. Status: prospective design; no M1.6 GPU results. All quality, latency and token measurements for these new conditions are null/not measured. Candidate/test manifests must be sealed before confirmatory execution.

## Condition matrix

In this document **D1 = D1_matched** (`CONCISE_MATCHED_GENERIC_RULES`), not `CONCISE_GENERIC_RULES`. Historical D2 replaced generic directives; B4 below adds rules to D1 and is a new experiment. Exact rule texts appear in the [integrity audit](M1_6_INTEGRITY_AUDIT.md).

| ID | Required condition | Payload / response | Identified contrast |
|---|---|---|---|
| B0 | D0 | Original neutral Base Protocol v1.0, no demonstrations | Original anchor retained |
| B1 | D1_matched | Original matched generic two directives, one-sentence reason then verdict | Strong generic primary control |
| B2 | D1 + Rule 1 | Add media-debate directive | B2−B1; rule effect with length companion |
| B3 | D1 + Rule 2 | Add missing-context directive | B3−B1; potential negative transfer |
| B4 | D1 + Rule 1 + Rule 2 | Both exact directives | Interaction: (B4−B3)−(B2−B1), descriptive on both metrics |
| E0 | D1 + Gold-only demonstrations | Source-error claim + source evidence + gold verdict; Base wrong and explanation hidden | Conventional demonstration control on identical error IDs |
| E1 | D1 + Historical Wrong + Gold | E0 plus actual frozen D0 wrong verdict, labeled as wrong | E1−E0: incremental wrong verdict |
| E2 | D1 + Historical Wrong + Gold + Verified Correction | E1 plus evidence-grounded verified explanation | E2−E1: incremental explanation |
| E3 | D1 + Rule 1 + Corrective Memory | E2 payload plus Rule 1 | E3−E2: incremental rule |
| E4 | D1 + Corrective Memory + Short Structured Reasoning | E2 payload; short scope/evidence-relation/limitation fields then verdict | E4−E2: response structure |
| E5 | D1 + Corrective Memory + CoT | E2 payload; explicit multi-step evidence-linked rationale then verdict | E5−E2 and E5−E4: longer rationale |

“Gold-only” means no historical wrong verdict or correction explanation; it still contains source claim/evidence. “Correction” in E1 means the **source gold verdict**. “Verified Correction” in E2 means an additional **verified evidence-grounded explanation**. Corrective Memory always denotes the complete E2 payload. This prevents relabeling a gold label as a verified rationale.

E5 uses an explicit evidence-linked rationale with `enable_thinking=False`, keeping the same model chat mode. It does not silently activate native thinking. If native thinking is later investigated, name it E5_native and compare to a matching generic reasoning control at the same cap/mode; it is outside the core matrix here.

## Field and non-error controls

| Control | Construction | What it tests |
|---|---|---|
| G0 | E0 with source verdict field masked | Gold verdict value: E0−G0 |
| G1 | E1 with gold field masked; label historical prediction “previous outcome,” not “wrong” | Wrong+gold versus previous prediction alone; “wrong” wording itself reveals inequality and must not remain |
| X0 | E2 with verified explanation masked; same record/slate as E2 | Replicates E1 and checks renderer consistency |
| N0 | D1 + SOURCE-Base-correct claim/evidence/gold + verified evidence explanation, previous-outcome field withheld | Main matched non-error control for E2/selected HEAL |
| N1 | Same as N0 but show its actual correct Base verdict | Controls mere visibility of a previous model prediction |
| B1_L | Exact B1 plus length-control slots corresponding to the candidate | Added input length / placeholder sensitivity |
| R0 | E2 with predeclared source-only random error slate | Retrieval sensitivity on development only |

For selected E3/E4/E5, N0 inherits the **same rule, response format, k, slots, output cap and evidence**; only the demonstration pool changes to matched correct examples. “N0” in confirmatory analyses means this candidate-matched rendering, whose hash must be frozen. Do not compare E4 to a short-rationale N0 or E3 to a no-rule N0 and attribute all differences to errors.

E0/E1/E2 and their field-mask controls use **the same IDs in the same order**, drawn from the explanation-verified error-bank intersection. This holds correction eligibility fixed; an explanation-free arm cannot take easier/different records. G1 is diagnostic because semantic removal also changes the meaning of the previous-outcome field. Do not fabricate an incorrect verdict for a correct-source control.

Construct error↔correct source matches before target scoring: exact source gold class where available, same broad topic and evidence-sufficiency annotation, nearest claim/evidence length bins (±20%), same retrieval procedure, and matched rendered token count. Source gold can be used to build this source-only match table; target labels cannot. Freeze paired IDs, tie-breaking, replacement policy (deterministic reuse allowed and logged), distance and unmatched cases. Do not drop target claims with unavailable matches: use an empty slot in all matched arms for that target and include it in all denominators. Report match coverage/imbalance; insufficient coverage triggers redesign before confirmation.

## Prompt and compute controls

- **Frozen model:** Qwen/Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`; tokenizer, chat template, dtype, transformers/backend versions pinned at execution. All embedding/reranking models, if used later, remain frozen. No weights/features are fit.
- **Decoding:** greedy `do_sample=False`, `enable_thinking=False`; no sampling, no best-of-N, no label-dependent retry. Record effective generation config; temperature is irrelevant to greedy decoding and should not produce an unintended sampling mode. Fixed seed 42 and deterministic ordering; GPU nondeterminism is logged, not erased by rerunning bad results.
- **Output cap:** **512 tokens for every core B/E/N arm**, including original B0. This preserves the historical D0 budget and gives reasoning arms the same upper bound. Concise prompts usually stop sooner; realized tokens/latency are outcomes to report. No condition silently moves to 64 tokens. Optional 64-token deployment experiments use a distinct suffix on every comparison arm and never replace B0. E5 with a larger cap is E5_long plus matched larger-cap controls, outside the main comparison.
- **Target evidence:** frozen R4 Top-8, same passage IDs/text/order, source URLs and truncation policy (existing formatter: 1,200 chars/passage) in every paired arm. No gold QA/justifications or target label-derived retrieval. Corpus/cache/model/config hashes verified first. Adding memory never truncates target evidence. If total context fails, stop before inference and revise the whole protocol, never remove evidence per arm.
- **Exemplars:** initial k=1, same source slates for all field ablations; k=2 is a new development experiment only if resources/coverage justify it. Freeze slates from target claim/evidence before running any arm. Match target IDs/evidence across all arms, including invalid outputs.
- **Length:** reserve 512 tokenizer tokens for one memory block and up to 256 for rule/format additions. Use the pinned Qwen tokenizer, not character counts. Eligibility is determined by a full rendered E2 record fitting the slot without losing decisive cited spans. Allocate field lengths in a source-only renderer; every E0/E1/E2 uses the same source text and visible-field spans. Hidden fields are replaced by a predeclared label-free placeholder so paired rendered input lengths differ by ≤2 tokens. B2–B4 use fixed rule slots with generic placebo text in corresponding B1_L companions. Publish exact token counts, padding strings and prompt hashes.
- **Native anchors:** B0 and B1 retain exact native prompt text; they are not silently padded. Report their native package comparison and also B1_L companions. The native HEAL−B1 comparison measures the full package, including context; E1−E0, E2−E1 and HEAL−N0 are length-matched semantic comparisons. Placeholders can influence the model, so retain unpadded development sensitivity runs and do not claim that equal token counts perfectly remove all prompt effects.
- **Response structure:** B1–E3 retain the same rationale/verdict order. E4 changes only requested structure; E5 changes only the explicit rationale request. Both retain ontology, cap and chat mode. Include generic/no-memory companions with the same response structure on development to distinguish memory from reasoning effects.
- **Execution/timing:** batch size 1 for comparable per-claim timing, fixed hardware, warm-up excluded and documented. Interleave conditions by fixed randomized claim blocks to reduce load/temperature drift; do not select output order from correctness. Report end-to-end latency and generation-only latency separately if both exist; missing measurements remain null. No reuse across conditions unless full model-input/decoder identity matches, and record cache hits separately from latency measurements.

Before any GPU launch, render full messages and effective tokenized chat inputs, assert evidence equality, IDs/field masks, slot lengths, output caps, group disjointness and absence of forbidden target fields. Hash manifests and deny inference-process access to target gold files. Synthetic runner tests verify software behavior only.

## Development selection and confirmatory preregistration

Development is exploratory. Report every arm, seed/slate, run failure and changed protocol; do not hide B3 or E5 if harmful. Rank eligible HEAL candidates E1–E5 by the smaller of `ΔAccuracy/0.02` and `ΔMacro-F1/0.02` versus native B1, subject to the method proposal's validity/safety/cost gates. Tie-break lower measured total generated tokens, then latency, then condition ID. This is discrete protocol selection, with no learned parameters. Do not select per target, fit routers, or optimize error labels against test outcomes.

**Confirmatory preregistration v0 (design-level):** choose exactly one candidate H using development only, then bind its exact hash, N0 rendering/matches, final target group IDs, sample size, stopping rule, analysis code and power report in a signed/committed execution manifest before unsealing test outputs. This document alone is not a claim that a complete run-specific preregistration already exists.

Primary family P: H versus **native D1_matched**, separately for Accuracy and Macro-F1, Holm family-wise α=0.05 over these two tests. Require both deltas positive and both Holm-adjusted p<0.05. Report paired effect sizes and CIs plus the practical criteria (Accuracy ≥+0.03, Macro-F1 ≥+0.02). No test-time winner selection among multiple HEAL arms.

Secondary family S is gated: only if **both** primary nulls are rejected, test H vs original D0 and H vs matched non-error N0, each on Accuracy and Macro-F1 (four tests), Holm α=0.05 within S. This serial gate prevents a secondary significance claim when the primary package comparison fails. Report secondary estimates regardless, but label untested/not significant when the gate is closed. A gain over D0 alone cannot validate HEAL. A gain over D1 without N0 evidence cannot establish error-specific contribution.

All other ablations are exploratory estimates in the initial design. If mechanistic E0/E1/E2 tests are promoted to confirmation, enlarge/re-register the family and redo power **before** observing target outcomes. Auxiliary M4 never enters the core family; a later formal M4 claim needs a separate prespecified family/allocation, not a post-hoc exemption.

For independent singleton claims, use exact two-sided McNemar for Accuracy and a paired label-swap permutation test for Macro-F1 (100,000 fixed-seed swaps, add-one p calculation). For correlated groups, use group-level paired swaps for both primary tests; individual McNemar is a descriptive sensitivity statistic. Swap the complete paired prediction vectors within a group, preserving label dependence. Use 10,000 paired group bootstrap resamples for effect CIs, with all four classes fixed and zero-division F1=0; report missing-class bootstrap frequency. Two-sided tests plus a positive-effect requirement are frozen. The permutation exchangeability assumption must be stated; these are paired predictive comparisons, not randomized treatment assignments. Check null calibration on development before confirmation. Report both ordinary 95% effect CIs and conservative simultaneous intervals (P: 97.5% each; S: 98.75% each) to avoid treating an unadjusted interval as multiplicity-adjusted evidence.

## Mandatory result record

Per condition: admitted N and group count; correct count; Accuracy and Macro-F1; four-class support/precision/recall/F1 and confusion matrix; validity and invalid reasons; CC/CW/WC/WW plus separate invalid transitions; correction/regression IDs and source exemplar IDs; matched non-error coverage; mean/median/p95 latency; prompt/generated/total tokens; token-cap hits and rationale completeness; GPU seconds, number of calls and memory-build/retrieval costs (amortized separately). Monetary cost is null unless a documented price is supplied. Invalid outputs count as wrong; no valid-only denominator replaces the main population. No filtering on gold rarity, outcome changes, or favorable classes.

Preserve raw attempts, errors and retries in separate append-only run IDs. Any infrastructure recovery affecting decoding, template, parser or population gets a new identity and disclosed rerun reason. Missing predictions cannot disappear from N. Recompute paired metrics by identical claim/group IDs, not row position.
