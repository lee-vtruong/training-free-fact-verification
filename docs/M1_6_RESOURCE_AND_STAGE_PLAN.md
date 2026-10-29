# M1.6 — Stage, RTX 5090 Budget and Decision Plan

Date: 2026-10-27. Prospective resource plan, **not GPU authorization**. Current work is M1.6.0 CPU audit; no new retrieval, model inference, historical memory construction or queued job has started. The [data audit](M1_6_DATA_PARTITION_PLAN.md) establishes no certified clean confirmatory train population.

## Measured resources and historical timing

Read-only snapshot: one RTX 5090, **32,607 MiB**, 3 MiB used, 0% utilization; approximately **198 GiB disk free**, project outputs 184 GiB, existing embedding cache 89,203,539,712 bytes across 350 files (about 83.1 GiB). This does not reserve the GPU. New-group cache provenance/reuse is unverified; budget cold retrieval.

[Resource/provenance evidence](audits/M1_6_RESOURCE_PROVENANCE.json):

| Historical own-project retrieval | Claims | End-to-end seconds | Seconds/claim |
|---|---:|---:|---:|
| Fold C | 50 | 13,440.5 | 268.8 |
| M1.1 100-claim run | 100 | 33,156.9 | 331.6 |
| Fold D | 100 | 37,802.8 | 378.0 |
| Fold E cached path | 100 | 2,176.5 | 21.8 |

Fold E's R4 per-claim metric is only 2.43 s; that is not the end-to-end cold preparation cost. Its delivered summary copies the underlying retrieval summary, not a second independent timing replicate. The 3-claim cold smoke took about 1,006 s/claim, illustrating startup/store-size risk beyond the larger-run range.

Own-project verifier historical means: D0 Fold D 5.14 s; D1_matched 1.81 s; M0 Fold E 1.51 s. These involve different populations/templates and are planning references, not measurements of future HEAL or CoT. ERRORFlow throughput is not substituted. GPU-active time for new stages is unmeasured: table estimates use end-to-end wall time including CPU/I/O as single-GPU session reservations.

## M1.6.1 coverage-driven source expansion

SOURCE contains 973 eligible records: 50 legacy A plus a **923-record expansion pool in 529 new blocks**. The [sealed schedule](partitions/m16-execution-plan-v1/phase_schedule.json) visits one claim per block before a second pass. The pool is not a quota; there is no fixed requirement to process 400 claims.

After a later execution instruction:

1. Audit own-project A D0/B1 prompts/evidence/model/provenance before reuse. The historical count of 25 A errors is not a count of reviewed verified corrections. Do not reuse ERRORFlow predictions, gold corrections or learned components.
2. Propose a **12-hour GPU session envelope**, initially admit only **25 new source claims**. Validate candidate-store provenance/leakage, prepare frozen evidence, generate frozen D0 non-thinking with 512-token cap, seal predictions, then join source-only gold.
3. Two independent reviewers check correction explanations against the exact displayed evidence and cited spans. Resolve disagreements; unsupported corrections stay rejected/unreviewed. Reviewer availability/completion is unmeasured and is an execution gate.
4. Stop at the first 25-claim checkpoint with **at least 30 verified errors overall, at least three prespecified mechanisms with five verified examples each, and 80% source-only matched-correct-control coverage**. Mechanisms: scope/quantifier, sufficiency, attribution, negation/polarity, reported conflict, partial truth/context. No target performance is consulted.
5. Otherwise admit the next fixed batch only if its conservative projection fits remaining time, disk and reviewer capacity. Stop INCOMPLETE at a resource limit or pool exhaustion. A larger envelope requires a documented source-only revision before another batch, never a response to test outcomes.

Current SOURCE/DEVELOPMENT are externally prediction-exposed and support exploratory work. The proposed source stage is disclosed before any decision to start M1.6.1; it does not manufacture a clean confirmation population.

## Stage costs and proposed limits

[Calculator](../scripts/m16_cost_plan.py), [exact estimates](audits/M1_6_COST_ESTIMATES.json). Larger historical runs give 268.8–378.0 s/claim cold retrieval; verifier planning range is 1.5–5.2 s/call. Reservations are twice the upper total plus 15 minutes startup, **not statistical confidence bounds**. Large stores/long CoT can exceed them; update telemetry before admitting another batch.

| Scenario | New retrieval claims | Verifier calls | Estimated total wall hours | Conservative reservation |
|---|---:|---:|---:|---:|
| M1.6.0 current audit | 0 | 0 | CPU only; duration not benchmarked | 0 GPU h |
| Initial source batch | 25 | 25 | 1.88–2.66 | 5.57 h |
| Cumulative source example | 50 | 50 | 3.75–5.32 | 10.90 h |
| Cumulative source example | 100 | 100 | 7.51–10.65 | 21.54 h; exceeds 12-hour envelope |
| Initial development, nine arms | 50 | 450 | 3.92–5.90 | 12.05 h |
| Development upper example with companions | 200 | 2,800 | 16.10–25.05 | 50.34 h; exceeds 24-hour envelope |
| Hypothetical confirmation per 100 targets/four arms | 100 | 400 | 7.63–11.08 | 22.41 h; zero current targets eligible |

Do not add cumulative examples together. Proposed development envelope: **24 hours**, initially 50 claims, expansion only after benefit/cost checks. It is a stopping limit, not a promise to complete 200 claims; pilot/evidence/controls cannot be dropped silently to fit it. Exact JSON values take precedence over rounded table values. Confirmation budget is null until clean provenance, pilot benefit and joint power justify it; no 1,400-claim run is scheduled.

Planning token assumptions: 2,500–3,500 input and 64–200 **realized output** tokens/call. Every core arm retains a **512-token cap**; 64 is not a cap imposed on CoT. For 450 calls: 1.125–1.575 million input, 28,800–90,000 expected-range output, maximum output 230,400. These are assumptions, not new measurements; exact tokenizer counts precede execution, and E5 may be slower. Monetary cost is null without a supplied GPU/electricity rate.

Correction review assumption: 8–12 minutes per reviewer per candidate, two reviewers; 16–24 reviewer-minutes total. Fifty explanations cost about 13–20 reviewer-hours, excluding disputes. Actual reviewer effort is not measured. Do not label unchecked corrections verified to satisfy a budget.

## Disk, acquisition and execution controls

Existing 100-claim Fold D paths use about 7.3 GiB raw stores, 6.8 GiB candidates and 12 GiB retrieval outputs. The existing cache average suggests another 100 embeddings could add about 24 GiB. Scaling is uncertain; retaining all 923 new source intermediates could exceed current free space before development.

Propose one 25-claim preparation batch at a time, **30 GiB new scratch ceiling**, and stop before free disk falls below **100 GiB**. Keep raw predictions, exact rendered evidence, correction reviews and manifests. A later execution manifest must define archival/release of new task-owned intermediates/caches and resulting recomputation costs. No old cache, ERRORFlow environment/data, frozen M1.5 artifact or unrelated data is deleted/modified. No deletion occurs now.

Load frozen embedding/reranker and Qwen3-8B sequentially; do not assume simultaneous VRAM residency. Historical retrieval peak means are roughly 5.6–6.3 GB; new verifier/context peak is not measured. Check occupancy and peak memory per batch.

Existing acquisition plans cover three pinned train knowledge-store shards. The fold convenience runner hardcodes the first shard; source IDs ≥1,000 require a validated shard-aware CPU acquisition manifest before later retrieval. No new ZIP/store download occurred here. No cache reuse is promised without matching dataset/store/segmentation/model/config hashes.

## Gates for M1.6.2 and M1.6.3

Pilot: B0–B4, E0/E1/E2 and N0 on the locked first 50 development claims from 50 blocks. **Gold-only versus wrong+gold uses identical exemplar IDs**, evidence, order and matched fields. Preserve native D0/D1_matched, length-control companions and matched non-error controls. E3/E4 follow useful signals; E5 comes last with adequate budget/matched reasoning controls. M4 remains auxiliary.

Expand development only with validity ≥99%, positive direction on both primary metrics and feasible measured cost. Advancement toward confirmation requires development Accuracy gain ≥2 points and Macro-F1 ≥.02 over D1_matched, class-harm/uncertainty checks, one locked H and renewed joint power analysis. A pilot signal is not significance evidence on test.

Confirmation requires **all** of clean documented holdout, locked H/D0/D1, joint Accuracy/Macro-F1 power ≥80%, class support, preregistered comparison family/stopping and feasible GPU budget. Decide N after development but before any target output. Current decision: **NO-GO for confirmation on current train**, regardless of a future pilot's success, until a clean population is established. Source/development GPU stages remain unstarted and require their explicit execution instruction.
