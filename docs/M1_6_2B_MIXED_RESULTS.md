# M1.6.2b — Mixed gold-only k=2 pilot results

Status: user-run GPU inference COMPLETE on the canonical DEVELOPMENT50 population. Scope is EXPLORATORY NOT CONFIRMATORY; this population was exposed and M1.6.3 stays NO-GO. [Results JSON](audits/M1_6_2B_MIXED_RESULTS.json).

## Outcome summary

`MIXED_GOLD_ONLY_K2` (one Base-correct + one Base-wrong source exemplar, order correct-then-wrong, gold-only blocks): Accuracy **0.700**, Macro-F1 **0.3373**, valid rate **1.00** (50/50 parseable, no Unparseable column).

| Comparison | Accuracy delta | Accuracy CI95 | Macro-F1 delta | Macro-F1 CI95 | Exact McNemar | F1 swap p | Transitions CC/CW/WC/WW | Corrections | Regressions |
|---|---:|---|---:|---|---:|---:|---|---|---|
| Mixed − P2 | +0.040 | [-0.04, +0.12] | +0.0155 | [-0.030, +0.064] | 0.6250 | 0.6226 | 32 / 1 / 3 / 14 | `train:1853`, `train:627`, `train:1728` | `train:1935` |
| Mixed − N0 | 0.000 | [0.00, 0.00] | +0.0079 | [-0.001, +0.025] | 1.0000 | 0.4916 | 35 / 0 / 0 / 15 | none | none |

Implied baselines on the same gold: P2 Accuracy 0.660 / Macro-F1 0.3219; N0 Accuracy 0.700 / Macro-F1 0.3294.

## Reading (descriptive, no tuning on this population)

- The mixed arm is accuracy-identical to **N0** (one-correct control) on all 50 claims, and its only corrections versus P2 are exactly N0's three known corrections. The Base-wrong exemplar in the pair added **zero** corrections and one net regression-in-set; in this pilot the observed benefit comes from the correct exemplar, not from the added wrong exemplar.
- The small Macro-F1 advantage of Mixed over N0 (+0.0079) despite identical accuracy implies some of the 15 jointly-wrong claims received different wrong classes between the two arms; per-class support is too small (4–8 per class) to interpret this beyond noise. Note Macro-F1 CI touches −0.001 and `missing_class_bootstrap_frequency` 0.0166, so mixed-vs-N0 Macro-F1 is not established as positive either.
- Neither comparison's delta is significant (all McNemar p ≥ 0.5, bootstrap CIs contain 0). The repeated directionality "exposing a Base-correct source example helps, refuted-prior collapse persists" is consistent with M1.6.2a but is not confirmation.
- Failure mode unchanged: per-class F1 is Refuted 0.849 / Supported 0.500 / NEI 0.000 / Conflicting 0.000. The model keeps collapsing NEI and Conflicting into Refuted (10+4 and 3+1 cross predictions), so the binary-collar pattern reported in earlier milestones persists under mixed memory.

## Cost and measured quantities

- Prompt tokens total **154,232** (≈3,085 per claim; k=2 nearly doubles exemplar content versus k=1), output tokens **3,515**, per-claim generation latency sum **100.30 s**, `gpu_active_seconds` **null/not measured** (only summed model-worker latency).
- Resume-segment wall time (after the checkpoint-bug interruption, 49 new calls plus caching/IO) was 282.08 s as checkpointed in `state.json`; the interrupted first claim (train:810) kept its receipt and was reused, so total fresh model calls remained 50 minus 1 reused = 49 + the original claim run once. No retrieval calls, no baseline reruns.
- Monetary cost null/not measured.

## Execution integrity

- 50/50 immutable receipts; every receipt revalidated against `mixed_batch.json` hashes, prompt token IDs and the canonical R4 Top-8 evidence hashes before and after inference. `completed.json` sealed before gold release; evaluation opened DEVELOPMENT gold only post-completion (existing gold release for this exposed population).
- One interruption occurred: the initial run hit `FileExistsError` at the first `state.json` checkpoint ([fixed in `cfa2a29`](WORK_LOG.md)); recovery resumed by receipt validation, executed exactly the 49 missing calls and duplicated nothing.
- No new source batch, no TEST_QUARANTINE access, no retrieval, no protocol change. M1.6.2a canonical artifacts (P0/P1/P2/P3/N0) were read but never regenerated.

## Next-step gates (unchanged)

- The >=30 verified-error/3-mechanism M1.6.1 review gate is still unmet; source review remains human-pending.
- Broad rationale studies and any M1.6.3 confirmation remain NO-GO pending clean-provenance resolution of TEST_QUARANTINE. The 0.700/0.3373 numbers must not be quoted as improvement evidence.
