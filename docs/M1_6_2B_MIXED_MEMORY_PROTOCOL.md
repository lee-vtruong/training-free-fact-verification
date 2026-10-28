# M1.6.2b — Cache-only mixed positive-negative memory pilot

Status: CPU preparation COMPLETE; GPU inference NOT RUN. This is an exploratory follow-up on the already exposed DEVELOPMENT50 population. It is not confirmation evidence and does not reopen TEST_QUARANTINE.

## Paired analysis of canonical M1.6.2a

All comparisons use the same 50 canonical predictions and the existing post-completion gold release. No method selection was changed from these results.

| Comparison, target minus base | Accuracy delta | Macro-F1 delta | CC/CW/WC/WW | Exact McNemar | Macro-F1 paired-swap p |
|---|---:|---:|---|---:|---:|
| P3 minus P2 | -0.040 | -0.0309 | 31 / 2 / 0 / 17 | 0.5000 | 0.49775 |
| N0 minus P2 | +0.040 | +0.0075 | 32 / 1 / 3 / 14 | 0.6250 | 0.71740 |

P3 versus P2 has accuracy bootstrap CI [-0.10, 0.00] and Macro-F1 bootstrap CI [-0.0919, 0.00]. It has no corrections and two regressions: `train:385`, `train:3026`. N0 versus P2 has accuracy CI [-0.04, 0.12] and Macro-F1 CI [-0.0376, 0.0526], with corrections `train:1853`, `train:627`, `train:1728` and regression `train:1935`. No prediction is invalid. These are descriptive exploratory observations; their p-values are not a license to tune repeatedly on this population.

## Frozen new condition

`MIXED_GOLD_ONLY_K2` is exactly D1_matched plus two historical source examples in this order:

1. One Base-correct, gold-only example.
2. One Base-wrong, gold-only example.

Each block displays only claim, fixed source evidence context and source gold verdict. It masks the Base verdict, corrective rationale and gold QA. Decoding is Qwen3-8B, frozen revision already used by the canonical run, non-thinking greedy, `max_new_tokens=512`.

The selection is frozen in [the condition config](../configs/m162b_mixed_gold_only.json): reuse each target's canonical source-BM25 error exemplar and its existing fixed same-gold/different-group correct control. The pair order is always correct then wrong. There is no new target similarity retrieval, random seed, source selection, target gold access or rule fitting.

| Source gold | Wrong ID(s), target uses | Correct ID | Mixed target count |
|---|---|---|---:|
| Refuted | train:6 (10), train:1824 (14) | train:10 | 24 |
| Supported | train:25 (12), train:978 (14) | train:718 | 26 |

No source group overlaps the target group. The cache-only preparation validates all 50 canonical R4 byte hashes and requires exactly eight passages per target. A missing/changed evidence receipt, changed prompt token IDs, mismatched source memory hash, duplicate target, invalid pair or group leakage stops before model loading. The runner contains no retrieval command.

## Controls and confounds

P2 is the one-wrong gold-only control; N0 is the one-correct gold-only control. Their 50 frozen prediction files are reused, never regenerated. Mixed k=2 changes both the number and identity of examples, so neither P2 nor N0 alone isolates every effect. The fixed pair keeps source gold class equal within a pair and pairs a correct and wrong source outcome, but it does not make the two historical claims/evidence texts identical.

The mixed prompt is necessarily longer than k=1 P2/N0. CPU preparation seals actual Qwen-token counts for every target before inference; these counts are **not measured yet** because the mixed batch has not been prepared on the server. They must be reported beside the result. Source gold labels are 26 Supported / 24 Refuted, while DEVELOPMENT target gold support is 8 Supported / 32 Refuted / 6 NEI / 4 Conflicting. This label prior and the restricted four-error/two-control source pool are material confounds. No claim is made that mixed memory covers all failure mechanisms or all target classes.

The planned result comparison is mixed minus P2 and mixed minus N0, reporting Accuracy, Macro-F1, per-class F1, validity, paired corrections/regressions, exact McNemar, paired-bootstrap CIs, prompt/output tokens, generation latency and null GPU-active/monetary cost unless actually measured. The evaluator may open DEVELOPMENT gold only after all 50 new receipts are complete.

## User-run command

The following creates exactly 50 new inference calls on an empty mixed run. It reuses canonical R4 evidence and canonical baselines; it does not retrieve or rerun P0/P1/P2/P3/N0.

```bash
ssh hvtham-server 'bash -s' <<'REMOTE'
set -euo pipefail
cd /home/stackops/whale/training-free-fact-verification
git fetch origin
git switch codex/m16-heal-design-integrity
git pull --ff-only origin codex/m16-heal-design-integrity
nohup bash scripts/run_m162b_mixed.sh > outputs/m162b-logs/launch-v1.log 2>&1 < /dev/null &
echo "PID=$!"
tail -f outputs/m162b-logs/launch-v1.log
REMOTE
```

The wrapper first creates/seals 50 mixed prompts on CPU, checks immutable R4 cache identity, then invokes Qwen only for missing mixed receipts. It resumes without duplicating completed records. Its output is `outputs/m162b-mixed-v1/evaluation-v1/results.json`. Do not run it concurrently with another GPU job or use it to expand to another population.
