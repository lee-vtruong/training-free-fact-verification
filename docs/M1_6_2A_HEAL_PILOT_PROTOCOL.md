# M1.6.2a — Minimal exploratory HEAL pilot protocol

2026-10-27. CPU implementation COMPLETE. Source bank and fixed DEV50 slates prepared without target labels/outcomes. **No GPU retrieval/inference has been launched by the agent; no pilot results exist.** Human review pending; P4 BLOCKED. This protocol narrows the earlier broad HEAL plan under the user's explicit instruction.

## Question, population and freeze

Does revealing a frozen Base's historical wrong verdict improve verification beyond ordinary gold-only demonstrations and D1_matched? Primary comparisons, in order: **P3 versus P2**, then **P3 versus P1**. P0 retains total improvement context. P4 versus P3 is eligible only if all identical selected error exemplars have two real human approvals before launch.

Use exactly `development_pilot_claims`, the first **50** entries of the sealed v2 DEVELOPMENT execution schedule, representing 50 distinct sealed leakage groups. SOURCE/DEVELOPMENT assignments, schedule and TEST_QUARANTINE_EXPOSED remain unchanged. Clean confirmatory eligibility remains zero. All outcomes are **EXPLORATORY**, including any statistically significant pilot result.

| Condition | Prompt | Historical information |
|---|---|---|
| P0 | Original native D0 builder | None |
| P1 | Original native D1_matched builder | None |
| P2 | D1 plus one source error example | Claim, fixed evidence context, gold verdict; wrong-verdict field masked |
| P3 | Same P2 source ID/context/order | Adds actual Base wrong verdict, explicitly marked historical/incorrect |
| P4 | Same P2/P3 source ID/context/order | Adds VERIFIED correction; currently BLOCKED |
| N0 | D1 plus one correct source control | Gold-only; same gold class, different source group, length matched |

**k=1, seed=42**, fixed before target retrieval. No k/seed sweep. P2 and P3 both mask the correction-explanation field. Unreviewed drafts can help source-only eligibility/context selection but never appear as a “verified correction.” N0 controls are semantically human-unreviewed; this limitation accompanies its results. The immediate P3/P2 comparison isolates disclosure of the wrong-verdict field conditional on this fixed source bank, not all possible error banks.

## Source-only selection and prompt-budget controls

The 75-record bank contains five pending cited error drafts and 28 assistant-insufficient errors, which are excluded. Same-source-gold, different-group correct controls must differ by at most 20% in unpadded demonstration tokens. An error demonstration and its control must each fit 768 tokens under the byte-pinned actual Qwen tokenizer. Ties use stable source ID order. This excludes train:11 without looking at development outcomes.

Eligible errors: **train:6, train:25, train:978, train:1824**. Fixed lexical BM25 over their source claim text (`k1=1.2, b=.75`, no learned parameters) selects one for each target claim. Actual pre-outcome usage: train:6=10, train:25=12, train:978=14, train:1824=14. Selected source gold labels comprise Supported=26 and Refuted=24 demonstrations, not a statement about target labels. Four source groups and this label prior limit transfer/coverage claims.

Source context is fixed from exact cited spans with 180-character surrounding windows. Controls use the first 600 rendered characters of their Top-1 source passage. All context keeps passage ID, URL and original offsets; full source Top-8 remains in the memory/review packets. P2/P3 use identical context and ordering. N0 is a different, matched non-error example, not identical factual content.

After all 50 shared R4 Top-8 results exist, `seal-prompts` renders **every condition for every target** before a verifier call. P2/P3/N0 full effective prompt lengths are matched within two actual tokenizer tokens using an explicitly labeled punctuation filler. This neutral padding is a control with its own possible effect, not proof that semantic prompt-budget confounding disappears. P0/P1 stay native. P3/P1 therefore includes a demonstration/budget difference; P3/P2 and P3/N0 are the stronger information/length controls. P4's extra rationale cost is reported separately if eligible; it is not length-matched to a fake rationale.

Tokenizer files are pinned to the frozen Qwen revision. The original D0/D1 evidence formatter renders at most 1,200 characters per passage; that existing rule is preserved for every condition. There is no further evidence truncation to fit a demonstration. Any context overflow, incomplete Top-8, missing control or failed token match stops preparation before inference and requires a documented prospective revision, not selective dropping after outcomes.

## Frozen execution and leakage gates

Backbone: Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, bf16, non-thinking, greedy `do_sample=false`, **512 max_new_tokens**, context8192 with the full 512-token reservation. This preserves native D0's cap; 64 tokens is not imposed. P0 and P1 call their original builders unchanged; appended demonstration conditions retain the same verdict output instruction/schema and parser. Actual output length, truncation and validity must be reported.

Retrieval uses the unchanged M1.6.1 frozen segmentation/dense/RRF/reranking configuration, digest `7f716c4fe2888ba311a4edba9ca8dd618c368de43707264dde1d5cf129e907ec`. The underlying pipeline emits R1–R4 as required to construct shared R4; it is **not** a retrieval factorial and no separate verifier arms use R1/R2. The pinned acquisition plan dispatches all three shards by ordinal, including IDs >=1000. Reuse only fully sealed task receipts under identical code/inputs/environment; no unverified legacy shared embedding cache reuse is promised.

Preparation reads label-free claim fields only. No DEVELOPMENT gold, gold QA, justifications, outcomes, test labels or ERRORFlow predictions select examples, prompts or rules. Target groups cannot overlap SOURCE examples/controls. Worker inputs have no gold argument. Source gold is permitted historical information.

Before execution, verify sealed manifests/schedule, source bank/claims hashes, exact condition files, generation/retrieval settings and code. Linux/CUDA, at least 24 GiB free VRAM, pinned model bytes and dependency versions are checked before GPU work. Models load sequentially. Retrieval and inference use locked subprocesses, process-group timeouts, per-step file hashes and new attempt directories after incomplete work. Live locks refuse execution; `--recover-stale-lock` is available only for verified dead processes. Old outputs are never overwritten. Do not change checkout/config/bank while resuming.

Condition order per target is deterministically shuffled using seed42 plus claim ordinal. Save raw text, parsed verdict, source IDs, effective input-token hash, actual input/output tokens, synchronized generation wall latency and peak GPU memory. Receipts preserve exact evidence and all prediction files. Both retrieval and prediction phases share a **24-hour cumulative** wall envelope, scratch ceiling30GiB and minimum free disk100GiB. Stop INCOMPLETE at a limit; never shorten evidence or omit arms to finish.

## Scoring and decision gates

`evaluate` verifies all 50 predictions for all frozen active conditions and their upstream receipts/completion seals **before opening the gold release**. It joins only the selected DEV50 labels. No incomplete-condition scoring that could guide remaining predictions is implemented.

Primary quality metrics: Accuracy and four-class Macro-F1, including invalid predictions in the denominator. Report per-class precision/recall/F1/support, validity, CC/CW/WC/WW paired transitions, correction/regression IDs, invalid transitions, actual tokens, synchronized generation wall latency, peak memory/raw telemetry and total session wall time. GPU-active time and monetary cost remain null unless separately measured/rated; generation latency is not GPU-active time.

Inference: exact two-sided McNemar for correctness, 10,000 paired bootstrap resamples for Accuracy/Macro-F1 delta CIs, and 19,999 paired label-swap permutations for Macro-F1, seed42. The four p-values for P3/P2 and P3/P1 across both metrics form one Holm family. Add the two P4/P3 tests to that family only if eligible **before launch**; no result-dependent family selection. P3/P0 and P3/N0 are descriptive secondary diagnostics. CIs are unadjusted 95% intervals; Holm-adjusted p-values control the declared test family. Missing-class bootstrap frequency is reported; all four classes remain in the macro average.

N=50 has 2-percentage-point Accuracy granularity and weak power for small effects or rare classes. Even five one-direction corrected pairs give two-sided McNemar p=.0625 before adjustment. With four selected source exemplars, target-paired intervals are conditional on that fixed memory, not uncertainty over alternative memory banks. No power or confirmation claim follows from this pilot.

- Consider further research only if P3 improves **both** metrics over both P2 and P1, validity >=99%, costs fit budget and per-class/copying diagnostics do not reveal unacceptable harm. Software's positive-direction signal is not an automatic advancement authorization or proof of significance.
- If signals are mixed, choose MODIFY; if absent/adverse, NO-GO for automatic expansion. First inspect negative transfer, wrong/gold verdict copying, source/target prediction label priors, prompt length, class errors and control behavior. Do not launch CoT variants or tune repeatedly on these same cases.
- If P4 is unavailable, retain P0/P1/P2/P3/N0 population and record P4 metrics as null. Do not substitute a pending draft.
- No additional source batch, k/seed sweep, R1/R2 factorial, long CoT or M1.6.3 is authorized by this preparation. Confirmation remains blocked on clean holdout and a later joint-power/budget decision.

## Costs, current readiness and user commands

Currently **250** planned verifier calls (50 x P0/P1/P2/P3/N0), not yet executed. P4 would add50 only in a new eligible pre-outcome preparation. Output cap totals128,000 tokens; actual prompt/output totals are unmeasured until prompt sealing/generation. Cold retrieval historical range268.8–378.0s/claim plus1.5–5.2s/verifier call suggests roughly3.84–5.61 hours before unmeasured additional startup/large-store risk. A conservative planning reservation is about11.5 hours within the24-hour hard envelope, not a throughput guarantee or confidence bound. Actual rates may differ. Read-only server snapshot: about188GiB free, RTX5090 idle; this reserves no resources. Money/GPU-active cost is null.

CPU implementation complete; real human review pending; **user-run retrieval/preparation ready** for the provisional pilot. **Verifier readiness is conditional** on successful DEV50 retrieval, actual prompt sealing and runtime preflight. P4 BLOCKED. Pilot evaluation/research conclusion INCOMPLETE. [Preparation evidence](audits/M1_6_2A_PREPARATION.json) contains no DEV gold or synthetic quality scores.

The prepared source-only bank/review packets are at server `outputs/m162a-inputs-v1/`. The [implemented wrapper](../scripts/run_m162a_pilot.sh) performs hash/seal checks, exact DEV50 preparation, environment preflight, shared retrieval, prompt sealing, inference and only then scoring. It was syntax-checked, not executed by the agent.

```bash
ssh hvtham-server 'bash -s' <<'REMOTE'
set -euo pipefail
cd /home/stackops/whale/training-free-fact-verification
git fetch origin
git switch codex/m16-heal-design-integrity
git pull --ff-only origin codex/m16-heal-design-integrity
mkdir -p outputs/m162a-logs
nohup bash scripts/run_m162a_pilot.sh > outputs/m162a-logs/launch-v1.log 2>&1 < /dev/null &
echo "PID=$!"
tail -f outputs/m162a-logs/launch-v1.log
REMOTE
```

The wrapper sources existing `.venv`, exports the project `PYTHONPATH`, prints logs/paths/counts and refuses modified protocol/code. Existing unrelated server wrapper changes are preserved; no force reset is suggested. On ordinary failure rerun the same wrapper once the cause is resolved, using the same checkout/bank/inputs. Do not start concurrent wrappers or relaunch while one is live. For a stale lock, inspect the saved parent/child PIDs before using the documented `--recover-stale-lock` flag on the relevant CLI command. Results appear at `outputs/m162a-dev50-v1/evaluation-v1/results.json` and `per_claim.jsonl`; runtime receipts/state retain resume information. No additional commands/scripts are implied.
