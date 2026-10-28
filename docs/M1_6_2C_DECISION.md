# M1.6.2c — Historical memory coverage & label-prior audit (decision)

Status: CPU audit COMPLETE (no GPU in this milestone). Machine-readable evidence: [M1_6_2C_AUDIT.json](audits/M1_6_2C_AUDIT.json). Raw-train SHA verified `ae5eda7c…`. Decision: **GO** for the three-arm cache-only coverage pilot (150 new calls, exploratory); **NO-GO** for history-separation arms until human review exists.

## A. SOURCE coverage audit (75 historical cases)

Crosstab of gold verdict × Base history (records with guarded evidence text + source URLs on every evidence item = usable; 75/75 usable, 0 zero-evidence):

| Gold class | Base-correct | Base-wrong | usable correct | usable wrong |
|---|---:|---:|---:|---:|
| Supported | 15 | 10 | train:0, 13, 1429, 22, 28, 30, 32, 33, 40, 42, 426, 45, 46, 467, 718 | train:2, 3, 14, 17, 25, 29, 37, 38, 978, 1881 |
| Refuted | 24 | 9 | 24 hits incl. train:10 | 9 incl. train:6, 1824 |
| Not Enough Evidence | 2 | 7 | train:23, train:2811 | train:21, 26, 27, 31, 35, 39, 49 |
| Conflicting Evidence/Cherrypicking | 1 | 7 | train:7 (only) | train:8, 12, 18, 19, 24, 979, 2449 |

- Review status (post-replay bank, authoritative): 47 PENDING_REVIEW (42 correct controls + 5 error drafts `train:6, 11, 25, 978, 1824`), 28 assistant-INSUFFICIENT errors, **0 human reviews, 0 VERIFIED**. Assistant drafts are never verification.
- Provenance: every record carries per-item source URLs and rendered evidence text; record-level `citations` exist for 1/75 (legacy layout uses stored spans instead) so added-exemplar excerpts use the deterministic guards-window fallback on stored rendered text.
- Leakage: 75 records span **46 leakage groups**; 5 groups bundle multiple bank records (max 15 in one block); **0 overlap** with any DEVELOPMENT group (v2 partitions remain sealed).

**Why the P2/N0/Mixed selectors only show Supported/Refuted exemplars:** all three arms draw exclusively from the frozen M1.6.2a DEV50 slates — error exemplars `train:6, 25, 1824, 978` and controls `train:10, 718` — whose gold classes are Supported/Refuted by construction of that preregistered pilot. The bank itself *does* contain NEI and Conflicting exemplars (9 and 8 usable records respectively). So the limitation is a **selector/slate bottleneck, not a true source-coverage bottleneck**: coverage exists but the frozen pool doesn't reach it.

## B. Pilot failure analysis (DEVELOPMENT50, exact verdicts from immutable receipts)

- **Mixed vs N0 identity:** the two arms produce the same exact predicted verdict on **48 of 50 claims**. The only two differences are `train:1249` (gold Conflicting: Mixed predicts Refuted, N0 predicts Supported) and `train:1217` (gold NEI: Mixed Refuted, N0 Supported). Both are wrong-but-other-class, and both moved toward Refuted — the dominant class of the target prior. Accuracy is therefore identical (0.700) while Macro-F1 differs by +0.0079 from these two swaps.
- **Corrections come entirely from the correct exemplar side:** Mixed vs P2 corrections are exactly N0's known three (`train:1853, 627, 1728`), all targets whose exemplar pair has Refuted gold; the single regression `train:1935` is a Supported-pair target. Adding the Base-wrong exemplar produced **no new correction and one extra regression-in-set**, supporting (correlationally) "no incremental benefit from the wrong-case gold-only exemplar".
- **Confusion and priors (unchanged failure mode):** per-class F1 Mixed = Refuted 0.849 / Supported 0.500 / NEI 0.000 / Conflicting 0.000; predictions collapse NEI (10/12 misnamed) and Conflicting (4/4) mostly into Refuted. Predicted priors stay Refuted-heavy in all three arms (P2 0.66 accuracy with Refuted-saturated errors).
- **k/token/order:** mean prompt tokens P2=N0=2803 → Mixed 3085 (k=1→2). The Content-vs-history insight: in gold-only blocks the Base verdict/correction is masked, so "Base-correct vs Base-wrong" is **not displayed to the model** — the mixed-vs-N0 identity above is exactly what that predicts. Any effect attributed to the exemplar's history cannot be causal here; it is a pure exemplar-claim/evidence-content effect. Full confusion and token tables: audit JSON (`pilot_failure`).
- All of B is descriptive on the exposed DEV50; no causal claim and no tuning of any future arm was done from these values.

## C. Next experiment — preregistered coverage pilot (branch 1, coverage sufficient)

New condition family `m162c-coverage-v1`, [frozen config](../configs/m162c_coverage_gold_only.json) and runner [`tffv.experiments.m162c`](../src/tffv/experiments/m162c.py), sealed CPU-side per arm before any model load. All arms are k=3 displays of the fixed Mixed pair (canonical correct-then-wrong gold-only blocks) plus exactly one added Base-correct gold-only exemplar (1 excerpt, header strictly neutral and identical across arms), same canonical target R4 Top-8, same frozen Qwen3-8B decoding (non-thinking greedy, 512), no rerun of any existing arm:

| Arm | Added exemplar (class) | Purpose | New calls |
|---|---|---|---:|
| `COV_PLACEBO_G3` | train:0 for Supported-pair targets, train:1 for Refuted-pair targets (pair-gold class) | token/(time)-matched composition control without new class | 50 |
| `COV_NEI_G3` | train:23 (Not-Enough-Evidence class; 2-record pool, tie count 8-8 broken ascending ID) | add NEI class display | 50 |
| `COV_CONF_G3` | train:7 (Conflicting; forced, pool n=1) | add Conflicting class display | 50 |

Selection rule (preregistered, label-free for DEVELOPMENT): usable SOURCE Base-correct exemplar of the arm's gold class; highest evidence-item count, ties ascending sample_id; pair IDs excluded; zero target/exemplar-group overlap rechecked per target at seal.

- **Label-coverage vs history separation:** the three arms hold the added exemplar's Base history constant (Base-correct everywhere), so `COV_NEI − COV_PLACEBO` and `COV_CONF − COV_PLACEBO` isolate *adding a class-coverage display* at matched k, order, header and template; the placebo arm's 2-sides variation is disclosed as a within-control factor. The Base-correct/wrong-history contrast is explicitly **parked**: gold-only blocks cannot display history, and history-revealing blocks require human-verified corrective rationales (currently 0). Attempts to invent history signals would violate the work rules.
- Cache-first: preparation validates canonical batch/index/memory hashes, per-target R4 Top-8 bytes, prompt token IDs, and group leakage; `run` fails closed before model load; resumes only missing receipts (checkpoints now use the atomic `update()` fix from `cfa2a29`).
- **Branch 2 (documented contingency, not triggered):** if usable coverage had been insufficient — for example if the Conflicting pool (n=1) or the Near-Miss pool is later judged too thin for any interpretation — the source-only expansion path is planned but NOT executed: the sealed SOURCE partition holds 1013 claims of which **973 are memory-eligible** with gold distribution Refuted 533 / Supported 272 / NEI 97 / Conflicting 71; any expansion batch must go through the same legacy-audit → retrieval → batch pipeline as reviewed batches, source-only, with predeclared stopping, and stays far from DEV/TEST.

User-run commands (exactly 3×50 new calls on first execution, resumable):

```bash
ssh hvtham-server 'bash -s' <<'REMOTE'
set -euo pipefail
cd /home/stackops/whale/training-free-fact-verification
git fetch origin
git switch codex/m16-heal-design-integrity
git pull --ff-only origin codex/m16-heal-design-integrity
nohup bash scripts/run_m162c_coverage.sh > outputs/m162c-logs/launch-v1.log 2>&1 < /dev/null &
echo "PID=$!"
REMOTE
```

Outputs: `outputs/m162c-COV_*-v1/evaluation-v1/results.json` per arm. Planned reads: each arm vs P2 vs N0 vs Mixed (Accuracy, Macro-F1, per-class, exact McNemar, paired bootstrap, corrections/regressions, prompt tokens, latency).

## D. Correction memory status

33 historical errors: 5 authoritative PENDING correction drafts (train:6, 11, 25, 978, 1824 — all inside the mixed-arm exemplar pool), 28 assistant-INSUFFICIENT dispositions, **0 human reviews submitted** (batch25, legacy, and reviews75 sheets on server all remain blank), **0 verified rationales, 0 verified mechanisms**. The ≥30-verified-errors/3-mechanism gate remains UNMET. No review status was upgraded and no explanation was fabricated by this audit.

## E. Decision

- **GO**: preregistered `COV_PLACEBO_G3` / `COV_NEI_G3` / `COV_CONF_G3` cache-only coverage pilot — 150 new inference calls, EXPLORATORY on the exposed DEVELOPMENT50, never confirmation evidence.
- **NO-GO**: history-revealing (Base correct/wrong separation) arms until ≥30 verified-error rationales exist; automatic source expansion; M1.6.3 confirmation via TEST_QUARANTINE; any tuning of the mixed/verdict-free predictors based on B.
- Known limitations: DEV50 exposure persists; Conflicting pool n=1 (single-group exemplar identity confound in that arm); NEI pool tie 8-8 broken ascending; placebo adds same-class exemplar identity; added exemplars are not human-verified (gold-only display, masked rationale); single-exemplar-per-arm content confound; GPU-active and monetary costs stay null unless separately measured.
