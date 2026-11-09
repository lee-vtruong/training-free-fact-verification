# M1.6.2c - Coverage pilot results (EXPLORATORY)

Status: **EXPLORATORY_NOT_CONFIRMATORY** — inference COMPLETE 2026-10-28, user-executed on `hvtham-server`
(log `outputs/m162c-logs/coverage-20261028T033928Z.log`, code HEAD `8a1f99b`, protocol `m162c-coverage-v1`,
frozen config [`m162c_coverage_gold_only.json`](../configs/m162c_coverage_gold_only.json), Qwen3-8B non-thinking
greedy 512, zero retrieval calls, no baseline rerun). `COV_CONF_G3` did not run (NO-GO: empty non-colliding pool,
fail-closed). Population is the already-exposed DEVELOPMENT50; nothing here may tune future exemplar selection,
and no further prompt sweep on DEV50 is authorized. Machine-readable results:
[placebo](audits/M1_6_2C_PLACEBO_RESULTS.json), [NEI](audits/M1_6_2C_NEI_RESULTS.json) (schema-locked by
[`schemas/m162c_results.schema.json`](../schemas/m162c_results.schema.json)).

## 1. Headline (50 claims, gold Refuted 32 / Supported 8 / NEI 6 / Conflicting 4)

| Arm | k | Added exemplar | Acc | Macro-F1 | Predicted prior (R/S/NEI/C) | Prompt tokens |
|---|---:|---|---:|---:|---|---:|
| P2 (canonical ref) | 2 | correct+wrong pair | 0.660 | 0.3219 | 39/9/0/2 | ~2803 mean |
| N0 (canonical ref) | 1 | correct only | 0.700 | 0.3294 | 39/10/0/1 | ~2803 mean |
| MIXED_GOLD_ONLY_K2 | 2 | pair | 0.700 | 0.3373 | 41/8/0/1 | 154,232 |
| **COV_PLACEBO_G3** | 3 | pair + same-gold (train:0 ×26 / train:1231 ×24) | **0.700** | **0.3400** | 43/7/0/0 | 165,186 |
| **COV_NEI_G3** | 3 | pair + NEI (train:23 ×38 / train:2811 ×12) | **0.660** | **0.3248** | 38/9/2/1 | 169,710 |

Per-class F1 (Refuted / Supported / NEI / Conflicting): placebo **0.827 / 0.533 / 0.000 / 0.000**;
NEI **0.829 / 0.471 / 0.000 / 0.000**. Valid rate 1.00 both arms. Adding a NEI-gold display did **not** create a
single true-positive NEI or Conflicting prediction: the only new-class outputs in the NEI arm are 2 NEI verdicts —
both on gold-Refuted claims.

## 2. Paired outcomes (transitions describe the correct/wrong PARTITION, not verdict identity)

Reading rule (make this explicit once): `CC/CW/WC/WW` count whether each arm was Correct/Wrong per claim. A
0-discordant row (CW=WC=0) means both arms are right on exactly the same claims; they may still emit **different
wrong verdicts** on the jointly-wrong claims, which is what the Macro-F1 delta captures.

| Comparison | Acc delta [CI95] | McNemar | Transitions (CC/CW/WC/WW) | Verdict-level diffs | Macro-F1 delta [CI95] (swap p) |
|---|---|---:|---|---|---|
| Placebo vs Mixed | 0.000 [0, 0] | 1.0 | 35/0/0/15 | 2 claims, all wrong-side | +0.0027 [−0.0086, +0.0198] (p 1.0) |
| Placebo vs N0 | 0.000 [0, 0] | 1.0 | 35/0/0/15 | 4 claims, all wrong-side | +0.0106 [−0.0080, +0.0366] (p 0.243) |
| Placebo vs P2 | +0.040 [−0.040, +0.120] | 0.625 | 32/1/3/14 | corrections = N0 trio; regression train:1935 | +0.0181 [−0.031, +0.071] (p 0.501) |
| NEI vs Mixed | −0.040 [−0.100, 0.000] | 0.5 | 33/2/0/15 | 5 claims: 2 regressions + 3 wrong-side | −0.0125 [−0.0352, +0.0059] (p 0.380) |
| NEI vs N0 | −0.040 [−0.100, 0.000] | 0.5 | 33/2/0/15 | 5 claims: same 2 regressions + 3 wrong-side | −0.0046 [−0.0225, +0.0142] (p 0.563) |
| NEI vs P2 | 0.000 [−0.100, +0.100] | 1.0 | 30/3/3/14 | corrections = N0 trio; regressions train:1935, 385, 1733 | +0.0029 [−0.046, +0.053] (p 0.965) |

- **Correct-set invariance:** {N0, Mixed, Placebo} are right on the *same 35 claims* (0 discordant pairs);
  NEI is right on a strict 33-subset of that set (2 losses, 0 gains).
- **Placebo wrong-side reshuffle (vs N0):** train:1284 (gold Supported, N0→Conflicting, placebo→Refuted),
  train:1249 (gold Conflicting, N0→Supported, placebo→Refuted), train:1217 and train:1201 (gold NEI, N0→Supported,
  placebo→Refuted). All four moved **toward Refuted**: the same-gold third exemplar tightened the Refuted collapse
  (prior R 39→43) without touching correctness. Vs Mixed the same effect is smaller (2 claims).
- **NEI-arm harm is fully identified:** both regressions `train:385` and `train:1733` are gold-**Refuted** claims
  that P2, N0, Mixed and placebo all answer Refuted (correct) and the NEI arm answers **Not Enough Evidence**.
  Those are the arm's only 2 NEI predictions. The NEI display therefore did not teach NEI detection on gold-NEI
  claims (NEI recall stays 0/6); it leaked NEI verdicts onto Refuted claims.
- **Corrections never move:** vs P2, both coverage arms reproduce exactly N0's three corrections
  (`train:1853, 627, 1728`) and no others. Across all five arms the correction set is N0-derived and closed.

## 3. Confound separation (the preregistered purpose of the pilot)

1. **Class-coverage association: null-to-negative.** Adding a NEI-class exemplar left NEI recall at 0.0 and
   Conflicting recall at 0.0, and the accuracy point estimate is *negative* (−0.040, exact McNemar 0.5, bootstrap
   CI [−0.10, 0.00], Macro-F1 swap p 0.38/0.56). At n=50 this is absence of evidence, not evidence of absence —
   but there is **no positive class-coverage signal** on this population, and the observed direction is harm
   through false-NEI on Refuted claims.
2. **Individual exemplar identity effect: disclosed, not separable.** The added exemplar identity varies per
   target in both arms (placebo: 2 identities split by pair gold 26/24; NEI: 2 identities split 38/12 by
   leakage-group collision). Because the placebo arm (same-class content, varying identity) produced **zero**
   partition change while the NEI arm (different-class content) produced the only partition change, the
   partition-level effect tracks the added exemplar's **class/content difference**, not k or order. Which NEI
   identity (train:23 vs train:2811) drives it cannot be separated in this design; isolating it would need
   identity-swept arms, i.e. further DEV50 exposure — not authorized.
3. **k/token/order confounds: jointly controlled and null.** The placebo arm raises k 2→3 (+219.1 mean tokens,
   range 185..256) with uniform order `(pair_correct, added, pair_wrong)` and produces exactly zero accuracy
   movement; its Macro-F1 delta (+0.0027) is within noise (swap p 1.0 vs Mixed). The NEI arm's residual over
   placebo is +90.5 mean tokens (67..138) plus the class content; a pure token-count artifact cannot explain two
   specific Refuted→NEI flips, so the harm is content-associated, not budget-associated.
4. **Historical Base outcome information effect: NOT claimable.** All arms here are gold-only: Base verdicts and
   corrections are masked (`[not provided]`) and no rationale text is displayed, so nothing in this pilot can
   confirm or refute a history effect. History-revealing arms stay parked until human-verified corrections exist
   (0 of 33 reviewed; preregistered gate ≥30 unmet).

## 4. Cost and integrity

- Cost: placebo prompt 165,186 tokens (equals the sealed per-prompt total exactly — the GPU run consumed the
  sealed prompts unmodified), output 3,487, wall 100.26 s; NEI 169,710 / 3,456 / 100.79 s. `gpu_active_seconds`
  stays null-locked; monetary cost not measured.
- Integrity: sealed `coverage_batch.json` SHAs (`26e62271…` placebo, `ffe9f336…` NEI) re-verified by `check()`
  before inference; 50/50 receipts per arm; `results.json` written only after the locked-shape validator passed;
  archived copies sha256 `ae509618…` / `6ca62083…`, byte-identical local/server; `COV_CONF_G3` fail-closed banner
  printed as designed; zero retrieval commands; the pre-run permission incident was resolved before any GPU call
  (sealed state hash unchanged at that point), so no calls were spent or lost.
- Statistics: exact McNemar on discordant pairs, 10,000-resample bootstrap CIs, 19,999 paired-swap permutations,
  `missing_class_bootstrap_frequency` 0.0166 in every comparison (small-support classes destabilize resamples).

## 5. Reading and next gates

- The Refuted-collapse prior is **robust to everything tried so far**: correct-only, correct+wrong, k=3
  same-gold, and NEI-class displays all leave NEI/Conflicting recall at 0. The failure is not a display-coverage
  gap reachable by re-arranging frozen exemplars.
- All four pilot arms on DEV50 share N0's closed correction set; no configuration produced a new correction.
- Parked / next gates (unchanged, none authorized by these results): (a) human review of the 33 error
  corrections — gate ≥30 verified before any history-separation arm; (b) `COV_CONF_G3` re-run only after a
  separately authorized source expansion supplies a non-colliding Conflicting-correct exemplar; (c) no further
  prompt sweeps on DEVELOPMENT50; (d) no M1.6.3 activation from exploratory evidence.
