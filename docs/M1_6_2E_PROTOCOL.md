# M1.6.2e — Three-Arm Feasibility Study Protocol (Locked Before Outcomes)

Status: LOCKED PROTOCOL DRAFT (section D of M1.6.2e) · prepared 2026-10-28 · **no new-population
inference has been run or may run under this protocol until the Tier-1 gate in
`docs/M1_6_2E_GATE_POLICY.md` is met from human review bookkeeping.**

This document, `configs/m162e_feasibility_protocol.json`, `src/tffv/experiments/m162e.py` and its
tests are committed **before** any new-population outcome exists. Any later change to selection
rules, arms, decoding, or statistics invalidates the lock and must be re-registered as a new
protocol version with the change stated. Locked lineages: arm templates descend from M1.6.2a
(P2 `gold_only`, P3 `wrong_gold`, P4 `wrong_gold_verified`); decoding is the project-frozen
temperature 0.0 / max_new_tokens 512.

## 1. Question and locked hypothesis

**H1 (pre-registered, direction fixed):** on new development targets whose predicted failure
mechanism class matches a covered exemplar class, adding a human-verified corrective explanation
of a same-mechanism historical SOURCE error to the exemplar block (arm B) makes Base's verdict on
the target agree with gold more often than a gold-only exemplar of the same error (arm A). The
feasibility study tests whether this comparison is *executable and interpretable at Tier-1 scale*;
it is not a confirmatory test and cannot activate M1.6.3.

## 2. Arms (fixed)

All arms share one prompt skeleton (task instructions, verdict-label menu with fixed option
order, target claim, target R4 evidence) and differ **only** in the exemplar-history segment:

| arm | lineage | exemplar-history segment |
|---|---|---|
| **A** `gold_only` | M1.6.2a P2 | claim + displayed evidence + gold verdict |
| **B** `gold_verified` | new (P4 without the wrong-verdict display) | claim + displayed evidence + gold verdict + **VERIFIED corrective explanation** (human-review text verbatim, including any `edited_text` accepted by reviewers) |
| **C** `wrong_gold_verified` | M1.6.2a P4 | claim + displayed evidence + **Base's wrong verdict** + gold verdict + verified corrective explanation |
| **D** `gold_filler_length_matched` | M1.6.2a P1 placebo lineage (control for B) | claim + displayed evidence + gold verdict + **neutral filler matched to B's explanation length** |

**D (length-matched control) is planned, not optional scaffolding:** it separates the effect of
*corrective information* from the effect of *adding tokens*. The filler is built deterministically
from a fixed neutral-sentence bank (format/procedure statements containing no verdict-relevant
content), cycled and trimmed to exactly B's explanation word count (pure function in m162e.py;
invariance and content tests in `tests/test_m162e_protocol.py`). D is scored in the same paired
framework as A/B/C; B vs D is the token-effect contrast, B vs A the information-effect contrast.

## 3. Fixed across all arms (invariance contract)

1. **Same exemplar IDs** for A/B/C/D — instantiated per target by the deterministic assignment in
   §5; all four arms of one target use the same exemplar record.
2. **Same exemplar evidence display** — only the exact Top-8 spans cited in the *verified* draft
   (decisive quotes from `bank_top8_asserted` records), same rendered text, same order.
3. **Same exemplar order** in the prompt (the single exemplar slot is per H1's same-mechanism
   design; if the pool ever supports multi-exemplar blocks, order = ascending sample_id ordinal,
   frozen here in advance).
4. **Same targets, same target R4 evidence** — each target's evidence comes from exactly one
   retrieval pass (the single permitted pass for the new population), computed once, hashed, and
   reused by every arm; no per-arm re-retrieval.
5. **Frozen decoding** — temperature 0.0, greedy, max_new_tokens 512, identical sampling config
   object across arms; no per-arm tuning of anything.
6. **Gold access timing** — target gold is read only at scoring, after all arms have generated.
   Selection, assignment, and prompt construction never read new-population gold.

## 4. Target inclusion rule (locked, gold-blind)

Frame at lock time: unused development claims in **fully-unused leakage groups** (readiness:
482 claims in 314 groups; re-derived at prepare time from the immutable partition manifests —
DEV50 members and the 9 claims whose groups intersect DEV50 are excluded unconditionally).

- T1. **Class match:** the frozen, non-learned heuristic of the M1.6.2 triage lineage assigns each
  candidate target a predicted mechanism class from Base's verdict and evidence checks only
  (never gold, never any learned router). Include a target only if its predicted class ∈ classes
  covered by ≥2 verified exemplars (Tier-1 F3).
- T2. **count-then-ID:** per covered class, take the first `n_per_class = 6` targets ordered by
  ascending `sample_id` ordinal. Total planned n = 12 for two covered classes. If a covered class
  has fewer eligible targets than 6, the study runs short of its plan only if total n ≥ 12 still
  holds; otherwise the run is blocked (fail-closed), never backfilled by loosening T1.
- T3. **Exclusions:** any DEV50 or shared-group collision (hard error), any target without a
  complete R4 evidence set after its single retrieval pass, any target whose claim text was
  previously exported in an outcome artifact (`artifact_outcome_exposed_directly` flags from the
  manifest are honored).
- T4. **No DEVELOPMENT50 tuning:** DEV50 outcomes are frozen history used nowhere in this
  protocol — not for thresholds, prompts, arm selection, or class weights.

## 5. Exemplar assignment (deterministic)

Pool per covered class, sorted by `sample_id` ordinal; targets within a class, ordered as in T2.
Target at within-class rank *i* receives the class exemplar at position *i mod (pool size)*
(count-then-ID rotation). The assignment is a pure function of (target list, verified pool) —
same inputs, same outputs, verified by test. No randomness anywhere in selection or assignment.

## 6. Statistics plan (locked)

- Primary paired contrasts: A vs B, A vs C, B vs D (per-target paired outcomes).
- Exact McNemar two-sided on discordant pairs; exact permutation sign count reported alongside.
- Bootstrap CI (10,000 resamples, seed 20261028) on the paired accuracy difference.
- Metrics: Accuracy and Macro-F1 (four-class AVeriTeC ontology, official label order) per arm;
  pooled and per covered class; all results labeled EXPLORATORY, small-n.
- Interpretation ceiling: a positive B-vs-A signal at Tier-1 scale justifies only "proceed to
  Tier-2 planning with human-verified pool expansion"; it never activates M1.6.3 or any
  TEST_SEALED access.

## 7. Environment and data rules (restated for lock)

- No GPU execution and no retrieval inside M1.6.2e; the single retrieval pass for targets is a
  step-4 action, gated on Tier-1 + human authorization.
- TEST_SEALED untouched; M1.6.3 not activated; no prompt sweeps; no refitting.
- All exemplar content is verified SOURCE-correction text; assistant drafts enter only as
  candidate text pending human review and are never rendered as verified.

## 8. Lock record

- Config: `configs/m162e_feasibility_protocol.json` (validated shape in m162e.py).
- Selection/assignment/arm/filler functions: `src/tffv/experiments/m162e.py` (CPU-only).
- Tests: `tests/test_m162e_gate.py`, `tests/test_m162e_protocol.py`.
- Fail-closed behavior: with 0 verified records (today's state) the pool loader returns an empty
  pool and `feasibility_ok == False`; every downstream entry point raises before any prompt is
  built for a real target.
