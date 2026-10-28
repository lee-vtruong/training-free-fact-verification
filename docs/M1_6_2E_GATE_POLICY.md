# M1.6.2e — Corrected Milestone Gate Policy (Verified Error Memory)

Status: LOCKED PROPOSAL (section A of M1.6.2e) · CPU-only derivation · 2026-10-28
Supersedes: the single ≥30-verified gate as the *milestone* gate for M1.6.2e feasibility work.
Scope of correction: gate arithmetic and tiering only. No protocol, threshold, or label was
adjusted to manufacture a GO; every number below is audited against immutable artifacts.

## 1. Audit of the infeasible ≥30 gate

The prior milestone gate required ≥30 human-verified error corrections before any
error-memory feasibility experiment. Audited against `docs/audits/M1_6_2D_EVIDENCE_TRIAGE.json`
(sha256 `c966fad7…`, triage E1=8 / E2=4 / E3=16) and `outputs/m1611-correction-audit-v3/memory75.json`:

| quantity | value | provenance |
|---|---|---|
| historical Base errors (memory75) | 33 | 5 drafted + 28 INSUFFICIENT |
| existing drafts (originals, untouched per C.1) | 5 | train:6, train:11, train:25, train:978, train:1824 |
| E3 candidates (reasoning/ontology failures, decisive evidence in frozen Top-8) | 16 | M1.6.2d triage |
| E2 candidates (retrieval/extraction failures; decisive information missing/truncated) | 4 | M1.6.2d triage |
| E1 (unsupported; no grounded correction possible) | 8 | M1.6.2d triage |
| **maximum plausible candidates, pre-rejection** | **25** | 5 + 16 + 4 |
| **ceiling under unchanged Top-8 evidence** | **21** | 5 + 16; E2 quarantined (C.6) |
| human-verified corrections today | **0** | review packets issued, not returned |

Structural conclusion: **30 > 25 ≥ any achievable pool.** Even with zero human rejection and
mixing E2 store-passage drafts into the pool, 25 < 30. Under the unchanged-evidence rule that the
primary reasoning-error experiment requires, the ceiling is 21. The ≥30 gate is therefore
*structurally unsatisfiable* from the current error population without authorized new source
batches. A gate that cannot be met by any permitted action is not a quality bar; it is a dead
branch, and leaving it in place would force either (a) a dishonest GO by silent gate-lowering or
(b) permanent deadlock. This document replaces it with an explicit two-tier gate. Nothing here
lowers a bar to force GO: the Tier-1 gate below is *not yet met* (0 verified), and the
GO conditions are stated in `docs/M1_6_2E_DECISION.md`.

## 2. Two-tier gate

### Tier 1 — Small feasibility gate (this milestone's experiment)

**Trigger condition (all must hold, fail-closed):**

- F1. ≥ **4** human-verified SOURCE corrections in the candidate pool;
- F2. covering ≥ **2** distinct mechanism classes;
- F3. with ≥ **2** verified corrections per covered class;
- F4. each verified record: both reviewers returned decisions (`ACCEPT` or
  `ACCEPT_WITH_EDITS`), recorded in human review bookkeeping (never authored by the assistant);
- F5. E2-quarantined records excluded from the primary reasoning-error experiment regardless of
  verification status (evidence-availability change declared per C.6); a separate declared arm
  could study them, but not this one.

**Why exactly these numbers — derivation from design requirements, not expected accuracy.**

- *R1 (exemplar identity, F1 ≥ 4 overall).* The A/B/C design requires the **same exemplar IDs**
  in every arm. Feasibility means: for a target whose predicted mechanism class is covered, at
  least one verified same-class exemplar exists. That alone would need 1 per class; R2/R3 raise it.
- *R2 (class coverage, F2 ≥ 2 classes).* The locked hypothesis is about *corrective explanations
  changing verdicts on same-mechanism targets*. With one covered class the result cannot even be
  stated as mechanism-level; it is a single-class anecdote and cannot support the Tier-2
  mechanism-coverage study either. Two classes is the minimum at which a class-contrast
  (same result in both classes vs class-dependent result) is observable.
- *R3 (identity rotation / non-degeneracy, F3 ≥ 2 per class).* With one exemplar per class, every
  B-arm prompt for that class reuses the identical text; any effect is confounded with that
  specific document. Two verified exemplars per class permit deterministic rotation
  (count-then-ID) so the paired comparison separates mechanism-level correction from
  text-specific idiosyncrasy, and permit a within-class leave-one-exemplar-out contrast.
- *R4 (paired statistics).* The exact McNemar design requires a target population, not more
  exemplars: the protocol locks n ≥ 12 targets (6 per covered class) so discordant pairs can
  occur at all. This constrains the target side (readiness: 482 unused claims in 314
  fully-unused groups) and is stated in `docs/M1_6_2E_PROTOCOL.md`.
- *F4/F5 (integrity, no numeric content).* Verification is human-only; E2 stays quarantined.
  These conditions carry the C.4/C.6 rules into the gate so the experiment cannot instantiate a
  pool from assistant-only drafts.

Deliberately **not** used: any expected-Accuracy or power-to-detect-a-presupposed-effect
argument. The gate asks "can the study be designed and executed honestly at n verified = k?", not
"will the study succeed?" — the answer to the latter is unknown by design (exploratory).

**Reachability from the current pool:** 21 unchanged-Top-8 candidates ≥ 4 (F1 ceiling), and the
16 E3 candidates already span mechanism classes mapped to ≥2 covered classes (e.g.,
direct-match/metric-mismatch/selective-reading families), so F2/F3 are reachable **if and only if
human review verifies at least 4 candidates in ≥2 classes with ≥2 each.** Current status: 0 of 33
records verified → **Tier 1 not met; experiment does not run** (fail-closed by construction in
`src/tffv/experiments/m162e.py`, which loads verified pool at runtime from the review records and
yields an empty pool today).

### Tier 2 — Full memory gate (unchanged, for the broader study)

**≥ 30 human-verified errors with mechanism coverage across the AVeriTeC-relevant classes**
remains the gate for any broader error-memory study beyond the small feasibility experiment.
Reaching it requires authorized new source batches (new error populations through the same
triage + human-review pipeline). No batch expansion is authorized in M1.6.2e; requesting it is a
next-step decision, not a milestone artifact.

## 3. Explicit anti-ratchet clauses

1. No gate in this document may be lowered after any new-population outcome is observed.
2. Human rejection counts against the pool: the 25/21 numbers are ceilings, not entitlements;
   verified counts come only from reviewer bookkeeping.
3. Assistant-authored draft text (including this milestone's 20 candidate drafts) is **never**
   counted as verified for any gate tier.
4. E2 passages may never be mixed into the unchanged-evidence pool without a declared
   evidence-availability change and a separate arm.
5. If Tier 1 cannot be met after review returns, the honest verdict is NO-GO for the feasibility
   experiment (with the pool shortfall reported), not a threshold edit.

## 4. Provenance

- Triage: `docs/audits/M1_6_2D_EVIDENCE_TRIAGE.json` sha256
  `c966fad71fc5bbb3fc19d69ff449d28dbb5a0579af071cbe5eb6cd104f2fb5ac`.
- Candidate drafts file sha256
  `add7a65c91d2283d20b1e44229c8234a5ff2555fda2f84b7df3ee532d44f65d5`.
- Candidate drafts + review records (0 verified, blank reviewer fields):
  `docs/audits/M1_6_2E_CANDIDATE_DRAFTS.json`, `docs/audits/M1_6_2E_REVIEW_RECORDS_REVIEWER1.json`,
  `docs/audits/M1_6_2E_REVIEW_RECORDS_REVIEWER2.json`.
- Gate arithmetic is machine-checked by `tests/test_m162e_gate.py` against
  `configs/m162e_feasibility_protocol.json` (locked).
- Readiness of the target-side population: `docs/M1_6_2E_READINESS.md` (491 unused claims;
  482 in 314 fully-unused groups; 9 shared-group claims excluded; R4 cache hit 0/491).
