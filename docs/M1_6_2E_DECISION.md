# M1.6.2e — Milestone Decision

Milestone: M1.6.2e — Verified Error-Memory Feasibility & Review Gate Correction
Date: 2026-10-28 · Scope: CPU-only (no GPU, no retrieval, no new-population inference)

## Review-bookkeeping update (2026-10-28)

Two human-review submissions supplied by the user have now been imported for the five original SOURCE drafts (`train:6`, `train:11`, `train:25`, `train:978`, `train:1824`). The immutable candidate bank remains untouched; imported records are in `docs/audits/M1_6_2E_REVIEWED_RECORDS.json`, with a separate evidence reconciliation in `docs/audits/M1_6_2E_TOP8_COMPLETION_AUDIT.md`.

All five are qualified `VERIFIED` corrections under the recorded two-reviewer bookkeeping. This does **not** meet Tier-1: the five records occupy five distinct mechanism classes, so none has the required two verified exemplars per class. The gate report is therefore `verified_now=5`, `covered_classes={}`, `TIER1_UNMET_FAIL_CLOSED`.

## Tier-1 review completion update (2026-10-28)

Two further user-attested human-review submissions were imported for `train:21`, `train:31`, `train:47`, and `train:48`. Their text is byte-identical (sha256 `7769b128b5f7037512d40ca64f8fcdc695e7f3b57bffa39f98b812b99c394668`), and that fact plus the user's independence attestation is retained in `docs/audits/M1_6_2E_TIER1_E3_REVIEW_SUBMISSIONS.json`; no personal reviewer identity was invented.

The current machine-checked gate is now `verified_now=9`, `metric_mismatch=2`, `missed_direct_ranking=2`, `TIER1_MET_PENDING_LOCKED_EXECUTION`. Tier-1 review sufficiency is met. This is **not** authorization to execute: the locked protocol still requires separate human authorization for the one new-population R4 retrieval pass and the frozen GPU arms. No retrieval, inference, or test access occurred during import.

## Verdict: **MODIFY**

The corrected gate framework, candidate correction drafts, two-reviewer review artifacts, and the
locked three-arm feasibility protocol are complete and internally consistent — but the Tier-1
feasibility gate is **not met** (0 of 33 candidate records human-verified), so the feasibility
experiment cannot be instantiated and the milestone cannot be GO.

## What MODIFY means here (conditions, not aspirations)

**GO conditions (all required; none currently satisfied):**

1. Human review bookkeeping records ≥ 4 verified SOURCE corrections (both reviewers,
   `ACCEPT`/`ACCEPT_WITH_EDITS`) from the M1.6.2e packets, covering ≥ 2 mechanism classes with
   ≥ 2 verified per class (Tier-1 F1–F5 in `docs/M1_6_2E_GATE_POLICY.md`).
2. The protocol lock (`configs/m162e_feasibility_protocol.json` + m162e.py + tests, committed
   before outcomes) remains unchanged; any edit requires a re-registered version.
3. Explicit human authorization for (a) the single permitted R4 retrieval pass over the selected
   target population and (b) the step-4 GPU arm executions under frozen decoding.
4. Environment readiness re-confirmed at execution time (server state, store availability).

**Why not GO:** 0 verified corrections exist. Twenty assistant-authored candidate drafts (16 E3
from exact Top-8 spans; 4 E2 quarantined) plus 5 untouched originals are *candidates*; under
C.4/C.6 and the anti-ratchet clauses, assistant-generated text is never counted as human-verified.
Claiming GO would require exactly the silent-gate-lowering this milestone was commissioned to
eliminate.

**Why not NO-GO:** the pool is structurally sufficient — 21 candidates under unchanged Top-8
evidence ≥ the Tier-1 requirement of 4, spanning ≥2 mechanism classes — so the gate is reachable
by completing the human review that is already prepared, exported, and addressed to two reviewers.
The pipeline is end-to-end ready: readiness report (`docs/M1_6_2E_READINESS.md`: 482 clean
claims / 314 fully-unused groups for targets; 0 cache hits → one retrieval pass defined),
fail-closed selection code, and locked statistics. NO-GO would be the correct verdict only if
review returns with fewer than 4 qualifying verifications, or if the pool's class coverage
collapses (see anti-ratchet clause 5).

## Corrections delivered (vs the infeasible prior state)

1. **Gate policy corrected and split** into Tier 1 (small feasibility: ≥4 verified, ≥2 classes,
   ≥2/class — derived from design requirements: same-exemplar identity, class coverage,
   identity rotation, paired-stats target minimum) and Tier 2 (≥30 verified + mechanism coverage,
   unchanged, requires authorized source-batch expansion). The ≥30 milestone gate is retired as
   structurally unsatisfiable (max plausible 25; unchanged-Top-8 ceiling 21) — documented, not
   silently lowered.
2. **Evidence-classification integrity held:** E1/E2/E3 distinctions preserved; E2 store-passage
   drafts quarantined with declared evidence-availability change and store-probe provenance
   (`verified_in: server_store_probe_20261028`); E1 stays no-draft; originals pass through
   verbatim; ontology re-review notes added to all sensitive E3 cases (NEI-protection for the
   metric/scope divergence cases 21/31/35/49; documented sign-off flags) — no disagreement
   between sources auto-labeled Conflicting without an ontology note.
3. **Drafts + review artifacts:** 33 records (5 ORIGINAL + 16 E3 + 4 E2-quarantined + 8 E1),
   each with claim, Base verdict, gold verdict, decisive quotes (all E3 quotes machine-asserted
   verbatim against bank Top-8 spans), mechanism, limitations, provenance; two identical
   blank-decision reviewer sheets; rendered review packet; per-mechanism support report showing
   0 verified everywhere.
4. **Protocol locked before outcomes:** A/B/C arms (+D length-matched control) with shared
   skeleton, exemplar IDs, evidence, order, target R4 evidence, frozen decoding; gold-blind
   count-then-ID target selection from fully-unused groups; exact McNemar + seeded bootstrap;
   fail-closed implementation.

## Immediate next steps (outside this milestone)

1. Two human reviewers return decisions on `docs/M1_6_2E_REVIEW_PACKET.md` (sign-offs mandatory
   on flagged records).
2. Bookkeeping records verified counts → Tier-1 check re-runs (`m162e.py` pool loader).
3. If Tier-1 met: request authorization for the single retrieval pass + step-4 executions.
4. If Tier-1 unmet after review: report pool shortfall honestly (NO-GO for the feasibility
   experiment) and, separately, decide whether to request authorized new source batches toward
   the Tier-2 gate.
