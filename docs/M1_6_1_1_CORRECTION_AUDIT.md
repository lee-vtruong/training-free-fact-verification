# M1.6.1.1 — Source corrective memory audit

2026-10-27. CPU preparation COMPLETE; human adjudication INCOMPLETE. No new retrieval or GPU inference was run by the agent. The user's explicit M1.6.1.1 instruction supersedes the obsolete M1.1-only scope sentence in AGENTS.md. Frozen/source-only rules still apply.

## Audited population and results

The exact own-project A50 plus completed source batch25 remains unchanged: **75 SOURCE claims, 33 Base incorrect verdicts, 42 correct verdicts, zero parser errors**. All record identities, Base raw/stored/reparsed verdicts, claim/group assignments, evidence text hashes and retained provenance pass the source contract. This is a serialized-artifact audit, not retrospective proof of legacy runtime weights or evidence entailment. [Legacy provenance limitations](M1_6_1_SOURCE_AUDIT.md) remain in force.

| Result | Count | Interpretation |
|---|---:|---|
| Reproducible Base/gold verdict mismatches | 33 | Includes insufficient-evidence cases; not 33 reasoning failures |
| Assistant-drafted, cited corrections | 5 | PENDING_REVIEW, never VERIFIED |
| Assistant insufficient-to-ground-gold dispositions | 28 | INSUFFICIENT_EVIDENCE, excluded from pilot memory |
| Correct controls | 42 | Raw/structural/provenance audit passed; semantic review pending |
| Human reviews / verified rationales / verified mechanisms | 0 / 0 / 0 | No invented reviewer identities or approvals |

Five errors plus 42 controls have PENDING_REVIEW status; the other 28 have an explicitly assistant-authored insufficient-evidence disposition. **All 75 are still human-unreviewed.** The audit's `incomplete_evidence_cases` count is assistant screening, not human adjudication.

## Five candidate explanations

| Source ID | Candidate evidence-grounded issue | Important limit |
|---|---|---|
| train:6 | States' 26.72% allocation versus claimed 35%; revenue-basket confusion | Intended period and revenue basket require review |
| train:11 | Federal 52.68% formula versus claimed 45%; statutory revenue and VAT differ | Do not interchange the separate VAT/oil examples |
| train:25 | Base acknowledges 0.43%/0.44% but misreads “not even 1%” | Period and denominator still matter |
| train:978 | Weekend 15:00–05:00 curfew is 14 hours; Base uses weekday 12-hour interval | Date/weekday and pet-walking exception must match |
| train:1824 | Specific falsely-attributed Churchill evidence versus repeated quote pages | Secondary citation; original authorship is not established |

These support candidate scope/quantifier, polarity, temporal/context and attribution explanations. They do **not** establish validated mechanism frequencies. Citation matching proves a span exists, not that the conclusion follows. The drafting decisions, exact source record IDs, explanations, limitations and literal span selectors are replayable in [the source-only decision file](../configs/m1611_correction_decisions.json).

The 28 exclusions include compound claims, incompatible years/denominators, missing decisive passages, ambiguous propositions and source-label/evidence tension. For example, train:2224 contains an embassy reversal allowing old passports, so a Refuted explanation must not be forced from its gold label. Train:1881 is a topic/title without a clear proposition. Train:17 has a bibliographic ratio fragment without the needed paired counts. No new web evidence, official gold QA, gold justifications or ERRORFlow predictions were used to fill gaps.

## Actual artifacts and reproducibility

Current local directory: `outputs/m1611-correction-audit-v3/`.

- `memory75.json`: full source bank; SHA-256 `b3f7275112aa49b71595d94c7313a4ae101b3ed263c277bb277f97def2565581`.
- `audit.json`: counts, parent-bank and decision-file hashes.
- `reviews75/reviewer_1.json`, `reviewer_2.json`: independent 75-case sheets with human fields unfilled.
- `reviews75/candidate_annotations_template.json`: revision template, not completed annotations.
- `review_packets/train-<ordinal>.md`: 75 readable packets with claim, exact Base generation, source gold, draft/limits, exact citation spans, all eight original rendered evidence passages and provenance.

The packets and bank were copied into the new server directory `outputs/m162a-inputs-v1/`; no old bank/review/artifact was overwritten. [Committed audit manifest](audits/M1_6_1_1_CORRECTION_AUDIT.json) records hashes of prepared files. Full source evidence remains in ignored outputs rather than a bulk Git addition.

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m tffv.memory.corrections \
  --memory outputs/m161-review-preparation-v1/combined75.json \
  --decisions configs/m1611_correction_decisions.json \
  --output outputs/m1611-correction-audit-replay-v1
python -m tffv.experiments.m161 validate-memory \
  --memory outputs/m1611-correction-audit-replay-v1/memory75.json
```

Replay requires the existing local input bank. The server instead has the prepared, hash-checked bank at `outputs/m162a-inputs-v1/memory75.json`. The exporter refuses an existing output directory; preserve revisions.

## Real review and P4 gate

Follow the [two-human protocol](../templates/m161-review/REVIEW_PROTOCOL.md). Prioritize the five drafts and selected controls; all 75 packets remain available for independent inspection. Reviewers must inspect all Top-8 evidence, record real identity/timestamp/reasons and affirm evidence sufficiency, gold relation and correction grounding. Null fields are not reviews. Software validates provenance/attestations, not real-world identity or semantic truth.

After two humans fill their separate actual sheets, the implemented import command is:

```bash
python -m tffv.experiments.m161 import-reviews \
  --memory outputs/m162a-inputs-v1/memory75.json \
  --review-1 outputs/m162a-inputs-v1/reviews75/reviewer_1.json \
  --review-2 outputs/m162a-inputs-v1/reviews75/reviewer_2.json \
  --output outputs/m162a-reviewed75-v1.json
```

The current unfilled sheets add no approvals: their records remain pending or insufficient. Rows can be reviewed in pairs while the other rows stay blank; a one-sided completed decision is rejected. Do not edit sealed bank context or stamp VERIFIED directly. A changed draft needs a new bank and new independent reviews. Do not replace the source bank of an already prepared pilot; human-reviewed P4 requires a separately versioned preparation before any target outcomes are inspected.

## Readiness and cost

**MODIFY for a broad rationale-based HEAL study; GO for preparation of the explicitly requested minimal, provisional P0–P3 plus N0 comparison. P4 BLOCKED.** This user-requested narrower pilot replaces the previous broad-study threshold of 30 verified errors/three supported mechanisms for this initial comparison only. It does not claim that threshold was met. Four matched error exemplars are insufficient for conclusions about all historical failure modes; no further source expansion or GPU job starts automatically.

This milestone made zero new model calls. Retained source-generation costs are 192,665 prompt tokens, 17,271 output tokens and 412.816 seconds generation latency across the prior 75 predictions; these are not new costs or whole-pipeline timings. The already completed user-run new25 batch took 9,168.7 seconds wall time. GPU-active time and monetary cost are unknown. Reviewing five drafts at the prior planning rate costs about 80–120 reviewer-minutes in total for two people, excluding disputes; actual human time remains unmeasured.

The research outcome remains unknown until the controlled exploratory development comparison. No Accuracy or Macro-F1 improvement is asserted.
