# M1.6.1 — Review after the completed source batch

Date: 2026-10-27. The user completed the fixed 25-source batch. Current combined bank: **75 records, 33 incorrect verdicts, 42 correct verdicts, zero parser errors**. All 75 remain PENDING_REVIEW. One assistant correction draft exists for train:25; the other 32 incorrect cases have no drafted correction yet. Human/verified rationale counts remain **0**.

## Prepared CPU artifacts

Local root: `outputs/m161-review-preparation-v1/`.

| Artifact | Contents |
|---|---|
| `combined75.json` | Own-project A50 plus fixed new25; source/group/hash checks passed |
| `errors33.json` | Priority incorrect-source bank, retaining the one unverified draft |
| `correct-controls42.json` | Correct-source controls, separate from the error bank |
| `error-reviews/reviewer_1.json`, `reviewer_2.json` | Two independent 33-case sheets; human fields unfilled |
| `control-reviews/reviewer_1.json`, `reviewer_2.json` | Two independent 42-case control sheets; unfilled |

All three banks and four sheets pass their formal JSON schemas. [Audit counts/hashes](audits/M1_6_1_POST_BATCH_REVIEW.json) preserve provenance without committing raw source claims/evidence. These files were prepared locally, not uploaded to overwrite server outputs. They contain source-only gold/evidence; no development/test labels or ERRORFlow predictions are used.

## Next sequence

1. Draft evidence-grounded candidate corrections for the remaining 32 wrong cases. Use exact displayed evidence/citation offsets; do not force a rationale where retrieval cannot support the source gold. The draft for train:25 also needs review, not automatic acceptance.
2. Apply candidate annotations to a NEW errors33 bank revision, then export NEW independent review sheets. The annotation template has blank correction fields for every row: include only completed annotations and omit the existing draft if it is not being revised. Do not apply blank rows as corrections. Keep the original banks/sheets intact.
3. Two distinct humans review the same candidate revision independently under [the protocol](../templates/m161-review/REVIEW_PROTOCOL.md). Record rejection or insufficient evidence honestly. Import completed sheets; no author/assistant identity is substituted for a reviewer.
4. Prepare/review appropriate non-error controls from the 42 correct cases and measure matching coverage. Availability of 42 records is not proof of adequate matching or verified rationales.
5. Reassess source coverage: at least 30 verified errors, at least three mechanisms with five verified cases each, and at least 80% source-only matched-control coverage. Only then decide the controlled development pilot. If coverage falls short, report which mechanisms/examples are missing before proposing another separately authorized source batch.

No further GPU source batch is justified merely because reviews are incomplete. There are already 33 raw errors, but their verified count is still zero. M1.6.2 readiness is **MODIFY**; no HEAL inference, quality improvement claim or confirmation is made here.

Candidate/review commands, after authoring actual annotation content:

```bash
python -m tffv.experiments.m161 apply-candidates \
  --memory outputs/m161-review-preparation-v1/errors33.json \
  --annotations outputs/m161-review-preparation-v1/candidate_annotations_filled.json \
  --output outputs/m161-review-preparation-v1/errors33-candidates-v2.json
python -m tffv.experiments.m161 export-reviews \
  --memory outputs/m161-review-preparation-v1/errors33-candidates-v2.json \
  --output outputs/m161-review-preparation-v1/error-review-round1
# Two humans fill separate reviewer_1.json / reviewer_2.json in that NEW directory.
python -m tffv.experiments.m161 import-reviews \
  --memory outputs/m161-review-preparation-v1/errors33-candidates-v2.json \
  --review-1 outputs/m161-review-preparation-v1/error-review-round1/reviewer_1.json \
  --review-2 outputs/m161-review-preparation-v1/error-review-round1/reviewer_2.json \
  --output outputs/m161-review-preparation-v1/errors33-reviewed-v1.json
python -m tffv.experiments.m161 validate-memory \
  --memory outputs/m161-review-preparation-v1/errors33-reviewed-v1.json
```

The filled annotation/review files do not exist yet; these commands document the next human-content workflow. CPU validators cannot replace semantic review. No new GPU work was launched during this preparation.
