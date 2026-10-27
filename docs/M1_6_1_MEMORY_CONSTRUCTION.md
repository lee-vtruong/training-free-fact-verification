# M1.6.1 — Exploratory Historical Error Memory Construction

Date: 2026-10-27. CPU implementation and legacy serialized-artifact audit delivered. **Human correction review and batch-25 GPU execution INCOMPLETE / NOT RUN.** M1.6.2 readiness: **MODIFY**. No HEAL quality result is claimed.

The user's M1.6.1 instruction authorizes exploratory source work independently of clean-holdout resolution. The stale M1.1-only AGENTS scope line does not block this explicitly requested task. All other frozen-model/source-only rules apply. Clean confirmatory test eligibility remains **0**, and no v2 partition/schedule or TEST_QUARANTINE_EXPOSED assignment changes.

## Implementation and artifacts

- [Source-memory contracts](../src/tffv/memory/heal.py): sealed SOURCE reader; strict source-only gold join; immutable record/content hashes; legacy raw/prompt/evidence audit; candidate and two-reviewer pipelines; explicit-ID views and duplicate-safe bank merge.
- [CLI and fixed batch runner](../src/tffv/experiments/m161.py): CPU commands separated from `run-batch --execute`. Private inference worker has scrubbed claims/evidence, no gold argument. Post-inference gold join is a different process.
- [Memory JSON schema](../schemas/heal_memory_v1.schema.json), [review-sheet schema](../schemas/heal_review_v1.schema.json), [review protocol](../templates/m161-review/REVIEW_PROTOCOL.md), and [CPU safety tests](../tests/test_m161_memory.py).
- Local actual bank: `outputs/m161-legacy-audit-v3/memory.json`; source summary: `outputs/m161-legacy-audit-v3/summary.json`; independent blank sheets and annotation template: `outputs/m161-legacy-review-v2/`. These raw source artifacts are ignored outputs; the runbook reproduces them on the server. Only audit summaries/hashes/templates are committed.

Each record retains source claim/ID/group, full retrieved evidence and exact displayed text, original provenance/offsets/hashes, Base stored/reparsed/raw prediction and cost fields, official source gold, **wrong -> gold** confusion type, candidate mechanism flags, nullable explanation/citations, review state, review-file/parent-bank hashes and limitations. Source gold joins select only the explicitly admitted IDs and never copy QA/justifications. The legacy gold-file JSON is parsed offline, but only selected source claim/label values are accessed for construction; no development/test gold influences selection or inference.

One concrete **assistant-authored, unverified** candidate was prepared for `train:25`: the Base output acknowledges a below-1% budget but reads “not even 1%” as a minimum. Exact displayed 0.43%/0.44% spans are cited; intended period/denominator still need human review. Files: `outputs/m161-assistant-candidates-v1.json`, `outputs/m161-candidate-bank-v1.json`, and two blank sheets in `outputs/m161-candidate-review-v1/`. It stays PENDING_REVIEW with zero human decisions. Original blank-bank exports remain separate; no other error is assigned an invented explanation.

`record_id` binds immutable source facts, Base output, evidence and provenance. `content_sha256` additionally covers the evolving candidate/review state. A review pins the source record revision and candidate hash. Candidate changes or edited sheet context invalidate older reviews. New files/revisions are required; old outputs are not overwritten.

## States and source-error interpretation

| State | Meaning / permitted use |
|---|---|
| PENDING_REVIEW | No completed independent correction adjudication; never a verified rationale |
| VERIFIED | Same candidate approved by two distinct attested humans, with valid citations and all grounding checks |
| REJECTED | At least one reviewer rejects; excluded from demonstrations |
| INSUFFICIENT_EVIDENCE | Missing evidence or reviewer judgment that retrieval cannot support the gold correction; excluded |
| PROVENANCE_UNKNOWN | Required serialized-artifact checks fail; excluded |

A Base incorrect verdict is a reproducible raw verdict/gold mismatch. Parser error is raw/stored disagreement or invalid parse, kept out of semantic error examples. Incomplete evidence requires evidence review; eight passages alone do not establish sufficiency. A genuinely supported reasoning failure requires two human judgments on the failure category, in addition to a verified rationale. None is inferred automatically from a confusion pair or regex.

Legacy runtime weights/effective-token fingerprints were not originally sealed. Available raw artifacts can still be checked and queued for review, but no new hash is represented as proof of historical runtime bytes. Exact original raw-store/document offsets have not been re-audited here; these limitations stay in each record. Recovery is specified in the [source audit](M1_6_1_SOURCE_AUDIT.md).

## Correction workflow

Use actual exported `candidate_annotations_template.json`, fill a selected subset with source-evidence-grounded candidate text/citations, and save it as a separate annotations file. Empty candidates remain null; do not automatically copy gold justifications. Then:

```bash
python -m tffv.experiments.m161 apply-candidates \
  --memory outputs/m161-legacy-source-v1/memory.json \
  --annotations outputs/m161-candidates.json \
  --output outputs/m161-candidate-bank-v1.json
python -m tffv.experiments.m161 export-reviews \
  --memory outputs/m161-candidate-bank-v1.json --output outputs/m161-review-round1
# Two humans fill separate sheets following templates/m161-review/REVIEW_PROTOCOL.md.
python -m tffv.experiments.m161 import-reviews \
  --memory outputs/m161-candidate-bank-v1.json \
  --review-1 outputs/m161-review-round1/reviewer_1.json \
  --review-2 outputs/m161-review-round1/reviewer_2.json \
  --output outputs/m161-reviewed-bank-v1.json
python -m tffv.experiments.m161 validate-memory --memory outputs/m161-reviewed-bank-v1.json
```

Reviewing a case as insufficient/rejected is possible without inventing a correction. Both sheets cover all original IDs; paired unfilled rows remain pending. One completed review alone does not promote a record. This software checks attestations, not real-world identity; semantic review is not automated.

## M1.6.2 interface preparation only

`view --ids ... --mode gold_only|wrong_gold|wrong_gold_verified|contrastive|mechanism` exposes fixed source IDs. Gold-only and wrong+gold hold claim/evidence/gold/record IDs constant and hide the explanation; the latter adds the actual Base verdict. Verified views require completed reviews; mechanism views additionally require both reviewers to confirm the requested mechanism. Contrastive requires exactly one correct and one incorrect record. No target query, target label, exemplar retrieval, HEAL prompt generation or HEAL inference is implemented here.

`merge-memory --memory <legacy> <batch> --output <new-bank>` combines source banks without silently replacing duplicate IDs or reviews. Select one exact bank revision for matched information ablations; do not transfer review sheets to a regenerated bank with different record hashes. Later common-bank/length/decoding controls remain those of M1.6, with D0/D1_matched retained and M4 auxiliary.

## Current decision

50 legacy records, 25 reproducible incorrect verdicts, 0 parser errors, **0 verified corrective rationales / 0 verified reasoning failures / 0 reviewed mechanism support**. All 50 evidence-sufficiency assessments remain pending. The 25-error bank cannot reach the ≥30 verified-error coverage gate even if every old error passes review; controlled source expansion is needed for that registered target, and review is independently necessary. Run only the first fixed 25 when the user elects to execute the [batch runbook](M1_6_1_BATCH25_RUNBOOK.md). Do not process the full source pool automatically.

Status **MODIFY** means CPU tooling and provisional gold/wrong information views are ready; full verified-rationale/mechanism HEAL pilot is not. First obtain real reviews, source coverage and matched controls, then decide M1.6.2. Actual new GPU cost is zero; future batch cost remains the historical estimate (about 1.88–2.66 h, conservative reservation 5.57 h), not measured new compute. Monetary cost and new GPU-active time remain unmeasured. Clean-holdout resolution does not gate this exploratory source phase.
