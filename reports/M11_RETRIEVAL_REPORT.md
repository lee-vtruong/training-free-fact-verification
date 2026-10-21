# M1.1 — Retrieval pipeline implementation report

Updated: 2026-10-21, Asia/Saigon. **INCOMPLETE: empirical benchmark awaits user-run
server commands and verified corpus/model inputs.** M0 is understood as setup
and work-rule adoption, per the user's clarification.

## Delivered locally

R1 BM25 Top-50; R2 frozen dense Top-50; R3 RRF Top-40; R4 frozen reranking of
R3 Top-40. Shared segmentation, Top-8 schema, provenance, numeric/polarity-aware
deduplication and separate Top-10 evaluation ranking are implemented.
Preflight, snapshot hashing, immutable corpus-vector reuse, sealed run caches,
offline qrels evaluation, split-aware summaries, leakage flags and manual case
review queue are available. Real Qwen adapters require server smoke validation.

Latest local checks: **87 tests passed**. Initial implementation checks were
46 tests passed; compilation and CLI help succeeded. Offline
synthetic smoke used 3 explicit fixture claims and 64 synthetic documents,
with non-neural test doubles. It produced four Top-8 caches and a comparison
report under `outputs/m11-synthetic-smoke-001/`. No actual Qwen inference,
checkpoint download or server GPU job was performed.

## Actual AVeriTeC results

| Method | Recall@5/8/10 | MRR@10 | nDCG@10 | Evidence coverage/diversity | Latency / VRAM |
|---|---|---|---|---|---|
| R1 | not measured | not measured | not measured | not measured | not measured |
| R2 | not measured | not measured | not measured | not measured | not measured |
| R3 | not measured | not measured | not measured | not measured | not measured |
| R4 | not measured | not measured | not measured | not measured | not measured |

No admitted quality-benchmark evidence caches, valid relevance annotations, or
superiority claims exist yet. An unadmitted real-source CPU R1 operational cache
now exists as reported below. Synthetic caches are software validation and do not satisfy
the empirical evidence-cache deliverable. Do not infer a best retrieval method.

## User-confirmed CPU R1 operational result — 2026-10-21

Run: `outputs/m11-r1-operational-20261021-060217`; scope
operational_smoke_not_quality_benchmark, corpus_admitted=false. User-reported
available RAM 98 GiB of 125 GiB; no swap, disk available 125G (df units).
Candidate manifest SHA256 d1c42758ca4e2773f6921e781adf7c55c7c6291cc8a3de3af74d0dc12c66d397;
configuration SHA256 95ef8b96a04df1b5dd5ed97f824297f250fbace9a2ac834d8a4ce6e06b3d37da.

| Source claim | Documents | Passages | Segmentation s | Index s | Query ms |
|---|---:|---:|---:|---:|---:|
| train:0 | 424 | 27122 | 1.8521 | 2.3714 | 70.17 |
| train:1 | 752 | 50151 | 3.0621 | 4.3967 | 104.73 |
| train:2 | 423 | 32979 | 1.9123 | 2.8179 | 97.19 |

Total 110252 passages; elapsed reported 18.9617 s, CPU-only, no GPU. The elapsed
timer starts after input loading/validation, so is not shell end-to-end runtime.
Query timing excludes segmentation/index/dedup/serialization. Peak CPU memory
was not measured. All three claims received 8 passages (24 evidence slots total).
Means: slot coverage 1.0, selected exact normalized text redundancy 0.0,
pairwise token Jaccard 0.177099, unique domains 6.0, unique documents/URLs
6.666667, domain entropy 1.711066. These are sample means of proxies, not proof
of relevance, semantic sufficiency or absence of near-duplicates.

Recall@5/8/10, MRR@10, nDCG@10 and semantic coverage remain null. Only R1 ran;
no R2/R3/R4 comparison, verified quality gain or final baseline selection. Full
24-passage source review was performed below. This exploratory three-claim run does
not establish leakage clearance or complete the empirical milestone.

## Evidence Leakage Review on Retrieved Evidence — 2026-10-21

Artifact: `outputs/m11-leakage-review-20261021-070305/review_report.json` and `review_report.md`.
Policy: `evidence_leakage_review_v2` (v2.0.0).
Input: 24 retrieved evidence passages from CPU R1 operational smoke.
Status: `complete_evidence_leakage_review`, label-free audit.

| Total items reviewed | Flagged items | Quarantined items | Flag percentage |
|---:|---:|---:|---:|
| 24 | 5 | 0 (flag_only mode) | 20.8% |

Flag breakdown:
- `fact_check_url_path`: 3 passages (`freep.com/.../rnc-2020-fact-check/...`, `abcnews.go.com/.../state-union-fact-check-...`, `forbes.com/.../fact-checking-trumps-recent-economic-tweets/...`)
- `claim_verdict_coupling`: 2 passages (`usatoday.com/...what-pam-bondis-rnc-attacks-hunter-biden-got-right-and-wrong/...`)
- `verdict_pattern`: 2 passages (`usatoday.com/...` quoting *"Bondi also re-aired a false claim..."*, and ABC News State of the Union fact-check).

These findings provide direct empirical evidence that approximately 20.8% of retrieved
passages in raw web-scraped stores contain third-party verdict contamination.
Corpus remains strictly `not_admitted`.

## Generalized Claim-Scoped Retrieval Runner (R1 Execution) — 2026-10-21

Run: `outputs/m11-store-r1-20261021-071247`; scope `claim_scoped_operational_smoke_not_admitted`.
Candidate manifest: `outputs/m11-store-candidates-20261021-055753/candidate_manifest.json`.
Execution mode: CPU R1, strictly isolated per-claim retrieval across 3 candidate stores.
Total documents: 1599; total segmented passages: 110252; total elapsed: 20.6621 seconds.

| Store | Documents | Passages | Segmentation s | BM25 Index s | Query ms |
|---|---:|---:|---:|---:|---:|
| train:0 | 424 | 27122 | 1.8832 | 2.3189 | 74.32 |
| train:1 | 752 | 50151 | 3.6378 | 4.7987 | 155.31 |
| train:2 | 423 | 32979 | 2.3749 | 3.0953 | 143.86 |

All 3 claims received 8 evidence passages (24 evidence slots total).
Means: slot coverage 1.0, exact text redundancy 0.0, mean pairwise token Jaccard 0.177099,
unique domains 6.0, unique URLs 6.666667, domain entropy 1.711066.
Relevance metrics remain null. R2-R4 await checkpoint and environment readiness.

## Initial infrastructure snapshot and limits (historical)

Current local probe: RTX 3050 Laptop 6 GB; about 5.52 GiB free workspace disk;
Python 3.13.11, torch 2.11.0, transformers 4.45.2, numpy 2.4.4. Local transformers
is too old for the Qwen3 adapter. No environment was modified. Server resources,
model revisions, corpus origin and actual AVeriTeC paths remain unverified.
The earlier SSH attempt was contrary to the user's intended workflow and timed
out; it is not evidence of server access. Further server execution is user-only.

Qrels are optional and must have validated passage mapping. Without them,
relevance stays null; coverage/redundancy/diversity alone cannot establish which
method finds better factual evidence. Real good/bad retrieval assessment and
numerical/temporal/conflicting case review remain pending. Slot coverage is not
semantic sufficiency. Timing definitions and lexical leakage limitations are
explicit in `docs/M11_RETRIEVAL_PROTOCOL.md`.

## Server evidence update — 2026-10-21

The user's synthetic smoke completed at
`outputs/m11-synthetic-20261021-040558`, exporting all four configurations with
slot coverage 1.0. These are non-neural test doubles, not Qwen results. Relevance
metrics remain not measured. No unit-test summary from the server has been
supplied yet; latest local suite after the follow-up probe is **50 passed**.

User-supplied resource inventory now confirms RTX 5090 GPU metadata (32,607 MiB
total), about 124.40 GiB free storage, and old AVeriTeC train/dev file paths.
The active venv has numpy/pytest but no torch/transformers. Required Qwen 0.6B
retrieval checkpoints were not found in the two scanned model roots; other
models were found. No independent AVeriTeC corpus or valid passage qrels have
been verified. The earlier infrastructure paragraph records the initial local
snapshot, not the updated server findings.

A source-schema/corpus/environment metadata probe is provided next. Read-only
old-environment reuse and any later GPU package installation require actual
dependency/CUDA checks. No backbone substitution or new corpus choice is made.
Empirical M1.1 remains INCOMPLETE.

Follow-up source probe found an existing Qwen3-Embedding-0.6B snapshot in the
custom whale/cache/huggingface tree, revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`. It also confirmed 3068 raw and
processed source-train rows with file hashes. Reranker-0.6B and an independent
AVeriTeC corpus remain unverified. GraphCURE has a torch/transformers environment,
but its transformers 5.15.0 is outside the current project range and no runtime
reuse is certified. A hash-checked source claims exporter is added; latest local
suite including metadata-only corpus planning is **62 passed**. The user confirms
the knowledge store has not been downloaded; exact acquisition metadata is the
next user-run step. These are setup advances, not retrieval quality results.

The user's acquisition plan resolves corpus revision
`2ca9dee23a2a6fa64c5bd918e0cd28ed0aa09031`, with 63,524,717,040 compressed
train-archive bytes (59.1620 GiB). No archive transfer/extraction occurred in
that metadata step. Bounded HTTP Range ZIP/ZIP64 inspection and source shape
sampling are now implemented, with **67 local tests passed**. Live archive
inspection, actual corpus admission and all real retrieval metrics remain pending.

User Range inspection has now succeeded: 296,894 payload bytes inspected all
train directories; declared raw size is 187.0848 GiB, compressed+raw 246.2468 GiB.
After cleanup the user reports 125.03 GiB free, so full extraction remains
infeasible. Source store schema is JSON records with claim_id/query/type/url/
url2text; some rows have no text. A capped complete-member smoke acquisition
tool is added, with **72 local tests passed**. This advances data preparation,
not the full retrieval evaluation or corpus admission.

The user's complete-store smoke acquisition succeeded for train ordinals 0/1/2:
130,960,407 raw bytes, 42,228,262 range payload bytes, selected-member CRC and
raw hashes recorded. Initial audits found 546 rows/115 empty/2 verdict-pattern
flags in store 0 and 603 rows/178 empty/1 flag in store 2. Store 1's audit is
unavailable at the original 8 MiB record cap; no row counts are guessed.
An explicit 64 MiB-record offline audit and bounded source review queue are
added, with **76 local tests passed**. No source exclusion, corpus admission,
Qwen execution or retrieval-quality inference is performed on these observations.

## M1.2 candidates

Latest server candidate conversion succeeded: retained records 424/752/423
for train:0/1/2 (1599 total), flagged quarantine 25, empty text 473. Corpus
admission remains false; these are preprocessing counts, not retrieval quality.
A CPU-only R1 operational runner tests claim-scoped indexing and the shared
output schema without relaxing admitted benchmark guards. Its execution has
succeeded as reported above; Recall/MRR/nDCG remain not measured. No R2-R4 runs or
method recommendation can be inferred from this setup progress.

Source claims export is now user-verified: 3067 valid claims, one quarantined
empty claim train:1948, no renumbering. Claims hash is
f1382c032a1c5746deae74fed4c6b870263754da0380c76b20f9e020bbaae52d.
This is a valid-claim train subset, not full official-split coverage. A separate
offline candidate converter now preserves per-claim pools and accounts for all
records while optionally quarantining the existing heuristic flags. This policy
has false positives and unknown residual leakage; no benchmark admission occurs.
Server conversion and the operational R1 cache now exist. Admitted R1-R4
quality-benchmark caches remain pending.

Latest user-run full-record audit completed in 8.6962 seconds for source stores
0/1/2: 2097 rows, 473 empty-text rows, 44 within-store exact duplicate rows,
8 verdict-pattern flags and 17 fact-check URL flags. All 25 flagged rows were
exported; no ID/schema/URL/annotation-field failures were observed. Estimated
passages before filtering: 127147. These are corpus diagnostics, not retrieval
metrics or evidence that the full corpus is clean. Source 1 is now fully audited
(948 rows); its previous parser-cap failure is resolved without raw-data changes.
Excerpts contain both actual fact-check verdict prose and legal-verdict false
alarms. Later claim-only context confirms same-claim BBC answer-bearing content
for train:1 in the raw store; residual leakage after filtering remains unresolved.
Corpus admission and empirical M1.1 remain INCOMPLETE.

Keep R1–R4 as candidates pending matched real evaluation. R1 supplies the lexical
reference; R2 tests frozen semantic retrieval; R3 tests complementary rank fusion;
R4 tests reranking beyond the same R3 candidate set. Recommend a shortlist only
after inspecting real relevance, case reviews, latency and memory. Do not select
R4 automatically. No verifier inference or historical error-memory work is done.

## Reproduce and hand off

Run `python scripts/run_tests.py` locally. User-run server commands are in
`docs/M11_SERVER_COMMANDS.md`; start with metadata-only resource reconnaissance
and send its output back. Full benchmark is not automatically launched.
The repository is on `main` with GitHub origin; user review/commit/push and
server synchronization remain separate user actions. Codex does not run them.
