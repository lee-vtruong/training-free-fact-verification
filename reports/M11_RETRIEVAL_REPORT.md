# M1.1 — Retrieval pipeline implementation report

Date: 2026-10-20, Asia/Saigon. **INCOMPLETE: empirical benchmark awaits user-run
server commands and verified corpus/model inputs.** M0 is understood as setup
and work-rule adoption, per the user's clarification.

## Delivered locally

R1 BM25 Top-50; R2 frozen dense Top-50; R3 RRF Top-40; R4 frozen reranking of
R3 Top-40. Shared segmentation, Top-8 schema, provenance, numeric/polarity-aware
deduplication and separate Top-10 evaluation ranking are implemented.
Preflight, snapshot hashing, immutable corpus-vector reuse, sealed run caches,
offline qrels evaluation, split-aware summaries, leakage flags and manual case
review queue are available. Real Qwen adapters require server smoke validation.

Local checks: **46 tests passed**; compilation and CLI help succeeded. Offline
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

No real evidence caches, valid relevance annotations, error cases, or superiority
claims exist yet. Synthetic caches are software validation and do not satisfy
the empirical evidence-cache deliverable. Do not infer a best retrieval method.

## Infrastructure and limits

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

Source claims export is now user-verified: 3067 valid claims, one quarantined
empty claim train:1948, no renumbering. Claims hash is
f1382c032a1c5746deae74fed4c6b870263754da0380c76b20f9e020bbaae52d.
This is a valid-claim train subset, not full official-split coverage. A separate
offline candidate converter now preserves per-claim pools and accounts for all
records while optionally quarantining the existing heuristic flags. This policy
has false positives and unknown residual leakage; no benchmark admission occurs.
Server conversion and all R1-R4 real evidence caches remain pending.

Latest user-run full-record audit completed in 8.6962 seconds for source stores
0/1/2: 2097 rows, 473 empty-text rows, 44 within-store exact duplicate rows,
8 verdict-pattern flags and 17 fact-check URL flags. All 25 flagged rows were
exported; no ID/schema/URL/annotation-field failures were observed. Estimated
passages before filtering: 127147. These are corpus diagnostics, not retrieval
metrics or evidence that the full corpus is clean. Source 1 is now fully audited
(948 rows); its previous parser-cap failure is resolved without raw-data changes.
Excerpts contain both actual fact-check verdict prose and legal-verdict false
alarms. Same-claim answer leakage remains unresolved pending claim-only context.
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
