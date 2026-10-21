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

## M1.2 candidates

Keep R1–R4 as candidates pending matched real evaluation. R1 supplies the lexical
reference; R2 tests frozen semantic retrieval; R3 tests complementary rank fusion;
R4 tests reranking beyond the same R3 candidate set. Recommend a shortlist only
after inspecting real relevance, case reviews, latency and memory. Do not select
R4 automatically. No verifier inference or historical error-memory work is done.

## Reproduce and hand off

Run `python -m pytest -q` locally. User-run server commands are in
`docs/M11_SERVER_COMMANDS.md`; start with metadata-only resource reconnaissance
and send its output back. Full benchmark is not automatically launched.
The repository currently has no commit or remote and remains on `master`;
user review/commit/push and server synchronization are separate user actions.
