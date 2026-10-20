# M1.1 retrieval protocol and contracts

Status: implementation and offline synthetic tests available; **real AVeriTeC
benchmark INCOMPLETE**. User runs all server commands. Codex does not SSH,
launch server jobs, push/pull repositories, or download checkpoints.

## Frozen configurations

| ID | Candidate ranking | Shared evidence output |
|---|---|---|
| R1 | BM25, Top-50, k1=1.2, b=.75 | Deduplicated Top-8 |
| R2 | Frozen Qwen3-Embedding-0.6B cosine, Top-50 | Deduplicated Top-8 |
| R3 | RRF of R1 Top-50 and R2 Top-50, constant=60, Top-40 | Deduplicated Top-8 |
| R4 | Frozen Qwen3-Reranker-0.6B scores on exactly R3 Top-40 | Deduplicated Top-8 |

No quality selection has been performed. Query instruction is fixed in code,
label-free, and asks for evidence relevant to verifying **or refuting** the claim.
R1 returns only passages with lexical matches; no zero-score padding. A Top-8
shortfall is explicit. All four methods use identical claims, document corpus,
segmentation, passage IDs, and final deduplication policy. R4 reorders R3's
candidates; it never searches a different corpus.

Default segmentation: 180 whitespace-delimited words, overlap 30, exact original
character offsets. This is a reproducible initial configuration, not a tuned
claim about optimal segmentation. IDs hash document ID, offsets and text hash.
Corpus text is not rewritten. Official splits are retained; no folds/random
resplits are introduced, and individual claim labels are absent from retrieval.

Deduplication is greedy in retrieval order, comparing normalized text, token
Jaccard and character similarity at .95. Numeric counts and selected polarity/
temporal terms must match for near-duplicate suppression. It is a lexical
heuristic and can miss or misclassify semantic duplicates. Suppression IDs and
their retained representative are recorded; judgments are never inferred from
duplicate similarity. Different URLs remain visible in the full passage index
and raw candidate list even if a duplicate passage is suppressed from evidence.

## Claims input: strict label-free JSONL

Each object has exactly these string fields:

```json
{"sample_id":"train:ORIGINAL_ID","dataset":"AVeriTeC","split":"train","claim":"Original claim text"}
```

Allowed splits: `train`, `dev`, `test`. Namespace IDs to prevent cross-split ID
collisions and document the mapping to official IDs. Do not invent a claim ID
mapping or split membership without checking the release. Input counts and hashes
must match the data manifest. Ground-truth QA, gold labels, label-derived tags
and justifications are forbidden fields. The four-class ontology is recorded
in the manifest; it is not projected to three classes or used for retrieval.

## Independent corpus input: strict document JSONL

Exactly these non-empty string fields are required:

```json
{"document_id":"ORIGINAL_DOCUMENT_ID","text":"Independent source document text","source_url":"https://source.example/document","source_kind":"web_document","retrieved_at":"ACTUAL_ACQUISITION_TIME","acquisition_id":"VERSIONED_ACQUISITION_RUN"}
```

`source_kind` is `web_document` or `knowledge_base_document`. Use actual
retrieval timestamps and acquisition IDs; never fill unknown provenance with
guesses. Raw AVeriTeC annotations, processed ground-truth QA, gold rationales,
gold justifications, and verdict-bearing records are **not** an eligible corpus.
Existing ERRORFlow processed evidence is not automatically admissible. A corpus
selected using annotation URLs/answers must be identified as an oracle/restricted
condition and cannot be silently substituted for this label-free experiment.

`retrieval_data_manifest.template.json` fails until hashes, split counts and
manual lineage/leakage review are completed. `true` flags are attestations
requiring evidence, not a way to bypass review. Preserve raw corpus hashes,
acquisition pipeline/version, extraction rules and review details in provenance.
Do not set all flags true merely to make the command run.

## Leakage checks and limitations

- Strict field whitelists reject annotation metadata at the retrieval boundary.
- Verdict-pattern text blocks a run; sanitize/exclude upstream only with recorded
  evidence and a new input hash. Nothing is silently removed inside retrieval.
- Claim substrings, exact duplicate documents, known fact-check URL markers,
  normalized duplicate claims and cross-split overlaps are flagged for review.
- Near-duplicate claims use token-set Jaccard >=.90. Flags may include numerical
  or polarity changes. Official splits are never changed by this screen.
- The lexical screen is O(number of claims squared); it is intended for the
  AVeriTeC-size claim population. Corpus loading/BM25 indexing are in memory;
  report RAM pressure before scaling to a much larger corpus.
- Pattern scans and origin attestations do not prove absence of paraphrased
  verdicts or contamination. Review raw text and acquisition lineage. No corpus
  has been certified leakage-free in this local milestone.

## Models and numerical behavior

Adapters follow the primary model documentation:
[Qwen3 Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) and
[Qwen3 Reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B).
Embedding uses instructed queries, last-token pooling and L2 normalization;
documents receive no query instruction. Reranking uses the model's yes/no
template and a two-logit probability. This score is relevance-oriented, not
confidence that a claim is factually correct.

Models load only from local paths, with `local_files_only=True` and no remote
code. They run in eval/inference mode with gradients disabled for every
parameter. No training, classifier fitting, learned index or prompt tuning.
Transformers >=4.51 is required; the inspected local 4.45.2 is incompatible.
Do not modify old environments to fix this. Real model execution remains
untested until the user runs server smoke tests with verified snapshots.

Pin full 40-character revisions plus content fingerprints. `fingerprint` reads
snapshot files without inference; paths named as HF snapshot commits must
match the configured revision. Other local paths require original revision
provenance from the acquisition record. A content hash alone does not prove
which upstream revision supplied the weights. Hashes bind all safetensors and
supported tokenizer/config/template files, excluding hidden downloader state.

Maximum model length is explicitly 1024 tokens in the initial template, batch
size 1, float16 on CUDA. Truncation counts and token totals are saved. Embedding
and reranker are loaded sequentially to avoid simultaneous model residency.
This does not guarantee a particular GPU can run the workload. Backend weights,
tokenization, GPU behavior and precision still require real smoke verification.

## Metrics and output interpretation

Each `R?.jsonl` includes raw Top-50/40 candidates, component scores/ranks,
`evaluation_ranking_top10` after shared deduplication, Top-8 evidence with exact
passage/source provenance, shortfall, timing and GPU-memory fields.
Top-8 is the prefix of that deduplicated Top-10; Recall@5/8/10, MRR@10 and
nDCG@10 therefore use the same final ranking. A reranker changes order, not
the candidate set before deduplication.

Coverage means filled evidence slots and fraction of claims with any evidence.
Semantic evidence completeness remains null. Redundancy includes exact text
duplication and mean pairwise token Jaccard; diversity includes unique documents,
URLs, domains and domain entropy. None is a substitute for judged relevance.

Relevance annotations are opened **only** by `evaluate`, after checking sealed
retrieval hashes. Qrels must be validated passage-level human/official judgments
with provenance and an exact passage mapping. Annotation URLs or QA overlap
cannot be automatically promoted to passage relevance. The evaluator never
manufactures labels, translates claim verdicts into relevance, or uses an LLM
to judge relevance.

With incomplete qrels, recall uses known judged positives and unjudged passages
are explicitly treated as nonrelevant. No-positive and unjudged queries receive
null metrics and are excluded with measured counts. This is not exhaustive
recall. Means are reported overall and per official split. No qrels means all
relevance metrics remain null. `manual_review_queue.jsonl` flags numerical,
temporal and conflicting-language cases for human review; these are heuristic
inspection tags, not gold categories or good/bad retrieval judgments.

Query latency is synchronized wall time for ranking/model inference. R3/R4
sum reused BM25/dense/fusion/reranking components, so these are reconstructed
pipeline latencies, not four independently timed cold deployments. Model load,
snapshot verification and index build are separate. Dedup/serialization and
evaluation are excluded from query timings. CUDA peak is PyTorch allocator peak
for sequential model residency, not whole-machine VRAM. CPU/unmeasured GPU values
are null. A cached index's original build peak is explicitly historical metadata.

## Reproducibility and cache lifecycle

Inputs are read-only. All new outputs are under this checkout's `outputs/`.
Fresh run directories are mandatory; no overwriting or writes to ERRORFlow.
Content-addressed corpus vectors are shared under the new index cache, binding
passages, checkpoint/revision/hash, precision, context length, seed, batch size
and code hash. Reuse verifies metadata and vector-file hash. Interrupted indexes
are rejected; use a different index-cache directory for a retry, preserving the
failed artifact. Query/reranker outputs remain in the completed evidence cache;
reuse them directly rather than rerunning an already completed experiment.

`manifest.json` is written last. It seals selected IDs, all evidence caches,
input hashes, code/config hashes, package versions, model settings and GPU info.
A failure writes `failure.json` and no completion seal. Evaluation has a separate
manifest and never edits retrieval outputs. Unknown costs stay null.

The bounded `--max-claims` / `--max-passages` modes are explicitly smoke scope;
they are not full-benchmark results. Every method receives the identical bounded
corpus. The deterministic prefix can be unrepresentative and must not select
the final method. No Qwen3-8B verifier or Historical Error Memory is part of M1.1.
