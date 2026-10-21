# M1.1 corpus readiness

Status: **independent AVeriTeC corpus still unverified**, 2026-10-21.
The source probe established 3068 train annotations and a reusable embedding
snapshot candidate. Annotation QA and processed gold evidence are not the corpus.
The user has now confirmed the knowledge store has **not been downloaded**.
The next step is public API metadata planning, not another local path scan.

Primary resources checked:

- [Official AVeriTeC page](https://fever.ai/dataset/averitec.html) links annotations
  separately from the provided evidence collections.
- [Official shared-task repository card](https://huggingface.co/chenxwh/AVeriTeC)
  describes pre-retrieved web documents for each claim and records a corrected
  test knowledge-store release. No sample gold labels are used for method design.
- [Train knowledge-store listing](https://huggingface.co/chenxwh/AVeriTeC/tree/main/data_store/knowledge_store/train)
  currently lists `train_0_999.zip` (~20.7 GB), `train_1000_1999.zip` (~21.3 GB)
  and `train_2000_3067.zip` (~21.5 GB). These are published rounded compressed
  sizes, not a measured local footprint or an estimate of extraction space.
- [Dev archive page](https://huggingface.co/chenxwh/AVeriTeC/blob/main/data_store/knowledge_store/dev_knowledge_store.zip)
  lists ~11.5 GB. Its existence is not permission to inspect dev labels or tune
  a retrieval pipeline on external outcomes.

Do not bulk-download or extract all archives from a floating `main`. Before
acquisition, confirm whether a suitable local copy exists, select a documented
release/revision, list exact files/sizes/hashes, inspect archive member metadata,
and budget compressed data + extraction + passage vectors + headroom. Current
free space (~124.40 GiB) alone does not prove all archives and derived outputs
will fit. The user executes downloads; Codex only prepares code/commands.

The official knowledge store is organized per claim. The current canonical
benchmark consumes one declared document pool. Do not silently union all stores
or switch between global and claim-scoped retrieval: document the eligible corpus
scope and apply the identical scope to R1–R4. An adapter needs actual archive
schema and source lineage before implementation. No experiment scope or default
corpus has been changed in this step.

Admission review must distinguish raw scraped web text from gold-annotated QA,
generated QA and verdict-bearing documents. Preserve URLs, original claim/store
mapping, release and filtering lineage. A corrected test release does not imply
every train document is automatically leakage-free. Gold source URLs must not
be used to select the retrieval pool. No URL-match is promoted to passage qrels.

Safe work while corpus availability is clarified: export all source train claims
without annotation fields using the verified source hash; fingerprint the existing
embedding snapshot later; inspect existing runtime health without installing into
old environments. Reranker-0.6B acquisition and runtime selection remain separate
pending tasks. No real retrieval or comparative quality claim is possible yet.

## Metadata-only acquisition plan

`scripts/m11_corpus_plan.py` uses only the Python standard library. It requests
the public repository-info JSON with blob metadata, caps response reads at 4 MiB,
resolves the requested ref into a full commit, and selects only the three known
train ZIPs. Every file must have an exact size and 64-character LFS SHA256;
missing/changed metadata fails closed. Plan URLs use the resolved commit, never
floating main. Public API response and planner code are hashed in the plan.
See [Hub API documentation](https://huggingface.co/docs/hub/api).

It never opens a `/resolve/` archive URL, invokes a downloader, installs packages,
extracts archives or reads annotation labels. Network behavior is validated with
offline HTTP test doubles; live server API access is still for the user to test.
An optional `--metadata-json` mode replays a previously saved metadata response
without network access.

`compressed_only_fits` compares only archive bytes with current free disk.
`ready_for_bulk_download` remains false regardless of that result because
extraction, passage/vector storage and headroom remain unmeasured. This is a
readiness report, not an admission certificate or automatic download decision.
After inspecting the plan, prepare a bounded archive-member/schema inspection
and explicit acquisition commands; avoid indiscriminate full-repository cloning.

## Metadata result supplied by the user — 2026-10-21

Plan: `outputs/m11-corpus-plan-20261021-042129/acquisition_plan.json`.
Resolved revision: `2ca9dee23a2a6fa64c5bd918e0cd28ed0aa09031`.

| Train archive | Exact compressed bytes | SHA256 from repository metadata |
|---|---:|---|
| train_0_999.zip | 20,730,339,734 | 389d2284c5f25410205ec530c2101c3e6d575eae460731e1909ad19192dcb810 |
| train_1000_1999.zip | 21,273,604,733 | e2bf5c1dccc855cf3474ccd3f14678fe60f21e6c69685a34d7bc884996fec446 |
| train_2000_3067.zip | 21,520,772,573 | d3e27cc4072cd085dbac44272f1ea918dd3ff970b39d94f143e40c51bbde43d0 |

Total compressed: 63,524,717,040 bytes (59.1620 GiB). Free at plan generation:
133,567,811,584 bytes. These are actual metadata-plan results, not evidence that
archives have been downloaded or that extraction/storage fits.

## Bounded ZIP directory and source-schema inspection

`python -m tffv.retrieval.remote_zip` consumes the pinned acquisition plan.
It uses HTTP Range with a 4 MiB successful-payload budget and 64-request cap per
archive (at most 12 MiB across three archives). The server must return matching
206/Content-Range/identity-encoding headers; otherwise it stops before reading
the file body. No fallback to full download. Connection/redirect/header overhead
is separate from this payload budget.

Python zipfile reads ZIP/ZIP64 central directories through the bounded seekable
reader. Outputs list member names, compressed/declared uncompressed sizes,
CRC metadata, compression methods and suspicious paths/links/encryption flags.
No member is extracted to disk. Declared uncompressed sizes are not filesystem
allocation measurements, and no whole-archive SHA256 is claimed verified.

Optional `--sample-first-record` opens one eligible JSON/JSONL source member per
train archive and reads at most a 256 KiB first-line prefix under the same range
budget. It exports only shape/field names and a prefix hash, never primitive
annotation/text values. It does not claim full-member CRC verification. If a
record cannot be parsed within the cap, schema remains unavailable; no oversized
read or alternate data format is silently inferred. Corpus admission stays false.

Local HTTP fixtures cover ZIP64, regular ZIP, byte-budget/cache behavior,
refusal of ignored/mismatched ranges, pinned URL validation and shape-only
sampling. Live HTTP Range behavior on the user's server remains untested.

## Live ZIP inspection result and current disk decision

User output `outputs/m11-zip-inspection-20261021-042533/zip_summary.json`
confirms all three pinned archives support bounded Range inspection. Total
range payload was 296,894 bytes. Declared uncompressed train bytes:
200,880,771,274 (187.0848 GiB). Compressed + uncompressed:
264,405,488,314 bytes (246.2468 GiB), excluding derived artifacts/headroom.
All-archive extraction does not fit the subsequently reported 125.03 GiB free.
Inodes are not the observed storage bottleneck (2% used).

The directories contain 1000 / 1000 / 1068 files, with one additional directory
entry in the third archive. Samples show JSON object records with `claim_id`,
`query`, `type`, `url`, and `url2text` (list of strings). `1000.json` starts with
an empty text list; do not assume every retrieved row provides evidence. Third
archive member names include a `data_store/train/` prefix, so raw basenames alone
must not be treated as output paths. No suspicious member flags appeared in this
listing; this does not prove text-level absence of verdict leakage.

For eventual complete processing, consider retained compressed archives and
sequential per-store reading rather than materializing all 187 GiB at once.
That is a storage strategy, not a reduced evaluation population or a method
selection result. Per-claim eligible pools must still be documented identically
for R1–R4; current benchmark does not silently change from global to scoped mode.
Vector/cache sizing and schema/provenance admission remain pending.

## Bounded complete-store acquisition for smoke

`python -m tffv.retrieval.acquire_subset` acquires complete source stores for
1–3 explicit source ordinals, chosen before labels. Default mode is preflight
only. `--execute` explicitly enables member transfers using pinned Range URLs.
Limits: 128 MiB total successful range payload, 512 MiB total raw output,
256 MiB per store, and 1 GiB free-space reserve. It never downloads the full ZIP
as a fallback. If limits are exceeded it fails and preserves an unsealed output.

Raw member names are mapped to fixed safe local names under a fresh output;
there is no extractall or use of remote paths as filesystem targets. A full
member stream reaches EOF for zipfile CRC validation; its raw bytes are SHA256
hashed. Whole-archive SHA256 is explicitly not claimed verified. Archive revision,
expected archive hash, member CRC/name and all range hashes remain in provenance.

The companion schema audit checks claim ID consistency, empty/invalid text lists,
URL validity and verdict patterns without printing source passage content.
Neither an official source nor a clean heuristic profile automatically admits
the corpus. `query`/`type` remain in raw lineage and do not select stores or
populate retrieval queries. Keep the claim→store mapping; no global union is
constructed here. This subset is smoke-only, not a quality benchmark or M1.2
selection set. No model loading, optimizer or inference occurs.

## Live three-store acquisition and offline audit follow-up

User result: `outputs/m11-source-store-smoke-20261021-053505/subset_manifest.json`,
server commit `5928a39`. Acquired source stores 0/1/2 at the pinned corpus
revision: 130,960,407 raw bytes, 42,228,262 range payload bytes. Every selected
member reached EOF/CRC verification and has a raw SHA256. Whole ZIP SHA256
remains unverified, as explicitly recorded in the acquisition manifest.

| Source store | Source rows audited initially | Empty text rows | Verdict-pattern rows |
|---|---:|---:|---:|
| 0 | 546 | 115 | 2 |
| 1 | not measured | not measured | not measured |
| 2 | 603 | 178 | 1 |

Store 1's initial schema audit stopped because a source record exceeded its
8 MiB parser bound. Acquisition itself completed and CRC passed; that audit
status is not evidence of a corrupt store. The flags in other stores are
observations, not proof of answer leakage. Preserve all raw files and review
sources before final evidence filtering.

`python -m tffv.retrieval.audit_subset` performs a separate offline audit with
an **explicit** encoded-record limit up to 64 MiB. This changes no model context,
tokenization or source text. It verifies source manifest/file identities first,
recomputes raw SHA256 and CRC during full-record reading, and fails rather than
truncating/skipping a record. Encoded-record size is not an RSS memory bound.

The audit exports aggregate counts, large-record statistics, exact-text duplicate
counts, domain frequencies and a pre-filter word-passage workload estimate.
`review_queue.json` includes at most 20 flagged source records per store, with
URL, line number, record hash and short excerpts. Only verdict-pattern and
fact-check-domain heuristics are applied; no correctness labels or causes are
inferred. Large unbroken strings can still make a word-based passage estimate
misleading, so inspect the large-record report before indexing.

No network, retransfers, raw-file rewriting, automatic exclusion, corpus admission
or inference occur. A successful audit still leaves corpus_admitted=false.
