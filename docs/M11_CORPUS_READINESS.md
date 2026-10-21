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
