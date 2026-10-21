# Resource inventory

Audit date: 2026-10-20, Asia/Saigon. Read-only observations in this session.
Server paths quoted from old configuration are **unverified candidates**, not
verified resources. No research dataset counts or metrics have been replayed.

## Server update — 2026-10-21, user-supplied inventory

Source: `resources.json` pasted in attachment
`1fdc32a9-8ec3-490d-b03e-1d6728b6feff`. This is evidence from the user's server
command, not direct Codex access. Earlier unknown-server statements below refer
to the initial 2026-10-20 inspection.

- Server Git: `main`, clean relative to `origin/main`, commit
  `0f69d5251250a14578f7d016f3649cd2064b42ec`.
- Active environment: Python 3.13.13, numpy 2.5.4, pytest 9.1.1. Torch,
  transformers and huggingface-hub are not installed in this environment;
  CUDA through PyTorch has not been verified.
- GPU metadata: NVIDIA RTX 5090, total 32,607 MiB, free 32,107 MiB at probe,
  driver 580.159.03. This is not a model-capacity or CUDA-compatibility test.
- Filesystem free: 133,568,360,448 bytes, approximately 124.40 GiB at probe.
- `/home/stackops/whale/ERRORFlow` exists. Bounded scan lists 280 files without
  hitting its file-count limit. The scan excludes old outputs, hidden directories,
  other file formats and paths beyond its depth limit.
- Source candidates exist: `/home/stackops/whale/ERRORFlow/data/raw/averitec/train.json`
  (10,184,813 bytes), `dev.json` in the same directory (1,785,475 bytes), and
  `/home/stackops/whale/ERRORFlow/data/processed/v2/averitec/train_all.jsonl`
  (5,050,424 bytes). Schema, record counts, raw ID mapping and content hashes
  are not established by this metadata scan. Smaller development/screen subsets
  also exist; they must not replace official splits silently.
- `/home/stackops/whale/cache/models` contains Qwen3-4B-Instruct-2507 config,
  tokenizer and three safetensors shards, plus bge-reranker-v2-m3 artifacts.
  Content/revision completeness has not been verified. These are not the two
  Qwen 0.6B retrieval checkpoints requested in M1.1.
- The HF cache root `/home/stackops/.cache/huggingface/hub` contains other
  embedding/image/Roberta models. Neither Qwen3-Embedding-0.6B nor
  Qwen3-Reranker-0.6B appears in either scanned model root. This is a scoped
  absence observation, not proof they are absent from the entire server.
- AVeriTeC source annotations are available as candidate claim sources, but no
  independent AVeriTeC web-document corpus has been established. The scan also
  lists FEVER Wikipedia and SciFact corpus files; they cannot silently stand in
  for the declared AVeriTeC corpus. Processed QA is not admitted as independent
  evidence merely because it is cached.

Next inspection is `scripts/m11_probe_inputs.py`: source-train schema/counts,
candidate corpus/archive paths across formats, and existing environment package
metadata. It avoids printing annotation values and does not inspect dev labels.
No downloads, installs, inference, data conversion or old-repository writes.

## Verified local resources

| Resource | Exact path / observation | Format and coverage | Reuse decision |
|---|---|---|---|
| New workspace | `C:/Users/ASUS/Downloads/training-free-fact-verification` | Initially empty, initially not a Git repository | Independent code and artifacts only |
| Historical local checkout | `C:/Users/ASUS/Downloads/ERRORFlow` | HEAD `252ddd545eb164481accb49477d82636a14d3ff2`; existing dirty reports, figures and untracked files | Read-only reference; no imports or writes |
| Requested historical path | `C:/Users/ASUS/whale/ERRORFlow` (local expansion of `~/whale/ERRORFlow`) | Does not exist locally | Does not establish anything about server |
| Local data | `C:/Users/ASUS/Downloads/ERRORFlow/data` | Absent | No source records available here |
| Non-temporary local predictions | `C:/Users/ASUS/Downloads/ERRORFlow/outputs/step2_demo/predictions.jsonl` | JSONL, 3 records, model `demo`, checkpoint `fixture`; no gold labels or actual research checkpoint | Reject for scientific Base audit |
| Demo traces | `C:/Users/ASUS/Downloads/ERRORFlow/outputs/step2_demo/traces.jsonl` | JSONL; see machine inventory for count and fields | Synthetic, no scientific reuse |
| AVeriTeC adapter | `C:/Users/ASUS/Downloads/ERRORFlow/src/errorflow/adapters/averitec.py` | Python, Q/A evidence concatenation and four-label normalization | Inspect before conversion; original evidence must match cache |
| SciFact adapter | `C:/Users/ASUS/Downloads/ERRORFlow/src/errorflow/adapters/scifact.py` | Python, supplied text evidence and three-label normalization | Oracle vs retrieved evidence must remain distinct |
| CLIMATE-FEVER specification | `C:/Users/ASUS/Downloads/ERRORFlow/configs/v3/climate_fever_external.json` | JSON, four-label ontology and five official evidence candidates | Specification only; release hash/count not independently verified |
| Model/scoring runner | `C:/Users/ASUS/Downloads/ERRORFlow/scripts/run_v3_qwen_action.py` | Python, mean label log probability, one forward pass per label | Reference only; historical `model_hash` hashes identifier, not weights |
| Prompt definitions | `C:/Users/ASUS/Downloads/ERRORFlow/src/errorflow/v3_prompts.py` | Python; `conflict_audit_v1`, `standard_verdict_v1` and others | Audit-style prompt cannot be called a conventional Base |
| Group helper | `C:/Users/ASUS/Downloads/ERRORFlow/src/errorflow/v3_folds.py` | Python, normalized exact-claim groups | Insufficient by itself for semantic near-duplicate safety |
| Near-duplicate audit spec | `C:/Users/ASUS/Downloads/ERRORFlow/configs/v3/averitec_reproducibility_audit.json` | JSON, declared token Jaccard threshold .90, five folds, seed 2026 | Actual groups/folds and their hashes unavailable locally |
| Local Python | `C:/Users/ASUS/miniconda3/python.exe` | Python 3.13.11; no active VIRTUAL_ENV/CONDA_PREFIX reported | Used for standard-library audit and pytest |
| Old local environment | `C:/Users/ASUS/Downloads/ERRORFlow/.venv` | Absent | Not reused |
| Local HF cache | `C:/Users/ASUS/.cache/huggingface` | `hub`, `datasets`, `xet`; hub entries include SIGLIP and NER datasets | No Qwen/Mistral or fact-verification datasets found in top-level entries |

The machine-readable scan is `outputs/local_resource_inventory.json` (ignored
by Git). It enumerates selected code/config/cache formats, file sizes, JSONL
record counts and field names, Git state, environment path overrides, and GPU.
It excludes Git internals, virtual environments, symlink traversal, and large
binary assets. It is not an exhaustive drive-wide search.

Tiny JSONL files also exist under historical `.tmp/pytest-*` directories,
including deliberately malformed fixtures. They are test artifacts, not research
Base caches. Demo traces contain 3 records; none are scientific observations.

Additional verified code references:

- `C:/Users/ASUS/Downloads/ERRORFlow/src/errorflow/verifier.py` scores labels
  under `torch.inference_mode()`. Confidence is a softmax over mean candidate
  label log probabilities, not calibrated correctness. Its `token_count=0` is
  a placeholder and must not be treated as measured zero inference cost.
- `C:/Users/ASUS/Downloads/ERRORFlow/scripts/evaluate_v3_grouping_sensitivity.py`
  implements normalized token-Jaccard edges, connected components and balanced
  whole-component folds. Actual saved group files and input hashes are absent.
- `C:/Users/ASUS/Downloads/ERRORFlow/scripts/hash_model_snapshot.py` implements
  per-file and aggregate snapshot SHA256 fingerprints. A historical output may
  resolve checkpoint identity; no such output or model snapshot was verified.
  A newly computed hash alone cannot prove historical execution identity.

## Label ontologies observed in code, not verified against data

- AVeriTeC: SUPPORTED, REFUTED, NOT ENOUGH INFO, CONFLICTING. The adapter maps
  NOT_ENOUGH_EVIDENCE and CONFLICTING_EVIDENCE/CHERRYPICKING variants explicitly.
- CLIMATE-FEVER: SUPPORTS→SUPPORTED, REFUTES→REFUTED,
  NOT_ENOUGH_INFO→NOT ENOUGH INFO, DISPUTED→CONFLICTING in the local specification.
- SciFact: SUPPORT→SUPPORTED, CONTRADICT→REFUTED, NEI→NOT ENOUGH INFO.
  The retrieved-evidence configuration includes a historical four-to-three label
  projection CONFLICTING→NOT ENOUGH INFO. This is not equivalent to native
  three-label scoring and must be a separately named protocol.

## Server access and candidate resources

The documented `stackops@hvtham-server` hostname did not resolve. The user
provided SSH alias `nvidia`, HostName `10.44.101.230`, User `hvtham`. A direct
read-only SSH attempt with batch mode and strict known-host checking first
encountered sandbox network denial; the approved network retry timed out on
port 22. No authentication or server command executed successfully. WSL
enumeration also returned access denied. Server existence, Git state, storage,
GPU, environments and caches remain unknown.

Candidate paths from local historical configurations, relative to the
**unverified server** `~/whale/ERRORFlow`:

| Candidate path | What historical configuration says | Missing verification |
|---|---|---|
| `data/processed/v2/averitec/train_all.jsonl` | Expected 3,068 AVeriTeC train records | Existence, count, labels, evidence and content hash |
| `outputs/v3/averitec/development/baseline_scores.jsonl` | `conflict_audit_v1` predictions | Completeness, alignment, scores, checkpoint bytes and exact prompt |
| `data/processed/v3/climate_fever/all_official_candidates.jsonl` | Expected 1,535 official claims | Existence and evidence/label mapping; previously evaluated externally |
| `outputs/v3/climate_fever/full/conflict_audit_v1.jsonl` | Audit-style Base predictions | Same checks; cannot be treated as untouched target |
| `outputs/v3/reviewer_round5/four_label_standard/averitec.jsonl` | Proposed conventional Base control | Protocol document is not evidence this cache exists |
| `outputs/v3/reviewer_round5/four_label_standard/climate_fever.jsonl` | Proposed conventional Base control | Same limitation |
| `/home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507` | Old protocol's model directory | Absolute path may differ for user hvtham; no checkpoint inspected |
| `~/whale/ERRORFlow/.venv`, `~/whale/GraphCURE/.venv` | Historical environment candidates | Existence and dependencies unknown; do not modify either |

SciFact preparation scripts and retrieved/oracle configurations exist locally;
no actual SciFact dataset, Base cache or group file was found here. Mistral-7B
weights, scoring protocol and caches are unverified. No historical metric quoted
in the paper/reports is promoted to a replay-verified result in this project.

## Hardware and storage

Local `nvidia-smi`: RTX 3050 laptop GPU, 6,144 MiB VRAM, driver 582.05,
reported CUDA compatibility 13.0, zero compute utilization at inspection.
This does not verify PyTorch CUDA compatibility or model capacity.
Workspace drive probe: total 294,973,861,888 bytes, free 5,943,947,264 bytes
(approximately 5.54 GiB) at inspection; free storage changes over time.
Installed packages include pytest 9.1.1, torch 2.11.0, transformers 4.45.2,
datasets 4.1.1, numpy 2.4.4. Only Python standard library and pytest are used
here; historical inference compatibility is untested. No downloads or GPU jobs.

## Integrity and reuse decision

No old file was edited, removed, staged, committed, reset, or rerun. No links
were created. Once verified, data/caches may be opened read-only through local
configuration. Mount them read-only on the server where possible. New artifacts
must live under this new project. Reject caches with unresolved input/protocol
identity; first recover sidecars and provenance before requesting new inference.
