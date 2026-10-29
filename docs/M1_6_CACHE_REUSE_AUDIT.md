# M1.6 — Cache-first reuse audit

Date: 2026-10-28. This audit covers the completed exploratory canonical run `outputs/m162a-dev50-v1` on `hvtham-server`. It does not change that run, its evidence, source memory, predictions, metrics, partitions, or protocol.

## Measured runtime decomposition

The pilot state records 20,948.032 seconds (5h49m08s) of session wall time. Its receipts provide the following measured stage times. These are elapsed wall times, not GPU-active times.

| Stage | Claims | Measured seconds | Mean seconds/claim | Reuse state |
|---|---:|---:|---:|---|
| Store acquisition/preparation | 50 | 2,454.394 | 49.088 | Sealed whole-step receipts reusable only with matching acquisition/input contract |
| Store audit | 50 | 231.321 | 4.626 | Sealed whole-step receipts reusable with same store artifact |
| Candidate-pool preparation | 50 | 280.219 | 5.604 | Sealed whole-step receipts reusable with same policy/input |
| R1–R4 retrieval, including BM25, dense, RRF and reranking | 50 | 16,346.063 | 326.921 | R4 Top-8 and all receipt hashes reusable as a unit |
| P0/P1/P2/P3/N0 generation, including per-claim Qwen load | 50 x 5 | 887.051 | 17.741 per claim / 3.548 per condition-call | Reusable only for exactly matching prompt/inference contract |
| Accounted receipts subtotal | — | 20,199.047 | — | — |
| Session minus receipt subtotal | — | 748.985 | — | Prompt sealing, environment/preflight, parent waits, scoring and I/O; no finer instrumentation |

Model loading is included in retrieval and prediction worker times because each worker loads its needed frozen model. The logs do not separate checkpoint loading from BM25, dense embedding, RRF, reranking, generation or I/O. Those individual durations are **unknown**, not inferred. The old run creates one embedding-cache directory per retrieval attempt; this was not admitted as a cross-run cache because its identity is insufficient alone. GPU-active time and monetary cost remain null.

For an exactly matching 50-claim condition, reuse removes the measured 19,311.997 seconds of acquisition/audit/candidate/retrieval work. It also reuses a baseline's measured inference. The remaining measured work is only new-condition inference: 50 calls at the observed mean is about 177.4 seconds, plus unmeasured model startup/sealing/I/O. This is an accounting projection from this one run, not a promised speedup.

## Actual artifact inventory and admission

The canonical run has 50 each of acquire, audit, candidate, retrieval and prediction receipts; 50 `R4.jsonl` files containing exactly eight passages; `sealed_prompts.json`; 250 prediction records for P0/P1/P2/P3/N0; completion/evaluation seals; and an exploratory results file. It occupies about 24 GiB. The cache index validates every file covered by each receipt and rejects a missing or modified one.

The current reusable IDs are `train:810, 1062, 521, 2298, 780, 368, 1025, 2185, 1812, 2473, 1935, 385, 954, 1284, 3026, 898, 1102, 1382, 1733, 2826, 2935, 832, 2763, 1249, 1517, 1011, 1299, 1024, 2450, 1923, 971, 2137, 2073, 1895, 1952, 1853, 1592, 1447, 1455, 1932, 1217, 2100, 627, 1384, 2770, 1728, 3054, 2374, 493, 1201`. Each has `evidence=REUSABLE`, `P0/P1/P2/P3/N0=REUSABLE` only under matching keys, `P4=MISSING/BLOCKED`, and no missing receipt. The full machine-readable per-ID matrix is emitted by the `index` command, rather than duplicated from a mutable server output into Git.

| Artifact | Current 50 | Next scheduled 50 | Admission |
|---|---|---|---|
| Pinned SOURCE memory `memory75.json` | Read-only, SHA recorded by canonical batch | Same source-only bank may be read only if its hash and review version match | Valid source memory; P4 remains blocked by review state |
| Source corrections/review state | Five pending drafts, zero verified rationales | Same | Valid as unverified metadata; cannot satisfy a verified-rationale condition |
| R4 Top-8 evidence | 50/50 present and receipt-sealed | 0/50 prepared | Reusable for the same 50 only |
| P0/P1/P2/P3/N0 predictions | 250/250 present and receipt-sealed | 0/250 | Reusable only when the matching cache key exists |
| Evaluation results | Canonical `evaluation-v1` present | Missing | Reproducible from canonical predictions without GPU |
| ERRORFlow predictions | Not admitted | Not admitted | Invalid for HEAL memory and cache reuse |

The next scheduled DEVELOPMENT claims are `train:1741, 370, 1860, 1215, 2973, 1635, 1499, 1917, 900, 881, 3003, 3019, 781, 2022, 745, 429, 1042, 2942, 1590, 2990, 2174, 415, 431, 856, 2170, 485, 847, 1830, 2163, 1496, 2238, 682, 2831, 1530, 1456, 2925, 749, 1032, 2405, 3042, 1149, 2418, 1864, 1050, 2438, 1140, 1810, 2024, 980, 1467`. They have no evidence/prediction cache and require a separately locked protocol before retrieval.

## Immutable contracts

`tffv.experiments.m16_cache` builds content-addressed keys, never an ID-only key.

- Evidence key: dataset release hash; claim ID; original and whitespace-normalized claim hashes; acquisition/store receipt hash; candidate-pool receipt hash; R1–R4 configuration; embedding/reranker checkpoint specifications; candidate policy; Top-8 count; and the byte hash of rendered R4 Top-8 evidence.
- Prediction key: evidence key; frozen verifier snapshot; rendered-message hash; effective token-ID hash; chat-template/thinking mode; generation configuration; and parser version.
- Memory contract: SOURCE-only partition statement; source Base-run provenance retained in records; source-gold provenance; fixed exemplar selection/order hash; and correction-review version. It is read-only and never includes DEVELOPMENT/TEST labels or ERRORFlow predictions.

A mismatch in any of these values is a cache miss, not a reason to reuse a condition ID. Missing evidence, incomplete Top-8, wrong claim hash, modified receipt, changed prompt, changed retrieval configuration, cross-partition target/source group overlap, or a changed model/parser/generation contract fails closed.

## Cache-first modes

The new runner is deliberately CPU/read-only. It does not retrieve, load Qwen, or silently write into canonical directories.

| Mode | Purpose | GPU/retrieval behavior |
|---|---|---|
| `index` | Admit a canonical run into an immutable cache index after receipt/hash checks | None |
| `prepare-evidence` | List reusable/missing evidence IDs | Never retrieves; missing IDs are explicit |
| `inference-only` | List claims with a matching prediction cache key for a condition | Never retrieves or loads a model |
| `resume` | Produce the same missing-only inference plan without duplicate records | Never retrieves or loads a model |
| `evaluate-only` | Recompute Accuracy/Macro-F1/per-class metrics from admitted predictions | No GPU/model; opens gold only after full prediction cache admission |

For actual evidence creation, retain the existing sealed retrieval runner and run it only after a new population/protocol has been locked. The cache tool will then admit its receipts. This avoids making a new generic retrieval framework that could bypass the M1.6 partition and provenance gates.

## Direct answers

1. A new experiment using the exact same 50 claims and matching evidence contract does **not** need retrieval.
2. All **250** canonical P0/P1/P2/P3/N0 predictions can be reused only under their exact prompt/model/parser/generation contracts. P4 has no prediction cache.
3. A genuinely new condition on those claims needs **50** GPU inference calls after its prompts are frozen and evidence is admitted. It needs zero retrieval calls. A partial resume executes only missing cache keys.
4. Retrieval runs again for a missing/changed evidence key: different claim/release/store/candidate/segmentation/Top-k/retriever checkpoint/configuration, unsealed or altered receipt, incomplete Top-8, or a new target population such as the next 50.
5. Measured avoidable work for a matching 50-claim condition is 5h21m52s of evidence preparation/retrieval; reusing an existing condition additionally avoids its observed generation time. A new condition's only measured comparable cost is about 2m57s of generation across 50 calls, plus unmeasured startup/sealing/I/O. This is not a claimed end-to-end speedup.
