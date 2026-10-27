# M1.6.1 — Legacy Source Audit

Date: 2026-10-27. Scope: own-project Fold A `train:0..49` only. CPU construction/audit local; read-only SSH/scp copied existing own-project artifacts. No ERRORFlow predictions/memory used. No GPU acquisition/retrieval/inference launched.

## Recomputed results

| Check | Result |
|---|---:|
| Admitted legacy SOURCE records | 50/50 |
| Reproducible incorrect Base verdicts | **25** |
| Correct verdicts | 25 |
| Raw/stored parser errors | **0** |
| Claim hashes / saved prompt hashes / evidence-ID ordering | 50/50 each |
| Evidence URL ordering / text fingerprints / decoder checks | 50/50 each |
| Top-8 serialized evidence present | 50/50 |
| Human reviews / verified corrective rationales | **0 / 0** |
| Verified reasoning failures / verified mechanisms | **0 / 0** |
| Evidence-sufficiency review pending | **50** |

No case has yet been human-adjudicated as insufficient; this is not evidence that all evidence is sufficient. A valid verdict disagrees with source gold in 25 cases, but the cause could be retrieval incompleteness, nuanced ontology, or reasoning failure. Do not silently label every mismatch a reasoning failure.

The source-only original Accuracy is 25/50, not a HEAL result. No new method has been evaluated on development or test. [Machine-readable audit](audits/M1_6_1_SOURCE_AUDIT.json) records hashes, per-ID artifact checks, confusion counts and limitations.

## Original provenance checked

- Own-project `outputs/m121-b1-preds-20261024-152746/predictions.jsonl`: SHA-256 `83d5b6a9f7828b51796691afcbc2c79211dc966a228e532cc5ffa11257e02af6`, matching B1 manifest and M1.3's referenced prediction artifact.
- Original R4 `outputs/m11-store-r14-pilot100-20261023-162716/R4.jsonl`: SHA-256 `f5e47540925d0b7e635d5c906aa12723d2fab04eb704449e83686375b5c11cb1`, matching prediction manifest and retrieval output manifest.
- Source train SHA-256 `ae5eda7c42ddf1695ef185a7ba1bc716928f5adf57103e4f78aae5f9afe00f9c`, matching v2. Only Fold A claim/label values join the legacy bank.
- B1 manifest declares Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, greedy non-thinking, 512-token cap, bfloat16. B1 is the historical Base prompt used as D0; reconstructing its messages reproduces every saved prompt hash.
- Retrieval manifest/config hash `7f716c4fe2888ba311a4edba9ca8dd618c368de43707264dde1d5cf129e907ec`; R4 evidence retains original passage/document/acquisition IDs, URLs, source offsets and text hashes. The new batch uses the same retrieval parameters/model revisions/fingerprints, not a shortened protocol.

Full original files contain 100 predictions/evidence rows; construction explicitly filters to the 50 SOURCE IDs. B rows are not joined into memory. Original serialized evidence is preserved, and the actual prompt formatter's 1,200-character passage display is recorded separately. New correction citations must refer to exact displayed text, not truncated-away content.

One assistant-authored candidate for `train:25` is preserved separately in `outputs/m161-candidate-bank-v1.json`, with exact displayed-evidence span citations and independent blank review sheets in `outputs/m161-candidate-review-v1/`. It diagnoses the Base's explicit below-1% / “not even 1%” interpretation and asks humans to check temporal/denominator assumptions. This is **one unverified draft**, not a human judgment or an additional verified mechanism. Human/verified counts remain zero; the machine-readable audit preserves its authoring status and hashes.

## Trust issue and limits

The old M1.3 builder assigns `gold_explanation_verified=True` unconditionally and stores gold justifications. All 50 old records carry that flag; **it is not human-review evidence**. The new builder does not import that field or the justification text. Every new correction explanation starts null and every original case awaits review.

The historical generation manifest records model revision and paths but no original runtime weight-content fingerprint or effective input-token hash. The new CPU audit cannot recover those missing execution facts. Raw store bytes and source document offsets are not independently re-audited here. The checked boundary is **serialized own-project artifacts plus declared runtime provenance**, not complete original byte-level execution certification.

Recovery: locate any existing original tokenizer/effective-input/model fingerprint and raw-store/candidate receipts; verify them against the original hashes before upgrading the provenance claim. If absent, retain null/unknown fields. A later explicitly authorized rerun may use the same own-source claims/evidence and frozen model, but must create a new condition/run identity; it cannot replace the original outputs or be called recovered original data. Do not fetch today's web text and call it the historical retrieved passage.

These gaps do not require inventing new data or blocking exploratory source expansion. They limit the strength of the legacy provenance claim. Cases with actual serialized-artifact mismatches become PROVENANCE_UNKNOWN and are excluded; none was found among these 50.

## Mechanism support

Frozen lexical candidate flags over source claim plus raw Base output: scope/quantifier 19; sufficiency 11; attribution 2; negation/polarity 21; reported conflict 6; partial truth/context 8. Flags overlap and may match words in correct reasoning; they are **review triage, not verified mechanism frequencies**. They differ from old manually assigned taxonomy counts and must not be pooled with them. Human-confirmed support under the new protocol is zero for every mechanism.

## Expansion preparation and costs

The first fixed 25 new source IDs require **8 / 12 / 5** stores from shards 0–999 / 1000–1999 / 2000–3067. Route-boundary and actual-schedule tests pass. The pinned acquisition plan is copied unchanged into [config](../configs/m161_acquisition_plan.json). The existing fold wrapper is not used.

Read-only filename inventory found 350 existing `.npy` cache files, **0 candidate filenames for the 25 new source IDs**. No cache hit or speedup is claimed. The old loader's bare filename does not certify model/config/text provenance; new execution reuses only fully hash-sealed retrieval steps. Interrupted unsealed attempts use new directories and do not trust the old size-only raw-store reuse branch.

Actual newly executed GPU work: **0**. New knowledge-store acquisition: **0**. Human reviews: **0**. CPU tests/audits are software/provenance evidence only. Historical token/latency totals retained in the JSON audit belong to the original Base predictions and are not charged as new M1.6.1 compute. Future hardware preflight/model-fingerprint verification is prepared for the user's server run, not reported as passed locally.

M1.6.2 readiness: **MODIFY**. Source bank construction and same-ID interfaces are usable; verified rationale/mechanism coverage is not. Obtain independent reviews and controlled source expansion. Keep development/test gold out of the builder and do not alter/recycle sealed partitions.
