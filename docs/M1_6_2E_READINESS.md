# M1.6.2e — Section E: New-Population Readiness (report only)

Generated: 2026-10-28 04:42Z · CPU-only, read-only · no retrieval, no GPU, no TEST_QUARANTINE access.

## Sealed partition frame

- Dataset: 3,068 claims (`outputs/m16-partitions-v1`, seal dataset_sha256 `ae5eda7c42ddf1695ef185a7ba1bc716928f5adf57103e4f78aae5f9afe00f9c`).
- Assignment intervals: SOURCE 0.0–0.3 · DEVELOPMENT 0.3–0.5 · TEST_SEALED 0.5–1.0.
- Development manifest: **541 claims** (official_split train). TEST_SEALED untouched (`test_sealed_manifest.json` not opened beyond count).

## Used vs unused development claims

| set | n | note |
|---|---|---|
| DEV50 (M1.4/M1.6.2 development population) | 50 | ordinals 368–3054; outcomes known and frozen |
| unused development claims | **491** | manifest minus DEV50 |

Exposure/eligibility flags over the 491 unused claims:

| flag | true count |
|---|---|
| `artifact_outcome_exposed_directly` | 0 |
| `artifact_outcome_exposure_propagated` | 0 |
| `claim_text_previously_exported` | 491 |
| `confirmatory_eligible` | 0 |
| `source_memory_eligible` | 0 |
| `legacy_source_A` | 0 |
| `external_exposure_status == UNKNOWN` | 491 |

## Leakage-group readiness

- Distinct leakage groups touched by the unused claims: **320**.
- Fully-unused groups (no DEV50 member): **482** claims in 314 groups — clean candidate pool.
- Groups shared with DEV50 (contamination risk for historical-outcome reasoning): **9** claims in 6 groups — excluded from the candidate pool by the locked selection rule.
- Group size distribution (fully-unused groups): [[1, 262], [2, 24], [3, 8], [4, 8], [5, 4], [6, 2], [7, 1], [8, 1], [11, 1], [12, 1], [15, 1], [31, 1]].

## R4 cache coverage (server probe 2026-10-28, read-only)

| cache root | n claim dirs | ordinals/IDs |
|---|---|---|
| `outputs/m11-store-r14-pilot100-20261023-162716` | 100 | 0–99 contiguous |
| `outputs/m162a-dev50-v1` | 50 | the 50 DEV50 IDs |
| `outputs/m142-retrieval-folde-20261026-030724` | 100 | 250–349 contiguous |
| `outputs/m161-source-batch25-v1` | 25 | 426–3012 contiguous |

- Unused development claims **with** an existing retrieval cache: **0**.
- Unused development claims **without** cache (would need the single permitted retrieval pass): **491**.
- Practical reading: a fresh population drawn from fully-unused groups will be a **cache MISS** almost everywhere; per the M1.6.2d step-4 rule the new population gets exactly one retrieval pass. No retrieval is executed in this milestone.

## Readiness verdict

**READY-PENDING-GATES**: a sufficient unused population exists (482 claims in 314 fully-unused groups), the sealed frame is intact, and cache strategy is defined. Readiness does NOT authorize execution: step-4 switch remains blocked by the M1.6.2e gates (human verification of corrections; protocol lock). No class-distribution read of unused development gold was performed at planning time; class-stratified inclusion happens only at prepare-time under the locked selection rule.

## Provenance

- `outputs/m16-partitions-v1/development_manifest.json` sha256 `d6524e37aa7826ecc326fb41348d889f067d8d4536b83cf3ff64cb7993ec8c91`
- `outputs/m16-partitions-v1/claims_manifest.json` sha256 `476888b7522347f0ff62453d9e07a38f0dd7c8804e14cf13f52970b7a763a70f`
- `outputs/m16-partitions-v1/seal.json` sha256 `4114804838716888ac5a321f416ef9f479f1f26034904a1e88372c79bdeeb97b`
- `outputs/m16-partitions-v1/protocol.json` sha256 `202445033bc747e195361b2440c5cb45958f5036284e2c4da82de1f812c0a9f8`
- DEV50 IDs: `outputs/m162a-dev50-preparation-v4/batch.json` sha256 `dcac8a3f3ff0f017853ddbcf9137f72840371852acb34c2a9cccb392782b087e`