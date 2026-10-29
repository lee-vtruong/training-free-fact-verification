# M1.6 v2 ID-only partition seal

Created by CPU audit on 2026-10-27. No raw claim text, gold labels, QA, justifications, verdicts or external memory content is included. Exact byte hashes: [seal.json](seal.json). Validate with `python scripts/m16_validate_seal.py docs/partitions/m16-v2`.

| File | Purpose |
|---|---|
| [claims_manifest.json](claims_manifest.json) | All 3,068 IDs, hashes, roles, eligibility and exposure ledger |
| [groups_manifest.json](groups_manifest.json) | All 1,952 complete leakage blocks |
| [group_edges.json](group_edges.json) | Retained lexical and hashed URL/event edges |
| [source_manifest.json](source_manifest.json) | 1,013 source-side records; 973 memory-eligible |
| [development_manifest.json](development_manifest.json) | 541 exploratory claims / 364 blocks |
| [test_sealed_manifest.json](test_sealed_manifest.json) | Empty: no certified clean train test |
| [test_quarantine_exposed_manifest.json](test_quarantine_exposed_manifest.json) | 1,230 former candidates / 835 exposed blocks |
| [legacy_exploratory_manifest.json](legacy_exploratory_manifest.json) | 283 other A–E-related exploratory records |
| [quarantine_invalid_manifest.json](quarantine_invalid_manifest.json) | Empty-claim ordinal preserved |
| [protocol.json](protocol.json) | Applied policy/seed/thresholds |

All train claims have external prediction artifacts. Human method-use history is not fully certified. `confirmation_ready=false`; every `confirmatory_eligible=false`. Hashes and read-only permissions cannot certify untouched data. Original v1 stays on server with its invalidated seal versioned separately. Do not restore v1 as a clean test or recycle its exposed reservoir into source/development to improve a pilot.
