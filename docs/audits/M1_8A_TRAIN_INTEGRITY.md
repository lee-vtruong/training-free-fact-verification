# M1.8a Full-Train Data & Integrity Audit Report

- **Total Records in Official Train:** `3068`
- **Valid Claims:** `3067` (non-empty text, valid label)
- **Invalid Claims:** `1` (`train:1948` has empty claim text `""`)

## 1. Historical M1.6 Partition Mapping

| Partition | Count | Description |
| :--- | :---: | :--- |
| **DEVELOPMENT** | 541 | Historical partition retained for provenance |
| **LEGACY_EXPLORATORY** | 283 | Historical partition retained for provenance |
| **QUARANTINE_INVALID** | 1 | Historical partition retained for provenance |
| **SOURCE** | 1013 | Historical partition retained for provenance |
| **TEST_SEALED** | 1230 | Historical partition retained for provenance |

## 2. Label Distribution Across Train

| Gold Label | Count | Percentage |
| :--- | :---: | :---: |
| **Conflicting Evidence/Cherrypicking** | 195 | 6.36% |
| **Not Enough Evidence** | 282 | 9.19% |
| **Refuted** | 1742 | 56.78% |
| **Supported** | 849 | 27.67% |

## 3. Official Train vs Official Dev Overlap Audit

- **Exact Normalized Claim Text Matches:** `4`
- **Fact-Checking Article URL Overlaps:** `3`

| Train ID | Dev ID | Overlapping Claim Text |
| :--- | :--- | :--- |
| `train:1539` | `dev:458` | the south african police service kills three times more people per capita than t... |
| `train:1340` | `dev:194` | at independence, nigeria had a population of 45 million... |
| `train:1341` | `dev:216` | 52% of nigeria’s current population lives in urban areas... |
| `train:1536` | `dev:463` | 558 people were killed by the police in 2018, while 201 people died in police cu... |