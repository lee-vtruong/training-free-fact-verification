# Milestone M2.0h: Phase D — K4 Knowledge Provenance Audit

## 1. Classification Summary

| Category | Description | Count | Percentage |
|:---|:---|:---:|:---:|
| **P1_VALIDATED_GROUNDED** | Traceable / Audited | 1 | 10.0% |
| **P2_SOURCE_TRACEABLE_UNVALIDATED** | Traceable / Audited | 9 | 90.0% |

## 2. Critical Text Invariance Discovery

In CERTIFIED_EXEMPLAR_BANK, for 100% (10/10) of exemplars, `corrective_resolution` is verbatim identical to `neutral_explanation`. The only text differences distinguishing K4 from K2 in the prompt are: (1) Header tag '[HISTORICAL ERROR CONTRAST]' vs '[VERIFICATION REASONING]', and (2) The inclusion of the single line '- Observed Mistake: <observed_mistake>'. This proves that the performance jump is not caused by differing ground-truth evidence explanations, but specifically by the contrastive error framing.

## 3. Per-Exemplar Detailed Provenance Table

| Sample ID | Gold | Baseline Mistake | Classification | Origin & Validator Status |
|:---|:---:|:---:|:---:|:---|
| `train:2956` | Supported | Not Enough Evidence | `P1_VALIDATED_GROUNDED` | M1.8g Substantive Recovery (Qwen3-14B Judge Revalidated) (LLM_JUDGE_VALIDATED) |
| `train:37` | Supported | Refuted | `P2_SOURCE_TRACEABLE_UNVALIDATED` | M1.6.2d/M1.6.2e Candidate Triage Bank (E3 Top-8 Reasoning Error, 0 Human Reviews) (TRIAGE_DRAFT_PENDING_REVIEW) |
| `train:622` | Supported | Conflicting Evidence/Cherrypicking | `P2_SOURCE_TRACEABLE_UNVALIDATED` | AVeriTeC Train Fold-1 Primary Agency Report (USDA Press Release) (UNVALIDATED_PRIMARY_SOURCE) |
| `train:47` | Refuted | Supported | `P2_SOURCE_TRACEABLE_UNVALIDATED` | M1.6.2d/M1.6.2e Candidate Triage Bank (E3 Top-8 Reasoning Error, 0 Human Reviews) (TRIAGE_DRAFT_PENDING_REVIEW) |
| `train:48` | Refuted | Supported | `P2_SOURCE_TRACEABLE_UNVALIDATED` | M1.6.2d/M1.6.2e Candidate Triage Bank (E3 Top-8 Reasoning Error, 0 Human Reviews) (TRIAGE_DRAFT_PENDING_REVIEW) |
| `train:1863` | Refuted | Conflicting Evidence/Cherrypicking | `P2_SOURCE_TRACEABLE_UNVALIDATED` | M1.6.2d/M1.6.2e Candidate Triage Bank (E3 Top-8 Reasoning Error, 0 Human Reviews) (TRIAGE_DRAFT_PENDING_REVIEW) |
| `train:35` | Not Enough Evidence | Supported | `P2_SOURCE_TRACEABLE_UNVALIDATED` | M1.6.2d/M1.6.2e Candidate Triage Bank (E3 Top-8 Reasoning Error, 0 Human Reviews) (TRIAGE_DRAFT_PENDING_REVIEW) |
| `train:49` | Not Enough Evidence | Supported | `P2_SOURCE_TRACEABLE_UNVALIDATED` | M1.6.2d/M1.6.2e Candidate Triage Bank (E3 Top-8 Reasoning Error, 0 Human Reviews) (TRIAGE_DRAFT_PENDING_REVIEW) |
| `train:8` | Conflicting Evidence/Cherrypicking | Refuted | `P2_SOURCE_TRACEABLE_UNVALIDATED` | M1.6.2d/M1.6.2e Candidate Triage Bank (E3 Top-8 Reasoning Error, 0 Human Reviews) (TRIAGE_DRAFT_PENDING_REVIEW) |
| `train:12` | Conflicting Evidence/Cherrypicking | Refuted | `P2_SOURCE_TRACEABLE_UNVALIDATED` | M1.6.2d/M1.6.2e Candidate Triage Bank (E3 Top-8 Reasoning Error, 0 Human Reviews) (TRIAGE_DRAFT_PENDING_REVIEW) |
