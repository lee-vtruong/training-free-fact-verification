# Milestone M2.0i: Development Exposure Audit

## 1. Context and Scope

This audit evaluates the prior exposure of the 550 evaluation claims used in Milestone M2.0i.

- **Primary Evaluation Set:** 550 claims from Benchmark Fold-2 (`track_c_split_manifest.json`).
- **Exemplar Bank Partition:** 10 exemplars drawn strictly from Train Fold-1.
- **Group Isolation:** 100% group-isolated (zero group overlap between exemplar memory and evaluation set).

## 2. Exposure History Across Milestones

| Milestone | Role of 550 Fold-2 Claims | Method Intervention | Potential Risk |
|:---|:---|:---|:---|
| **M2.0d** | Evaluation partition | Forced Evidence Attribution vs Canonical B0 | Model observed failures under forced attribution; Fold-2 established as fixed testbed. |
| **M2.0f** | Evaluation partition | Factorial 2x2 Ablation (Error x Attribution) | Used to confirm recovery upon removing attribution. |
| **M2.0g** | Evaluation partition | Multi-seed robustness & Component ablation | Used across Seeds 1..5 and k in {2, 4, 6}. |
| **M2.0h** | Audit target | Forensic verification of all 20,200 predictions | Recomputed metrics, confirmed zero parser drift and zero leakage. |
| **M2.0i** | Mechanistic ablation | Factorial 5-arm study (F0, F1, F2, F3, F4) | **Exploratory mechanistic study**, not a blind out-of-distribution test. |

## 3. Methodological Determination

Milestone M2.0i is formally designated as a **Mechanistic Exploratory Ablation Study**.
Because the 550 claims were previously evaluated in M2.0f/g, this evaluation directly isolates the internal prompt mechanism (mistake content vs framing vs placebo) under identical conditions, but cannot be claimed as an untouched blind holdout test.
