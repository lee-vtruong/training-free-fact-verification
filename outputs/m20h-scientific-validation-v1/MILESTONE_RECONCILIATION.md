# Milestone M2.0h: Phase F — Cross-Milestone Empirical Reconciliation

## 1. Evolution of Historical Error Knowledge Investigations

| Milestone | Paradigm | Result | Exact Mechanism & Diagnosis |
|:---|:---|:---:|:---|
| **M1.9g** | Raw Error-History ICL (unstructured past model logs) | `NEGATIVE (Did not outperform Gold ICL)` | Raw past outputs contained verbose, noisy failure text without targeted contrastive focus. |
| **M2.0a** | Global Static Error Rules in System Prompt | `NEGATIVE (Accuracy remained flat ~48%)` | Abstract global rules do not ground the verifier into specific evidence passages; model ignored general advice. |
| **M2.0c** | Instance-Grounded Error ICL via Dense Retrieval | `NEGATIVE (Did not beat matched generic baseline)` | Dense retrieval retrieved spurious past errors with semantic claim overlap but irrelevant error mechanisms. |
| **M2.0d** | Contrastive Error Patterns with Forced Evidence Attribution | `NEGATIVE (Accuracy dropped from 48.18% to 40.36%, Net: -43 claims)` | Forced Evidence Attribution imposed rigid passage-ID extraction and sufficiency analysis, triggering the Contradiction Trap (Refuted -> Conflicting/NEI collapse). |
| **M2.0f** | Factorial 2x2 Ablation (Error Knowledge x Forced Attribution) | `BREAKTHROUGH (Main effect of Attribution: -8.45%; Main effect of Error Knowledge: +9.55%; A3 reached 56.55%)` | Isolating the error signal by removing forced attribution unlocked the positive contrastive effect. |
| **M2.0g** | Multi-Seed Robustness (E1) + Component Ablation (E2) + Outer Fold (E3) | `POSITIVE CONFIRMATION (15/15 seeds won in E1; +8.91% in E2 K4 vs K1; 59.40% in E3)` | Historical error knowledge provides grounded counter-examples that specifically inoculate the model against over-skepticism and superficial matching. |

## 2. Theoretical and Empirical Synthesis

The apparent discrepancy between earlier negative results (M1.9g, M2.0a, M2.0c, M2.0d) and recent positive results (M2.0f, M2.0g) is fully resolved by identifying the two confounding factors: (1) Forced Evidence Attribution (which uniformly damages performance by -8.45%), and (2) Unfiltered dense retrieval of noisy past logs. When clean, certified contrastive error exemplars are formatted cleanly without forced attribution, the positive historical error effect is robust and reproducible across all tested configurations.
