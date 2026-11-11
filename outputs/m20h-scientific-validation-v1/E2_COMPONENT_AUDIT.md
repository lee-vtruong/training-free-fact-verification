# Milestone M2.0h: Phase C — E2 Knowledge Component Ablation Audit

## 1. Metric Progression Across Knowledge Levels

| Component | Description | Accuracy | Macro-F1 | Balanced Acc | Mean Prompt Tokens |
|:---|:---|:---:|:---:|:---:|:---:|
| **E2_K1_gold_only** | Level K1 | 46.00% | 0.3762 | 0.4102 | 1238.9 |
| **E2_K2_neutral_expl** | Level K2 | 46.18% | 0.3786 | 0.4217 | 1333.9 |
| **E2_K3_wrong_pred** | Level K3 | 50.00% | 0.3905 | 0.4266 | 1331.9 |
| **E2_K4_grounded_error** | Level K4 | 54.91% | 0.4024 | 0.4210 | 1396.9 |

## 2. Stepwise Contrastive Differences

| Step | Delta Acc | Delta Macro-F1 | 95% Bootstrap CI | Net Benefit | McNemar Stat |
|:---|:---:|:---:|:---:|:---:|:---:|
| **K2_vs_K1** | **+0.18%** | **+0.0024** | `[-3.27%, +3.64%]` | **+1** | 0.00 |
| **K3_vs_K2** | **+3.82%** | **+0.0119** | `[+0.55%, +7.09%]` | **+21** | 5.06 |
| **K4_vs_K3** | **+4.91%** | **+0.0119** | `[+1.09%, +8.55%]` | **+27** | 6.20 |
| **K4_vs_K1** | **+8.91%** | **+0.0262** | `[+5.45%, +12.55%]` | **+49** | 20.76 |

## 3. Confounding Assessment: Prompt Length

- **Status:** `NOT_IDENTIFIED`
- **Assessment:** K4 prompt length (mean 1410 tokens) exceeds K1 (mean 1210 tokens) by ~200 tokens. Because M2.0g did not include a length-matched neutral filler control for K4 (unlike M1.6.2c/e), the causal lift cannot be completely disentangled from token length / salience overhead.
