# Milestone M2.0h: Phase E — E3 Source-Fold Generalization Audit

## 1. Generalization Metrics on 500 Outer Claims (Train Fold-1)

| Arm | Description | Accuracy | Macro-F1 | Balanced Acc | Delta vs B0 | Delta vs Gold |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **E3_B0_baseline** | Zero-Shot Direct | 52.20% | 0.3920 | 0.4087 | 0.00% | +3.80% |
| **E3_A1_gold** | Matched Gold ICL (k=2) | 48.40% | 0.3907 | 0.4525 | -3.80% | 0.00% |
| **E3_A3_error** | **Historical Error ICL (k=2)** | **59.40%** | **0.4159** | **0.4631** | **+7.20%** | **+11.00%** |

## 2. Statistical Significance

- **Paired Bootstrap 95% CI (A3 vs B0):** `[+3.20%, +11.40%]`
- **McNemar Chi-Square Statistic:** `10.94`
- **Net Benefit:** `+36 claims` (Corrected: 74, Harmed: 38)
- **Group Leakage Check:** `0 leaks detected` (100% group isolated).
