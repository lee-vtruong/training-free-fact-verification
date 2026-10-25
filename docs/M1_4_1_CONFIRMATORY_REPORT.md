# M1.4.1 — Confirmatory Fact Verification Report (Fold C)

**Milestone:** M1.4.1 (Confirmatory Fact Verification)
**Date Generated:** 2026-10-25 10:25:43
**Design:** Independent Confirmatory Evaluation on Untouched Fold C
**Research Principle:** Quality-First, Strictly Training-Free (Zero parameter updates)

---

## 1. Population & Anti-Leakage Partition

- **Source Memory Bank (Fold A):** 50 claims (`train:0..49`).
- **Evaluation Population (Fold C):** 50 claims admitted.
- **Quarantined Near-Duplicates:** 0 claims quarantined.
- **Exemplar Disjointness:** 100% of exemplars strictly from Fold A. Overlap with Fold C is 0.0%.

---

## 2. Executive Performance Summary

| Condition | Description | Valid Verdicts | Accuracy | Macro-F1 | Precision | Recall | Mean Latency | Mean Tokens (Prompt / Gen) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **D0** | Frozen Base Protocol v1.0 (Zero-shot B1) | 50/50 (100.0%) | **54.0%** (27/50) | **0.3801** | 0.4312 | 0.5448 | 4.25s | ~ / 179.1 |
| **D3** | Random Error Demonstrations ($k=2$) | 50/50 (100.0%) | **68.0%** (34/50) | **0.3739** | 0.3833 | 0.3675 | 1.98s | ~ / 78.0 |
| **D4** | Similarity-Retrieved Error Demonstrations ($k=2$) | 50/50 (100.0%) | **62.0%** (31/50) | **0.3333** | 0.3415 | 0.3305 | 1.79s | ~ / 70.0 |
| **D5** | Similarity-Retrieved Error Demonstrations with Corrections ($k=2$) | 50/50 (100.0%) | **52.0%** (26/50) | **0.3603** | 0.4226 | 0.5221 | 1.90s | ~ / 75.9 |
| **D2** | Concise Source-Derived Error Rules | 50/50 (100.0%) | **68.0%** (34/50) | **0.4365** | 0.4187 | 0.6104 | 1.62s | ~ / 64.2 |

---

## 3. Paired Comparisons & Confirmatory Hypothesis Testing

| Comparison | Net Gain | Accuracy Delta [95% CI] | Macro-F1 Delta [95% CI] | Exact McNemar $p$ | Significant at $\alpha=0.05$? |
|:---|:---:|:---:|:---:|:---:|:---:|
| **D4 vs D0** | **+4** | +0.0800 [-0.0200, +0.2000] | -0.0468 [-0.1332, +0.0246] | **0.2891** | NO |
| **D4 vs D3** | **-3** | -0.0600 [-0.1600, +0.0400] | -0.0406 [-0.0954, +0.0038] | **0.4531** | NO |
| **D4 vs D5** | **+5** | +0.1000 [-0.0200, +0.2200] | -0.0269 [-0.1296, +0.0694] | **0.1797** | NO |
| **D2 vs D0** | **+7** | +0.1400 [+0.0600, +0.2400] | +0.0564 [-0.0145, +0.1361] | **0.0156** | **YES** |

---

## 4. Mutually Exclusive Paired Transitions (9 Categories)

| Comparison | C→C | C→W | C→U | W→C | W→W | W→U | U→C | U→W | U→U | Sum |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **D4 vs D0** | 25 | 2 | 0 | 6 | 17 | 0 | 0 | 0 | 0 | **50** |
| **D4 vs D3** | 29 | 5 | 0 | 2 | 14 | 0 | 0 | 0 | 0 | **50** |
| **D4 vs D5** | 24 | 2 | 0 | 7 | 17 | 0 | 0 | 0 | 0 | **50** |

---

## 5. Methodological Safeguards & Scientific Conclusion

- **Untouched Evaluation Split:** Fold C was completely unobserved during prompt formulation and D4 pilot evaluation.
- **Deterministic Retrieval:** Exemplars in D4 and D5 are selected with frozen 3-gram lexical similarity with zero random seed perturbation.
- **Strict Parameter Freeze:** Zero parameters updated, frozen Qwen3-8B snapshot, non-thinking greedy decoding.
