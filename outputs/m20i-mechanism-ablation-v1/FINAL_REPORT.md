# Milestone M2.0i: Scientific Decision Report

## Official Determination: `FRAMING_SUPPORTED`

**Rationale:** F3 outperforms F0 (Neutral Explanation) but fails to significantly outperform F4 (Matched Placebo). The empirical benefit is predominantly driven by contrastive framing and token salience rather than specific historical error content.

### Key Metrics Summary:
- **Primary Endpoint:** $\Delta\text{Macro-F1}$ (F3 vs F4) = `+0.0806` (95% CI: `[-0.0238, +0.1085]`)
- **Accuracy Comparison:** F3 = `54.91%` vs F4 = `55.64%` ($\Delta$ = `-0.73%`)
- **McNemar Chi-Square Statistic:** `0.12` (Net Benefit: `-4 claims`)


# Milestone M2.0i: Mechanism Analysis & Factorial Decomposition

## 1. Experimental Design: 2x2 Factorial + Placebo

| Arm | Description | Contrastive Framing | Mistake Content |
|:---|:---|:---:|:---:|
| **F0_neutral_expl** | Standard reasoning + Neutral explanation | No | No |
| **F1_header_only** | Contrastive header + Neutral explanation | Yes | No |
| **F2_observed_mistake** | Standard reasoning + Actual observed mistake | No | Yes |
| **F3_full_historical_error** | Contrastive header + Actual observed mistake | Yes | Yes |
| **F4_matched_placebo** | Contrastive header + Length-matched placebo mistake | Yes | Placebo |

## 2. Factorial Effects Decomposition
- **Main Effect of Framing:** Accuracy $\Delta = +4.73%$, Macro-F1 $\Delta = +0.0251$
- **Main Effect of Mistake Content:** Accuracy $\Delta = +4.00%$, Macro-F1 $\Delta = -0.0013$
- **Interaction (Framing x Content):** Accuracy $\Delta = +1.82%$, Macro-F1 $\Delta = -0.0021$

## 3. Length-Matched Control: F3 vs F4
- **Delta Accuracy:** `-0.73%`
- **Delta Macro-F1:** `+0.0806`
- **Paired Bootstrap 95% CI:** `[-0.0238, +0.1085]`
