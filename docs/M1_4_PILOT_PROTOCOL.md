# M1.4 Pilot Protocol: Controlled Error-Informed Fact Verification (D0–D6)

**Milestone:** M1.4 — Controlled Fact Verification Pilot  
**Date:** 2026-10-25  
**Core Scientific Question:** Can training-free, frozen LLM fact verification benefit from historical error signals when demonstrations and rules are constrained to concise, topically retrieved, and context-bounded formulations?  
**Constraint:** Strictly Training-Free & Quality-First (Zero parameter fine-tuning, zero gradients, zero learned neural routers).  
**Pilot Designation:** **EXPLORATORY DEBUG PILOT** (Conducted on exploratory Fold B claims to debug mechanisms; confirmatory claims require an untouched evaluation split).

---

## 1. Experimental Conditions Matrix (D0–D6)

| Condition | Protocol Name | Prompt Strategy | Context Budget | Purpose & Hypothesis |
|:---|:---|:---|:---:|:---|
| **D0** | **Frozen Base Protocol v1.0** | Zero-shot standard prompt | ~2,500 tokens | Baseline anchor (identical to M1.2.1 B1 and M1.3 C0). |
| **D1** | **Concise Generic Rules** | Core verification directives in 2 concise sentences | ~2,580 tokens | Test whether concise directives improve calibration without triggering verbose generation. |
| **D2** | **Concise Error Rules** | 2 source-derived error guards (debate $\neq$ contradiction; missing detail $\neq$ cherrypicking) | ~2,600 tokens | Test whether distilling Fold A error patterns concisely prevents both truncation and error pitfall traps. |
| **D3** | **Random Error Demos ($k=2$)** | 2 random error exemplars from Fold A (compact: Top-1 passage, $\le 250$ chars) | ~2,850 tokens | Test whether reducing demonstration count ($k=4 \to k=2$) prevents prompt bloat and negative transfer. |
| **D4** | **Similar Error Demos ($k=2$)** | Top-2 error exemplars retrieved via lexical 3-gram similarity to target claim | ~2,850 tokens | Test whether topically/lexically aligned error exemplars provide more relevant guidance than random exemplars. |
| **D5** | **Similar Demos + Corrections ($k=2$)** | Top-2 similar error exemplars with explicit base mistake and verified gold label | ~2,950 tokens | Test whether explicit contrast between base error and gold truth provides effective in-context calibration. |
| **D6** | **Contrastive Demos ($k=2$)** | 1 nearest Correct exemplar + 1 nearest Error exemplar from Fold A | ~2,900 tokens | Test whether pairwise boundary contrast (Correct vs Error) provides balanced decision guidance. |

---

## 2. Invariant Scientific Controls

To ensure strict experimental validity, all conditions adhere to identical foundations:
1. **Backbone:** Frozen `Qwen/Qwen3-8B` (`snapshots/b968826d9c46dd6066d109eabc6255188de91218`).
2. **Decoding Parameters:** Greedy decoding (`temperature=0.0`, `do_sample=False`, `enable_thinking=False`, `max_new_tokens=512`).
3. **Evidence Protocol:** Frozen R4 Top-8 passages from M1.1 retrieval.
4. **Label Ontology:** AVeriTeC four-class ontology (`Supported`, `Refuted`, `Not Enough Evidence`, `Conflicting Evidence/Cherrypicking`).
5. **Output Parsing:** Deterministic regex normalization (`Verdict: <Label>`).
6. **Token Cap & Conciseness Guard:** Every prompt instructs: *"State a concise 1-sentence factual reason, then output Verdict: <Label>."* This directly resolves the truncation failure observed in C6.

---

## 3. Data Partition & Anti-Leakage Protocol

- **Source Memory Bank (Fold A):** Claims `train:0` through `train:49` ($N=50$).
  * Built using verbatim predictions from Base Protocol B1 and ground truth from AVeriTeC `train.json`.
  * Demonstrations are strictly sampled or retrieved from Fold A.
- **Evaluation Claims (Fold B):** Claims `train:50` through `train:99` ($N=50$).
  * Quarantined: 4 near-duplicates (`train:67, train:70, train:77, train:95`).
  * Admitted: Exactly 46 evaluation claims.
- **Leakage Guards:**
  * Absolute fold disjointness asserted before runner initialization (`assert_fold_disjointness`).
  * Exemplar overlap with evaluation claims is strictly asserted to be 0.0%.
  * Gold labels for evaluation claims are inaccessible during inference; predictions are sealed with SHA-256 hashes prior to scoring.

---

## 4. Analysis & Evaluation Framework

1. **Primary Metrics:** Accuracy and Macro-F1 across the 46 evaluation claims.
2. **Paired Transition Categories:** All 9 mutually exclusive transitions against D0:
   - $C \to C$, $C \to W$, $C \to U$ (Sum = 28)
   - $W \to C$, $W \to W$, $W \to U$ (Sum = 18)
   - $U \to C$, $U \to W$, $U \to U$ (Sum = 0)
   - Total transitions identically $= 46$.
3. **Statistical Significance:**
   - Two-sided exact binomial McNemar test on discordant pairs.
   - Multiplicity adjustment via Holm-Bonferroni.
   - Paired Bootstrap 95% Confidence Intervals (2,000 resamples).
4. **Token Usage & Truncation Accounting:**
   - Tracking prompt tokens and generated tokens for each claim.
   - Measuring valid verdict rate and detecting any truncation before the verdict token.

---

## 5. Server Execution Guide

Execute on `stackops@hvtham-server`:

```bash
# 1. Update repository
cd ~/whale/training-free-fact-verification
git pull origin main

# 2. Activate virtual environment
source .venv/bin/activate
export HF_HOME=/home/stackops/whale/cache/huggingface

# 3. Grant execution permissions and run M1.4 Pilot
chmod +x scripts/run_m14_pilot.sh
bash scripts/run_m14_pilot.sh 2>&1 | tee outputs/m14_run.log
```
