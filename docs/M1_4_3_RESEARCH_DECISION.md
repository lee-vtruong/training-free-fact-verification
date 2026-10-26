# M1.4.3 — Strategic Research Decision & Roadmap

**Milestone:** M1.4.3 Research Direction Audit & Strategic Roadmap  
**Author:** AI Research Engineer (Antigravity)  
**Date:** 2026-10-26  
**Repository:** `lee-vtruong/training-free-fact-verification`  
**Decision Scope:** Post-M1.4.2 Failure Mode Analysis & Transition to M1.5  
**Core Constraint:** Strictly training-free. Zero parameter updates. Zero learned neural routers. Frozen models only.

---

## Executive Decision: Deprecating Static In-Context Error Injection

Following three independent evaluation iterations across **187 unseen claims** (Fold B: $N=46$, Fold C: $N=50$, Fold D: $N=91$ clean), the empirical evidence is decisive:

1. **`D4` (Similarity-Retrieved Error Demonstrations) is Terminated:**
   Lexical similarity retrieval of error exemplars failed replication on both Fold C (Macro-F1 0.3333 vs D0 0.3801) and Fold D (57.1% vs generic 63.7%). Topic similarity does not correlate with reasoning failure modes.
2. **`D2` (Static Empirical Error Rules) is Terminated:**
   On Fold D, `D2` was defeated by `D1_matched` (59.3% vs 63.7%, Net Gain = $-4$ claims, 95% CI `[-0.09, -0.01]`). Static negative rules create blind spots and overfit to the narrow distributions of the source fold.
3. **The Current Best Training-Free Baseline is `D1_matched`:**
   A concise, format-controlled generic prompt ([`CONCISE_MATCHED_GENERIC_RULES`](file:///c:/Users/ASUS/Downloads/training-free-fact-verification/src/tffv/verification/m14_prompting.py#L50-L55)) achieves **63.7% accuracy** on Fold D by curbing output verbosity (64 tokens vs 179 tokens), outperforming all historical error-injection methods.

**Strategic Pivot:** We will NOT build larger error memory banks or craft more static prompt heuristics. Future training-free work must address **concrete semantic and structural failure mechanisms** with clear failure-mode telemetry.

---

## Proposed Training-Free Research Directions

We propose exactly **three rigorous, strictly training-free directions**. Each direction is fully specified with research hypotheses, unlearned mechanisms, mandatory baselines, minimal ablations, resource costs, go/no-go gates, and novelty risk assessments.

---

### Direction 1: Deterministic Permutation-Consistency Calibration (Semantic Calibration)

#### 1. Research Hypothesis
Un-finetuned LLMs (Qwen3-8B) suffer from high-variance verdict flips when exposed to arbitrary input permutations (e.g. reversing the order of retrieved web passages or slight prompt paraphrasing). Claims where the verdict is invariant across permutations represent high-confidence ground truth predictions. Conversely, claims that flip verdicts under superficial permutation represent unstable hallucinations. Aggregating across deterministic permutations will eliminate high-variance errors without parameter tuning.

#### 2. Training-Free Mechanism
For each claim $c$ with Top-8 retrieved evidence passages $E = [p_1, p_2, \dots, p_8]$:
1. **Pass 1 (Canonical):** Evidence order $[p_1, \dots, p_8]$ with `D1_matched` prompt.
2. **Pass 2 (Inverted Evidence):** Evidence order reversed $[p_8, \dots, p_1]$ with `D1_matched` prompt.
3. **Pass 3 (Syntactic Paraphrase):** Evidence order $[p_1, \dots, p_8]$ with `D2_para1` prompt (action-guideline style).
- **Decision Rule (Deterministic Greedy, $T=0.0$):**
  - If $\text{verdict}_1 = \text{verdict}_2 = \text{verdict}_3$ (Unanimous): Emit consensus verdict.
  - If 2 out of 3 agree (Majority): Emit majority verdict.
  - If all 3 disagree (Tie / High Uncertainty): Abstain to `Not Enough Evidence` or verify against top-1 passage.

#### 3. Mandatory Baselines
- `D0` (Single-pass verbose baseline).
- `D1_matched` (Single-pass concise generic control — current champion: 63.7%).

#### 4. Minimal Ablations
- **Ablation 1:** Evidence order permutation only (Pass 1 + Pass 2).
- **Ablation 2:** Prompt paraphrase permutation only (Pass 1 + Pass 3).
- **Ablation 3:** Permutations vs Stochastic temperature sampling ($T=0.3$, 3 draws).

#### 5. Resource & GPU Cost
- $3 \text{ forward passes} \times 91 \text{ claims} = 273 \text{ forward passes}$.
- At $\approx 2.0\text{s}$ per forward pass on RTX 5090: **$\approx 9.1 \text{ minutes}$** total GPU runtime.
- Zero disk index overhead; purely inference-time aggregation.

#### 6. Go/No-Go Decision Criteria
- **GO:** Achieves statistically significant improvement over single-pass `D1_matched` (Exact McNemar $p < 0.05$, Net Gain $\ge +4$ claims on Fold D) AND Macro-F1 $\ge 0.4200$.
- **NO-GO:** If Net Gain $\le +1$ claim or McNemar $p \ge 0.10$ relative to `D1_matched`.

#### 7. Novelty Risk Assessment
Self-consistency (Wang et al., 2022) is standard in reasoning. Permutation consistency across retrieved evidence has been studied in multi-document QA (Peysakhovich & Lerer, 2023). **Novelty cannot be claimed for the consistency concept itself.** The contribution must be strictly framed as an empirical calibration benchmark for training-free 4-class fact verification.

---

### Direction 2: Two-Stage Attribution-Fact Decomposition (Quote vs World Fact Separation)

#### 1. Research Hypothesis
A primary failure mode identified in Folds A, C, and D is the model's inability to distinguish between **quote attribution** (*"Politician X claimed Y"*) and **empirical factual truth** (*"Y is true"*). A claim asserting that a politician made a false statement is `Supported` if the speech occurred, but base models frequently evaluate the underlying proposition Y and wrongly predict `Refuted`. Conversely, on empirical claims, models treat reported political quotes as factual evidence. Decomposing verification into an attribution detection step followed by a targeted verification prompt will resolve this confusion.

#### 2. Training-Free Mechanism
- **Stage 1 (Claim Type Classification — Frozen Qwen3-8B, 15 tokens):**
  Classify the claim into:
  - `[Type A: Attribution / Direct Speech Statement]`
  - `[Type B: Empirical World / Statistical Assertion]`
- **Stage 2 (Targeted Verification Prompt):**
  - If `Type A`: Deploy an Attribution-Grounded Prompt instructing the model to verify whether the source confirmed that the speaker made the statement.
  - If `Type B`: Deploy `D1_matched` (Standard Verification Directives).

#### 3. Mandatory Baselines
- `D1_matched` (Single-stage generic prompt).
- `D0` (Single-stage base prompt).

#### 4. Minimal Ablations
- **Ablation 1:** Rule-based regex parser (verbs: "said", "claimed", "tweeted", "pledged") vs LLM Stage 1 classifier.
- **Ablation 2:** Unified prompt with attribution clause vs Two-stage branched execution.

#### 5. Resource & GPU Cost
- Stage 1: 91 claims $\times 15$ tokens $\approx 36$ seconds.
- Stage 2: 91 claims $\times 60$ tokens $\approx 180$ seconds.
- Total GPU Runtime: **$\approx 3.6 \text{ minutes}$** on RTX 5090.

#### 6. Go/No-Go Decision Criteria
- **GO:** Outperforms `D1_matched` by $\ge +3$ net claims AND reduces attribution-induced misclassifications by $\ge 40\%$.
- **NO-GO:** If Stage 1 classification accuracy is $< 85\%$ or fails to improve overall accuracy over `D1_matched`.

#### 7. Novelty Risk Assessment
Claim decomposition is well explored in QA-based fact-checking (e.g. FACTIFY, MiniCheck). Training-free attribution routing on frozen models is clean engineering, but not an algorithmic breakthrough. Must be reported as a failure-mode-driven heuristic.

---

### Direction 3: Retrieval-Density Gated Evidence Sufficiency Filtering (Calibration for NEI)

#### 1. Research Hypothesis
Across all three evaluation folds, ALL LLM prompts (`D0`, `D1_matched`, `D2`, `D3`, `D4`) achieved an **F1 score of 0.0000 on `Not Enough Evidence` (NEI)**. LLMs possess a strong generation bias toward either refuting or supporting claims when presented with arbitrary text. The decision for `Not Enough Evidence` is fundamentally an **information-retrieval density property**, not a generation property. Evaluating retrieval cross-encoder score distributions can deterministically identify NEI claims without LLM prompting.

#### 2. Training-Free Mechanism
1. From the frozen retrieval pipeline (R1 BM25 + R4 Qwen3-Reranker-0.6B), extract the maximum reranker cross-encoder score among the top-8 evidence passages:
   $$S_{\text{max}} = \max_{p \in \text{Top8}} \text{Score}_{\text{R4}}(p)$$
2. Compute an unlearned threshold $\tau_{\text{NEI}}$ from source Fold A (e.g. 10th percentile of true positive evidence scores).
3. **Decision Rule:**
   - If $S_{\text{max}} < \tau_{\text{NEI}}$: Deterministically assign **`Verdict: Not Enough Evidence`** (bypassing the generator).
   - If $S_{\text{max}} \ge \tau_{\text{NEI}}$: Pass evidence to `D1_matched` to classify between `Supported`, `Refuted`, and `Conflicting`.

#### 3. Mandatory Baselines
- `D1_matched` (Pure LLM prompt without retrieval gating).
- Random class-prior baseline.

#### 4. Minimal Ablations
- **Ablation 1:** BM25 max score vs Reranker cross-encoder max score.
- **Ablation 2:** Hard deterministic gating vs Soft score injection into prompt.
- **Ablation 3:** Sensitivity sweep across $\tau \in [0.1, 0.5]$.

#### 5. Resource & GPU Cost
- **0 additional GPU forward passes**! Reuses already persisted retrieval scores in `R4.jsonl`.
- CPU Execution Time: **$< 0.1 \text{ seconds}$**.

#### 6. Go/No-Go Decision Criteria
- **GO:** Increases `Not Enough Evidence` F1 from 0.0000 to $\ge 0.3000$ (identifying at least 2 true NEI claims) without degrading overall accuracy below `D1_matched` (63.7%).
- **NO-GO:** If false positive NEI rate exceeds 10% (corrupting valid Supported/Refuted claims).

#### 7. Novelty Risk Assessment
Retrieval score thresholding is standard practice in open-domain question answering (Karpukhin et al., 2020). Novelty cannot be claimed for thresholding. The contribution is limited to calibrating retrieval density as an inductive prior for the 4th class in AVeriTeC.

---

## Strategic Recommendation for Milestone M1.5

We recommend prioritizing **Direction 1 (Deterministic Permutation-Consistency Calibration)** as the primary research investigation for Milestone M1.5, with **Direction 3 (Retrieval-Density Gated NEI)** running as an orthogonal zero-cost filter:

1. **Direction 1 directly addresses the variance problem** that plagued D0, D2, and D3, requiring only ~9 minutes of GPU time.
2. **Direction 3 costs 0 GPU seconds** and directly targets the glaring 0.0000 F1 score on `Not Enough Evidence`.
3. Combined, these two directions require zero parameter training, zero learned routers, and zero data leakage, while providing clean, testable, and falsifiable hypotheses.
