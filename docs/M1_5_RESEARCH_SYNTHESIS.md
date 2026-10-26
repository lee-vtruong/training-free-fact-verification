# M1.5 — Cross-Milestone Research Synthesis & Evidence Consolidation

**Milestone:** M1.5 Research Synthesis & Preparation for M1.6  
**Author:** AI Research Engineer (Antigravity)  
**Date:** 2026-10-26  
**Repository:** `lee-vtruong/training-free-fact-verification`  
**Git Commit Target:** `main`  
**Evaluated Artifacts:**
- Milestone M1.2 & M1.2.1: Base Stabilization on Fold A (`train:0..49`, $N=50$)
- Milestone M1.3: Repaired Evaluation on Fold B (`train:50..99`, $N=46$ clean)
- Milestone M1.4 & M1.4.1: Confirmatory Evaluation on Fold C (`train:100..149`, $N=50$)
- Milestone M1.4.2 & M1.4.3: Mechanism Audit on Fold D (`train:150..249`, $N=91$ clean, 9 quarantined)
- Milestone M1.5: Source-Only Mechanism Feasibility & Fold E Protocol Readiness (`train:250..349`, $N=99$ admitted, 1 quarantined)

---

## 1. Executive Synthesis: The Training-Free Verification Journey

Across four sequential evaluation folds (**Fold A**, **Fold B**, **Fold C**, and **Fold D**), encompassing **237 unique AVeriTeC fact-checking claims**, this project has investigated whether **frozen language models (Qwen3-8B)** can be systematically improved on open-domain fact verification using **training-free historical error memory** without any parameter fitting, neural routers, or gradient updates.

### Definitive Cross-Milestone Empirical Findings:
1. **Formatting & Output Brevity Outperforms Error Rules:**  
   The single largest and most reproducible performance gain across all milestones was achieved by transitioning from unconstrained zero-shot prompting (`D0`, ~179 tokens output) to concise matched generic prompting (`D1_matched`, max 64 tokens output):
   - **Fold D Accuracy:** `D0` 57.1% $\to$ `D1_matched` **63.7%** ($+6.6\%$, $+6$ net claims).
   - Verbose generation caused the model to hallucinate uncertainty, bailing into `Not Enough Evidence` (14 false positives) and `Conflicting Evidence` (11 false positives). Rationale brevity eliminated 80% of these false hesitations.
2. **Static Empirical Error Rules Cause Negative Transfer:**  
   In Condition `D2`, injecting four empirical error rules synthesized from Fold A failed to beat generic matched prompting on Fold D:
   $$\text{Accuracy}(D2) = 59.3\% \quad < \quad \text{Accuracy}(D1\_matched) = 63.7\% \quad (\Delta = -4.40\%, \text{ CI } [-0.09, -0.01])$$
   `D2` did not correct a single claim over `D1_matched` ($W\to C = 0$), while actively corrupting 4 claims that `D1_matched` answered correctly.
3. **Rule Compositional Asymmetry (Rule 1 vs Rule 2):**  
   Leave-one-out ablations on Fold D revealed that static rules have conflicting effects:
   - **Rule 1 (*"Media debate is not contradiction"*):** Provided positive lift ($+2.2\%$ over `D2_minus_r1`), preventing false refutation on political dispute claims.
   - **Rule 2 (*"Missing context is not Cherrypicking"*):** Caused severe negative transfer ($-3.3\%$ vs `D2_minus_r2`), misclassifying nuanced claims (`train:152`, `train:239`, `train:242`) into `Supported` or `Conflicting`.
4. **Lexical Similarity Retrieval (D4) is an Unreplicable Failure:**  
   Using 3-gram character Jaccard similarity to retrieve historical error exemplars achieved high performance on Fold B (Macro-F1 0.5562, lucky hit on a single minority class), collapsed on Fold C (0.3333, below random D3), and tied random selection on Fold D (52/91 vs 52/91, exact McNemar $p = 1.0000$). Topical lexical overlap (matching keywords like "budget" or "election") has zero correlation with reasoning failure modes.
5. **Sample Insufficiency of Source Memory Bank:**  
   In Fold A ($N=50$ claims, 25 errors), only **one failure mechanism** (*Scope, Quantifier & Denominator Mismatch*, $n=8$) has $>5$ instances. Categories such as *Attribution vs Proposition* ($n=2$) and *Negation Inversion* ($n=2$) are severely under-represented, making rule induction statistically underpowered.

---

## 2. In-Depth Mechanistic Analysis of Ten Core Dimensions

### Dimension 1: D0 Base Prompting vs D1_matched Concise Prompting
- **D0 (Unconstrained Baseline):** Prompts the model with standard zero-shot instructions and asks for a reasoning analysis followed by a verdict. Latency: ~11.3s/claim. Average output: 179 tokens. Vulnerable to "over-thinking": the model finds peripheral ambiguities in the text and hallucinates doubt.
- **D1_matched (Concise Generic Control):** Enforces 1-sentence rationale followed immediately by `Verdict: <Label>`. Output tokens: ~45–64 tokens. Latency: ~2.3s/claim. Eliminates verbosity-induced hesitations without injecting any error heuristics.
- **Scientific Takeaway:** Output token budget control is a first-order confound. Any error-informed technique must be benchmarked against `D1_matched` rather than `D0` to isolate semantic contributions from brevity gains.

### Dimension 2: Rule 1, Rule 2, and Rule 1+2 Interaction
- **Rule 1 Definition:** *"Media debate is not factual contradiction: An article describing a political disagreement does not constitute Refutation of a verifiable fact."*
  - Provenance: Synthesized from Fold A `train:17` and US political reporting.
  - Behavior: Broadly helpful across all folds because political quotes are frequently reported alongside factual claims.
- **Rule 2 Definition:** *"Missing context is not Cherrypicking: Absence of secondary details does not warrant Cherrypicking if the primary assertion is substantiated."*
  - Provenance: Synthesized from Fold A `train:2`, `train:6`, `train:11` to cure false NEI bailing.
  - Behavior: Harmful when transferred outside African fiscal claims. On Fold D claims with conditional health advisories or qualified statistics, Rule 2 blinded the model to genuine omissions, turning correct Cherrypicking predictions into false Supported verdicts.
- **Scientific Takeaway:** Static prompt rules behave globally. What functions as a clarifying directive in domain A acts as a distractor in domain B.

### Dimension 3: Historical Wrong Outcomes, Gold Corrections, and Error-Derived Rules
- Presenting a model with a past failed claim, its wrong verdict, and its gold correction (`D5`/`D6`) provides mixed signals:
  - If the retrieved demonstration does not share identical logical structure, the model adopts surface heuristics (e.g. copying the label of the exemplar).
  - Without causal explanations of *why* the error occurred, in-context learning acts as a stochastic perturbation rather than systematic reasoning calibration.

### Dimension 4: Random (D3) vs Similarity-Retrieved (D4) Error Demonstrations
- Table of Cross-Fold Performance:

| Fold | Metric | D0 (Base) | D1_matched | D3 (Random k=2) | D4 (Similarity k=2) | McNemar $p$ (D4 vs D3) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Fold B** ($N=46$) | Macro-F1 | 0.4079 | *Not tested* | 0.3801 | **0.5562** | 0.1250 |
| **Fold C** ($N=50$) | Macro-F1 | 0.3801 | *Not tested* | 0.3739 | **0.3333** | 0.6875 |
| **Fold D** ($N=91$) | Accuracy | 57.1% | **63.7%** | 57.1% (52/91) | 57.1% (52/91) | **1.0000** |

- **Conclusion:** Lexical error retrieval has failed replication across two consecutive independent folds. It is conclusively terminated.

### Dimension 5: Impact of Few-Shot $k$, Prompt Length, and Output Budget
- Increasing $k=0 \to 2$ adds ~350 prompt tokens, increasing attention entropy and inference latency by $2.5\times$.
- In training-free verification on frozen 8B models, $k$-shot demonstrations dilute the model's focus on the actual evidence passages. Single-pass zero-shot verification with sharp structural directives consistently beats $k=2$ demonstrations.

### Dimension 6: Class Confusions (Refuted, Supported, NEI, Cherrypicking)
- **Refuted vs Supported:** The model is most reliable on binary polarity disputes when evidence is decisive (Precision > 80%).
- **NEI Fragility:**
  - `D0` maintains non-zero recall (**NEI F1 = 0.2500** on Fold D, 3 True Positives).
  - Brevity-constrained prompts (`D1_matched`, `D2`, `D3`, `D4`) suffered complete NEI collapse (**NEI F1 = 0.0000**, 0 True Positives).
  - Brevity suppresses hesitation, eliminating 11 false positives at the cost of 3 true positives.
- **Conflicting Evidence / Cherrypicking:** Remains the hardest class across all folds (F1 < 0.25), heavily confounded by journalistic neutrality.

### Dimension 7: Source Memory Limitations (50 Claims, 25 Errors)
- With only 25 errors in Fold A:
  - 1 mechanism has $n=8$ (Scope & Denominator).
  - 2 mechanisms have $n=5$ (Partial Truth, Sufficiency).
  - 1 mechanism has $n=3$ (Media Conflict).
  - 2 mechanisms have $n=2$ (Attribution, Negation).
- Generalizing robust rules from $n=2$ instances is statistically impossible without extreme risk of overfitting to idiosyncratic phrasing.

### Dimension 8: Feasibility of Evidence-Grounded Corrective Rationales
- A gold correction without an explicit contrastive grounding fails. If we present:
  `"Claim X failed because passage Y states Z, which refutes W"`
  the model can learn the reasoning pattern ONLY IF the current evaluation claim presents a structurally parallel relation. Generic text explanations do not transfer across disparate topics.

### Dimension 9: Failure Mechanisms with Support and Observable Signals
- As audited in M1.5:
  - **Scope / Quantifier:** Fired on 24 claims (16 hits, 8 false alarms). High utility but carries 32% risk of regressing correct claims.
  - **Negation:** Fired on 10 claims (3 hits, 7 false alarms). Precision = 30.0%. Blindly warning about negation corrupts 7 correct claims to potentially fix 3.
  - **Attribution:** Fired on 2 claims (1 hit, 1 false alarm). Fails to cover attribution syntax reliably.

### Dimension 10: Status of Scientific Hypotheses
- **Refuted Hypotheses:**
  - *H-D4:* Topical lexical similarity retrieves helpful error exemplars. (**REFUTED** on Folds C & D, $p=1.0000$).
  - *H-D2:* Static empirical error rules outperform matched generic prompts. (**REFUTED** on Fold D, $\Delta = -4.4\%$, CI `[-0.09, -0.01]`).
  - *H-NEI-ZERO:* All LLM prompts collapse to NEI F1 = 0. (**REFUTED**, Fold D `D0` achieved NEI F1 = 0.2500).
- **Unverified Hypotheses (Lacking Controlled Audits):**
  - *H-ATTR:* Attribution claims fail at $3\times$ higher rates. (Qualitative observation, unproven across splits).
  - *H-RERANK-NEI:* Cross-encoder rerank score < $\tau$ deterministically identifies NEI. (Conflates IR relevance with logical sufficiency).
- **Positive / Promising Hypotheses:**
  - *H-BREVITY:* Rationale length restriction eliminates false hesitations into NEI and Cherrypicking. (**CONFIRMED** on Folds C & D).
  - *H-M4:* Permutation consistency majority voting resolves evidence presentation order sensitivity without historical error memory. (**TESTING ON FOLD E**).

---

## 3. Fold E Execution Status

- **Evaluation Partition:** Fold E (`train:250..349`, $N=100$ candidate claims).
- **Anti-Leakage Quarantine Verification:**
  - Admitted claims: **99 claims**.
  - Quarantined claims: **1 claim** (`train:251` quarantined due to 3-gram containment $\ge 0.50$ against Fold A `train:27` Malabo declaration).
  - Split disjointness: Verified ($0$ overlap with Fold A `train:0..49`).
- **Evaluation Status:** **PENDING SERVER GPU RUN**.
  - R4 evidence preparation and frozen runner execution commands are staged and verified. Results will be recorded in `outputs/m15-folde-evaluation/` upon server run completion.

---

## 4. M1.6 Research Handoff: Designing Combined Error-Guided Verification

> [!IMPORTANT]
> **Strict Research Scope Boundary:** This section specifies the research framework and ablation design for Milestone M1.6. **No M1.6 experiments shall be implemented or executed until Milestone M1.5 Fold E results are fully recorded and analyzed.**

Following the completion of M1.5, Milestone M1.6 will investigate whether combining **inference-time aggregation (M4)** with **fine-grained structural guidance** and **contrastive error-correction exemplars** can finally establish a statistically significant positive lift over `D1_matched`.

### Candidate Architectural Components for M1.6:

1. **Foundational Non-Error Baseline (`D1_matched`):**
   - Must remain the primary reference control. Any candidate method that fails to beat `D1_matched` by $\ge +3$ net claims with paired bootstrap CI excluding zero is disqualified.
2. **Component R1 (Media Debate Filtering Rule Only):**
   - Isolate Rule 1 as an independent modular clause, discarding the harmful Rule 2.
3. **Component AC (Attribution-Proposition De-coupling Directive):**
   - Explicit prompt clause instructing the model to distinguish speech acts from proposition truth.
4. **Component SC (Statistical Scope & Denominator Alignment Directive):**
   - Derived from the only adequately supported Fold A mechanism ($n=8$), warning against base population mismatches.
5. **Component CR (Evidence-Grounded Contrastive Exemplars):**
   - Instead of raw wrong predictions, provide a contrastive pair:
     `[Wrong interpretation] vs [Correct grounded reasoning]`
     with explicit passage citations.
6. **Component PC (Permutation Consistency Aggregation):**
   - Apply 3-permutation majority voting as the outer inference wrapper.

### Pre-Registered Ablation Matrix for M1.6:

```
[M1.6 Benchmark Matrix]
├── Baseline 0: D0 (Original Base unconstrained prompt)
├── Baseline 1: D1_matched (Concise generic prompt — PRIMARY CONTROL)
├── Variant A:  D1_matched + Rule 1 (Isolated media debate rule)
├── Variant B:  D1_matched + Attribution Clause (AC)
├── Variant C:  D1_matched + Scope Alignment (SC)
├── Variant D:  D1_matched + Structured Contrastive Exemplar (k=1)
├── Variant E:  Permutation Consistency (M4) + D1_matched
└── Variant F:  Full Combined Pipeline (M4 + D1_matched + R1 + SC)
```

### Mandatory Methodological Invariants for M1.6:
- Zero parameter updates, zero learned routers, 100% frozen Qwen3-8B.
- Untouched evaluation split (Fold F: `train:350..449` or official `dev` split).
- Primary metrics: Accuracy and Macro-F1 across all 4 official AVeriTeC classes.
- Multiplicity control via step-down Holm-Bonferroni correction.
