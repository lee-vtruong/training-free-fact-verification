# M1.2.1 — Frozen Base Stabilization Protocol & Comparison

**Milestone:** M1.2.1  
**Status:** IMPLEMENTED & READY FOR SERVER EXECUTION  
**Target Hardware:** NVIDIA GeForce RTX 5090 (32 GB VRAM) on `hvtham-server`  
**Dataset:** 100 AVeriTeC Exploratory Claims (`train:0` .. `train:99`)  
**Evidence Source:** M1.1 R4 Frozen Reranked Top-8 Passages  
**Strictly Training-Free:** 100% frozen parameters, zero fine-tuning, zero gradients, zero learned routers.

---

## 1. Context & Objective

In Milestone M1.2, zero-shot verification using frozen `Qwen/Qwen3-8B` with `max_new_tokens=512` achieved:
- **Overall Accuracy:** 28.0% (28/100)
- **Macro-F1:** 0.3076
- **Valid Verdict Rate:** 65.0% (65/100)
- **Conditional Accuracy on Parsed Claims:** 43.1% (28/65)
- **High Binary Precision:** `Refuted` 87.5%, `Supported` 80.0%

However, **35 out of 100 claims failed due to output truncation**: the model spent its entire 512-token generation budget producing detailed per-passage reasoning traces inside `<think>...</think>`, running out of tokens before emitting `Verdict: <Class>`.

**Milestone M1.2.1 Goal:** Stabilize the Frozen Base Verification pipeline by systematically comparing three controlled configurations (B0, B1, B2) on the exact same 100 exploratory claims, isolating the impact of thinking mode versus generation token budget.

---

## 2. Inference Audit Findings

### A. Checkpoint & Tokenizer Inspection
- **Model Backbone:** `Qwen/Qwen3-8B` (Apache-2.0, Revision `b968826d9c46dd6066d109eabc6255188de91218`).
- **Tokenizer:** Vocabulary size 151,669 tokens.
- **EOS & Stopping Conditions:**
  - EOS tokens: `<|im_end|>` (id: 151645) and `<|endoftext|>` (id: 151643).
  - PAD token: `<|endoftext|>` (id: 151643).
  - Generation mode: Strictly deterministic greedy decoding (`do_sample=False`, `temperature=None`, `top_p=None`).

### B. Chat Template & Thinking Mechanism Discovery
Inspection of the official `tokenizer_config.json` chat template revealed first-class native support for thinking control via Jinja2:
```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
```

#### Operational Verification:
1. **When `enable_thinking=True` (or omitted):**
   The prompt ends with `<|im_start|>assistant\n`. Qwen3-8B generates `<think>` and enters step-by-step reasoning over the evidence passages.
2. **When `enable_thinking=False`:**
   The prompt ends with `<|im_start|>assistant\n<think>\n\n</think>\n\n`. The pre-closed empty `<think>` block signals to the model that reasoning is already complete, forcing it to generate a direct, concise response without wasting tokens on intermediate reasoning traces.

### C. Output Parsing Refinement
In M1.2, reasoning inside `<think>` often discussed multiple potential verdicts ("this supports X but other sources refute Y"), creating ambiguity when full-text regex scanning was used on truncated outputs.
The parser has been upgraded to a **two-tier architecture**:
1. **Tier 1 (Post-Think Priority):** If `</think>` is present, extract the segment following `</think>` and parse the explicit verdict. This eliminates all reasoning noise.
2. **Tier 2 (Fallback):** If no `</think>` tag is present (e.g. non-thinking mode B1 or unclosed trace), parse the full output using strict high-confidence patterns (`Verdict: <Class>`, `The claim is <Class>`).

---

## 3. Experimental Configurations (B0, B1, B2)

All three configurations use identical frozen weights, identical prompt text, and identical Top-8 evidence passages:

| Config ID | Description | Thinking Mode | Max New Tokens | Generation Prefix |
|---|---|:---:|---:|---|
| **B0** | Baseline M1.2 (Reused Cache) | `True` (Native) | 512 | `<|im_start|>assistant\n` |
| **B1** | Non-Thinking Stabilized | `False` (Suppressed) | 512 | `<|im_start|>assistant\n<think>\n\n</think>\n\n` |
| **B2** | Extended Budget Thinking | `True` (Native) | 1,024 | `<|im_start|>assistant\n` |

### Configuration Files
- `configs/m121_b0_thinking_512.json`
- `configs/m121_b1_nonthinking_512.json`
- `configs/m121_b2_thinking_1024.json`

---

## 4. Evaluation & Scientific Protocols

1. **Strict Label-Free Execution:**
   Predictions are materialized in `predictions.jsonl` with token counts, latencies, and SHA-256 digests sealed in `manifest.json` before `train.json` is opened.
2. **Group-Safe Knowledge Isolation:**
   The 100 pilot claims remain designated as `exploratory_profiling_only`. They must never be used as a blind evaluation benchmark for future milestones.
3. **Paired Verdict Transitions:**
   Every claim is tracked across configurations to measure:
   - **Improved Claims:** Claims where B0 was unparsed or wrong, but the target configuration produced the correct verdict.
   - **Regressed Claims:** Claims where B0 was correct, but the target configuration produced an incorrect or unparsed verdict.
   - **Net Gain:** `Improved - Regressed`.

---

## 5. Experimental Results

### Performance Summary Table

| Metric | B0 (Thinking, 512) | B1 (Non-thinking, 512) | B2 (Thinking, 1024) |
|---|:---:|:---:|:---:|
| **Valid Verdict Rate** | 65.0% (65/100) | *PENDING SERVER EXECUTION* | *PENDING SERVER EXECUTION* |
| **Overall Accuracy** | 28.0% (28/100) | *PENDING SERVER EXECUTION* | *PENDING SERVER EXECUTION* |
| **Accuracy (Parsed Only)** | 43.1% (28/65) | *PENDING SERVER EXECUTION* | *PENDING SERVER EXECUTION* |
| **Macro-F1** | 0.3076 | *PENDING SERVER EXECUTION* | *PENDING SERVER EXECUTION* |
| **Macro-Precision** | 0.5023 | *PENDING SERVER EXECUTION* | *PENDING SERVER EXECUTION* |
| **Macro-Recall** | 0.2804 | *PENDING SERVER EXECUTION* | *PENDING SERVER EXECUTION* |
| **Supported F1** | 0.3404 | *PENDING SERVER EXECUTION* | *PENDING SERVER EXECUTION* |
| **Refuted F1** | 0.4912 | *PENDING SERVER EXECUTION* | *PENDING SERVER EXECUTION* |
| **Not Enough Evidence F1** | 0.1765 | *PENDING SERVER EXECUTION* | *PENDING SERVER EXECUTION* |
| **Cherrypicking F1** | 0.2222 | *PENDING SERVER EXECUTION* | *PENDING SERVER EXECUTION* |
| **Mean Latency per Claim** | 11.31s | *PENDING SERVER EXECUTION* | *PENDING SERVER EXECUTION* |
| **Mean Generated Tokens** | 499.11 | *PENDING SERVER EXECUTION* | *PENDING SERVER EXECUTION* |

### Paired Transitions vs Baseline B0
- **B0 $\to$ B1:** *PENDING SERVER EXECUTION*
- **B0 $\to$ B2:** *PENDING SERVER EXECUTION*

---

## 6. Artifact Hashes & Verified File Paths

| Artifact Description | Workspace Path | Status |
|---|---|---|
| **B0 Config** | `configs/m121_b0_thinking_512.json` | Pushed to GitHub |
| **B1 Config** | `configs/m121_b1_nonthinking_512.json` | Pushed to GitHub |
| **B2 Config** | `configs/m121_b2_thinking_1024.json` | Pushed to GitHub |
| **Stabilization Evaluator** | `src/tffv/verification/stabilization.py` | Pushed to GitHub |
| **Automated Runner** | `scripts/run_m121_stabilization.sh` | Pushed to GitHub |
| **Unit Test Suite** | `tests/test_base_stabilization.py` | 119/119 Tests Passed |
| **B0 Predictions Cache** | `outputs/m12-base-qwen3-8b-20261024-132900/` | Available on Server |
| **B1 Predictions** | `outputs/m121-b1-preds-<timestamp>/` | To be generated on Server |
| **B2 Predictions** | `outputs/m121-b2-preds-<timestamp>/` | To be generated on Server |
| **Comparison Report** | `outputs/m121-comparison-<timestamp>/comparison_report.md` | To be generated on Server |
