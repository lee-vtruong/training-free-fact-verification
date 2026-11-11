# Milestone M2.0h: Independent Scientific Audit Decision Report

## Final Determination: `VALIDATED_WITH_LIMITATIONS`

**Rationale:** Empirical evidence robustly confirms that Historical Error Knowledge significantly outperforms Matched Gold ICL across 15/15 seed-k configurations (+10.62% at k=2, +4.98% at k=4, +6.15% at k=6) and generalizes to out-of-sample outer folds (+7.20% over B0, +11.00% over Gold ICL) with zero data or group leakage. However, the finding is qualified by two scientific limitations: (1) Exemplar rationales are source-traceable from AVeriTeC train data and M1.6.2d triage but lack formal human review (P2 classification), and (2) K4 adds ~200 prompt tokens without an explicit token-matched filler control, so part of the gain may reflect attention allocation.

## Multi-Dimensional Evaluation Breakdown

- **1. Engineering & Integrity Validity:** `PASS` (100% count conservation, 0 parser mismatches, 0 group leaks).
- **2. Statistical Robustness Evidence:** `ROBUST` (15/15 seed-k configurations won by Historical Error ICL).
- **3. Knowledge Provenance Classification:** `P2_DOMINANT` (90% P2 source-traceable from AVeriTeC train & M1.6.2d triage, 10% P1).
- **4. Generalization Evidence:** `SUPPORTED` (+7.20% accuracy gain on 500 outer claims with zero group overlap).
- **5. Causal Attribution:** `QUALIFIED_BY_LENGTH_CONFOUND` (Contrastive error effect confirmed, qualified by prompt-length asymmetry).

## Key Scientific Limitations

- Provenance: 9/10 exemplars in CERTIFIED_EXEMPLAR_BANK are P2 (source-traceable to AVeriTeC train and M1.6.2d/e triage with exact Top-8 quotes, but LLM-drafted without formal human reviewer sign-off).
- Prompt Length Confounding: E2 K4 prompt is longer than K1/K2 by ~200 tokens. While M1.6.2c/e showed that pure word-length filler without error content adds +0.00% accuracy, M2.0g itself did not rerun an explicit length-matched placebo arm.
- Demonstration Saturation: Accuracy at k=4 (51.05%) and k=6 (52.36%) drops relative to k=2 (55.75%), demonstrating prompt dilution as context length expands beyond 2,000 tokens.
- Ontology Asymmetry: Conflicting Evidence recall remains challenging across all training-free approaches without neural reranking.

## Recommended Next Actions

1. Conduct a single targeted 2-arm length-controlled experiment comparing K4 against Length-Matched Neutral Filler (K4-Placebo) to mathematically isolate semantic error contrast from token overhead.
1. Subject the 10 certified exemplar rationales to two-reviewer blind human verification following the M1.6.2e Gate Policy.
1. Lock k=2 as the canonical demonstration budget for fact verification with frozen Qwen3-8B.
