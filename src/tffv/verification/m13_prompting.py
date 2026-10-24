"""Prompt builders for M1.3 Error-Informed In-Context Learning and Rule Prompting (C0–C6)."""
from typing import Any, Dict, List, Optional, Tuple
from tffv.provenance import canonical, digest
from tffv.verification.error_memory import ErrorMemoryBank
from tffv.verification.ontology import VERDICTS
from tffv.verification.prompting import (
    SYSTEM_PROMPT,
    VERDICT_DEFINITIONS,
    build_verification_prompt,
    format_evidence_block,
)

GENERIC_TASK_INSTRUCTIONS = (
    "Detailed Verification Guidelines:\n"
    "1. Thoroughly read each evidence passage and identify statements directly relevant to the claim.\n"
    "2. If the passages explicitly affirm the claim's core facts, select 'Supported'.\n"
    "3. If the passages explicitly deny or contradict the claim's core facts, select 'Refuted'.\n"
    "4. If the passages do not provide enough specific details to confirm or deny the claim, select 'Not Enough Evidence'.\n"
    "5. If reliable sources present conflicting facts, or if the claim takes an isolated fact out of context "
    "to imply something misleading, select 'Conflicting Evidence/Cherrypicking'.\n"
    "6. Base your decision solely on the provided passages without relying on unstated assumptions."
)


def format_exemplar_evidence(compact_evidence: List[Dict[str, Any]], max_chars: int = 400) -> str:
    """Format compact evidence passages for demonstration exemplars."""
    if not compact_evidence:
        return "[No evidence passages provided.]"
    blocks = []
    for idx, ev in enumerate(compact_evidence, 1):
        text = ev.get("text", "").strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        source_url = ev.get("source_url", "source")
        blocks.append(f"  [Passage {idx}] ({source_url}): {text}")
    return "\n".join(blocks)


def format_standard_demonstration(record: Dict[str, Any], example_num: int) -> str:
    """Format standard demonstration for C2 (random) and C3 (correct-only)."""
    ev_str = format_exemplar_evidence(record.get("compact_evidence", []))
    label = record.get("gold_label", "")
    return (
        f"--- Example {example_num} ---\n"
        f"Claim: \"{record.get('claim', '')}\"\n"
        f"Evidence Passages:\n{ev_str}\n"
        f"Analysis: The evidence passages directly ground the factual status of this claim.\n"
        f"Verdict: {label}\n"
    )


def format_corrected_error_demonstration(record: Dict[str, Any], example_num: int) -> str:
    """Format corrected error demonstration for C4 (error-only with gold correction)."""
    ev_str = format_exemplar_evidence(record.get("compact_evidence", []))
    wrong_pred = record.get("base_predicted_verdict", "Unknown")
    gold_label = record.get("gold_label", "")
    justification = record.get("gold_justification", "")
    if len(justification) > 300:
        justification = justification[:300] + "..."

    return (
        f"--- Example {example_num} (Error Caution Example) ---\n"
        f"Claim: \"{record.get('claim', '')}\"\n"
        f"Evidence Passages:\n{ev_str}\n"
        f"Common Pitfall: Baseline models commonly misclassify this as '{wrong_pred}'.\n"
        f"Correction Analysis: {justification or 'Careful examination reveals the evidence establishes the correct label.'}\n"
        f"Verdict: {gold_label}\n"
    )


def format_contrastive_pair_demonstrations(
    pair: Tuple[Dict[str, Any], Dict[str, Any]],
    pair_num: int,
) -> str:
    """Format a contrastive pair for C5: one correct case and one corrected error case on the same boundary."""
    correct_rec, error_rec = pair
    ev_corr = format_exemplar_evidence(correct_rec.get("compact_evidence", []))
    ev_err = format_exemplar_evidence(error_rec.get("compact_evidence", []))

    boundary_name = f"Boundary Analysis {pair_num} (Distinguishing {correct_rec.get('gold_label')} vs Confusions)"
    
    return (
        f"=== {boundary_name} ===\n"
        f"[Case A: Canonical {correct_rec.get('gold_label')}]\n"
        f"Claim: \"{correct_rec.get('claim', '')}\"\n"
        f"Evidence Passages:\n{ev_corr}\n"
        f"Key Pattern: Evidence directly and unequivocally establishes the claim.\n"
        f"Verdict: {correct_rec.get('gold_label')}\n\n"
        f"[Case B: Subtle Counterpart - Avoid Common Mistake]\n"
        f"Claim: \"{error_rec.get('claim', '')}\"\n"
        f"Evidence Passages:\n{ev_err}\n"
        f"Common Pitfall: Erroneously labeled as '{error_rec.get('base_predicted_verdict')}'.\n"
        f"Correction Distinction: Look past superficial media quotes to the verifiable factual assertion.\n"
        f"Verdict: {error_rec.get('gold_label')}\n"
    )


def build_c0_prompt(claim: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """C0: Base Protocol v1.0 (Frozen B1 zero-shot)."""
    return build_verification_prompt(claim, evidence)


def build_c1_prompt(claim: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """C1: Generic Task Instruction (Enhanced guidelines, zero exemplars)."""
    evidence_block = format_evidence_block(evidence)
    
    user_content = (
        f"{GENERIC_TASK_INSTRUCTIONS}\n\n"
        f"Claim to verify:\n\"{claim}\"\n\n"
        f"Retrieved Evidence Passages:\n{evidence_block}\n\n"
        f"Based strictly on the evidence above, evaluate the truthfulness of the claim.\n"
        f"Choose exactly one of the following four official verdicts:\n"
        f"{VERDICT_DEFINITIONS}\n\n"
        f"Instructions:\n"
        f"- Analyze the relationship between the claim and the evidence briefly according to the guidelines.\n"
        f"- Conclude your analysis on the final line with the exact format:\n"
        f"Verdict: <Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking>"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    payload = {
        "condition": "C1",
        "system": SYSTEM_PROMPT,
        "user": user_content,
        "messages": messages,
    }
    return {**payload, "prompt_sha256": digest(payload)}


def build_c2_prompt(
    claim: str,
    evidence: List[Dict[str, Any]],
    memory_bank: ErrorMemoryBank,
    k: int = 4,
    seed: int = 42,
) -> Dict[str, Any]:
    """C2: Random Source In-Context Learning (k balanced demonstrations from D_source)."""
    exemplars = memory_bank.sample_random_source_exemplars(k=k, seed=seed)
    demo_blocks = [format_standard_demonstration(rec, i + 1) for i, rec in enumerate(exemplars)]
    demos_text = "\n".join(demo_blocks)
    evidence_block = format_evidence_block(evidence)

    user_content = (
        f"Here are reference fact-verification examples demonstrating how to evaluate claims:\n\n"
        f"{demos_text}\n"
        f"Now, evaluate the following new claim using the same standard:\n\n"
        f"Claim to verify:\n\"{claim}\"\n\n"
        f"Retrieved Evidence Passages:\n{evidence_block}\n\n"
        f"Choose exactly one of the following four official verdicts:\n"
        f"{VERDICT_DEFINITIONS}\n\n"
        f"Conclude your analysis on the final line with the exact format:\n"
        f"Verdict: <Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking>"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    payload = {
        "condition": "C2",
        "exemplar_sample_ids": [r["sample_id"] for r in exemplars],
        "system": SYSTEM_PROMPT,
        "user": user_content,
        "messages": messages,
    }
    return {**payload, "prompt_sha256": digest(payload)}


def build_c3_prompt(
    claim: str,
    evidence: List[Dict[str, Any]],
    memory_bank: ErrorMemoryBank,
    k: int = 4,
    seed: int = 42,
) -> Dict[str, Any]:
    """C3: Correct-Only In-Context Learning (k balanced demonstrations from D_correct)."""
    exemplars = memory_bank.sample_correct_only_exemplars(k=k, seed=seed)
    demo_blocks = [format_standard_demonstration(rec, i + 1) for i, rec in enumerate(exemplars)]
    demos_text = "\n".join(demo_blocks)
    evidence_block = format_evidence_block(evidence)

    user_content = (
        f"Here are verified reference fact-verification examples:\n\n"
        f"{demos_text}\n"
        f"Now, evaluate the following new claim based strictly on its retrieved evidence:\n\n"
        f"Claim to verify:\n\"{claim}\"\n\n"
        f"Retrieved Evidence Passages:\n{evidence_block}\n\n"
        f"Choose exactly one of the following four official verdicts:\n"
        f"{VERDICT_DEFINITIONS}\n\n"
        f"Conclude your analysis on the final line with the exact format:\n"
        f"Verdict: <Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking>"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    payload = {
        "condition": "C3",
        "exemplar_sample_ids": [r["sample_id"] for r in exemplars],
        "system": SYSTEM_PROMPT,
        "user": user_content,
        "messages": messages,
    }
    return {**payload, "prompt_sha256": digest(payload)}


def build_c4_prompt(
    claim: str,
    evidence: List[Dict[str, Any]],
    memory_bank: ErrorMemoryBank,
    k: int = 4,
    seed: int = 42,
) -> Dict[str, Any]:
    """C4: Error-Only In-Context Learning with Gold Corrections (k demonstrations from D_error)."""
    exemplars = memory_bank.sample_error_only_exemplars(k=k, seed=seed)
    demo_blocks = [format_corrected_error_demonstration(rec, i + 1) for i, rec in enumerate(exemplars)]
    demos_text = "\n".join(demo_blocks)
    evidence_block = format_evidence_block(evidence)

    user_content = (
        f"Below are historical fact-checking examples showing frequent error traps and their corrections:\n\n"
        f"{demos_text}\n"
        f"Now, carefully verify the following new claim, taking care to avoid these common verification traps:\n\n"
        f"Claim to verify:\n\"{claim}\"\n\n"
        f"Retrieved Evidence Passages:\n{evidence_block}\n\n"
        f"Choose exactly one of the following four official verdicts:\n"
        f"{VERDICT_DEFINITIONS}\n\n"
        f"Conclude your analysis on the final line with the exact format:\n"
        f"Verdict: <Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking>"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    payload = {
        "condition": "C4",
        "exemplar_sample_ids": [r["sample_id"] for r in exemplars],
        "system": SYSTEM_PROMPT,
        "user": user_content,
        "messages": messages,
    }
    return {**payload, "prompt_sha256": digest(payload)}


def build_c5_prompt(
    claim: str,
    evidence: List[Dict[str, Any]],
    memory_bank: ErrorMemoryBank,
    num_pairs: int = 2,
    seed: int = 42,
) -> Dict[str, Any]:
    """C5: Contrastive Correct/Wrong In-Context Learning."""
    pairs = memory_bank.sample_contrastive_pairs(num_pairs=num_pairs, seed=seed)
    pair_blocks = [format_contrastive_pair_demonstrations(p, i + 1) for i, p in enumerate(pairs)]
    pairs_text = "\n\n".join(pair_blocks)
    evidence_block = format_evidence_block(evidence)

    user_content = (
        f"Examine these contrastive fact-checking boundary pairs that contrast correct reasoning with common pitfalls:\n\n"
        f"{pairs_text}\n\n"
        f"Now, evaluate the target claim by applying these boundary distinctions:\n\n"
        f"Claim to verify:\n\"{claim}\"\n\n"
        f"Retrieved Evidence Passages:\n{evidence_block}\n\n"
        f"Choose exactly one of the following four official verdicts:\n"
        f"{VERDICT_DEFINITIONS}\n\n"
        f"Conclude your analysis on the final line with the exact format:\n"
        f"Verdict: <Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking>"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    used_sids = []
    for c_rec, e_rec in pairs:
        used_sids.extend([c_rec["sample_id"], e_rec["sample_id"]])

    payload = {
        "condition": "C5",
        "exemplar_sample_ids": used_sids,
        "system": SYSTEM_PROMPT,
        "user": user_content,
        "messages": messages,
    }
    return {**payload, "prompt_sha256": digest(payload)}


def build_c6_prompt(
    claim: str,
    evidence: List[Dict[str, Any]],
    memory_bank: Optional[ErrorMemoryBank] = None,
) -> Dict[str, Any]:
    """C6: Source-Derived Error Rules (4-point checklist derived from empirical failure modes)."""
    rules_text = (
        memory_bank.get_source_derived_rules()
        if memory_bank
        else (
            "Verification Checklist (Derived from Historical Verification Errors):\n"
            "1. Media Debate vs Factual Contradiction: If retrieved articles quote political opponents "
            "arguing about a claim, verify whether the underlying factual statement itself is substantiated. "
            "Do NOT classify political disagreement as Conflicting Evidence if official records confirm the fact.\n"
            "2. Evidence Sufficiency Threshold: If evidence directly confirms or contradicts the core proposition, "
            "do not default to Not Enough Evidence simply because minor peripheral details are unmentioned.\n"
            "3. Cherrypicking Precision: Choose 'Conflicting Evidence/Cherrypicking' only when the claim quotes true "
            "facts out of context to convey a demonstrably false implication. Outright false claims are Refuted.\n"
            "4. Polarity Check: Carefully evaluate conditional denials and double negatives to ensure the direction "
            "of entailment matches the claim."
        )
    )
    evidence_block = format_evidence_block(evidence)

    user_content = (
        f"{rules_text}\n\n"
        f"Claim to verify:\n\"{claim}\"\n\n"
        f"Retrieved Evidence Passages:\n{evidence_block}\n\n"
        f"Based strictly on the evidence above and applying the verification checklist, evaluate the truthfulness of the claim.\n"
        f"Choose exactly one of the following four official verdicts:\n"
        f"{VERDICT_DEFINITIONS}\n\n"
        f"Instructions:\n"
        f"- Explicitly apply the checklist to the claim and evidence.\n"
        f"- Conclude your analysis on the final line with the exact format:\n"
        f"Verdict: <Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking>"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    payload = {
        "condition": "C6",
        "system": SYSTEM_PROMPT,
        "user": user_content,
        "messages": messages,
    }
    return {**payload, "prompt_sha256": digest(payload)}


def build_prompt_for_condition(
    condition: str,
    claim: str,
    evidence: List[Dict[str, Any]],
    memory_bank: Optional[ErrorMemoryBank] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Universal dispatcher for M1.3 prompt construction across C0–C6."""
    cond = condition.upper()
    if cond == "C0":
        return build_c0_prompt(claim, evidence)
    elif cond == "C1":
        return build_c1_prompt(claim, evidence)
    elif cond == "C2":
        if memory_bank is None:
            raise ValueError("ErrorMemoryBank required for condition C2")
        return build_c2_prompt(claim, evidence, memory_bank, k=4, seed=seed)
    elif cond == "C3":
        if memory_bank is None:
            raise ValueError("ErrorMemoryBank required for condition C3")
        return build_c3_prompt(claim, evidence, memory_bank, k=4, seed=seed)
    elif cond == "C4":
        if memory_bank is None:
            raise ValueError("ErrorMemoryBank required for condition C4")
        return build_c4_prompt(claim, evidence, memory_bank, k=4, seed=seed)
    elif cond == "C5":
        if memory_bank is None:
            raise ValueError("ErrorMemoryBank required for condition C5")
        return build_c5_prompt(claim, evidence, memory_bank, num_pairs=2, seed=seed)
    elif cond == "C6":
        return build_c6_prompt(claim, evidence, memory_bank)
    else:
        raise ValueError(f"Unknown condition ID: {condition}. Must be one of C0, C1, C2, C3, C4, C5, C6.")
