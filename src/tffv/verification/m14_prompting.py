"""Prompt builders for M1.4 Controlled Error-Informed Fact Verification Pilot (D0–D6).

Conditions:
- D0: Frozen Base Protocol v1.0 (Zero-shot B1)
- D1: Concise generic verification rules
- D2: Concise source-derived error rules
- D3: Random error demonstrations (k=2)
- D4: Similarity-retrieved error demonstrations (k=2)
- D5: Similarity-retrieved error demonstrations with verified corrections (k=2)
- D6: Correct/wrong contrastive demonstrations (k=2)
"""
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from tffv.provenance import canonical, digest
from tffv.verification.error_memory import ErrorMemoryBank
from tffv.verification.ontology import VERDICTS
from tffv.verification.prompting import (
    SYSTEM_PROMPT,
    VERDICT_DEFINITIONS,
    build_verification_prompt,
    format_evidence_block,
)

ALL_CONDITIONS_M14 = ["D0", "D1", "D2", "D3", "D4", "D5", "D6"]
ALL_CONDITIONS_M142 = [
    "D0",
    "D1",
    "D1_matched",
    "D2",
    "D2_minus_r1",
    "D2_minus_r2",
    "D2_para1",
    "D2_para2",
    "D3",
    "D4",
    "D5",
    "D6",
]

CONCISE_GENERIC_RULES = (
    "Core Verification Directives:\n"
    "- If the provided evidence directly substantiates the core claim, select Supported.\n"
    "- If the provided evidence directly contradicts the core claim, select Refuted.\n"
    "- If the evidence is insufficient to verify or refute the claim, select Not Enough Evidence.\n"
    "- If reputable sources present contradictory factual accounts, select Conflicting Evidence/Cherrypicking.\n"
    "State a concise 1-sentence factual reason, then output Verdict: <Label>."
)

CONCISE_MATCHED_GENERIC_RULES = (
    "Standard Verification Directives (applicable to general empirical claims):\n"
    "1. Primary assertion evaluation: Assess whether the central factual proposition is directly confirmed or denied by the provided sources.\n"
    "2. Substantive content focus: Base the final determination on the primary factual statement rather than on peripheral commentary or stylistic phrasing.\n"
    "State a concise 1-sentence factual reason, then output Verdict: <Label>."
)

CONCISE_SOURCE_ERROR_RULES = (
    "Empirical Error Directives (derived from observed verification pitfalls):\n"
    "1. Media debate is not factual contradiction: An article describing a political disagreement does not constitute Refutation of a verifiable fact.\n"
    "2. Missing context is not Cherrypicking: Absence of secondary details does not warrant Cherrypicking if the primary assertion is substantiated.\n"
    "State a concise 1-sentence factual reason, then output Verdict: <Label>."
)

CONCISE_ERROR_RULES_MINUS_R1 = (
    "Empirical Error Directives (derived from observed verification pitfalls):\n"
    "1. Missing context is not Cherrypicking: Absence of secondary details does not warrant Cherrypicking if the primary assertion is substantiated.\n"
    "State a concise 1-sentence factual reason, then output Verdict: <Label>."
)

CONCISE_ERROR_RULES_MINUS_R2 = (
    "Empirical Error Directives (derived from observed verification pitfalls):\n"
    "1. Media debate is not factual contradiction: An article describing a political disagreement does not constitute Refutation of a verifiable fact.\n"
    "State a concise 1-sentence factual reason, then output Verdict: <Label>."
)

CONCISE_SOURCE_ERROR_RULES_PARA1 = (
    "Empirical Verification Guidelines (derived from past error analysis):\n"
    "1. Distinguish reporting from refutation: When news outlets quote opposing political arguments, verify the underlying factual claim rather than treating political dispute as refutation.\n"
    "2. Separate context omission from selective framing: Do not mark a claim as cherrypicked solely because peripheral context is absent, provided the main factual claim holds.\n"
    "State a concise 1-sentence factual reason, then output Verdict: <Label>."
)

CONCISE_SOURCE_ERROR_RULES_PARA2 = (
    "Empirical Error Guidelines (based on observed verification pitfalls):\n"
    "1. Disagreement does not equal contradiction: Quotes showing political figures arguing about a statement do not disprove verified factual records.\n"
    "2. Unmentioned details do not equal cherrypicking: Missing non-essential facts do not justify a cherrypicking label when the core assertion is accurate.\n"
    "State a concise 1-sentence factual reason, then output Verdict: <Label>."
)


def compute_claim_similarity(claim_a: str, claim_b: str) -> float:
    """Compute character 3-gram Jaccard similarity between two claims."""
    def get_3grams(text: str) -> Set[str]:
        cleaned = "".join(c.lower() for c in text if c.isalnum() or c.isspace())
        words = cleaned.split()
        text_norm = " ".join(words)
        if len(text_norm) < 3:
            return {text_norm} if text_norm else set()
        return {text_norm[i:i+3] for i in range(len(text_norm) - 2)}

    set_a = get_3grams(claim_a)
    set_b = get_3grams(claim_b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def retrieve_top_k_similar_exemplars(
    target_claim: str,
    candidate_records: List[Dict[str, Any]],
    k: int = 2,
) -> List[Dict[str, Any]]:
    """Retrieve top-k candidate exemplars with highest lexical similarity to target_claim."""
    scored = []
    for rec in candidate_records:
        sim = compute_claim_similarity(target_claim, rec.get("claim", ""))
        scored.append((sim, rec.get("sample_id", ""), rec))
    # Sort descending by similarity, tie-break deterministically by sample_id
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [rec for _, _, rec in scored[:k]]


def format_compact_exemplar_evidence(compact_evidence: List[Dict[str, Any]], max_chars: int = 250) -> str:
    """Format a single compact evidence passage (Top-1) to avoid context expansion."""
    if not compact_evidence:
        return "[No evidence passage provided.]"
    top_ev = compact_evidence[0]
    text = top_ev.get("text", "").strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    source_url = top_ev.get("source_url", "source")
    return f"  [Passage 1] ({source_url}): {text}"


def format_compact_demonstration(record: Dict[str, Any], example_num: int) -> str:
    """Format compact demonstration for D3 and D4."""
    ev_str = format_compact_exemplar_evidence(record.get("compact_evidence", []))
    label = record.get("gold_label", "")
    return (
        f"--- Example {example_num} ---\n"
        f"Claim: \"{record.get('claim', '')}\"\n"
        f"Key Evidence:\n{ev_str}\n"
        f"Verdict: {label}\n"
    )


def format_compact_corrected_error_demonstration(record: Dict[str, Any], example_num: int) -> str:
    """Format compact corrected error demonstration for D5."""
    ev_str = format_compact_exemplar_evidence(record.get("compact_evidence", []))
    wrong_pred = record.get("base_predicted_verdict", "Unknown")
    gold_label = record.get("gold_label", "")
    return (
        f"--- Example {example_num} (Caution Reference) ---\n"
        f"Claim: \"{record.get('claim', '')}\"\n"
        f"Key Evidence:\n{ev_str}\n"
        f"Base Mistake: Misclassified as '{wrong_pred}'.\n"
        f"Verified Ground Truth: {gold_label}\n"
    )


def build_d0_prompt(claim: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """D0: Frozen Base Protocol v1.0 (Zero-shot B1)."""
    p = build_verification_prompt(claim, evidence)
    p["condition"] = "D0"
    p["exemplar_sample_ids"] = []
    return p


def build_d1_prompt(claim: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """D1: Concise generic verification rules."""
    user_content = (
        f"Claim: \"{claim}\"\n\n"
        f"Retrieved Evidence:\n{format_evidence_block(evidence)}\n\n"
        f"{CONCISE_GENERIC_RULES}\n\n"
        f"{VERDICT_DEFINITIONS}\n\n"
        "State your 1-sentence reasoning and output your verdict in the format:\n"
        "Verdict: <Supported|Refuted|Not Enough Evidence|Conflicting Evidence/Cherrypicking>"
    )
    prompt_obj = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "user": user_content,
        "condition": "D1",
        "exemplar_sample_ids": [],
    }
    prompt_obj["prompt_sha256"] = digest(canonical(prompt_obj["messages"]))
    return prompt_obj


def build_d2_prompt(claim: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """D2: Concise source-derived error rules."""
    user_content = (
        f"Claim: \"{claim}\"\n\n"
        f"Retrieved Evidence:\n{format_evidence_block(evidence)}\n\n"
        f"{CONCISE_SOURCE_ERROR_RULES}\n\n"
        f"{VERDICT_DEFINITIONS}\n\n"
        "State your 1-sentence reasoning and output your verdict in the format:\n"
        "Verdict: <Supported|Refuted|Not Enough Evidence|Conflicting Evidence/Cherrypicking>"
    )
    prompt_obj = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "user": user_content,
        "condition": "D2",
        "exemplar_sample_ids": [],
    }
    prompt_obj["prompt_sha256"] = digest(canonical(prompt_obj["messages"]))
    return prompt_obj


def build_d1_matched_prompt(claim: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """D1_matched: Matched concise generic rules (no error knowledge)."""
    user_content = (
        f"Claim: \"{claim}\"\n\n"
        f"Retrieved Evidence:\n{format_evidence_block(evidence)}\n\n"
        f"{CONCISE_MATCHED_GENERIC_RULES}\n\n"
        f"{VERDICT_DEFINITIONS}\n\n"
        "State your 1-sentence reasoning and output your verdict in the format:\n"
        "Verdict: <Supported|Refuted|Not Enough Evidence|Conflicting Evidence/Cherrypicking>"
    )
    prompt_obj = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "user": user_content,
        "condition": "D1_matched",
        "exemplar_sample_ids": [],
    }
    prompt_obj["prompt_sha256"] = digest(canonical(prompt_obj["messages"]))
    return prompt_obj


def build_d2_minus_r1_prompt(claim: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """D2_minus_r1: Leave-one-out ablation removing Rule 1 (Media debate). Keeps Rule 2 only."""
    user_content = (
        f"Claim: \"{claim}\"\n\n"
        f"Retrieved Evidence:\n{format_evidence_block(evidence)}\n\n"
        f"{CONCISE_ERROR_RULES_MINUS_R1}\n\n"
        f"{VERDICT_DEFINITIONS}\n\n"
        "State your 1-sentence reasoning and output your verdict in the format:\n"
        "Verdict: <Supported|Refuted|Not Enough Evidence|Conflicting Evidence/Cherrypicking>"
    )
    prompt_obj = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "user": user_content,
        "condition": "D2_minus_r1",
        "exemplar_sample_ids": [],
    }
    prompt_obj["prompt_sha256"] = digest(canonical(prompt_obj["messages"]))
    return prompt_obj


def build_d2_minus_r2_prompt(claim: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """D2_minus_r2: Leave-one-out ablation removing Rule 2 (Missing context). Keeps Rule 1 only."""
    user_content = (
        f"Claim: \"{claim}\"\n\n"
        f"Retrieved Evidence:\n{format_evidence_block(evidence)}\n\n"
        f"{CONCISE_ERROR_RULES_MINUS_R2}\n\n"
        f"{VERDICT_DEFINITIONS}\n\n"
        "State your 1-sentence reasoning and output your verdict in the format:\n"
        "Verdict: <Supported|Refuted|Not Enough Evidence|Conflicting Evidence/Cherrypicking>"
    )
    prompt_obj = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "user": user_content,
        "condition": "D2_minus_r2",
        "exemplar_sample_ids": [],
    }
    prompt_obj["prompt_sha256"] = digest(canonical(prompt_obj["messages"]))
    return prompt_obj


def build_d2_para1_prompt(claim: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """D2_para1: Paraphrase robustness variant 1 (guideline style)."""
    user_content = (
        f"Claim: \"{claim}\"\n\n"
        f"Retrieved Evidence:\n{format_evidence_block(evidence)}\n\n"
        f"{CONCISE_SOURCE_ERROR_RULES_PARA1}\n\n"
        f"{VERDICT_DEFINITIONS}\n\n"
        "State your 1-sentence reasoning and output your verdict in the format:\n"
        "Verdict: <Supported|Refuted|Not Enough Evidence|Conflicting Evidence/Cherrypicking>"
    )
    prompt_obj = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "user": user_content,
        "condition": "D2_para1",
        "exemplar_sample_ids": [],
    }
    prompt_obj["prompt_sha256"] = digest(canonical(prompt_obj["messages"]))
    return prompt_obj


def build_d2_para2_prompt(claim: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """D2_para2: Paraphrase robustness variant 2 (formal criteria style)."""
    user_content = (
        f"Claim: \"{claim}\"\n\n"
        f"Retrieved Evidence:\n{format_evidence_block(evidence)}\n\n"
        f"{CONCISE_SOURCE_ERROR_RULES_PARA2}\n\n"
        f"{VERDICT_DEFINITIONS}\n\n"
        "State your 1-sentence reasoning and output your verdict in the format:\n"
        "Verdict: <Supported|Refuted|Not Enough Evidence|Conflicting Evidence/Cherrypicking>"
    )
    prompt_obj = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "user": user_content,
        "condition": "D2_para2",
        "exemplar_sample_ids": [],
    }
    prompt_obj["prompt_sha256"] = digest(canonical(prompt_obj["messages"]))
    return prompt_obj


def build_d3_prompt(
    claim: str,
    evidence: List[Dict[str, Any]],
    memory_bank: ErrorMemoryBank,
    seed: int = 42,
) -> Dict[str, Any]:
    """D3: Random error demonstrations (k=2)."""
    # Sample k=2 from wrong_sids in memory bank
    error_sids = sorted(list(memory_bank.wrong_sids))
    rng = random.Random(seed)
    chosen_sids = rng.sample(error_sids, k=min(2, len(error_sids)))
    chosen_records = [memory_bank.records[sid] for sid in chosen_sids]

    demo_blocks = [
        format_compact_demonstration(rec, idx)
        for idx, rec in enumerate(chosen_records, 1)
    ]
    demos_text = "Verified Demonstration Examples:\n\n" + "\n".join(demo_blocks)

    user_content = (
        f"{demos_text}\n\n"
        f"--- Target Claim to Verify ---\n"
        f"Claim: \"{claim}\"\n\n"
        f"Retrieved Evidence:\n{format_evidence_block(evidence)}\n\n"
        f"{VERDICT_DEFINITIONS}\n\n"
        "State your 1-sentence reasoning and output your verdict in the format:\n"
        "Verdict: <Supported|Refuted|Not Enough Evidence|Conflicting Evidence/Cherrypicking>"
    )
    prompt_obj = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "user": user_content,
        "condition": "D3",
        "exemplar_sample_ids": chosen_sids,
    }
    prompt_obj["prompt_sha256"] = digest(canonical(prompt_obj["messages"]))
    return prompt_obj


def build_d4_prompt(
    claim: str,
    evidence: List[Dict[str, Any]],
    memory_bank: ErrorMemoryBank,
) -> Dict[str, Any]:
    """D4: Similarity-retrieved error demonstrations (k=2)."""
    error_records = [memory_bank.records[sid] for sid in memory_bank.wrong_sids]
    retrieved = retrieve_top_k_similar_exemplars(claim, error_records, k=2)
    chosen_sids = [rec["sample_id"] for rec in retrieved]

    demo_blocks = [
        format_compact_demonstration(rec, idx)
        for idx, rec in enumerate(retrieved, 1)
    ]
    demos_text = "Topically Similar Reference Cases:\n\n" + "\n".join(demo_blocks)

    user_content = (
        f"{demos_text}\n\n"
        f"--- Target Claim to Verify ---\n"
        f"Claim: \"{claim}\"\n\n"
        f"Retrieved Evidence:\n{format_evidence_block(evidence)}\n\n"
        f"{VERDICT_DEFINITIONS}\n\n"
        "State your 1-sentence reasoning and output your verdict in the format:\n"
        "Verdict: <Supported|Refuted|Not Enough Evidence|Conflicting Evidence/Cherrypicking>"
    )
    prompt_obj = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "user": user_content,
        "condition": "D4",
        "exemplar_sample_ids": chosen_sids,
    }
    prompt_obj["prompt_sha256"] = digest(canonical(prompt_obj["messages"]))
    return prompt_obj


def build_d5_prompt(
    claim: str,
    evidence: List[Dict[str, Any]],
    memory_bank: ErrorMemoryBank,
) -> Dict[str, Any]:
    """D5: Similarity-retrieved error demonstrations with verified corrections (k=2)."""
    error_records = [memory_bank.records[sid] for sid in memory_bank.wrong_sids]
    retrieved = retrieve_top_k_similar_exemplars(claim, error_records, k=2)
    chosen_sids = [rec["sample_id"] for rec in retrieved]

    demo_blocks = [
        format_compact_corrected_error_demonstration(rec, idx)
        for idx, rec in enumerate(retrieved, 1)
    ]
    demos_text = "Similar Error Caution & Correction Cases:\n\n" + "\n".join(demo_blocks)

    user_content = (
        f"{demos_text}\n\n"
        f"--- Target Claim to Verify ---\n"
        f"Claim: \"{claim}\"\n\n"
        f"Retrieved Evidence:\n{format_evidence_block(evidence)}\n\n"
        f"{VERDICT_DEFINITIONS}\n\n"
        "State your 1-sentence reasoning and output your verdict in the format:\n"
        "Verdict: <Supported|Refuted|Not Enough Evidence|Conflicting Evidence/Cherrypicking>"
    )
    prompt_obj = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "user": user_content,
        "condition": "D5",
        "exemplar_sample_ids": chosen_sids,
    }
    prompt_obj["prompt_sha256"] = digest(canonical(prompt_obj["messages"]))
    return prompt_obj


def build_d6_prompt(
    claim: str,
    evidence: List[Dict[str, Any]],
    memory_bank: ErrorMemoryBank,
) -> Dict[str, Any]:
    """D6: Correct/wrong contrastive demonstrations (k=2: 1 correct + 1 error)."""
    correct_records = [memory_bank.records[sid] for sid in memory_bank.correct_sids]
    error_records = [memory_bank.records[sid] for sid in memory_bank.wrong_sids]

    retrieved_correct = retrieve_top_k_similar_exemplars(claim, correct_records, k=1)
    retrieved_error = retrieve_top_k_similar_exemplars(claim, error_records, k=1)

    chosen_sids = []
    demo_blocks = []
    if retrieved_correct:
        chosen_sids.append(retrieved_correct[0]["sample_id"])
        demo_blocks.append(
            f"[Case 1 - Verified Correct Reference]\n"
            f"Claim: \"{retrieved_correct[0].get('claim', '')}\"\n"
            f"Key Evidence:\n{format_compact_exemplar_evidence(retrieved_correct[0].get('compact_evidence', []))}\n"
            f"Verified Verdict: {retrieved_correct[0].get('gold_label', '')}"
        )
    if retrieved_error:
        chosen_sids.append(retrieved_error[0]["sample_id"])
        demo_blocks.append(
            f"[Case 2 - Error Caution & Correction Reference]\n"
            f"Claim: \"{retrieved_error[0].get('claim', '')}\"\n"
            f"Key Evidence:\n{format_compact_exemplar_evidence(retrieved_error[0].get('compact_evidence', []))}\n"
            f"Base Mistake: Misclassified as '{retrieved_error[0].get('base_predicted_verdict')}'.\n"
            f"Verified Ground Truth: {retrieved_error[0].get('gold_label', '')}"
        )

    demos_text = "Contrastive Reference Demonstrations (1 Correct, 1 Error Caution):\n\n" + "\n\n".join(demo_blocks)

    user_content = (
        f"{demos_text}\n\n"
        f"--- Target Claim to Verify ---\n"
        f"Claim: \"{claim}\"\n\n"
        f"Retrieved Evidence:\n{format_evidence_block(evidence)}\n\n"
        f"{VERDICT_DEFINITIONS}\n\n"
        "State your 1-sentence reasoning and output your verdict in the format:\n"
        "Verdict: <Supported|Refuted|Not Enough Evidence|Conflicting Evidence/Cherrypicking>"
    )
    prompt_obj = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "user": user_content,
        "condition": "D6",
        "exemplar_sample_ids": chosen_sids,
    }
    prompt_obj["prompt_sha256"] = digest(canonical(prompt_obj["messages"]))
    return prompt_obj


def build_prompt_for_condition_m14(
    condition: str,
    claim: str,
    evidence: List[Dict[str, Any]],
    memory_bank: Optional[ErrorMemoryBank] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Unified dispatcher for M1.4 / M1.4.2 prompt building."""
    if condition == "D0":
        return build_d0_prompt(claim, evidence)
    elif condition == "D1":
        return build_d1_prompt(claim, evidence)
    elif condition == "D1_matched":
        return build_d1_matched_prompt(claim, evidence)
    elif condition == "D2":
        return build_d2_prompt(claim, evidence)
    elif condition == "D2_minus_r1":
        return build_d2_minus_r1_prompt(claim, evidence)
    elif condition == "D2_minus_r2":
        return build_d2_minus_r2_prompt(claim, evidence)
    elif condition == "D2_para1":
        return build_d2_para1_prompt(claim, evidence)
    elif condition == "D2_para2":
        return build_d2_para2_prompt(claim, evidence)
    elif condition == "D3":
        if memory_bank is None:
            raise ValueError("memory_bank required for D3")
        return build_d3_prompt(claim, evidence, memory_bank, seed=seed)
    elif condition == "D4":
        if memory_bank is None:
            raise ValueError("memory_bank required for D4")
        return build_d4_prompt(claim, evidence, memory_bank)
    elif condition == "D5":
        if memory_bank is None:
            raise ValueError("memory_bank required for D5")
        return build_d5_prompt(claim, evidence, memory_bank)
    elif condition == "D6":
        if memory_bank is None:
            raise ValueError("memory_bank required for D6")
        return build_d6_prompt(claim, evidence, memory_bank)
    else:
        raise ValueError(
            f"Unknown condition: {condition}. Expected one of {ALL_CONDITIONS_M142}"
        )
