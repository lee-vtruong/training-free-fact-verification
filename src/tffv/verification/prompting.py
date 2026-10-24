"""Zero-shot neutral prompting templates for training-free fact verification."""
from typing import Any, Dict, List, Optional
from tffv.provenance import canonical, digest

SYSTEM_PROMPT = (
    "You are an impartial, highly rigorous fact-checking assistant. "
    "Your task is to verify a factual claim against provided retrieved evidence passages. "
    "You must rely strictly and solely on the provided evidence passages. Do not extrapolate, "
    "speculate, or introduce outside claims not substantiated by the evidence."
)

VERDICT_DEFINITIONS = (
    "1. Supported: The claim is directly confirmed and fully substantiated by the evidence.\n"
    "2. Refuted: The claim is directly contradicted, disproven, or refuted by the evidence.\n"
    "3. Not Enough Evidence: The provided evidence is insufficient or lacks the facts needed to prove or refute the claim.\n"
    "4. Conflicting Evidence/Cherrypicking: Reliable evidence sources directly contradict each other on this claim, "
    "or the claim selects true facts out of context to present a misleading picture."
)


def format_evidence_block(evidence: List[Dict[str, Any]], max_chars_per_passage: int = 1200) -> str:
    """Format Top-8 evidence passages into structured numbered blocks."""
    if not evidence:
        return "[No evidence passages provided.]"

    blocks = []
    for idx, ev in enumerate(evidence, 1):
        text = ev.get("text", "").strip()
        if len(text) > max_chars_per_passage:
            text = text[:max_chars_per_passage] + "..."
        source_url = ev.get("source_url", "unknown")
        blocks.append(f"Passage [{idx}] (Source: {source_url}):\n{text}")

    return "\n\n".join(blocks)


def build_verification_prompt(claim: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Construct a neutral, zero-shot verification prompt for a claim and evidence.
    
    Returns structured dict with system prompt, user prompt, messages, and prompt digest.
    """
    evidence_block = format_evidence_block(evidence)
    
    user_content = (
        f"Claim to verify:\n\"{claim}\"\n\n"
        f"Retrieved Evidence Passages:\n{evidence_block}\n\n"
        f"Based strictly on the evidence above, evaluate the truthfulness of the claim.\n"
        f"Choose exactly one of the following four official verdicts:\n"
        f"{VERDICT_DEFINITIONS}\n\n"
        f"Instructions:\n"
        f"- Analyze the relationship between the claim and the evidence briefly.\n"
        f"- Conclude your analysis on the final line with the exact format:\n"
        f"Verdict: <Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking>"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    prompt_payload = {
        "system": SYSTEM_PROMPT,
        "user": user_content,
        "messages": messages,
    }
    prompt_hash = digest(prompt_payload)

    return {
        **prompt_payload,
        "prompt_sha256": prompt_hash,
    }
