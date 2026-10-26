"""Source-Grounded Failure Mechanism Taxonomy and Observable Signal Extractors for M1.5.

Strictly TRAINING-FREE:
- Zero parameter updates, zero learned routers, zero neural classifiers.
- Observable signal extractors operate ONLY on claim text and retrieved evidence passages.
- Historical error provenance is strictly bounded to permitted source split (Fold A: train:0..49).
- Fallback to matched generic control (M0) when no observable signal fires.
"""
from dataclasses import dataclass, field
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple


# ==============================================================================
# 1. Canonical Taxonomy Mechanisms
# ==============================================================================

MECHANISM_ATTRIBUTION = "Attribution vs proposition confusion"
MECHANISM_NEGATION = "Negation / polarity inversion"
MECHANISM_SCOPE_QUANTIFIER = "Scope, quantifier and denominator mismatch"
MECHANISM_SUFFICIENCY = "Evidence sufficiency confusion"
MECHANISM_PARTIAL_TRUTH = "Partial truth vs cherrypicking"
MECHANISM_CONFLICT = "Apparent vs genuine evidence conflict"

ALL_MECHANISMS = [
    MECHANISM_SCOPE_QUANTIFIER,
    MECHANISM_PARTIAL_TRUTH,
    MECHANISM_SUFFICIENCY,
    MECHANISM_CONFLICT,
    MECHANISM_ATTRIBUTION,
    MECHANISM_NEGATION,
]


@dataclass(frozen=True)
class MechanismProfile:
    """Operational definition and historical provenance for a failure mechanism."""
    name: str
    definition: str
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    gold_error_characterization: str
    observable_signals: List[str]
    ambiguous_cases: str
    fold_a_support_count: int
    fold_a_example_ids: List[str]
    is_adequately_supported: bool  # True if fold_a_support_count >= 8 in N=50


TAXONOMY_PROFILES: Dict[str, MechanismProfile] = {
    MECHANISM_SCOPE_QUANTIFIER: MechanismProfile(
        name=MECHANISM_SCOPE_QUANTIFIER,
        definition=(
            "Failure occurring when claim veracity hinges on exact numerical thresholds, "
            "proportions, per-capita versus aggregate bases, or geographic/administrative "
            "jurisdiction bounds, and the model confuses differing denominators or subsets."
        ),
        inclusion_criteria=[
            "Claim contains explicit percentages, ratios, or quantifier phrases (e.g., '20%', 'per person', 'all').",
            "Evidence provides statistics for a related but non-identical denominator or scope.",
            "Model confirms or refutes without reconciling base populations.",
        ],
        exclusion_criteria=[
            "Claim is purely qualitative with no numerical or scope constraint.",
            "Disagreement is over reporting credibility rather than statistical bounds.",
        ],
        gold_error_characterization=(
            "On Fold A, Qwen3-8B repeatedly misjudged constitutional revenue allocations "
            "(train:2, train:6, train:11) and South African alcohol metrics (train:47, train:48, train:49), "
            "confusing 'drinkers' with 'total population' and 1999 constitutions with subsequent amendments."
        ),
        observable_signals=[
            "Regex matching percentages: r'(\\d+[\\.,]?\\d*[%％]|\\bpercent\\b)'",
            "Quantifier bounds: r'\\b(per capita|per person|average|proportion|all|none|more than|less than)\\b'",
            "Fiscal/monetary indicators: r'\\b(revenue|allocation|budget|gdp|share)\\b'",
        ],
        ambiguous_cases=(
            "Claims where a percentage is stated rhetorically (e.g. '100% committed') "
            "rather than as an empirical measurement."
        ),
        fold_a_support_count=8,
        fold_a_example_ids=[
            "train:2", "train:6", "train:11", "train:21",
            "train:27", "train:47", "train:48", "train:49",
        ],
        is_adequately_supported=True,
    ),
    MECHANISM_PARTIAL_TRUTH: MechanismProfile(
        name=MECHANISM_PARTIAL_TRUTH,
        definition=(
            "Failure occurring when a multi-clause or conjunctive claim has some true elements "
            "and some unverified or false elements, and the model bails into full Support or "
            "full Refutation without correctly identifying Cherrypicking / Conflicting Evidence."
        ),
        inclusion_criteria=[
            "Claim makes multiple distinct assertions joined by conjunctions ('and', 'as well as').",
            "Evidence substantiates a subset of clauses but contradicts or leaves unverified others.",
            "Gold verdict requires 'Conflicting Evidence/Cherrypicking' or precise clause accounting.",
        ],
        exclusion_criteria=[
            "Single-assertion claims.",
            "Claims where all sub-clauses share identical truth values.",
        ],
        gold_error_characterization=(
            "On Fold A, the model failed on multi-conflict foreign policy claims (train:8: Biden supported 2 of 3 wars) "
            "and multi-country voting age claims (train:18, train:19: true for Scotland, false for Wales), "
            "collapsing compound claims into binary verdicts."
        ),
        observable_signals=[
            "Conjunction structures with distinct entities: r'\\b(and|as well as|both .* and)\\b'",
            "Multiple distinct verbs or temporal sequences: r'\\b(inherited .* and then rebuilt|voted .* and supported)\\b'",
        ],
        ambiguous_cases=(
            "Compound subject phrases (e.g. 'Alice and Bob traveled') where the event is singular, "
            "versus compound propositions (e.g. 'Alice traveled and Bob won')."
        ),
        fold_a_support_count=5,
        fold_a_example_ids=["train:8", "train:12", "train:18", "train:19", "train:24"],
        is_adequately_supported=False,
    ),
    MECHANISM_SUFFICIENCY: MechanismProfile(
        name=MECHANISM_SUFFICIENCY,
        definition=(
            "Failure occurring when evidence is topically relevant but logically insufficient "
            "to establish or refute the specific claim proposition, and the model hallucinates "
            "support or refutation instead of predicting Not Enough Evidence."
        ),
        inclusion_criteria=[
            "Evidence discusses the claim topic, entities, and context extensively.",
            "Decisive verifiable fact or statistical figure is absent from all retrieved passages.",
            "Model predicts Supported or Refuted instead of Not Enough Evidence.",
        ],
        exclusion_criteria=[
            "Passages contain the complete logical rebuttal or proof.",
            "Evidence is completely off-topic (retrieval failure).",
        ],
        gold_error_characterization=(
            "On Fold A, the model predicted Supported/Refuted on claims about court statistics "
            "(train:29, train:31, train:38, train:39) and tax commitments (train:26), because the "
            "passages mentioned the judicial system or tax discussions without verifying the exact figures."
        ),
        observable_signals=[
            "Absence of key numerical entities in top-k evidence passages.",
            "High query-passage lexical overlap with low entity-level exact match.",
        ],
        ambiguous_cases=(
            "Differentiating between true logical insufficiency and implicit textual entailment."
        ),
        fold_a_support_count=5,
        fold_a_example_ids=["train:26", "train:29", "train:31", "train:38", "train:39"],
        is_adequately_supported=False,
    ),
    MECHANISM_CONFLICT: MechanismProfile(
        name=MECHANISM_CONFLICT,
        definition=(
            "Failure occurring when multiple retrieved sources present opposing statistics, "
            "diverging baseline years, or editorial disagreement, and the model resolves "
            "the disagreement unilaterally rather than identifying genuine conflict."
        ),
        inclusion_criteria=[
            "Top retrieved evidence passages cite contradictory numbers or conclusions.",
            "Disagreement reflects genuine divergence across reporting sources or baseline periods.",
            "Model takes a side or bails into Not Enough Evidence.",
        ],
        exclusion_criteria=[
            "Single source reporting on a political dispute (attribution debate).",
            "Outdated figures superseded by an official revision within the same document.",
        ],
        gold_error_characterization=(
            "On Fold A, Ghana economic growth and unemployment claims (train:35, train:37) and "
            "police lethality ratios (train:17) produced conflicting predictions due to differing survey baselines."
        ),
        observable_signals=[
            "Contradictory directional terms across top passages: r'\\b(increased|rose|grew)\\b' vs r'\\b(decreased|fell|dropped)\\b'.",
            "Differing reference years or multiple opposing statistical estimates.",
        ],
        ambiguous_cases=(
            "Distinguishing partisan media rhetoric from actual factual/methodological conflict."
        ),
        fold_a_support_count=3,
        fold_a_example_ids=["train:17", "train:35", "train:37"],
        is_adequately_supported=False,
    ),
    MECHANISM_ATTRIBUTION: MechanismProfile(
        name=MECHANISM_ATTRIBUTION,
        definition=(
            "Failure occurring when the claim asserts that an agent uttered, signed, or pledged "
            "a statement, and the model evaluates the factual truth of the proposition rather than "
            "the factual truth of the utterance/attribution act."
        ),
        inclusion_criteria=[
            "Claim contains explicit speech, commitment, or legislative attribution verbs ('said', 'pledged', 'signed', 'claimed').",
            "Evidence verifies that the statement was made, but the statement itself may be false or unverified.",
            "Model confuses verifying the speech act with verifying the speech content.",
        ],
        exclusion_criteria=[
            "Claims asserting empirical world facts directly without attribution syntax.",
        ],
        gold_error_characterization=(
            "On Fold A, only 2 claims exhibited this mode: train:3 (Biden border wall pledge) and "
            "train:14 (Buhari police bill signature). In both, the model struggled to decouple the "
            "official act from the underlying societal policy."
        ),
        observable_signals=[
            "Attribution verbs: r'\\b(said|pledged|announced|claims?|claimed|stated|tweeted|vowed|signed)\\b'",
            "Quotation marks wrapping propositional clauses.",
        ],
        ambiguous_cases=(
            "Claims that use 'announced' to state a factual event (e.g. 'NASA announced water on Mars') "
            "where the verification target is the discovery, not NASA's press release."
        ),
        fold_a_support_count=2,
        fold_a_example_ids=["train:3", "train:14"],
        is_adequately_supported=False,
    ),
    MECHANISM_NEGATION: MechanismProfile(
        name=MECHANISM_NEGATION,
        definition=(
            "Failure occurring when a claim or evidence passage contains explicit negation, "
            "double negatives, or restrictive particles, and the model inverts the polarity "
            "or misses negative scoping."
        ),
        inclusion_criteria=[
            "Claim or key evidence contains explicit syntactic negation ('not', 'never', 'does not even', 'without').",
            "Model fails to track negative polarity, resulting in an exact Supported <-> Refuted inversion.",
        ],
        exclusion_criteria=[
            "Affirmative claims with negative semantic connotations (e.g. 'poverty increased').",
        ],
        gold_error_characterization=(
            "On Fold A, only 2 claims exhibited this pattern: train:20 (state vs private project) and "
            "train:25 ('does not even get 1%'). Sample count is severely inadequate ($n=2$) to establish "
            "reliable operational rules."
        ),
        observable_signals=[
            "Syntactic negation particles: r'\\b(not|never|no|none|neither|nor|without|fail(ed|s)?|refus(ed|es)?|cannot|can\\'t|won\\'t|doesn\\'t|didn\\'t)\\b'",
            "Double negation or negative restrictive syntax: r'\\b(does not even|no other than|never without)\\b'",
        ],
        ambiguous_cases=(
            "Negation used idiomatically or within a title rather than as a proposition modifier."
        ),
        fold_a_support_count=2,
        fold_a_example_ids=["train:20", "train:25"],
        is_adequately_supported=False,
    ),
}


# ==============================================================================
# 2. Observable Label-Free Signal Extractors
# ==============================================================================

_RE_ATTRIBUTION = re.compile(
    r"\b(said|pledged|announced|claims?|claimed|stated|tweeted|vowed|signed|reported)\b",
    re.IGNORECASE,
)
_RE_SCOPE_QUANTIFIER = re.compile(
    r"(\d+[\.,]?\d*[%％]|\b(per capita|per person|average|percent|percentage|all|none|more than|less than|every|billion|million)\b)",
    re.IGNORECASE,
)
_RE_NEGATION = re.compile(
    r"\b(not|never|no|none|neither|nor|without|fail(?:ed|s)?|refus(?:ed|es)?|cannot|can't|won't|doesn't|didn't)\b",
    re.IGNORECASE,
)
_RE_CONJUNCTION = re.compile(
    r"\b(and|as well as|both .* and|also)\b",
    re.IGNORECASE,
)


def extract_attribution_signal(claim: str, evidence: Optional[List[Dict[str, Any]]] = None) -> bool:
    """Label-free detector for attribution syntax in claim text."""
    if _RE_ATTRIBUTION.search(claim):
        return True
    if '"' in claim or '“' in claim or '”' in claim:
        return True
    return False


def extract_scope_quantifier_signal(claim: str, evidence: Optional[List[Dict[str, Any]]] = None) -> bool:
    """Label-free detector for explicit quantitative, numerical, or scope bounds."""
    return bool(_RE_SCOPE_QUANTIFIER.search(claim))


def extract_negation_signal(claim: str, evidence: Optional[List[Dict[str, Any]]] = None) -> bool:
    """Label-free detector for explicit syntactic negation in claim text."""
    return bool(_RE_NEGATION.search(claim))


def extract_partial_truth_signal(claim: str, evidence: Optional[List[Dict[str, Any]]] = None) -> bool:
    """Label-free detector for multi-clause conjunctive claims."""
    # Matches claims with explicit 'and' or 'as well as' connecting multiple clauses
    return bool(_RE_CONJUNCTION.search(claim)) and len(claim.split()) > 10


def extract_evidence_sufficiency_signal(
    claim: str,
    evidence: Optional[List[Dict[str, Any]]] = None,
    min_word_count: int = 50,
) -> bool:
    """Label-free detector for sparse or missing evidence passages."""
    if evidence is None:
        return False
    if len(evidence) == 0:
        return True
    total_words = sum(len(str(p.get("text", "")).split()) for p in evidence)
    return total_words < min_word_count


def detect_active_mechanisms(
    claim: str,
    evidence: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """Detect all label-free observable signals triggered on a given claim/evidence pair.
    
    Returns ordered list of active mechanisms.
    """
    active: List[str] = []
    if extract_scope_quantifier_signal(claim, evidence):
        active.append(MECHANISM_SCOPE_QUANTIFIER)
    if extract_partial_truth_signal(claim, evidence):
        active.append(MECHANISM_PARTIAL_TRUTH)
    if extract_evidence_sufficiency_signal(claim, evidence):
        active.append(MECHANISM_SUFFICIENCY)
    if extract_attribution_signal(claim, evidence):
        active.append(MECHANISM_ATTRIBUTION)
    if extract_negation_signal(claim, evidence):
        active.append(MECHANISM_NEGATION)
    return active


# ==============================================================================
# 3. Controlled Guidance Rules (M0, M1, M2, M3)
# ==============================================================================

# M0: Matched generic control (identical to D1_matched)
GUIDANCE_M0 = (
    "Standard Verification Directives (applicable to general empirical claims):\n"
    "1. Primary assertion evaluation: Assess whether the central factual proposition is directly confirmed or denied by the provided sources.\n"
    "2. Substantive content focus: Base the final determination on the primary factual statement rather than on peripheral commentary or stylistic phrasing.\n"
    "State a concise 1-sentence factual reason, then output Verdict: <Label>."
)

# M1: Generic claim-type clarification (no error rules, broad taxonomy guidance)
GUIDANCE_M1 = (
    "Clarification Directives for Diverse Claim Types:\n"
    "1. Categorization: Distinguish statistical claims (requiring exact denominator alignment), attribution claims (verifying whether a statement was uttered), and compound claims.\n"
    "2. Resolution: Base your verdict strictly on whether the provided sources substantiate the primary claim proposition.\n"
    "State a concise 1-sentence factual reason, then output Verdict: <Label>."
)

# M2 conditional mechanism-specific guidance directives (derived from Fold A):
# Notice: Only mechanisms with sufficient empirical support should provide specific rules.
MECHANISM_GUIDANCE_RULES: Dict[str, str] = {
    MECHANISM_SCOPE_QUANTIFIER: (
        "Statistical Scope Directives:\n"
        "1. Denominator consistency: Verify that figures in the claim match the exact base population in the sources (e.g. per-capita vs total, drinkers vs whole population).\n"
        "2. Exact thresholds: If numbers differ substantially from sources, select Refuted.\n"
        "State a concise 1-sentence factual reason, then output Verdict: <Label>."
    ),
    MECHANISM_PARTIAL_TRUTH: (
        "Compound Assertion Directives:\n"
        "1. Clause breakdown: For claims asserting multiple facts, verify each clause independently.\n"
        "2. Selective truth: If sources confirm only a portion of the assertions while omitting or contradicting others, select Conflicting Evidence/Cherrypicking.\n"
        "State a concise 1-sentence factual reason, then output Verdict: <Label>."
    ),
    MECHANISM_SUFFICIENCY: (
        "Evidence Sufficiency Directives:\n"
        "1. Direct confirmation: General topical discussion does not substantiate specific figures or quotes.\n"
        "2. Absence of proof: If retrieved sources do not explicitly answer the claim's central question, select Not Enough Evidence.\n"
        "State a concise 1-sentence factual reason, then output Verdict: <Label>."
    ),
    MECHANISM_ATTRIBUTION: (
        "Attribution Directives:\n"
        "1. Utterance vs proposition: If the claim states a person said or signed something, verify whether the act occurred, not whether the statement itself is true.\n"
        "2. Sourced quotes: If sources confirm the speaker made the statement, select Supported.\n"
        "State a concise 1-sentence factual reason, then output Verdict: <Label>."
    ),
    MECHANISM_NEGATION: (
        "Polarity Directives:\n"
        "1. Scope of negation: Carefully track negative terms (e.g., 'not', 'never', 'without') in both the claim and sources.\n"
        "2. Direct polarity: Ensure double negatives or restrictive modifiers do not invert your verdict.\n"
        "State a concise 1-sentence factual reason, then output Verdict: <Label>."
    ),
    MECHANISM_CONFLICT: (
        "Source Divergence Directives:\n"
        "1. Genuine contradiction: If reputable sources report contradictory figures for the same event, select Conflicting Evidence/Cherrypicking.\n"
        "2. Editorial dispute: Political debate between parties is not factual contradiction.\n"
        "State a concise 1-sentence factual reason, then output Verdict: <Label>."
    ),
}


def build_controlled_guidance(
    condition: str,
    claim: str,
    evidence: Optional[List[Dict[str, Any]]] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[str, Optional[str]]:
    """Build guidance string for controlled conditions M0, M1, M2, M3.
    
    Returns:
        (guidance_text, selected_mechanism_or_none)
    """
    cond = condition.upper()
    if cond == "M0":
        return GUIDANCE_M0, None
    if cond == "M1":
        return GUIDANCE_M1, None
    if cond == "M2":
        # Source-derived mechanism recognition:
        # Check active signals; prioritize the mechanism with greatest Fold A support
        active = detect_active_mechanisms(claim, evidence)
        if not active:
            # Fallback to matched generic control
            return GUIDANCE_M0, None
        # Select highest-priority active mechanism (Scope > Partial Truth > Sufficiency > Attribution > Negation)
        selected = active[0]
        rule = MECHANISM_GUIDANCE_RULES.get(selected, GUIDANCE_M0)
        return rule, selected
    if cond == "M3":
        # Negative control: randomly assign one of the 6 mechanism guidance directives
        # independent of the claim's actual features
        if rng is None:
            rng = random.Random(42)
        random_mech = rng.choice(ALL_MECHANISMS)
        rule = MECHANISM_GUIDANCE_RULES[random_mech]
        return rule, random_mech
    if cond == "M4":
        # M4 uses M0 guidance, but evidence ordering is permuted at inference time
        return GUIDANCE_M0, None
    raise ValueError(f"Unknown condition: {condition}. Expected M0, M1, M2, M3, or M4.")
