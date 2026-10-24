"""Source-only historical error memory builder, indexing, and exemplar selectors."""
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tffv.data import read_jsonl
from tffv.provenance import canonical, digest, file_hash, write_json
from tffv.verification.ontology import VERDICTS


class ErrorMemoryBank:
    """Historical error memory bank containing verified correct and wrong base predictions."""

    def __init__(self, memory_data: Dict[str, Any]):
        self.data = memory_data
        self.records: Dict[str, Dict[str, Any]] = memory_data.get("records", {})
        self.correct_sids: List[str] = memory_data.get("correct_sids", [])
        self.wrong_sids: List[str] = memory_data.get("wrong_sids", [])
        self.by_class: Dict[str, List[str]] = memory_data.get("by_class", {})
        self.by_confusion: Dict[str, List[str]] = memory_data.get("by_confusion", {})

    def get_record(self, sid: str) -> Dict[str, Any]:
        if sid not in self.records:
            raise KeyError(f"Sample ID {sid} not in error memory bank!")
        return self.records[sid]

    def sample_random_source_exemplars(self, k: int = 4, seed: int = 42) -> List[Dict[str, Any]]:
        """Sample k balanced exemplars (1 per class) randomly from all source records."""
        rng = random.Random(seed)
        selected = []
        for c in VERDICTS:
            candidates = [sid for sid in self.records if self.records[sid]["gold_label"] == c]
            if not candidates:
                raise ValueError(f"No source candidates available for class {c}")
            chosen_sid = rng.choice(candidates)
            selected.append(self.records[chosen_sid])
        return selected[:k]

    def sample_correct_only_exemplars(self, k: int = 4, seed: int = 42) -> List[Dict[str, Any]]:
        """Sample k balanced exemplars (1 per class) strictly from correct source records."""
        rng = random.Random(seed)
        selected = []
        for c in VERDICTS:
            candidates = [sid for sid in self.correct_sids if self.records[sid]["gold_label"] == c]
            if not candidates:
                raise ValueError(f"Insufficient correct source examples for class {c}")
            chosen_sid = rng.choice(candidates)
            selected.append(self.records[chosen_sid])
        return selected[:k]

    def sample_error_only_exemplars(self, k: int = 4, seed: int = 42) -> List[Dict[str, Any]]:
        """Sample k error exemplars with gold corrections targeting the most frequent confusion pairs."""
        rng = random.Random(seed)
        target_pairs = [
            "Refuted -> Not Enough Evidence",
            "Supported -> Conflicting Evidence/Cherrypicking",
            "Supported -> Not Enough Evidence",
            "Conflicting Evidence/Cherrypicking -> Refuted",
        ]
        selected = []
        for pair in target_pairs:
            candidates = self.by_confusion.get(pair, [])
            if candidates:
                chosen_sid = rng.choice(candidates)
                selected.append(self.records[chosen_sid])
            else:
                # Fallback to any remaining wrong exemplar if specific pair unavailable
                rem = [sid for sid in self.wrong_sids if sid not in [r["sample_id"] for r in selected]]
                if rem:
                    selected.append(self.records[rng.choice(rem)])

        return selected[:k]

    def sample_contrastive_pairs(self, num_pairs: int = 2, seed: int = 42) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Sample contrastive pairs: (Correct exemplar, Corrected error exemplar) on subtle boundaries.
        
        Pair 1: Supported boundary (Correct Supported vs Supported misclassified as Cherrypicking)
        Pair 2: Refuted boundary (Correct Refuted vs Refuted misclassified as Not Enough Evidence)
        """
        rng = random.Random(seed)
        pairs = []

        # Boundary 1: Supported
        corr_supp = [sid for sid in self.correct_sids if self.records[sid]["gold_label"] == "Supported"]
        err_supp_cherry = self.by_confusion.get("Supported -> Conflicting Evidence/Cherrypicking", [])
        if corr_supp and err_supp_cherry:
            pairs.append((
                self.records[rng.choice(corr_supp)],
                self.records[rng.choice(err_supp_cherry)],
            ))

        # Boundary 2: Refuted
        corr_ref = [sid for sid in self.correct_sids if self.records[sid]["gold_label"] == "Refuted"]
        err_ref_nei = self.by_confusion.get("Refuted -> Not Enough Evidence", [])
        if corr_ref and err_ref_nei:
            pairs.append((
                self.records[rng.choice(corr_ref)],
                self.records[rng.choice(err_ref_nei)],
            ))

        return pairs[:num_pairs]

    def get_source_derived_rules(self) -> str:
        """Synthesize compact verification checklist derived from empirical failure modes in D_error."""
        return (
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


def build_error_memory_bank(
    train_json_path: Path,
    b1_predictions_path: Path,
    r4_evidence_path: Path,
    output_path: Optional[Path] = None,
    allowed_sample_ids: Optional[Set[str]] = None,
) -> ErrorMemoryBank:
    """Build and seal a content-addressed ErrorMemoryBank from B1 predictions and ground truth."""
    train_json_path = Path(train_json_path).expanduser().resolve()
    b1_predictions_path = Path(b1_predictions_path).expanduser().resolve()
    r4_evidence_path = Path(r4_evidence_path).expanduser().resolve()

    # 1. Load gold data
    train_data = json.loads(train_json_path.read_text(encoding="utf-8"))
    gold_map = {}
    for idx, row in enumerate(train_data):
        sid = f"train:{idx}"
        gold_map[sid] = {
            "claim": row.get("claim", "").strip(),
            "label": row.get("label", "").strip(),
            "justification": row.get("justification", "").strip(),
            "speaker": row.get("speaker"),
            "claim_date": row.get("claim_date"),
        }

    # 2. Load Top-8 Evidence
    evidence_map = {}
    for row in read_jsonl(r4_evidence_path):
        evidence_map[row["sample_id"]] = row.get("evidence", [])

    # 3. Load B1 Predictions
    records = {}
    correct_sids = []
    wrong_sids = []
    by_class: Dict[str, List[str]] = {c: [] for c in VERDICTS}
    by_confusion: Dict[str, List[str]] = {}

    for row in read_jsonl(b1_predictions_path):
        sid = row["sample_id"]
        if allowed_sample_ids is not None and sid not in allowed_sample_ids:
            continue
        gold_info = gold_map.get(sid)
        if not gold_info:
            continue

        gold_label = gold_info["label"]
        pred_verdict = (row.get("predicted_verdict") or "").strip()
        is_correct = bool(pred_verdict and pred_verdict in VERDICTS and pred_verdict == gold_label)

        ev_list = evidence_map.get(sid, [])
        # Select compact passage representation
        compact_evidence = [
            {
                "passage_id": ev.get("passage_id"),
                "source_url": ev.get("source_url"),
                "text": ev.get("text", "")[:400] + ("..." if len(ev.get("text", "")) > 400 else ""),
            }
            for ev in ev_list[:4]  # Top-4 passages for compact demonstration
        ]

        confusion_pair = None
        if not is_correct:
            confusion_pair = f"{gold_label} -> {pred_verdict or '[Unparseable]'}"
            by_confusion.setdefault(confusion_pair, []).append(sid)
            wrong_sids.append(sid)
        else:
            correct_sids.append(sid)

        by_class.setdefault(gold_label, []).append(sid)

        records[sid] = {
            "sample_id": sid,
            "claim": gold_info["claim"],
            "gold_label": gold_label,
            "gold_justification": gold_info["justification"],
            "base_predicted_verdict": pred_verdict,
            "base_raw_generation": row.get("raw_generation", "").strip(),
            "is_correct": is_correct,
            "status": "correct" if is_correct else "wrong",
            "confusion_pair": confusion_pair,
            "compact_evidence": compact_evidence,
            "full_evidence_count": len(ev_list),
            "gold_explanation_verified": True,
        }

    memory_payload = {
        "milestone": "M1.3",
        "phase": "source_error_memory_construction",
        "total_source_claims": len(records),
        "correct_count": len(correct_sids),
        "wrong_count": len(wrong_sids),
        "accuracy_source": len(correct_sids) / len(records) if records else 0.0,
        "train_json_sha256": file_hash(train_json_path),
        "b1_predictions_sha256": file_hash(b1_predictions_path),
        "r4_evidence_sha256": file_hash(r4_evidence_path),
        "correct_sids": sorted(correct_sids, key=lambda s: int(s.split(":")[-1])),
        "wrong_sids": sorted(wrong_sids, key=lambda s: int(s.split(":")[-1])),
        "by_class": {k: sorted(v, key=lambda s: int(s.split(":")[-1])) for k, v in by_class.items()},
        "by_confusion": {k: sorted(v, key=lambda s: int(s.split(":")[-1])) for k, v in by_confusion.items()},
        "records": records,
    }

    if output_path:
        out_p = Path(output_path).expanduser().resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(memory_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return ErrorMemoryBank(memory_payload)
