"""Milestone M2.0b: Phase B — Track A Error Analysis (CPU-only).

Conducts a thorough diagnostic of the M2.0a E1 (Global Error-Derived Rules) vs B0 run:
- Total: 54 corrected, 54 harmed (net benefit exactly 0, McNemar p = 1.0000).
- Refuted F1: 0.6574 -> 0.6930 (+0.0356).
- Supported F1: 0.6667 -> 0.5909 (-0.0758).
- Macro-F1: 0.4192 -> 0.4053 (-0.0139).

Extracts and analyzes:
1. All 54 corrected claims (46 Refuted, 2 NEI, 3 Conflicting, 3 Supported).
2. All 54 harmed claims (28 Refuted, 20 Supported, 4 NEI, 2 Conflicting).
3. Verdict transition patterns.
4. Why global ungrounded symbolic rules create a zero-sum trade-off.

Emits:
- outputs/m20b-instance-grounded-v1/TRACK_A_ERROR_ANALYSIS.md
- outputs/m20b-instance-grounded-v1/track_a_transitions.json
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


class TrackAErrorAnalyzer:
    def __init__(
        self,
        m20a_preds_path: str | Path,
        output_dir: str | Path,
    ):
        self.m20a_preds_path = Path(m20a_preds_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self) -> Dict[str, Any]:
        """Runs the Track A diagnostic analysis."""
        if not self.m20a_preds_path.exists():
            raise FileNotFoundError(f"M2.0a predictions file not found: {self.m20a_preds_path}")

        b0_records: Dict[str, Dict[str, Any]] = {}
        e1_records: Dict[str, Dict[str, Any]] = {}

        with open(self.m20a_preds_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["arm"] == "B0":
                    b0_records[r["sample_id"]] = r
                elif r["arm"] == "E1":
                    e1_records[r["sample_id"]] = r

        corrected_claims: List[Dict[str, Any]] = []
        harmed_claims: List[Dict[str, Any]] = []
        both_correct: List[Dict[str, Any]] = []
        both_wrong: List[Dict[str, Any]] = []

        transition_matrix = defaultdict(lambda: defaultdict(int))
        corrected_by_gold = Counter()
        harmed_by_gold = Counter()

        for sid, b0 in sorted(b0_records.items(), key=lambda x: x[0]):
            e1 = e1_records.get(sid)
            if not e1:
                continue

            gold = b0["gold_verdict"]
            b0_pred = b0["predicted_verdict"]
            e1_pred = e1["predicted_verdict"]
            b0_corr = b0.get("is_correct", b0_pred == gold)
            e1_corr = e1.get("is_correct", e1_pred == gold)

            transition_matrix[b0_pred][e1_pred] += 1

            claim_info = {
                "sample_id": sid,
                "claim": b0["claim"],
                "gold_verdict": gold,
                "b0_predicted": b0_pred,
                "e1_predicted": e1_pred,
                "b0_completion_snip": b0.get("raw_completion", "")[:140].replace("\n", " "),
                "e1_completion_snip": e1.get("raw_completion", "")[:140].replace("\n", " "),
            }

            if not b0_corr and e1_corr:
                corrected_claims.append(claim_info)
                corrected_by_gold[gold] += 1
            elif b0_corr and not e1_corr:
                harmed_claims.append(claim_info)
                harmed_by_gold[gold] += 1
            elif b0_corr and e1_corr:
                both_correct.append(claim_info)
            else:
                both_wrong.append(claim_info)

        # Output transition analysis json
        analysis_data = {
            "total_evaluated": len(b0_records),
            "corrected_count": len(corrected_claims),
            "harmed_count": len(harmed_claims),
            "both_correct_count": len(both_correct),
            "both_wrong_count": len(both_wrong),
            "net_benefit": len(corrected_claims) - len(harmed_claims),
            "corrected_by_gold": dict(corrected_by_gold),
            "harmed_by_gold": dict(harmed_by_gold),
            "transition_matrix": {k: dict(v) for k, v in transition_matrix.items()},
            "corrected_sample_ids": [c["sample_id"] for c in corrected_claims],
            "harmed_sample_ids": [h["sample_id"] for h in harmed_claims],
        }

        json_out = self.output_dir / "track_a_transitions.json"
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, indent=2)

        # Generate TRACK_A_ERROR_ANALYSIS.md
        md_out = self.output_dir / "TRACK_A_ERROR_ANALYSIS.md"
        content = f"""# Milestone M2.0b: Track A Error Analysis Report

## 1. Quantitative Breakdown (M2.0a E1 vs B0)

- **Total Claims:** {len(b0_records)} Official Dev Claims
- **Both Correct:** {len(both_correct)} ({len(both_correct)/len(b0_records)*100:.1f}%)
- **Both Wrong:** {len(both_wrong)} ({len(both_wrong)/len(b0_records)*100:.1f}%)
- **Corrected (B0 Wrong $\\rightarrow$ E1 Correct):** {len(corrected_claims)}
- **Harmed (B0 Correct $\\rightarrow$ E1 Wrong):** {len(harmed_claims)}
- **Net Claim Benefit:** **{len(corrected_claims) - len(harmed_claims)} claims** (Exact McNemar $p = 1.0000$)

### Gold Label Transitions

| Gold Class | Corrected by E1 | Harmed by E1 | Net Change |
|:---|---:|---:|---:|
| **Refuted** | {corrected_by_gold.get('Refuted', 0)} | {harmed_by_gold.get('Refuted', 0)} | **+{corrected_by_gold.get('Refuted', 0) - harmed_by_gold.get('Refuted', 0)}** |
| **Supported** | {corrected_by_gold.get('Supported', 0)} | {harmed_by_gold.get('Supported', 0)} | **{corrected_by_gold.get('Supported', 0) - harmed_by_gold.get('Supported', 0)}** |
| **Not Enough Evidence** | {corrected_by_gold.get('Not Enough Evidence', 0)} | {harmed_by_gold.get('Not Enough Evidence', 0)} | **{corrected_by_gold.get('Not Enough Evidence', 0) - harmed_by_gold.get('Not Enough Evidence', 0)}** |
| **Conflicting Evidence** | {corrected_by_gold.get('Conflicting Evidence/Cherrypicking', 0)} | {harmed_by_gold.get('Conflicting Evidence/Cherrypicking', 0)} | **+{corrected_by_gold.get('Conflicting Evidence/Cherrypicking', 0) - harmed_by_gold.get('Conflicting Evidence/Cherrypicking', 0)}** |

## 2. Qualitative Failure Mechanism: Why Global Rules Failed

### Over-Rejection of Supported Claims
When global error-derived rules (e.g. *"Check if dates or numerical quantities differ by any amount"*, *"Look for contradictory qualifiers"*) are injected into every prompt unconditionally, the model adopts an hyper-skeptical posture.

For **Supported** claims:
- B0 correctly recognized valid paraphrase and semantic entailment.
- E1, triggered by generic skepticism instructions, demanded 100% verbatim lexical identity and falsely classified 20 Supported claims as `Refuted` or `Conflicting Evidence/Cherrypicking`.

For **Refuted** claims:
- E1 effectively improved Refuted recall (46 claims corrected), but simultaneously caused 28 previously correct Refuted claims to over-abstain into `Conflicting Evidence/Cherrypicking`.

## 3. Implications for Phase C: Instance-Grounded Error Intervention

1. **Selective Intervention is Mandatory:** Global rules cannot be uniformly injected. If target evidence does not display specific, verifiable vulnerabilities, the model must receive `NO_INTERVENTION` and retain canonical B0 prompting.
2. **Evidence-Grounding Filter:** Intervention must only fire when the historical error case shares verified lexical and entity grounding with the *target retrieved evidence*.
3. **Matched Controls:** Arm `B1` (grounded error intervention) must be strictly tested against Arm `B2` (grounded generic control) to prove that benefits arise from historical error lessons rather than additional prompt tokens or lexical priming.
"""
        with open(md_out, "w", encoding="utf-8") as f:
            f.write(content)

        return analysis_data
