"""Milestone M2.0e: Phase C â€” Inference-Time Risk Features Construction.

Constructs feature table using strictly pre-gold inference-time signals:
- B0 predicted label characteristics (ambiguous vs binary).
- Output token counts and latency.
- Evidence characteristics (length, passage count).
- Historical pattern selector signals and grounding score.
- Surface linguistic signals (temporal, numerical, negation, conflict).
- Target offline variables (strictly segregated from feature space):
  * target_b0_wrong
  * target_c2_benefit (45 cases)
  * target_c2_harm (88 cases)
  * target_transition_class (C2_BENEFIT, C2_HARM, NEUTRAL)

Computes AUROC, AUPRC across candidate risk scores.
Emits:
- risk_features.csv
- risk_metrics.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


class RiskFeatureExtractor:
    def __init__(
        self,
        m20d_dir: str | Path = "outputs/m20d-contrastive-verification-v1",
        output_dir: str | Path = "outputs/m20e-selective-feasibility-v1",
    ):
        self.m20d_dir = Path(m20d_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_features(self) -> Dict[str, Any]:
        p_file = self.m20d_dir / "predictions.jsonl"
        t_file = self.m20d_dir / "selection_traces.jsonl"
        r_file = self.m20d_dir / "retrieval_receipts.jsonl"
        if not r_file.exists():
            r_file = Path("outputs/m20c-independent-holdout-v1/retrieval_receipts.jsonl")

        recs = self._load_jsonl(p_file)
        traces = {r["sample_id"]: r for r in self._load_jsonl(t_file)} if t_file.exists() else {}
        receipts = {r["sample_id"]: r.get("evidence", []) for r in self._load_jsonl(r_file)} if r_file.exists() else {}

        b0_map = {r["sample_id"]: r for r in recs if r["arm"] == "B0"}
        c2_map = {r["sample_id"]: r for r in recs if r["arm"] == "C2"}

        sample_ids = sorted(list(b0_map.keys()))
        rows = []

        for sid in sample_ids:
            r_b0 = b0_map[sid]
            r_c2 = c2_map.get(sid, {})
            tr = traces.get(sid, {})
            ev = receipts.get(sid, [])

            gold = r_b0.get("gold_label")
            p_b0 = r_b0.get("parsed_verdict")
            p_c2 = r_c2.get("parsed_verdict")

            # Targets (strictly for offline evaluation)
            corr_b0 = (p_b0 == gold)
            corr_c2 = (p_c2 == gold)

            is_benefit = int(not corr_b0 and corr_c2)
            is_harm = int(corr_b0 and not corr_c2)
            b0_wrong = int(not corr_b0)

            if is_benefit:
                trans_class = "C2_BENEFIT"
            elif is_harm:
                trans_class = "C2_HARM"
            else:
                trans_class = "NEUTRAL"

            # Inference-time features (available BEFORE gold)
            is_ambiguous_pred = 1 if p_b0 in ["Not Enough Evidence", "Conflicting Evidence/Cherrypicking"] else 0
            b0_tokens = r_b0.get("output_tokens", 0)
            b0_latency = r_b0.get("latency_seconds", 0.0)

            ev_text = " ".join([p.get("text", "") for p in ev])
            ev_len = len(ev_text)
            ev_passages = len(ev)

            pat_id = tr.get("selected_pattern_id") or "NONE"
            pat_score = tr.get("selection_score", 0.0)
            signals = tr.get("salient_signals", [])
            sig_count = len(signals)

            has_temp = 1 if any("temp" in s or "year" in s for s in signals) else 0
            has_num = 1 if any("num" in s or "quantity" in s for s in signals) else 0
            has_neg = 1 if any("neg" in s for s in signals) else 0
            has_conf = 1 if any("conflict" in s for s in signals) else 0

            # Composite heuristic risk score: combines ambiguity with signal density
            composite_risk = (is_ambiguous_pred * 0.6) + (pat_score * 0.4)

            rows.append({
                "sample_id": sid,
                "b0_predicted_label": p_b0,
                "is_b0_ambiguous_pred": is_ambiguous_pred,
                "b0_output_tokens": b0_tokens,
                "b0_latency": b0_latency,
                "evidence_char_length": ev_len,
                "evidence_passage_count": ev_passages,
                "selected_pattern_id": pat_id,
                "pattern_selection_score": pat_score,
                "salient_signal_count": sig_count,
                "has_temporal_signal": has_temp,
                "has_numerical_signal": has_num,
                "has_negation_signal": has_neg,
                "has_conflict_signal": has_conf,
                "composite_risk_score": round(composite_risk, 3),
                # Targets (Offline Only)
                "target_b0_wrong": b0_wrong,
                "target_c2_benefit": is_benefit,
                "target_c2_harm": is_harm,
                "target_transition_class": trans_class,
            })

        # Save risk_features.csv
        csv_path = self.output_dir / "risk_features.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        # Compute AUROC & AUPRC metrics across features
        y_b0_err = [r["target_b0_wrong"] for r in rows]
        y_ben = [r["target_c2_benefit"] for r in rows]
        y_harm = [r["target_c2_harm"] for r in rows]

        feature_keys = [
            "is_b0_ambiguous_pred",
            "composite_risk_score",
            "pattern_selection_score",
            "salient_signal_count",
            "b0_output_tokens",
            "evidence_char_length",
        ]

        metrics = {
            "total_samples": len(rows),
            "target_distribution": {
                "b0_wrong": sum(y_b0_err),
                "c2_benefit": sum(y_ben),
                "c2_harm": sum(y_harm),
                "neutral": len(rows) - sum(y_ben) - sum(y_harm),
            },
            "evaluations": {},
        }

        for f_name in feature_keys:
            scores = [r[f_name] for r in rows]
            # Predict B0 errors
            auc_err = float(roc_auc_score(y_b0_err, scores))
            ap_err = float(average_precision_score(y_b0_err, scores))
            # Predict C2 Benefit
            auc_ben = float(roc_auc_score(y_ben, scores))
            ap_ben = float(average_precision_score(y_ben, scores))
            # Predict C2 Harm
            auc_harm = float(roc_auc_score(y_harm, scores))
            ap_harm = float(average_precision_score(y_harm, scores))

            metrics["evaluations"][f_name] = {
                "predict_b0_error": {"auroc": round(auc_err, 4), "auprc": round(ap_err, 4)},
                "predict_c2_benefit": {"auroc": round(auc_ben, 4), "auprc": round(ap_ben, 4)},
                "predict_c2_harm": {"auroc": round(auc_harm, 4), "auprc": round(ap_harm, 4)},
            }

        json_path = self.output_dir / "risk_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        return {
            "csv_path": str(csv_path),
            "json_path": str(json_path),
            "metrics": metrics,
        }

    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        recs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    recs.append(json.loads(line))
        return recs
