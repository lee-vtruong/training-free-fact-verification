"""Evaluation of frozen base fact verification against AVeriTeC ground truth."""
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tffv.data import read_jsonl
from tffv.provenance import canonical, digest, file_hash, write_json
from tffv.verification.ontology import VERDICTS


def load_gold_labels(train_json_path: Path) -> Dict[str, Dict[str, Any]]:
    """Extract official gold labels from AVeriTeC train.json."""
    train_json_path = Path(train_json_path).expanduser().resolve()
    if not train_json_path.is_file():
        raise ValueError(f"train.json not found: {train_json_path}")

    raw = train_json_path.read_bytes()
    payload = json.loads(raw.decode("utf-8-sig"))

    gold_map = {}
    for idx, item in enumerate(payload):
        sid = f"train:{idx}"
        gold_map[sid] = {
            "sample_id": sid,
            "claim": item.get("claim", ""),
            "gold_label": item.get("label"),
            "speaker": item.get("speaker"),
            "claim_date": item.get("claim_date"),
        }
    return gold_map


def compute_metrics(
    eval_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute Accuracy, Macro-F1, per-class P/R/F1, and 4x4 Confusion Matrix."""
    total = len(eval_rows)
    all_pred_classes = list(VERDICTS) + ["Unparseable"]
    if total == 0:
        return {
            "total_evaluated": 0,
            "correct_verdicts": 0,
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "confusion_matrix": {g: {p: 0 for p in all_pred_classes} for g in VERDICTS},
            "per_class": {g: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for g in VERDICTS},
        }

    correct = sum(1 for r in eval_rows if r["is_correct"])
    accuracy = correct / total

    # Confusion matrix: rows = Gold, cols = Predicted
    # Include an 'Unparseable' column if any
    matrix = {g: {p: 0 for p in all_pred_classes} for g in VERDICTS}

    per_class = {}
    for c in VERDICTS:
        tp = sum(1 for r in eval_rows if r["gold_label"] == c and r["predicted_verdict"] == c)
        fp = sum(1 for r in eval_rows if r["gold_label"] != c and r["predicted_verdict"] == c)
        fn = sum(1 for r in eval_rows if r["gold_label"] == c and r["predicted_verdict"] != c)
        support = sum(1 for r in eval_rows if r["gold_label"] == c)

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    # Populate confusion matrix
    for r in eval_rows:
        g = r["gold_label"]
        p = r["predicted_verdict"] if r["predicted_verdict"] in VERDICTS else "Unparseable"
        if g in matrix:
            matrix[g][p] += 1

    macro_f1 = sum(per_class[c]["f1"] for c in VERDICTS) / len(VERDICTS)
    macro_precision = sum(per_class[c]["precision"] for c in VERDICTS) / len(VERDICTS)
    macro_recall = sum(per_class[c]["recall"] for c in VERDICTS) / len(VERDICTS)

    return {
        "total_claims": total,
        "correct_claims": correct,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def run_evaluation(
    predictions_dir: str,
    train_json_path: str,
    output_dir: str,
) -> Path:
    """Evaluate predictions against ground truth and produce report and artifacts."""
    pred_dir = Path(predictions_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    preds_file = pred_dir / "predictions.jsonl"
    if not preds_file.is_file():
        raise ValueError(f"Predictions file not found: {preds_file}")

    manifest_file = pred_dir / "manifest.json"
    pred_manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.is_file() else {}

    gold_map = load_gold_labels(Path(train_json_path))
    predictions = read_jsonl(preds_file)

    eval_rows = []
    for pred in predictions:
        sid = pred["sample_id"]
        gold_info = gold_map.get(sid, {})
        gold_label = gold_info.get("gold_label")
        pred_verdict = pred.get("predicted_verdict")

        is_correct = bool(gold_label and pred_verdict and gold_label == pred_verdict)
        row = {
            "sample_id": sid,
            "claim": gold_info.get("claim", ""),
            "gold_label": gold_label,
            "predicted_verdict": pred_verdict,
            "is_correct": is_correct,
            "latency_seconds": pred.get("latency_seconds", 0.0),
            "generated_tokens": pred.get("generated_tokens", 0),
            "prompt_tokens": pred.get("prompt_tokens", 0),
            "evidence_count": pred.get("evidence_count", 0),
            "raw_generation": pred.get("raw_generation", ""),
        }
        eval_rows.append(row)

    metrics = compute_metrics(eval_rows)

    # Calculate runtime & efficiency stats
    latencies = [r["latency_seconds"] for r in eval_rows if r["latency_seconds"] > 0]
    tokens = [r["generated_tokens"] for r in eval_rows]
    runtime_stats = {
        "mean_latency_seconds": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "total_latency_seconds": sum(latencies),
        "mean_generated_tokens": (sum(tokens) / len(tokens)) if tokens else 0,
        "total_generated_tokens": sum(tokens),
    }

    # Export per_claim_evaluation.csv
    csv_file = destination / "per_claim_evaluation.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["sample_id", "gold_label", "predicted_verdict", "is_correct", "latency_seconds", "generated_tokens", "evidence_count"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in eval_rows:
            writer.writerow(r)

    # Export evaluation_summary.json
    summary_data = {
        "milestone": "M1.2",
        "evaluation_target": "frozen_base_fact_verification",
        "predictions_dir": str(pred_dir),
        "predictions_sha256": file_hash(preds_file),
        "train_json_path": str(train_json_path),
        "train_json_sha256": file_hash(Path(train_json_path)),
        "metrics": metrics,
        "runtime_stats": runtime_stats,
    }
    write_json(destination / "evaluation_summary.json", summary_data)

    # Export evaluation_report.md
    cm = metrics["confusion_matrix"]
    matrix_header = "| Gold \\ Predicted | Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking | Unparseable |"
    matrix_sep = "|---|---:|---:|---:|---:|---:|"
    matrix_rows = []
    for g in VERDICTS:
        row_str = f"| **{g}** | {cm[g]['Supported']} | {cm[g]['Refuted']} | {cm[g]['Not Enough Evidence']} | {cm[g]['Conflicting Evidence/Cherrypicking']} | {cm[g]['Unparseable']} |"
        matrix_rows.append(row_str)

    model_spec = pred_manifest.get("model_spec") or {}
    md_lines = [
        "# M1.2 — Frozen Base Verification Evaluation Report",
        "",
        f"- **Evaluated Model:** `{model_spec.get('model_id', 'Synthetic / Qwen3-8B')}`",
        f"- **Model Revision:** `{model_spec.get('revision', 'N/A')}`",
        f"- **Mode:** Training-Free Zero-Shot (Greedy Decoding, Temperature 0.0)",
        f"- **Evaluated Claims:** {metrics['total_claims']}",
        f"- **Overall Accuracy:** {metrics['accuracy']:.1%} ({metrics['correct_claims']}/{metrics['total_claims']})",
        f"- **Macro-F1:** {metrics['macro_f1']:.4f}",
        f"- **Mean Latency per Claim:** {runtime_stats['mean_latency_seconds']:.2f}s",
        "",
        "## 1. Per-Class Performance Breakdown",
        "",
        "| Class | Support | Precision | Recall | F1-Score |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in VERDICTS:
        p_c = metrics["per_class"][c]
        md_lines.append(f"| **{c}** | {p_c['support']} | {p_c['precision']:.3f} | {p_c['recall']:.3f} | {p_c['f1']:.4f} |")

    md_lines.extend([
        f"| **Macro Average** | {metrics['total_claims']} | {metrics['macro_precision']:.3f} | {metrics['macro_recall']:.3f} | **{metrics['macro_f1']:.4f}** |",
        "",
        "## 2. Confusion Matrix (Gold Rows x Predicted Columns)",
        "",
        matrix_header,
        matrix_sep,
        *matrix_rows,
        "",
    ])

    report_file = destination / "evaluation_report.md"
    report_file.write_text("\n".join(md_lines), encoding="utf-8")

    # Seal manifest.json
    eval_manifest = {
        "evaluation_summary_sha256": file_hash(destination / "evaluation_summary.json"),
        "evaluation_report_sha256": file_hash(report_file),
        "per_claim_evaluation_sha256": file_hash(csv_file),
    }
    write_json(destination / "manifest.json", eval_manifest)

    return destination


def main():
    parser = argparse.ArgumentParser(description="M1.2 Verification Evaluator")
    parser.add_argument("--predictions-dir", required=True, help="Directory containing predictions.jsonl")
    parser.add_argument("--train-json", required=True, help="Path to AVeriTeC train.json")
    parser.add_argument("--output", required=True, help="Output directory for evaluation results")
    args = parser.parse_args()

    run_evaluation(args.predictions_dir, args.train_json, args.output)


if __name__ == "__main__":
    main()
