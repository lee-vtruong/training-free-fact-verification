"""Comprehensive error profiling and failure mode categorization for fact verification."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from tffv.data import read_jsonl
from tffv.provenance import file_hash, write_json
from tffv.verification.ontology import VERDICTS


def run_error_profiling(
    evaluation_dir: str,
    predictions_file: str,
    retrieval_eval_csv: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Path:
    """Analyze correct vs wrong predictions, cross-referencing retrieval quality."""
    eval_dir = Path(evaluation_dir).expanduser().resolve()
    preds_file = Path(predictions_file).expanduser().resolve()
    dest = Path(output_dir or eval_dir / "error_analysis").expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    # 1. Load evaluation rows
    per_claim_csv = eval_dir / "per_claim_evaluation.csv"
    if not per_claim_csv.is_file():
        raise ValueError(f"per_claim_evaluation.csv not found in {eval_dir}")

    eval_rows = []
    with per_claim_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["is_correct"] = (row["is_correct"].lower() == "true")
            row["latency_seconds"] = float(row.get("latency_seconds", 0.0))
            row["generated_tokens"] = int(row.get("generated_tokens", 0))
            row["evidence_count"] = int(row.get("evidence_count", 0))
            eval_rows.append(row)

    # 2. Load prediction details (raw reasoning)
    preds = {r["sample_id"]: r for r in read_jsonl(preds_file)}

    # 3. Load retrieval metrics if available (to get evidence_hit)
    retrieval_evidence_hit = {}
    if retrieval_eval_csv and Path(retrieval_eval_csv).is_file():
        with open(retrieval_eval_csv, encoding="utf-8") as f:
            r_reader = csv.DictReader(f)
            for r in r_reader:
                if r.get("method") == "R4":
                    retrieval_evidence_hit[r["sample_id"]] = (r.get("evidence_hit") == "1.0")

    # 4. Partition into correct and error sets
    d_correct = [r for r in eval_rows if r["is_correct"]]
    d_error = [r for r in eval_rows if not r["is_correct"]]

    # 5. Confusion pairs distribution
    confusion_pairs = Counter((r["gold_label"], r["predicted_verdict"]) for r in d_error)

    # 6. Evidence coverage correlation
    # - With gold evidence in Top-8 (Hit@8): How many correct vs wrong?
    # - Without gold evidence in Top-8 (Miss@8): How many correct vs wrong?
    coverage_stats = {
        "with_gold_evidence": {"total": 0, "correct": 0, "wrong": 0},
        "without_gold_evidence": {"total": 0, "correct": 0, "wrong": 0},
        "unmeasured": {"total": 0, "correct": 0, "wrong": 0},
    }

    # Anomalies
    false_negatives_with_gold_evidence = []  # Had gold URL in Top-8, but still failed
    correct_without_gold_evidence = []       # Did NOT have gold URL in Top-8, but succeeded

    for r in eval_rows:
        sid = r["sample_id"]
        hit = retrieval_evidence_hit.get(sid)
        status_key = "with_gold_evidence" if hit is True else ("without_gold_evidence" if hit is False else "unmeasured")

        coverage_stats[status_key]["total"] += 1
        if r["is_correct"]:
            coverage_stats[status_key]["correct"] += 1
            if hit is False:
                correct_without_gold_evidence.append(sid)
        else:
            coverage_stats[status_key]["wrong"] += 1
            if hit is True:
                false_negatives_with_gold_evidence.append(sid)

    # Compute error rates by coverage
    for k in coverage_stats:
        tot = coverage_stats[k]["total"]
        coverage_stats[k]["error_rate"] = (coverage_stats[k]["wrong"] / tot) if tot > 0 else 0.0

    # 7. Evidence length / token analysis
    correct_tokens = [r["generated_tokens"] for r in d_correct]
    error_tokens = [r["generated_tokens"] for r in d_error]
    length_analysis = {
        "mean_generated_tokens_correct": (sum(correct_tokens) / len(correct_tokens)) if correct_tokens else 0,
        "mean_generated_tokens_error": (sum(error_tokens) / len(error_tokens)) if error_tokens else 0,
    }

    # 8. Preliminary Error Taxonomy
    # We categorize errors into 4 observed empirical buckets:
    # 1. Evidence Shortfall (Not Enough Evidence confusion): Model fails on claims where evidence lacks proof
    # 2. Polarity Reversal (Supported <-> Refuted): Model reaches opposite factual conclusion
    # 3. Nuance & Cherrypicking Confusion: Conflicting Evidence conflated with Supported or Refuted
    # 4. Under-Retrieval Bottleneck: Gold URL was absent from retrieved Top-8 passages
    taxonomy = {
        "polarity_reversals": [
            r["sample_id"] for r in d_error
            if (r["gold_label"] == "Supported" and r["predicted_verdict"] == "Refuted") or
               (r["gold_label"] == "Refuted" and r["predicted_verdict"] == "Supported")
        ],
        "evidence_shortfall_misses": [
            r["sample_id"] for r in d_error
            if r["gold_label"] == "Not Enough Evidence" or r["predicted_verdict"] == "Not Enough Evidence"
        ],
        "cherrypicking_nuance_failures": [
            r["sample_id"] for r in d_error
            if r["gold_label"] == "Conflicting Evidence/Cherrypicking" or
               r["predicted_verdict"] == "Conflicting Evidence/Cherrypicking"
        ],
        "under_retrieval_errors": [
            r["sample_id"] for r in d_error
            if retrieval_evidence_hit.get(r["sample_id"]) is False
        ],
    }

    # 9. Dump analysis summary
    analysis_payload = {
        "total_claims": len(eval_rows),
        "total_errors": len(d_error),
        "total_correct": len(d_correct),
        "error_rate": len(d_error) / len(eval_rows) if eval_rows else 0.0,
        "confusion_pairs": [{"gold": g, "predicted": p, "count": cnt} for (g, p), cnt in confusion_pairs.most_common()],
        "coverage_correlation": coverage_stats,
        "false_negatives_with_gold_evidence_count": len(false_negatives_with_gold_evidence),
        "false_negatives_with_gold_evidence_sample_ids": false_negatives_with_gold_evidence,
        "correct_without_gold_evidence_count": len(correct_without_gold_evidence),
        "correct_without_gold_evidence_sample_ids": correct_without_gold_evidence,
        "length_analysis": length_analysis,
        "preliminary_taxonomy_counts": {k: len(v) for k, v in taxonomy.items()},
        "taxonomy": taxonomy,
    }
    write_json(dest / "error_analysis_summary.json", analysis_payload)

    # 10. Generate detailed Markdown Report
    lines = [
        "# M1.2 — Fact Verification Error Profiling Report",
        "",
        f"- **Total Evaluated Claims:** {len(eval_rows)}",
        f"- **Correct Predictions:** {len(d_correct)} ({len(d_correct)/len(eval_rows):.1%})",
        f"- **Error Predictions:** {len(d_error)} ({len(d_error)/len(eval_rows):.1%})",
        "",
        "## 1. Top Failure Modes (Gold vs Predicted Confusion)",
        "",
        "| Gold Class | Predicted Class | Error Count | Percentage of Errors |",
        "|---|---|---:|---:|",
    ]
    for (g, p), cnt in confusion_pairs.most_common():
        pct = cnt / len(d_error) if d_error else 0.0
        lines.append(f"| **{g}** | `{p}` | {cnt} | {pct:.1%} |")

    lines.extend([
        "",
        "## 2. Evidence Coverage vs Error Rate",
        "",
        "| Evidence Status in Top-8 | Total Claims | Correct | Wrong | Error Rate |",
        "|---|---:|---:|---:|---:|",
        f"| **With Gold Evidence (Hit@8=True)** | {coverage_stats['with_gold_evidence']['total']} | {coverage_stats['with_gold_evidence']['correct']} | {coverage_stats['with_gold_evidence']['wrong']} | {coverage_stats['with_gold_evidence']['error_rate']:.1%} |",
        f"| **Without Gold Evidence (Hit@8=False)** | {coverage_stats['without_gold_evidence']['total']} | {coverage_stats['without_gold_evidence']['correct']} | {coverage_stats['without_gold_evidence']['wrong']} | {coverage_stats['without_gold_evidence']['error_rate']:.1%} |",
        "",
        f"> **Key Finding:** Claims with gold evidence retrieved in Top-8 exhibited an error rate of {coverage_stats['with_gold_evidence']['error_rate']:.1%}, compared to {coverage_stats['without_gold_evidence']['error_rate']:.1%} when gold evidence was missing.",
        "",
        "## 3. Discrepancy Diagnostics: Gold Presence vs Prediction Success",
        "",
        f"- **Failed Despite Having Gold Evidence ({len(false_negatives_with_gold_evidence)} claims):**",
        f"  `{', '.join(false_negatives_with_gold_evidence) if false_negatives_with_gold_evidence else 'None'}`",
        f"- **Correct Despite Missing Gold URL ({len(correct_without_gold_evidence)} claims):**",
        f"  `{', '.join(correct_without_gold_evidence) if correct_without_gold_evidence else 'None'}`",
        "",
        "> [!NOTE]",
        "> Absence of a gold URL does NOT imply lack of equivalent factual evidence; auxiliary news articles or syndicated wires frequently confirm the same facts under distinct URLs.",
        "",
        "## 4. Preliminary Error Taxonomy",
        "",
        f"1. **Under-Retrieval Bottleneck ({len(taxonomy['under_retrieval_errors'])} errors):** Retrieval failed to place gold URL in Top-8.",
        f"2. **Evidence Shortfall Confusion ({len(taxonomy['evidence_shortfall_misses'])} errors):** Model struggles to distinguish between lack of proof (`Not Enough Evidence`) and explicit contradiction (`Refuted`).",
        f"3. **Nuance & Cherrypicking Confusion ({len(taxonomy['cherrypicking_nuance_failures'])} errors):** Model fails to catch subtle out-of-context true statements (`Conflicting Evidence/Cherrypicking`).",
        f"4. **Polarity Reversals ({len(taxonomy['polarity_reversals'])} errors):** Severe inverted reasoning between `Supported` and `Refuted`.",
    ])

    (dest / "error_profiling_report.md").write_text("\n".join(lines), encoding="utf-8")
    return dest


def main():
    parser = argparse.ArgumentParser(description="M1.2 Error Profiler")
    parser.add_argument("--eval-dir", required=True, help="Evaluation output directory")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl")
    parser.add_argument("--retrieval-csv", default=None, help="Path to retrieval per_claim_metrics.csv")
    parser.add_argument("--output", default=None, help="Output directory for error analysis")
    args = parser.parse_args()

    run_error_profiling(args.eval_dir, args.predictions, args.retrieval_csv, args.output)


if __name__ == "__main__":
    main()
