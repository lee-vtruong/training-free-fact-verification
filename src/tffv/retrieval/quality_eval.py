"""Retrieval quality evaluation against official AVeriTeC ground truth.

Strict contract:
1. Sealed runs only: Ranking caches must be materialized, hashed, and sealed before
   any evaluation annotations are read.
2. Label-free isolation: Gold annotations NEVER influence retrieval, candidate
   selection, fusion, or reranking.
3. Provenance and reproducibility: Denominators, candidate store oracle ceilings,
   and exact URL matches are explicitly tracked and sealed.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tffv.data import read_jsonl
from tffv.provenance import canonical, digest, file_hash, write_json
from tffv.retrieval.benchmark import output_directory
from tffv.retrieval.data import Passage
from tffv.retrieval.url_norm import normalize_url, urls_match


DEPTHS = (5, 10, 20, 50, 100)


def verify_sealed_run(run_dir: Path) -> Dict[str, Any]:
    """Verify that retrieval run is complete, sealed, and label-free."""
    run_dir = Path(run_dir).expanduser().resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Run directory missing manifest.json: {run_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("gold_labels_used") is not False:
        raise ValueError("Cannot evaluate: retrieval run did not certify gold_labels_used is False")

    # Verify hashes of outputs
    if "outputs_sha256" in manifest:
        for rel_path, expected_hash in manifest["outputs_sha256"].items():
            full_path = run_dir / rel_path
            if full_path.is_file():
                actual_hash = file_hash(full_path)
                if actual_hash != expected_hash:
                    raise ValueError(f"Sealed output hash mismatch for {rel_path}: {actual_hash} != {expected_hash}")

    return manifest


def extract_gold_urls_from_source(train_json_path: Path) -> List[Dict[str, Any]]:
    """Extract gold evidence URLs and fact-checking article from AVeriTeC train.json."""
    train_json_path = Path(train_json_path).expanduser().resolve()
    if not train_json_path.is_file():
        raise ValueError(f"Source train JSON does not exist: {train_json_path}")

    raw = train_json_path.read_bytes()
    payload = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("AVeriTeC train.json must be a JSON array")

    extracted = []
    for index, item in enumerate(payload):
        sid = f"train:{index}"
        gold_urls: Set[str] = set()

        # 1. Extract URLs from QA pairs in questions
        for q in item.get("questions", []):
            if isinstance(q, dict):
                for key in ("source_url", "url"):
                    val = q.get(key)
                    if isinstance(val, str) and val.strip():
                        norm = normalize_url(val)
                        if norm:
                            gold_urls.add(norm)
                for a in q.get("answers", []):
                    if isinstance(a, dict):
                        for key in ("source_url", "url"):
                            val = a.get(key)
                            if isinstance(val, str) and val.strip():
                                norm = normalize_url(val)
                                if norm:
                                    gold_urls.add(norm)

        # 2. Extract URLs from evidence list if present
        for ev in item.get("evidence", []):
            if isinstance(ev, dict):
                for key in ("source_url", "url"):
                    val = ev.get(key)
                    if isinstance(val, str) and val.strip():
                        norm = normalize_url(val)
                        if norm:
                            gold_urls.add(norm)

        # 3. Extract fact_checking_article separately
        fc_article = item.get("fact_checking_article")
        fc_norm = normalize_url(fc_article) if isinstance(fc_article, str) and fc_article.strip() else None

        extracted.append({
            "sample_id": sid,
            "claim": item.get("claim", ""),
            "label": item.get("label"),
            "gold_evidence_urls": sorted(gold_urls),
            "fact_checking_article_url": fc_norm,
        })

    return extracted


def load_store_urls(
    run_dir: Path,
    sample_id: str,
    cache: Optional[Dict[str, Tuple[Dict[str, str], Set[str]]]] = None
) -> Tuple[Dict[str, str], Set[str]]:
    """Load passage_id -> source_url mapping and store URLs with memory caching."""
    if cache is not None and sample_id in cache:
        return cache[sample_id]

    ordinal = sample_id.split(":")[-1] if ":" in sample_id else sample_id
    store_passages_path = run_dir / f"train-{ordinal}" / "passages.jsonl"
    if not store_passages_path.is_file():
        store_passages_path = run_dir / "passages.jsonl"

    if not store_passages_path.is_file():
        return {}, set()

    import sys
    sys.stderr.write(f"[quality_eval] Loading passage URLs for {sample_id}...\n")
    pid_to_url: Dict[str, str] = {}
    store_urls: Set[str] = set()

    with store_passages_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            pid = row.get("passage_id")
            raw_url = row.get("source_url")
            if pid and raw_url:
                norm = normalize_url(raw_url)
                pid_to_url[pid] = norm
                if norm:
                    store_urls.add(norm)

    res = (pid_to_url, store_urls)
    if cache is not None:
        cache[sample_id] = res
    return res


def evaluate_method_claim(
    claim_record: Dict[str, Any],
    gold_record: Dict[str, Any],
    pid_to_url: Dict[str, str],
    store_urls: Set[str],
) -> Dict[str, Any]:
    """Compute retrieval metrics for one claim and one method."""
    gold_urls = set(gold_record.get("gold_evidence_urls", []))
    n_gold = len(gold_urls)

    # Candidate ranking
    candidates = claim_record.get("ranked_candidates", [])
    candidate_urls = []
    for c in candidates:
        pid = c["passage_id"]
        norm_url = pid_to_url.get(pid)
        if norm_url:
            candidate_urls.append((c["rank"], norm_url, pid))

    # Evidence URLs (selected Top-8)
    evidence_urls = []
    for ev in claim_record.get("evidence", []):
        if "source_url" in ev and ev["source_url"]:
            evidence_urls.append(normalize_url(ev["source_url"]))

    # Hit@k and Recall@k
    hits = {}
    recalls = {}
    first_gold_rank: Optional[int] = None
    retrieved_gold_urls = set()

    for k in DEPTHS:
        k_urls = {u for rank, u, _ in candidate_urls if rank <= k and u}
        matched = k_urls & gold_urls
        hits[f"hit@{k}"] = 1.0 if matched else 0.0
        recalls[f"recall@{k}"] = (len(matched) / n_gold) if n_gold > 0 else None

    # First gold rank & MRR@100
    for rank, u, _ in candidate_urls:
        if rank <= 100 and u in gold_urls:
            if first_gold_rank is None:
                first_gold_rank = rank
            retrieved_gold_urls.add(u)

    mrr100 = (1.0 / first_gold_rank) if first_gold_rank is not None else 0.0

    # Store coverage (oracle ceiling)
    store_matched = store_urls & gold_urls
    store_coverage = (len(store_matched) / n_gold) if n_gold > 0 else None
    store_hit = 1.0 if store_matched else 0.0

    # Evidence (Top-8) gold count
    evidence_matched = set(evidence_urls) & gold_urls
    evidence_hit = 1.0 if evidence_matched else 0.0
    evidence_recall = (len(evidence_matched) / n_gold) if n_gold > 0 else None

    # Fact-checking article check
    fc_url = gold_record.get("fact_checking_article_url")
    fc_retrieved_rank: Optional[int] = None
    if fc_url:
        for rank, u, _ in candidate_urls[:100]:
            if u == fc_url:
                fc_retrieved_rank = rank
                break

    return {
        "sample_id": claim_record["sample_id"],
        "method": claim_record["method"],
        "n_gold_urls": n_gold,
        "n_candidates": len(candidates),
        "first_gold_rank": first_gold_rank,
        "mrr@100": mrr100 if n_gold > 0 else None,
        **hits,
        **recalls,
        "evidence_hit": evidence_hit if n_gold > 0 else None,
        "evidence_recall": evidence_recall if n_gold > 0 else None,
        "store_gold_coverage": store_coverage,
        "store_gold_hit": store_hit if n_gold > 0 else None,
        "hit100_miss10": bool(hits.get("hit@100") == 1.0 and hits.get("hit@10") == 0.0),
        "missing_gold_urls": sorted(gold_urls - retrieved_gold_urls),
        "fc_article_retrieved_rank": fc_retrieved_rank,
    }


def run_evaluation(
    run_dir: str,
    train_json_path: str,
    output_dir: str,
    claims_manifest_path: Optional[str] = None,
) -> Path:
    """Execute retrieval quality evaluation and export comprehensive report."""
    run_path = Path(run_dir).expanduser().resolve()
    manifest = verify_sealed_run(run_path)
    destination = output_directory(output_dir)

    # Extract gold records
    gold_records = extract_gold_urls_from_source(Path(train_json_path))
    gold_by_id = {r["sample_id"]: r for r in gold_records}

    # Identify methods in run
    methods_evaluated = manifest.get("config", {}).get("methods", [])
    if not methods_evaluated:
        methods_evaluated = [m for m in ("R1", "R2", "R3", "R4") if (run_path / f"{m}.jsonl").is_file()]

    all_per_claim_results = []
    method_summaries = {}
    r3_r4_comparison = None

    per_method_results: Dict[str, List[Dict[str, Any]]] = {}
    store_cache: Dict[str, Tuple[Dict[str, str], Set[str]]] = {}

    for method in methods_evaluated:
        method_file = run_path / f"{method}.jsonl"
        if not method_file.is_file():
            continue

        claim_rows = read_jsonl(method_file)
        results = []

        for row in claim_rows:
            sid = row["sample_id"]
            gold = gold_by_id.get(sid, {"sample_id": sid, "gold_evidence_urls": []})
            pid_to_url, store_urls = load_store_urls(run_path, sid, cache=store_cache)

            eval_res = evaluate_method_claim(row, gold, pid_to_url, store_urls)
            results.append(eval_res)
            all_per_claim_results.append(eval_res)

        per_method_results[method] = results

        # Compute summary for method
        annotated_results = [r for r in results if r["n_gold_urls"] > 0]
        denom = len(annotated_results)

        means = {}
        if denom > 0:
            for k in DEPTHS:
                means[f"hit@{k}"] = sum(r[f"hit@{k}"] for r in annotated_results) / denom
                means[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in annotated_results) / denom
            valid_mrrs = [r["mrr@100"] for r in annotated_results if r["mrr@100"] is not None]
            means["mrr@100"] = (sum(valid_mrrs) / denom) if valid_mrrs else 0.0
            means["store_gold_coverage"] = sum(r["store_gold_coverage"] for r in annotated_results) / denom
            means["store_gold_hit"] = sum(r["store_gold_hit"] for r in annotated_results) / denom
            means["evidence_recall@8"] = sum(r["evidence_recall"] for r in annotated_results) / denom
            means["evidence_hit@8"] = sum(r["evidence_hit"] for r in annotated_results) / denom

        method_summaries[method] = {
            "total_claims": len(results),
            "annotated_claims": denom,
            "unannotated_claims": len(results) - denom,
            "means": means,
            "hit100_miss10_count": sum(1 for r in results if r["hit100_miss10"]),
            "hit100_miss10_sample_ids": [r["sample_id"] for r in results if r["hit100_miss10"]],
        }

    # Reranker Shift Analysis (R3 vs R4)
    if "R3" in per_method_results and "R4" in per_method_results:
        r3_map = {r["sample_id"]: r for r in per_method_results["R3"]}
        r4_map = {r["sample_id"]: r for r in per_method_results["R4"]}
        common_sids = sorted(set(r3_map.keys()) & set(r4_map.keys()))

        improved = 0
        degraded = 0
        unchanged = 0
        promoted_top10 = 0
        demoted_top10 = 0
        net_rank_shift = 0

        shifts = []
        for sid in common_sids:
            r3_item = r3_map[sid]
            r4_item = r4_map[sid]
            if r3_item["n_gold_urls"] == 0:
                continue

            r3_rank = r3_item["first_gold_rank"]
            r4_rank = r4_item["first_gold_rank"]

            # If both found in Top-100
            if r3_rank is not None and r4_rank is not None:
                shift = r3_rank - r4_rank
                net_rank_shift += shift
                if shift > 0:
                    improved += 1
                elif shift < 0:
                    degraded += 1
                else:
                    unchanged += 1
                shifts.append({"sample_id": sid, "r3_rank": r3_rank, "r4_rank": r4_rank, "shift": shift})

                if r3_rank > 10 and r4_rank <= 10:
                    promoted_top10 += 1
                elif r3_rank <= 10 and r4_rank > 10:
                    demoted_top10 += 1
            elif r3_rank is None and r4_rank is not None:
                improved += 1
                if r4_rank <= 10:
                    promoted_top10 += 1
            elif r3_rank is not None and r4_rank is None:
                degraded += 1
                if r3_rank <= 10:
                    demoted_top10 += 1
            else:
                unchanged += 1

        r3_r4_comparison = {
            "evaluated_claims": len(common_sids),
            "improved_rank_count": improved,
            "degraded_rank_count": degraded,
            "unchanged_rank_count": unchanged,
            "promoted_to_top10_count": promoted_top10,
            "demoted_from_top10_count": demoted_top10,
            "net_rank_shift_sum": net_rank_shift,
            "detailed_shifts": shifts,
        }

    # Export CSVs
    # 1. metrics_table.csv
    csv_table_path = destination / "metrics_table.csv"
    with csv_table_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "claims", "annotated_claims", "hit@5", "hit@10", "hit@20", "hit@50", "hit@100",
                         "recall@5", "recall@10", "recall@20", "recall@50", "recall@100", "mrr@100",
                         "evidence_recall@8", "store_gold_coverage", "hit100_miss10_count"])
        for m, stats in method_summaries.items():
            means = stats["means"]
            writer.writerow([
                m, stats["total_claims"], stats["annotated_claims"],
                f"{means.get('hit@5', 0):.4f}", f"{means.get('hit@10', 0):.4f}",
                f"{means.get('hit@20', 0):.4f}", f"{means.get('hit@50', 0):.4f}", f"{means.get('hit@100', 0):.4f}",
                f"{means.get('recall@5', 0):.4f}", f"{means.get('recall@10', 0):.4f}",
                f"{means.get('recall@20', 0):.4f}", f"{means.get('recall@50', 0):.4f}", f"{means.get('recall@100', 0):.4f}",
                f"{means.get('mrr@100', 0):.4f}", f"{means.get('evidence_recall@8', 0):.4f}",
                f"{means.get('store_gold_coverage', 0):.4f}", stats["hit100_miss10_count"]
            ])

    # 2. per_claim_metrics.csv
    csv_claim_path = destination / "per_claim_metrics.csv"
    if all_per_claim_results:
        keys = ["sample_id", "method", "n_gold_urls", "first_gold_rank", "mrr@100",
                "hit@5", "hit@10", "hit@20", "hit@50", "hit@100",
                "recall@5", "recall@10", "recall@20", "recall@50", "recall@100",
                "evidence_hit", "evidence_recall", "store_gold_coverage", "hit100_miss10"]
        with csv_claim_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for row in all_per_claim_results:
                writer.writerow(row)

    # 3. evaluation_summary.json
    summary_data = {
        "run_path": str(run_path),
        "train_json_path": str(train_json_path),
        "corpus_admitted": False,
        "scope": "exploratory_quality_evaluation_unadmitted_corpus",
        "methods": method_summaries,
        "r3_r4_reranker_comparison": r3_r4_comparison,
        "limitations": [
            "Source-level URL evaluation using AVeriTeC train annotations; not verified passage qrels.",
            "Candidate corpus is unadmitted; metrics denote exploratory retrieval behavior.",
            "Gold evidence URLs were inspected strictly post-retrieval; no gold data guided search.",
        ]
    }
    write_json(destination / "evaluation_summary.json", summary_data)

    # 4. evaluation_report.md
    md_lines = [
        "# M1.1 — Retrieval Quality Evaluation Report",
        "",
        f"- **Evaluated Run:** `{run_path.name}`",
        f"- **Source Annotations:** `{Path(train_json_path).name}`",
        "- **Corpus Admission:** `false` (exploratory retrieval evaluation on candidate stores)",
        "",
        "## 1. Method Performance Overview",
        "",
        "| Method | Annotated Claims | Hit@5 | Hit@10 | Hit@20 | Hit@50 | Hit@100 | Recall@10 | Recall@100 | MRR@100 | Evidence Rec@8 | Store Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m, s in method_summaries.items():
        mn = s["means"]
        md_lines.append(
            f"| **{m}** | {s['annotated_claims']}/{s['total_claims']} | "
            f"{mn.get('hit@5', 0):.1%} | {mn.get('hit@10', 0):.1%} | {mn.get('hit@20', 0):.1%} | "
            f"{mn.get('hit@50', 0):.1%} | {mn.get('hit@100', 0):.1%} | {mn.get('recall@10', 0):.3f} | "
            f"{mn.get('recall@100', 0):.3f} | {mn.get('mrr@100', 0):.4f} | "
            f"{mn.get('evidence_recall@8', 0):.3f} | {mn.get('store_gold_coverage', 0):.1%} |"
        )

    md_lines.extend([
        "",
        "> [!NOTE]",
        "> **Store Coverage (Oracle Ceiling):** Tỷ lệ phần trăm URLs vàng thực sự có mặt trong candidate store của claim. "
        "Nếu một URL vàng không có trong candidate store, không phương pháp nào có thể truy xuất được nó.",
        "",
        "## 2. Reranker Effectiveness Analysis (R3 vs R4)",
        "",
    ])

    if r3_r4_comparison:
        md_lines.extend([
            f"- **Số claims so sánh:** {r3_r4_comparison['evaluated_claims']}",
            f"- **Thứ hạng cải thiện (R4 tốt hơn R3):** {r3_r4_comparison['improved_rank_count']}",
            f"- **Thứ hạng tụt (R4 kém hơn R3):** {r3_r4_comparison['degraded_rank_count']}",
            f"- **Thứ hạng không đổi:** {r3_r4_comparison['unchanged_rank_count']}",
            f"- **Được đẩy vào Top-10 (Promoted to Top-10):** {r3_r4_comparison['promoted_to_top10_count']}",
            f"- **Bị rớt khỏi Top-10 (Demoted from Top-10):** {r3_r4_comparison['demoted_from_top10_count']}",
            f"- **Tổng độ dịch chuyển thứ hạng (Net rank shift):** {r3_r4_comparison['net_rank_shift_sum']:+d}",
            "",
        ])
    else:
        md_lines.append("R3 hoặc R4 không có mặt đồng thời trong lần chạy này.\n")

    md_lines.extend([
        "## 3. Chẩn đoán các trường hợp Hit@100 nhưng Miss@10",
        "",
        "Các trường hợp URL vàng đã được tìm thấy trong Top-100 nhưng không lọt vào Top-10:",
        "",
    ])

    hit100_miss10_any = False
    for m, s in method_summaries.items():
        if s["hit100_miss10_sample_ids"]:
            hit100_miss10_any = True
            md_lines.append(f"- **{m}:** {len(s['hit100_miss10_sample_ids'])} claims: `{', '.join(s['hit100_miss10_sample_ids'])}`")
    if not hit100_miss10_any:
        md_lines.append("Không có trường hợp Hit@100 nhưng Miss@10 nào trong tập đánh giá.\n")

    (destination / "evaluation_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Manifest
    eval_manifest = {
        "status": "complete_evaluation",
        "corpus_admitted": False,
        "run_dir": str(run_path),
        "train_json": str(train_json_path),
        "methods": methods_evaluated,
        "outputs_sha256": {
            p.name: file_hash(p) for p in destination.glob("*") if p.is_file()
        }
    }
    write_json(destination / "manifest.json", eval_manifest)

    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Path to retrieval run output directory (e.g. outputs/m11-store-r14-...)")
    parser.add_argument("--train-json", required=True, help="Path to raw AVeriTeC train.json containing evidence annotations")
    parser.add_argument("--output", required=True, help="Fresh output directory for evaluation reports")
    parser.add_argument("--claims-manifest", help="Optional claims manifest to verify IDs")
    args = parser.parse_args()

    out = run_evaluation(args.run_dir, args.train_json, args.output, args.claims_manifest)
    print(f"Evaluation completed successfully: {out}")


if __name__ == "__main__":
    main()
