"""Configurable evidence leakage and verdict-bearing content review.

Detects potential answer leakage, fact-checking verdict prose, known fact-check
domains/paths, scraping boilerplate, and irrelevant evidence. Operates strictly
label-free without accessing ground-truth QA or gold labels.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from tffv.data import read_jsonl
from tffv.provenance import canonical, digest, file_hash, write_json
from tffv.retrieval.data import normalize, tokens


POLICY_VERSION = "2.0.0"
DEFAULT_POLICY_ID = "evidence_leakage_review_v2"

DEFAULT_VERDICT_PATTERNS = [
    r"\b(?:verdict|fact[ -]?check(?:ing)?(?:\s+rating)?|gold[_ ]label)\s*[:=]",
    r"\b(?:our\s+)?(?:rating|verdict)\s*[:=]\s*(?:false|true|mostly false|mostly true|half true|pants on fire|misleading|unproven|unsupported|correct|incorrect|debunked)",
    r"\b(?:we\s+rate|rated\s+(?:this|the)\s+claim|the\s+claim\s+is\s+(?:false|true|mostly false|mostly true|debunked|misleading|unfounded|unsupported|untrue|bogus))\b",
    r"\b(?:claim|statement)\s*:\s*.{5,100}?\s*(?:check|verdict|rating)\s*:\s*(?:false|true|misleading|untrue)",
    r"\b(?:false|misleading|debunked|unsubstantiated|unfounded|bogus|fabricated)\s+claim\b",
    r"\b(?:debunking|fact-checking|reality check)\b",
    r"\b(?:four pinocchios|pants on fire)\b",
]

DEFAULT_FACT_CHECK_DOMAINS = [
    "politifact.com",
    "snopes.com",
    "fullfact.org",
    "factcheck.org",
    "africacheck.org",
    "leadstories.com",
    "checkyourfact.com",
    "climatefeedback.org",
    "healthfeedback.org",
    "polygraph.info",
    "rappler.com",
    "boomlive.in",
    "vishvasnews.com",
    "thequint.com",
    "newsmobile.in",
]

DEFAULT_FACT_CHECK_URL_PATHS = [
    "fact-check",
    "factcheck",
    "reality-check",
    "fact_check",
    "debunk",
    "fact-checker",
]

DEFAULT_BOILERPLATE_PATTERNS = [
    r"\b(?:cookie policy|terms of service|privacy policy|terms of use)\b",
    r"\b(?:enable javascript to view|javascript is disabled|please enable javascript)\b",
    r"\b(?:subscribe to continue reading|sign in to your account|access denied|page not found|404 not found)\b",
]

DEFAULT_POLICY = {
    "policy_id": DEFAULT_POLICY_ID,
    "version": POLICY_VERSION,
    "action": "flag_for_review",
    "verdict_patterns": DEFAULT_VERDICT_PATTERNS,
    "fact_check_domains": DEFAULT_FACT_CHECK_DOMAINS,
    "fact_check_url_paths": DEFAULT_FACT_CHECK_URL_PATHS,
    "boilerplate_patterns": DEFAULT_BOILERPLATE_PATTERNS,
    "claim_verdict_coupling": {
        "enabled": True,
        "verdict_terms": ["false", "untrue", "debunked", "misleading", "hoax", "fake", "incorrect", "unfounded", "bogus"],
        "window_words": 30,
    },
    "irrelevance_detection": {
        "enabled": True,
        "min_claim_token_overlap": 1,
    },
}

STOPWORDS = {"the", "a", "an", "in", "on", "of", "to", "for", "and", "or", "is", "are", "was", "were", "that", "this", "it", "as", "by", "at", "with", "from"}


def load_policy(policy_spec=None, action_override=None):
    if policy_spec is None:
        policy = dict(DEFAULT_POLICY)
    elif isinstance(policy_spec, dict):
        policy = dict(policy_spec)
    elif isinstance(policy_spec, (str, Path)):
        path = Path(policy_spec)
        if path.is_file():
            policy = json.loads(path.read_text(encoding="utf-8"))
        elif str(policy_spec) == DEFAULT_POLICY_ID:
            policy = dict(DEFAULT_POLICY)
        else:
            raise ValueError(f"Unknown policy specification: {policy_spec}")
    else:
        raise ValueError("Policy must be None, a dict, or a file path")
    required = {"policy_id", "version", "verdict_patterns", "fact_check_domains", "fact_check_url_paths", "boilerplate_patterns"}
    if not required <= set(policy):
        raise ValueError(f"Missing required policy fields: {required - set(policy)}")
    if action_override:
        if action_override not in {"flag_for_review", "quarantine"}:
            raise ValueError(f"Invalid action override: {action_override}")
        policy["action"] = action_override
    policy["compiled_verdict"] = [re.compile(p, re.IGNORECASE) for p in policy["verdict_patterns"]]
    policy["compiled_boilerplate"] = [re.compile(p, re.IGNORECASE) for p in policy["boilerplate_patterns"]]
    raw_for_digest = {k: v for k, v in policy.items() if not k.startswith("compiled_")}
    policy["policy_sha256"] = digest(raw_for_digest)
    return policy


def inspect_evidence_item(item, policy):
    """Review an evidence or document item for potential leakage/noise flags."""
    sample_id = item.get("sample_id", "unknown")
    claim_text = item.get("claim", "")
    passage_id = item.get("passage_id") or item.get("document_id", "unknown")
    text = item.get("text", "")
    source_url = item.get("source_url", "")

    # Reject gold fields if present
    forbidden = {"label", "gold_label", "justification", "questions", "answers"}
    found_forbidden = forbidden & set(item)
    if found_forbidden:
        raise ValueError(f"Forbidden evaluation gold fields detected in retrieval evidence: {found_forbidden}")

    flags = []
    reasons = []
    matched_patterns = []
    excerpt = None

    # 1. Verdict patterns
    for pattern, compiled in zip(policy["verdict_patterns"], policy["compiled_verdict"]):
        match = compiled.search(text)
        if match:
            flags.append("verdict_pattern")
            matched_str = match.group(0)
            matched_patterns.append(matched_str)
            reasons.append(f"Matched verdict pattern: {pattern}")
            start = max(0, match.start() - 80)
            end = min(len(text), match.end() + 120)
            excerpt = text[start:end]
            break

    # 2. Fact-check domains & URL paths
    if source_url:
        try:
            parsed = urlsplit(source_url)
            hostname = (parsed.hostname or "").casefold()
            path = (parsed.path or "").casefold()
            for domain in policy["fact_check_domains"]:
                if domain in hostname:
                    flags.append("fact_check_domain")
                    reasons.append(f"Source URL hostname matches known fact-check domain: {domain}")
                    matched_patterns.append(domain)
                    break
            for marker in policy["fact_check_url_paths"]:
                if marker in path:
                    flags.append("fact_check_url_path")
                    reasons.append(f"Source URL path matches fact-check marker: {marker}")
                    matched_patterns.append(marker)
                    break
        except Exception:
            pass

    # 3. Boilerplate noise
    for pattern, compiled in zip(policy["boilerplate_patterns"], policy["compiled_boilerplate"]):
        match = compiled.search(text)
        if match:
            flags.append("boilerplate_noise")
            matched_str = match.group(0)
            matched_patterns.append(matched_str)
            reasons.append(f"Matched scraping boilerplate: {pattern}")
            if not excerpt:
                start = max(0, match.start() - 60)
                end = min(len(text), match.end() + 60)
                excerpt = text[start:end]
            break

    # 4. Claim-verdict co-occurrence
    coupling_cfg = policy.get("claim_verdict_coupling", {})
    if coupling_cfg.get("enabled") and claim_text and text:
        claim_words = [w for w in tokens(claim_text) if len(w) >= 3 and w not in STOPWORDS]
        if len(claim_words) >= 2:
            text_words = tokens(text)
            verdict_terms = set(coupling_cfg.get("verdict_terms", []))
            window = coupling_cfg.get("window_words", 30)
            claim_word_set = set(claim_words)
            for idx, word in enumerate(text_words):
                if word in verdict_terms:
                    w_start = max(0, idx - window)
                    w_end = min(len(text_words), idx + window + 1)
                    surrounding = set(text_words[w_start:w_end])
                    overlap = surrounding & claim_word_set
                    if len(overlap) >= 2:
                        flags.append("claim_verdict_coupling")
                        reasons.append(f"Verdict term '{word}' co-occurs within {window} words of claim terms: {sorted(overlap)}")
                        matched_patterns.append(f"'{word}' near {list(overlap)}")
                        if not excerpt:
                            excerpt = " ".join(text_words[w_start:w_end])
                        break

    # 5. Irrelevance / zero lexical overlap
    irrel_cfg = policy.get("irrelevance_detection", {})
    if irrel_cfg.get("enabled") and claim_text and text:
        claim_content = {w for w in tokens(claim_text) if len(w) >= 3 and w not in STOPWORDS}
        text_content = {w for w in tokens(text) if len(w) >= 3 and w not in STOPWORDS}
        if claim_content and not (claim_content & text_content):
            flags.append("zero_claim_token_overlap")
            reasons.append("Passage has zero lexical overlap with claim content tokens")

    if not excerpt and text:
        excerpt = text[:200]

    disposition = "retained"
    if flags:
        disposition = "quarantined" if policy["action"] == "quarantine" else "flagged_for_review"

    return {
        "sample_id": sample_id,
        "passage_id": passage_id,
        "source_url": source_url,
        "flags": flags,
        "reasons": reasons,
        "matched_patterns": matched_patterns,
        "bounded_excerpt": excerpt,
        "text_sha256": digest(text) if text else None,
        "disposition": disposition,
    }


def load_items_to_review(input_path):
    """Load items from review queue, operational run directory, or candidate manifest."""
    input_path = Path(input_path).resolve()
    items = []
    input_type = None

    if input_path.is_file():
        if input_path.name == "manual_review_queue.jsonl":
            input_type = "manual_review_queue"
            for row in read_jsonl(input_path):
                # Format: {"sample_id": ..., "claim": ..., "evidence": {...}}
                ev = row.get("evidence", {})
                items.append({
                    "sample_id": row.get("sample_id"),
                    "claim": row.get("claim"),
                    "passage_id": ev.get("passage_id"),
                    "text": ev.get("text"),
                    "source_url": ev.get("source_url"),
                    "document_id": ev.get("document_id"),
                })
        elif input_path.name == "candidate_manifest.json":
            input_type = "candidate_manifest"
            manifest = json.loads(input_path.read_text(encoding="utf-8"))
            for store in manifest.get("stores", []):
                store_dir = input_path.parent / store["directory"]
                claims = read_jsonl(store_dir / "claims.jsonl")
                claim_text = claims[0]["claim"] if claims else ""
                corpus = read_jsonl(store_dir / "corpus.jsonl")
                for doc in corpus:
                    items.append({
                        "sample_id": store["sample_id"],
                        "claim": claim_text,
                        "passage_id": doc["document_id"],
                        "document_id": doc["document_id"],
                        "text": doc["text"],
                        "source_url": doc["source_url"],
                    })
        else:
            input_type = "generic_jsonl"
            for row in read_jsonl(input_path):
                items.append({
                    "sample_id": row.get("sample_id", "unknown"),
                    "claim": row.get("claim", ""),
                    "passage_id": row.get("passage_id") or row.get("document_id"),
                    "document_id": row.get("document_id"),
                    "text": row.get("text", ""),
                    "source_url": row.get("source_url", ""),
                })
    elif input_path.is_dir():
        if (input_path / "manual_review_queue.jsonl").is_file():
            return load_items_to_review(input_path / "manual_review_queue.jsonl")
        elif (input_path / "candidate_manifest.json").is_file():
            return load_items_to_review(input_path / "candidate_manifest.json")
        else:
            raise ValueError(f"Directory {input_path} lacks recognizable review queue or candidate manifest")
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    return items, input_type, file_hash(input_path)


def review_evidence(input_path, output_dir, policy_spec=None, action=None):
    items, input_type, input_hash = load_items_to_review(input_path)
    policy = load_policy(policy_spec, action_override=action)

    output_path = Path(output_dir).resolve()
    allowed_root = (Path(__file__).resolve().parents[3] / "outputs").resolve()
    if output_path == allowed_root or not output_path.is_relative_to(allowed_root):
        raise ValueError("Output must be a fresh subdirectory under outputs/")
    output_path.mkdir(parents=True, exist_ok=False)

    flag_counts = Counter()
    results = []
    flagged_entries = []
    quarantined_ledger = []

    for item in items:
        res = inspect_evidence_item(item, policy)
        for f in res["flags"]:
            flag_counts[f] += 1
        results.append(res)
        if res["flags"]:
            flagged_entries.append(res)
            if res["disposition"] == "quarantined":
                quarantined_ledger.append({
                    "sample_id": res["sample_id"],
                    "passage_id": res["passage_id"],
                    "source_url": res["source_url"],
                    "text_sha256": res["text_sha256"],
                    "flags": res["flags"],
                    "reasons": res["reasons"],
                    "policy_id": policy["policy_id"],
                    "policy_sha256": policy["policy_sha256"],
                })

    report = {
        "status": "complete_evidence_leakage_review",
        "input_path": str(Path(input_path).resolve()),
        "input_type": input_type,
        "input_sha256": input_hash,
        "policy_id": policy["policy_id"],
        "policy_version": policy["version"],
        "policy_sha256": policy["policy_sha256"],
        "action": policy["action"],
        "total_records_reviewed": len(items),
        "flagged_records_count": len(flagged_entries),
        "quarantined_records_count": len(quarantined_ledger),
        "flag_breakdown": dict(flag_counts),
        "tool_sha256": file_hash(__file__),
        "limitations": [
            "Heuristic flags are not proof that unflagged passages are free of answer leakage.",
            "Legal verdicts and news reporting on claims may produce false alarms requiring human review.",
            "Does not alter official splits or invent ground-truth relevance labels.",
            "Corpus admission remains false pending complete experimental verification.",
        ],
    }

    # Write JSON report
    write_json(output_path / "review_report.json", {**report, "records": results})

    # Write flagged evidence JSONL
    with (output_path / "flagged_evidence.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for entry in flagged_entries:
            stream.write(canonical(entry) + "\n")

    # Write quarantined ledger JSONL
    if quarantined_ledger:
        with (output_path / "quarantined_ledger.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for entry in quarantined_ledger:
                stream.write(canonical(entry) + "\n")

    # Write Markdown summary report
    md_lines = [
        f"# Evidence Leakage Review Report — {policy['policy_id']} (v{policy['version']})",
        "",
        f"- **Input:** `{Path(input_path).name}` (SHA256: `{input_hash[:16]}...`)",
        f"- **Action:** `{policy['action']}`",
        f"- **Total items reviewed:** {len(items)}",
        f"- **Flagged items:** {len(flagged_entries)} ({len(flagged_entries) / (len(items) or 1) * 100:.1f}%)",
        f"- **Quarantined items:** {len(quarantined_ledger)}",
        "",
        "## Flag Breakdown",
        "",
        "| Flag | Count | Description |",
        "|---|---:|---|",
    ]
    descriptions = {
        "verdict_pattern": "Explicit rating or verdict prose (e.g., 'Verdict:', 'The claim is false')",
        "fact_check_domain": "Hostname belongs to a known fact-checking organization",
        "fact_check_url_path": "URL path contains fact-checking markers (e.g., '/fact-check/', '/reality-check/')",
        "boilerplate_noise": "Scraping boilerplate (cookies, javascript disabled, 404, paywall)",
        "claim_verdict_coupling": "Verdict term appears in close proximity to claim content keywords",
        "zero_claim_token_overlap": "Zero content-word lexical overlap with the claim",
    }
    for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
        desc = descriptions.get(flag, "Heuristic review flag")
        md_lines.append(f"| `{flag}` | {count} | {desc} |")

    md_lines.extend([
        "",
        "## Flagged Records Sample",
        "",
        "| Claim ID | Passage / Doc ID | Flags | Source URL | Excerpt |",
        "|---|---|---|---|---|",
    ])
    for entry in flagged_entries[:25]:
        flags_str = ", ".join(f"`{f}`" for f in entry["flags"])
        url_str = entry["source_url"] or "N/A"
        excerpt_clean = (entry["bounded_excerpt"] or "").replace("\n", " ").replace("|", "/")[:120]
        md_lines.append(f"| `{entry['sample_id']}` | `{entry['passage_id'][:12]}` | {flags_str} | `{url_str}` | {excerpt_clean}... |")

    if len(flagged_entries) > 25:
        md_lines.append(f"\n*(Showing 25 of {len(flagged_entries)} flagged records; see `flagged_evidence.jsonl` for full list)*")

    md_lines.extend([
        "",
        "> [!IMPORTANT]",
        "> This audit is strictly label-free. Heuristic flags indicate high-risk evidence passages for human inspection.",
        "> They do not automatically clear unflagged passages of semantic answer leakage, nor do they certify corpus admission.",
        "",
    ])
    (output_path / "review_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    return output_path / "review_report.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to manual_review_queue.jsonl, run directory, or candidate_manifest.json")
    parser.add_argument("--output", required=True, help="Fresh output directory under outputs/")
    parser.add_argument("--policy", help="Path to custom policy JSON or policy ID")
    parser.add_argument("--quarantine", action="store_true", help="Set disposition to quarantined for flagged records")
    args = parser.parse_args()
    action = "quarantine" if args.quarantine else None
    result = review_evidence(args.input, args.output, policy_spec=args.policy, action=action)
    print(f"Review completed successfully: {result}")


if __name__ == "__main__":
    main()
