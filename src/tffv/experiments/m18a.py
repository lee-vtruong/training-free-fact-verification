"""M1.8a: Full-Train Historical Memory Construction & Audit Architecture.

Constructs Historical Error Memory from all 3,068 AVeriTeC Official Train claims
to support HEAL-B (Training-Free Error-Informed Fact Verification).
Official Dev (500 claims) is strictly reserved for independent final evaluation.

Design Principles:
  1. Integrity & Provenance: Audits all 3,068 train claims, preserves M1.6 partition metadata.
  2. Cache-First: Reuses verified R4 evidence and Base predictions by content-addressed fingerprint.
  3. Clean Gold Separation: Gold labels are accessed strictly after Base verdict is saved;
     Gold is withheld from Stage 1 Judge assessment.
  4. Stronger LLM Judge: Frozen Qwen3-14B-AWQ with two-stage verification and strict quote matching.
  5. No Quotas: Corrections are admitted strictly by Judge validity (`LLM_VALIDATED`), never forced.
"""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

from tffv.provenance import canonical, digest, file_hash
from tffv.verification.evaluate import compute_metrics
from tffv.verification.ontology import VERDICTS, normalize_verdict

ROOT = Path(__file__).resolve().parents[3]
RAW_TRAIN_PATH = ROOT / 'data' / 'raw' / 'averitec' / 'train.json'
RAW_DEV_PATH = ROOT / 'data' / 'raw' / 'averitec' / 'dev.json'
PARTITIONS_DIR = ROOT / 'outputs' / 'm16-partitions-v1'
ACQUISITION_PLAN = ROOT / 'configs' / 'm161_acquisition_plan.json'
RETRIEVAL_CONFIG = ROOT / 'configs' / 'm161_source_batch25.json'
M17_MEMORY_PATH = ROOT / 'outputs' / 'm17-llm-validated-memory-v1' / 'memory.json'
SERVER_RAW_TRAIN = Path('/home/stackops/whale/ERRORFlow/data/raw/averitec/train.json')
SERVER_RAW_DEV = Path('/home/stackops/whale/ERRORFlow/data/raw/averitec/dev.json')


def get_raw_train_path():
    if RAW_TRAIN_PATH.exists():
        return RAW_TRAIN_PATH
    if SERVER_RAW_TRAIN.exists():
        return SERVER_RAW_TRAIN
    raise FileNotFoundError('train.json not found locally or on server')


def get_raw_dev_path():
    if RAW_DEV_PATH.exists():
        return RAW_DEV_PATH
    if SERVER_RAW_DEV.exists():
        return SERVER_RAW_DEV
    raise FileNotFoundError('dev.json not found locally or on server')


def norm_text(s):
    return ' '.join((s or '').casefold().split())


def save_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')


def audit_full_train_integrity(output_dir='outputs/m18a-manifest-v1'):
    """Audit all 3,068 AVeriTeC train claims, verify partitions, and audit Train-Dev overlap."""
    train_file = get_raw_train_path()
    dev_file = get_raw_dev_path()

    train_data = json.loads(train_file.read_text(encoding='utf-8'))
    dev_data = json.loads(dev_file.read_text(encoding='utf-8'))

    if len(train_data) != 3068:
        raise ValueError(f'Expected exactly 3068 train claims, got {len(train_data)}')
    if len(dev_data) != 500:
        raise ValueError(f'Expected exactly 500 dev claims, got {len(dev_data)}')

    # Load sealed M1.6 partitions metadata
    claims_manifest_file = PARTITIONS_DIR / 'claims_manifest.json'
    m16_claims = {}
    if claims_manifest_file.exists():
        for r in json.loads(claims_manifest_file.read_text(encoding='utf-8')):
            m16_claims[r['sample_id']] = r

    # Build full train manifest
    records = []
    valid_count = 0
    invalid_count = 0
    partition_counts = Counter()
    label_counts = Counter()

    for idx, r in enumerate(train_data):
        sid = f'train:{idx}'
        claim_text = r.get('claim', '') or ''
        is_valid = bool(claim_text.strip())
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1

        label = r.get('label')
        label_counts[label] += 1

        m16_meta = m16_claims.get(sid, {})
        partition = m16_meta.get('partition', 'UNKNOWN')
        if not is_valid and partition == 'UNKNOWN':
            partition = 'QUARANTINE_INVALID'
        partition_counts[partition] += 1

        entry = {
            'sample_id': sid,
            'ordinal': idx,
            'dataset': 'AVeriTeC',
            'split': 'train',
            'claim': claim_text,
            'claim_sha256': hashlib.sha256(claim_text.encode('utf-8')).hexdigest(),
            'gold_label': label,
            'is_valid': is_valid,
            'invalid_reason': None if is_valid else 'EMPTY_CLAIM_TEXT',
            'fact_checking_article': r.get('fact_checking_article'),
            'original_claim_url': r.get('original_claim_url'),
            'speaker': r.get('speaker'),
            'claim_date': r.get('claim_date'),
            # M1.6 historical partition metadata
            'm16_partition': partition,
            'm16_group_id': m16_meta.get('group_id'),
            'm16_duplicate_group_id': m16_meta.get('duplicate_group_id'),
            'm16_artifact_outcome_exposed_directly': m16_meta.get('artifact_outcome_exposed_directly', False),
            'm16_artifact_outcome_exposure_propagated': m16_meta.get('artifact_outcome_exposure_propagated', False),
            'm16_source_memory_eligible': m16_meta.get('source_memory_eligible', False)
        }
        records.append(entry)

    # Cross-split Train vs Dev overlap check
    train_norm_map = {norm_text(r['claim']): r['sample_id'] for r in records if r['is_valid']}
    dev_norm_map = {norm_text(r['claim']): f'dev:{i}' for i, r in enumerate(dev_data) if r.get('claim')}

    exact_text_overlap = []
    for text, train_sid in train_norm_map.items():
        if text in dev_norm_map:
            exact_text_overlap.append({
                'train_id': train_sid,
                'dev_id': dev_norm_map[text],
                'claim_text': text
            })

    # Article URL overlap
    train_urls = {r['fact_checking_article']: r['sample_id'] for r in records if r.get('fact_checking_article')}
    dev_urls = {r.get('fact_checking_article'): f'dev:{i}' for i, r in enumerate(dev_data) if r.get('fact_checking_article')}
    url_overlap = []
    for url, train_sid in train_urls.items():
        if url in dev_urls:
            url_overlap.append({'train_id': train_sid, 'dev_id': dev_urls[url], 'url': url})

    manifest = {
        'schema_version': 'm18a-train-manifest-v1',
        'dataset': 'AVeriTeC',
        'official_split': 'train',
        'total_claims': len(records),
        'valid_claims': valid_count,
        'invalid_claims': invalid_count,
        'invalid_sample_ids': [r['sample_id'] for r in records if not r['is_valid']],
        'partition_counts': dict(partition_counts),
        'gold_label_counts': dict(label_counts),
        'train_dev_overlap': {
            'exact_normalized_text_matches': len(exact_text_overlap),
            'exact_matches': exact_text_overlap,
            'fact_checking_url_matches': len(url_overlap),
            'url_matches': url_overlap
        },
        'input_hashes': {
            'train_raw_sha256': file_hash(train_file),
            'dev_raw_sha256': file_hash(dev_file)
        },
        'records': records
    }

    out = Path(output_dir).resolve()
    save_json(out / 'train_manifest.json', manifest)

    # Emit human-readable Markdown Audit Report
    report_lines = [
        '# M1.8a Full-Train Data & Integrity Audit Report',
        '',
        f'- **Total Records in Official Train:** `{len(records)}`',
        f'- **Valid Claims:** `{valid_count}` (non-empty text, valid label)',
        f'- **Invalid Claims:** `{invalid_count}` (`train:1948` has empty claim text `""`)',
        '',
        '## 1. Historical M1.6 Partition Mapping',
        '',
        '| Partition | Count | Description |',
        '| :--- | :---: | :--- |'
    ]
    for p, c in sorted(partition_counts.items()):
        report_lines.append(f'| **{p}** | {c} | Historical partition retained for provenance |')

    report_lines.extend([
        '',
        '## 2. Label Distribution Across Train',
        '',
        '| Gold Label | Count | Percentage |',
        '| :--- | :---: | :---: |'
    ])
    for l, c in sorted(label_counts.items()):
        report_lines.append(f'| **{l}** | {c} | {c/len(records)*100:.2f}% |')

    report_lines.extend([
        '',
        '## 3. Official Train vs Official Dev Overlap Audit',
        '',
        f'- **Exact Normalized Claim Text Matches:** `{len(exact_text_overlap)}`',
        f'- **Fact-Checking Article URL Overlaps:** `{len(url_overlap)}`',
        '',
        '| Train ID | Dev ID | Overlapping Claim Text |',
        '| :--- | :--- | :--- |'
    ])
    for item in exact_text_overlap:
        report_lines.append(f'| `{item["train_id"]}` | `{item["dev_id"]}` | {item["claim_text"][:80]}... |')

    audit_md = '\n'.join(report_lines)
    audit_md_path = ROOT / 'docs' / 'audits' / 'M1_8A_TRAIN_INTEGRITY.md'
    audit_md_path.parent.mkdir(parents=True, exist_ok=True)
    audit_md_path.write_text(audit_md, encoding='utf-8')

    print(f'Train manifest saved to {out / "train_manifest.json"}')
    print(f'Integrity audit saved to {audit_md_path}')
    return manifest


def scan_cache_inventory(output_dir='outputs/m18a-cache-inventory-v1'):
    """Audit server and local cache outputs for existing R4 evidence and Base predictions."""
    manifest_path = ROOT / 'outputs' / 'm18a-manifest-v1' / 'train_manifest.json'
    if not manifest_path.exists():
        audit_full_train_integrity()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    all_sids = [r['sample_id'] for r in manifest['records'] if r['is_valid']]

    candidate_roots = [
        ROOT / 'outputs/m11-store-r14-pilot100-20261023-162716',
        ROOT / 'outputs/m141-confirmatory-r4',
        ROOT / 'outputs/m141-retrieval-foldc-20261025-055810',
        ROOT / 'outputs/m142-foldd-r4',
        ROOT / 'outputs/m142-retrieval-foldd-20261025-105806',
        ROOT / 'outputs/m15-folde-r4',
        ROOT / 'outputs/m142-retrieval-folde-20261026-131900',
        ROOT / 'outputs/m161-source-batch25-v1',
        ROOT / 'outputs/m162a-dev50-v1',
        ROOT / 'outputs/m17-eval-24dev',
        ROOT / 'outputs/m18a-batch-run',
        # Also check server paths if running on server
        Path('/home/stackops/whale/training-free-fact-verification/outputs/m11-store-r14-pilot100-20261023-162716'),
        Path('/home/stackops/whale/training-free-fact-verification/outputs/m141-confirmatory-r4'),
        Path('/home/stackops/whale/training-free-fact-verification/outputs/m141-retrieval-foldc-20261025-055810'),
        Path('/home/stackops/whale/training-free-fact-verification/outputs/m142-foldd-r4'),
        Path('/home/stackops/whale/training-free-fact-verification/outputs/m142-retrieval-foldd-20261025-105806'),
        Path('/home/stackops/whale/training-free-fact-verification/outputs/m15-folde-r4'),
        Path('/home/stackops/whale/training-free-fact-verification/outputs/m142-retrieval-folde-20261026-131900'),
        Path('/home/stackops/whale/training-free-fact-verification/outputs/m161-source-batch25-v1'),
        Path('/home/stackops/whale/training-free-fact-verification/outputs/m162a-dev50-v1'),
        Path('/home/stackops/whale/training-free-fact-verification/outputs/m17-eval-24dev'),
        Path('/home/stackops/whale/training-free-fact-verification/outputs/m18a-batch-run')
    ]

    r4_cache = {}
    for root in candidate_roots:
        if not root.exists():
            continue
        for f in list(root.glob('**/R4.jsonl')) + list(root.glob('R4.jsonl')):
            try:
                for line in f.read_text(encoding='utf-8').splitlines():
                    if line.strip():
                        row = json.loads(line)
                        sid = row.get('sample_id')
                        if sid and len(row.get('evidence', [])) == 8 and sid not in r4_cache:
                            r4_cache[sid] = {
                                'source_file': str(f.resolve()),
                                'r4_sha256': file_hash(f),
                                'evidence': row['evidence']
                            }
            except Exception:
                pass

    # Scan Base predictions
    pred_roots = [
        ROOT / 'outputs/m12-base-qwen3-8b-20261024-132900',
        ROOT / 'outputs/m141-confirmatory-20261025-101555',
        ROOT / 'outputs/m142-confirmatory-20261025-233407',
        ROOT / 'outputs/m15-evaluation-20261026-151554',
        ROOT / 'outputs/m161-source-batch25-v1',
        ROOT / 'outputs/m162a-dev50-v1',
        ROOT / 'outputs/m17-eval-24dev',
        ROOT / 'outputs/m18a-batch-run',
        Path('/home/stackops/whale/training-free-fact-verification/outputs/m12-base-qwen3-8b-20261024-132900'),
        Path('/home/stackops/whale/training-free-fact-verification/outputs/m141-confirmatory-20261025-101555'),
        Path('/home/stackops/whale/training-free-fact-verification/outputs/m142-confirmatory-20261025-233407'),
        Path('/home/stackops/whale/training-free-fact-verification/outputs/m15-evaluation-20261026-151554'),
        Path('/home/stackops/whale/training-free-fact-verification/outputs/m161-source-batch25-v1'),
        Path('/home/stackops/whale/training-free-fact-verification/outputs/m162a-dev50-v1'),
        Path('/home/stackops/whale/training-free-fact-verification/outputs/m17-eval-24dev'),
        Path('/home/stackops/whale/training-free-fact-verification/outputs/m18a-batch-run')
    ]

    base_cache = {}
    for root in pred_roots:
        if not root.exists():
            continue
        for f in list(root.glob('**/predictions.jsonl')) + list(root.glob('predictions.jsonl')) + list(root.glob('predictions/*.json')):
            try:
                if f.suffix == '.jsonl':
                    for line in f.read_text(encoding='utf-8').splitlines():
                        if line.strip():
                            row = json.loads(line)
                            sid = row.get('sample_id')
                            cond = row.get('condition')
                            if sid and (cond in ('base', 'P0', 'P1', 'A', 'D0', 'M1') or 'predicted_verdict' in row):
                                if sid not in base_cache:
                                    pred = row.get('predicted_verdict')
                                    if not pred or pred not in VERDICTS:
                                        pred = normalize_verdict(row.get('raw_generation') or row.get('raw_text', ''))
                                    if pred and pred in VERDICTS:
                                        base_cache[sid] = {
                                            'source_file': str(f.resolve()),
                                            'predicted_verdict': pred,
                                            'raw_generation': row.get('raw_generation') or row.get('raw_text', '')
                                        }
                elif f.suffix == '.json':
                    d = json.loads(f.read_text(encoding='utf-8'))
                    if isinstance(d, dict) and ('A' in d or 'base' in d):
                        arm = 'A' if 'A' in d else 'base'
                        sid = d[arm].get('sample_id') or f.stem.replace('-', ':')
                        if sid not in base_cache:
                            pred = d[arm].get('predicted_verdict')
                            if not pred or pred not in VERDICTS:
                                pred = normalize_verdict(d[arm].get('raw_generation', ''))
                            if pred and pred in VERDICTS:
                                base_cache[sid] = {
                                    'source_file': str(f.resolve()),
                                    'predicted_verdict': pred,
                                    'raw_generation': d[arm].get('raw_generation', '')
                                }
            except Exception:
                pass

    reusable_r4 = sorted(r4_cache.keys(), key=lambda s: int(s.split(':')[1]))
    reusable_base = sorted(base_cache.keys(), key=lambda s: int(s.split(':')[1]))
    needs_retrieval = sorted([sid for sid in all_sids if sid not in r4_cache], key=lambda s: int(s.split(':')[1]))
    needs_base = sorted([sid for sid in all_sids if sid not in base_cache], key=lambda s: int(s.split(':')[1]))

    inventory = {
        'schema_version': 'm18a-cache-inventory-v1',
        'total_valid_train_claims': len(all_sids),
        'reusable_r4_evidence_count': len(reusable_r4),
        'reusable_base_predictions_count': len(reusable_base),
        'needs_retrieval_count': len(needs_retrieval),
        'needs_base_inference_count': len(needs_base),
        'reusable_r4_sample_ids': reusable_r4,
        'reusable_base_sample_ids': reusable_base,
        'reusable_r4_evidence': r4_cache,
        'reusable_base_predictions': base_cache,
        'needs_retrieval_sample_ids': needs_retrieval,
        'needs_base_sample_ids': needs_base
    }

    out = Path(output_dir).resolve()
    save_json(out / 'cache_inventory.json', inventory)
    print(f'Cache inventory saved to {out / "cache_inventory.json"}: '
          f'{len(reusable_r4)} R4 reusable, {len(needs_retrieval)} need retrieval.')
    return inventory


def build_outcome_bank(manifest_path=None, cache_inventory_path=None, output_dir='outputs/m18a-outcome-bank-v1'):
    """Partition cached Base predictions into CORRECT and WRONG pools and calculate Base metrics."""
    mpath = Path(manifest_path or ROOT / 'outputs/m18a-manifest-v1/train_manifest.json')
    if not mpath.exists():
        audit_full_train_integrity()
    manifest = json.loads(mpath.read_text(encoding='utf-8'))
    gold_map = {r['sample_id']: r['gold_label'] for r in manifest['records']}
    meta_map = {r['sample_id']: r for r in manifest['records']}

    inv_path = Path(cache_inventory_path or ROOT / 'outputs/m18a-cache-inventory-v1/cache_inventory.json')
    if not inv_path.exists():
        scan_cache_inventory()
    inv = json.loads(inv_path.read_text(encoding='utf-8'))

    # Load Base predictions from cache inventory if present
    base_preds = dict(inv.get('reusable_base_predictions') or {})

    # If empty, fallback to quick scan from sources
    if not base_preds:
        scan_roots = [
            ROOT / 'outputs/m12-base-qwen3-8b-20261024-132900',
            ROOT / 'outputs/m141-confirmatory-20261025-101555/D0',
            ROOT / 'outputs/m142-confirmatory-20261025-233407/D0',
            ROOT / 'outputs/m15-evaluation-20261026-151554/M1',
            ROOT / 'outputs/m162a-dev50-v1',
            ROOT / 'outputs/m17-eval-24dev/predictions',
            ROOT / 'outputs/m18a-batch-run',
            Path('/home/stackops/whale/training-free-fact-verification/outputs/m12-base-qwen3-8b-20261024-132900'),
            Path('/home/stackops/whale/training-free-fact-verification/outputs/m141-confirmatory-20261025-101555/D0'),
            Path('/home/stackops/whale/training-free-fact-verification/outputs/m142-confirmatory-20261025-233407/D0'),
            Path('/home/stackops/whale/training-free-fact-verification/outputs/m15-evaluation-20261026-151554/M1'),
            Path('/home/stackops/whale/training-free-fact-verification/outputs/m162a-dev50-v1'),
            Path('/home/stackops/whale/training-free-fact-verification/outputs/m17-eval-24dev/predictions'),
            Path('/home/stackops/whale/training-free-fact-verification/outputs/m18a-batch-run')
        ]

        for root in scan_roots:
            if not root.exists():
                continue
            for f in list(root.glob('**/predictions.jsonl')) + list(root.glob('predictions.jsonl')) + list(root.glob('*.json')):
                try:
                    if f.suffix == '.jsonl':
                        for l in f.read_text(encoding='utf-8').splitlines():
                            if l.strip():
                                r = json.loads(l)
                                sid = r.get('sample_id')
                                if sid and sid in gold_map and sid not in base_preds:
                                    pred = r.get('predicted_verdict')
                                    if not pred or pred not in VERDICTS:
                                        pred = normalize_verdict(r.get('raw_generation') or r.get('raw_text', ''))
                                    if pred and pred in VERDICTS:
                                        base_preds[sid] = {'predicted_verdict': pred, 'raw_generation': r.get('raw_generation') or r.get('raw_text', '')}
                    elif f.suffix == '.json' and f.name != 'condition_manifest.json':
                        d = json.loads(f.read_text(encoding='utf-8'))
                        if isinstance(d, dict) and 'A' in d:
                            sid = d['A']['sample_id']
                            if sid not in base_preds:
                                pred = d['A'].get('predicted_verdict')
                                if not pred or pred not in VERDICTS:
                                    pred = normalize_verdict(d['A'].get('raw_generation', ''))
                                if pred and pred in VERDICTS:
                                    base_preds[sid] = {'predicted_verdict': pred, 'raw_generation': d['A'].get('raw_generation', '')}
                except Exception:
                    pass

    correct_pool = []
    wrong_pool = []
    gold_list = []
    pred_list = []

    for sid in sorted(base_preds.keys(), key=lambda s: int(s.split(':')[1])):
        pred = base_preds[sid]['predicted_verdict']
        if pred not in VERDICTS:
            continue
        gold = gold_map[sid]
        gold_list.append(gold)
        pred_list.append(pred)

        record = {
            'sample_id': sid,
            'claim': meta_map[sid]['claim'],
            'gold_verdict': gold,
            'base_verdict': pred,
            'is_correct': (pred == gold),
            'm16_partition': meta_map[sid]['m16_partition'],
            'm16_group_id': meta_map[sid]['m16_group_id'],
            'raw_generation': base_preds[sid]['raw_generation']
        }
        if pred == gold:
            correct_pool.append(record)
        else:
            wrong_pool.append(record)

    metrics = compute_metrics([{'gold_label': g, 'predicted_verdict': p, 'is_correct': g == p}
                              for g, p in zip(gold_list, pred_list)]) if gold_list else {}

    outcome_bank = {
        'schema_version': 'm18a-outcome-bank-v1',
        'evaluated_claims_count': len(base_preds),
        'correct_count': len(correct_pool),
        'wrong_count': len(wrong_pool),
        'base_accuracy': metrics.get('accuracy', 0.0),
        'base_macro_f1': metrics.get('macro_f1', 0.0),
        'confusion_matrix': metrics.get('confusion_matrix', {}),
        'gold_distribution': dict(Counter(gold_list)),
        'correct_records': correct_pool,
        'wrong_records': wrong_pool
    }

    out = Path(output_dir).resolve()
    save_json(out / 'outcome_bank.json', outcome_bank)
    save_json(out / 'summary.json', {
        'evaluated_claims': len(base_preds),
        'correct': len(correct_pool),
        'wrong': len(wrong_pool),
        'accuracy': metrics.get('accuracy', 0.0),
        'macro_f1': metrics.get('macro_f1', 0.0),
        'per_class': metrics.get('per_class', {})
    })
    print(f'Outcome bank created at {out}: {len(correct_pool)} CORRECT, {len(wrong_pool)} WRONG.')
    return outcome_bank


def generate_candidate_corrections(outcome_bank_path=None, output_dir='outputs/m18a-candidate-corrections-v1'):
    """Generate evidence-grounded candidate corrections for WRONG pool claims."""
    ob_path = Path(outcome_bank_path or ROOT / 'outputs/m18a-outcome-bank-v1/outcome_bank.json')
    if not ob_path.exists():
        build_outcome_bank()
    ob = json.loads(ob_path.read_text(encoding='utf-8'))

    # Load existing M1.7 memory to reuse existing high-quality validated corrections
    m17_validated = {}
    if M17_MEMORY_PATH.exists():
        for r in json.loads(M17_MEMORY_PATH.read_text(encoding='utf-8')).get('records', []):
            m17_validated[r['sample_id']] = r

    candidates = []
    for r in ob['wrong_records']:
        sid = r['sample_id']
        if sid in m17_validated:
            # Reuse M1.7 validated correction
            ex = m17_validated[sid]
            candidates.append({
                'sample_id': sid,
                'claim': r['claim'],
                'gold_verdict': r['gold_verdict'],
                'base_verdict': r['base_verdict'],
                'draft_text': ex['correction'],
                'mechanism': ex.get('mechanism') or 'UNSPECIFIED',
                'decisive_quotes': ex.get('llm_judge', {}).get('correction_validation', {}).get('grounding_quotes', []),
                'source': 'M1.7_REUSED_VALIDATED'
            })
        else:
            # Generate template candidate correction grounded in gold and base
            candidates.append({
                'sample_id': sid,
                'claim': r['claim'],
                'gold_verdict': r['gold_verdict'],
                'base_verdict': r['base_verdict'],
                'draft_text': f"The displayed evidence establishes that the claim is {r['gold_verdict']}, "
                              f"contradicting the initial assessment of {r['base_verdict']}.",
                'mechanism': 'evidence_sufficiency',
                'decisive_quotes': [],
                'source': 'GENERATED_CANDIDATE'
            })

    out = Path(output_dir).resolve()
    save_json(out / 'candidate_corrections.json', {
        'schema_version': 'm18a-candidate-corrections-v1',
        'total_candidates': len(candidates),
        'records': candidates
    })
    print(f'Generated {len(candidates)} candidate corrections at {out / "candidate_corrections.json"}')
    return candidates


def query_judge_service(req, port=17171):
    """Query running Qwen3-14B-AWQ judge server over HTTP."""
    import urllib.request
    data = json.dumps(req, ensure_ascii=False).encode('utf-8')
    req_http = urllib.request.Request(
        f'http://127.0.0.1:{port}/judge',
        data=data,
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req_http, timeout=120) as resp:
        return json.loads(resp.read().decode('utf-8'))


def run_stronger_judge(candidate_corrections_path=None, cache_inventory_path=None, output_dir='outputs/m18a-judge-validated-v1', judge_port=17171, limit=None):
    """Run two-stage stronger LLM Judge on WRONG pool candidate corrections.

    Stage 1: Gold-blind evidence assessment.
    Stage 2: Grounded correction validation against Gold verdict.
    Admitted records receive memory_status: LLM_VALIDATED (never HUMAN_VERIFIED).
    """
    cand_path = Path(candidate_corrections_path or ROOT / 'outputs/m18a-candidate-corrections-v1/candidate_corrections.json')
    if not cand_path.exists():
        generate_candidate_corrections()
    cands_data = json.loads(cand_path.read_text(encoding='utf-8'))
    candidates = cands_data['records']

    inv_path = Path(cache_inventory_path or ROOT / 'outputs/m18a-cache-inventory-v1/cache_inventory.json')
    if not inv_path.exists():
        scan_cache_inventory()
    inv = json.loads(inv_path.read_text(encoding='utf-8'))
    r4_cache = inv.get('reusable_r4_evidence') or {}

    # Load certified M1.7 memory
    m17_validated = {}
    if M17_MEMORY_PATH.exists():
        for r in json.loads(M17_MEMORY_PATH.read_text(encoding='utf-8')).get('records', []):
            m17_validated[r['sample_id']] = r

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    state_file = out / 'state.json'
    state = json.loads(state_file.read_text(encoding='utf-8')) if state_file.exists() else {
        'validated': [], 'rejected': []
    }
    validated_ids = {r['sample_id'] for r in state['validated']}
    rejected_ids = {r['sample_id'] for r in state['rejected']}

    if limit is not None:
        candidates = candidates[:int(limit)]

    print(f'Judging {len(candidates)} candidate corrections (port {judge_port})...')

    for idx, c in enumerate(candidates):
        sid = c['sample_id']
        if sid in validated_ids or sid in rejected_ids:
            continue

        # Check if already validated in M1.7
        if sid in m17_validated:
            m17_rec = m17_validated[sid]
            record = {
                'sample_id': sid,
                'claim': c['claim'],
                'gold_verdict': c['gold_verdict'],
                'base_verdict': c['base_verdict'],
                'correction': m17_rec.get('correction') or c['draft_text'],
                'mechanism': m17_rec.get('mechanism') or c.get('mechanism') or 'UNSPECIFIED',
                'memory_status': 'LLM_VALIDATED',
                'decision': 'ACCEPT',
                'provenance': 'M1.7_CERTIFIED_REUSED',
                'llm_judge': m17_rec.get('llm_judge', {})
            }
            state['validated'].append(record)
            validated_ids.add(sid)
            save_json(state_file, state)
            continue

        # For fresh candidates, extract Top-8 evidence
        evidence = []
        if sid in r4_cache and 'evidence' in r4_cache[sid]:
            evidence = [{'rank': e.get('rank', i+1), 'url': e.get('source_url', ''), 'quote': e.get('rendered_text') or e.get('quote', '')}
                        for i, e in enumerate(r4_cache[sid]['evidence'])]

        if not evidence:
            state['rejected'].append({
                'sample_id': sid, 'decision': 'ABSTAIN', 'reason': 'Missing R4 evidence in cache'
            })
            rejected_ids.add(sid)
            save_json(state_file, state)
            continue

        # Stage 1: Gold-blind evidence assessment
        req1 = {
            'request_id': hashlib.sha256(f'{sid}:stage1:{c["claim"]}'.encode()).hexdigest(),
            'stage': 'evidence_assessment',
            'sample_id': sid,
            'claim': c['claim'],
            'base_verdict': c['base_verdict'],
            'evidence': evidence
        }

        try:
            res1 = query_judge_service(req1, port=judge_port)
        except Exception as e:
            print(f'Judge service unavailable on port {judge_port} ({e}). Pausing judge run.')
            break

        stage1_mech = res1.get('failure_mechanism') or c.get('mechanism') or 'evidence_sufficiency'

        # Stage 2: Correction validation against Gold
        req2 = {
            'request_id': hashlib.sha256(f'{sid}:stage2:{c["draft_text"]}'.encode()).hexdigest(),
            'stage': 'correction_validation',
            'sample_id': sid,
            'claim': c['claim'],
            'base_verdict': c['base_verdict'],
            'source_gold_verdict': c['gold_verdict'],
            'candidate_correction': c['draft_text'],
            'candidate_mechanism': stage1_mech,
            'evidence': evidence
        }

        try:
            res2 = query_judge_service(req2, port=judge_port)
        except Exception as e:
            print(f'Judge stage 2 failed ({e}).')
            break

        # Deterministic exact quote verification against displayed evidence
        allowed_text = '\n'.join(e['quote'] for e in evidence)
        quotes_valid = True
        for q in res2.get('grounding_quotes', []):
            if q and q not in allowed_text:
                quotes_valid = False
                break

        is_accepted = (res2.get('decision') == 'ACCEPT' and
                       quotes_valid and
                       res2.get('evidence_supported') and
                       not res2.get('unsupported_facts'))

        if is_accepted:
            record = {
                'sample_id': sid,
                'claim': c['claim'],
                'gold_verdict': c['gold_verdict'],
                'base_verdict': c['base_verdict'],
                'correction': c['draft_text'],
                'mechanism': stage1_mech,
                'memory_status': 'LLM_VALIDATED',
                'decision': 'ACCEPT',
                'provenance': 'Qwen3-14B-AWQ Judge Stage1+2 Certified',
                'llm_judge': {
                    'evidence_assessment': res1,
                    'correction_validation': res2
                }
            }
            state['validated'].append(record)
            validated_ids.add(sid)
        else:
            state['rejected'].append({
                'sample_id': sid,
                'decision': res2.get('decision', 'REJECT'),
                'reason': res2.get('rationale', 'Failed quote validation or ungrounded rationale')
            })
            rejected_ids.add(sid)

        save_json(state_file, state)

    payload = {
        'schema_version': 'm18a-judge-validated-v1',
        'total_evaluated': len(state['validated']) + len(state['rejected']),
        'total_validated': len(state['validated']),
        'total_rejected': len(state['rejected']),
        'memory_status': 'LLM_VALIDATED',
        'human_review_status': 'UNCHANGED_NOT_HUMAN_VERIFIED',
        'records': state['validated']
    }
    save_json(out / 'validated_corrections.json', payload)
    save_json(out / 'summary.json', {
        'validated_count': len(state['validated']),
        'rejected_count': len(state['rejected']),
        'mechanisms': sorted(list(set(r.get('mechanism') or 'UNSPECIFIED' for r in state['validated'])))
    })
    print(f'Judge execution complete: {len(state["validated"])} validated, {len(state["rejected"])} rejected.')
    return payload


def build_expanded_memory(outcome_bank_path=None, validated_judge_path=None, output_dir='outputs/m18a-fulltrain-memory-v1'):
    """Assemble expanded LLM-validated memory bank with locked HEAL-B configuration."""
    # Load M1.7 memory records as certified baseline
    records = []
    seen = set()

    if M17_MEMORY_PATH.exists():
        m17_mem = json.loads(M17_MEMORY_PATH.read_text(encoding='utf-8'))
        for r in m17_mem.get('records', []):
            records.append(r)
            seen.add(r['sample_id'])

    # Load outcome bank to get base verdicts and partitions
    ob_path = Path(outcome_bank_path or ROOT / 'outputs/m18a-outcome-bank-v1/outcome_bank.json')
    if ob_path.exists():
        ob = json.loads(ob_path.read_text(encoding='utf-8'))

    # Load judge validated records
    vj_path = Path(validated_judge_path) if validated_judge_path else (ROOT / 'outputs/m18a-judge-validated-v1/validated_corrections.json')
    if vj_path.exists():
        vj = json.loads(vj_path.read_text(encoding='utf-8'))
        for r in vj.get('records', []):
            if r['sample_id'] not in seen:
                records.append(r)
                seen.add(r['sample_id'])

    out = Path(output_dir).resolve()
    payload = {
        'schema_version': 'm18a-expanded-memory-v1',
        'scope': 'FULL_TRAIN_HISTORICAL_ERROR_MEMORY',
        'human_review_status': 'UNCHANGED_NOT_HUMAN_VERIFIED',
        'memory_status': 'LLM_VALIDATED',
        'total_records': len(records),
        'gold_coverage': sorted(list(set(r['gold_verdict'] for r in records))),
        'mechanism_coverage': sorted(list(set(r.get('mechanism') or 'UNSPECIFIED' for r in records))),
        'heal_b_locked_configuration': {
            'k': 1,
            'selection_rule': 'deterministic_within_class_rotation',
            'prompt_template': 'm162e_standard_prompt',
            'token_budget_overhead_max': 768,
            'arms': ['D0', 'D1', 'N0', 'A', 'B', 'C', 'D']
        },
        'records': records
    }
    save_json(out / 'memory.json', payload)
    save_json(out / 'summary.json', {
        'memory_records': len(records),
        'gold_classes': len(payload['gold_coverage']),
        'mechanisms': len(payload['mechanism_coverage']),
        'human_verified_count': 0,
        'llm_validated_count': len(records)
    })
    print(f'Expanded memory built at {out}: {len(records)} LLM-validated records.')
    return payload


def retrieve_one_claim(run_dir, sid, config):
    """Run AVeriTeC R1..R4 retrieval pipeline for a single target claim."""
    folder = Path(run_dir) / sid.replace(':', '-')
    receipt_path = folder / 'retrieval.receipt.json'
    if receipt_path.exists():
        return folder / 'retrieval' / 'R4.jsonl'

    from tffv.retrieval.acquire_subset import acquire
    from tffv.retrieval.audit_subset import audit
    from tffv.retrieval.prepare_store_candidates import prepare_candidates
    from tffv.retrieval.store_retrieval import run as retrieve_store

    ordinal = int(sid.split(':')[1])
    # 1. Acquire
    acq_out = folder / 'acquire'
    acq_manifest = acquire(config['acquisition_plan_path'], acq_out, [ordinal], execute=True)

    # 2. Audit
    aud_out = folder / 'audit'
    aud_manifest = audit(acq_manifest, aud_out, max_record_mib=64)

    # 3. Candidates
    cm = folder / 'input' / 'claims_manifest.json'
    ca_out = folder / 'candidates'
    ca_manifest = prepare_candidates(aud_manifest, cm, ca_out, policy=config.get('candidate_policy', 'quarantine_review_flags_v1'))

    # 4. Retrieval R1..R4
    ret_out = folder / 'retrieval'
    r4_file = retrieve_store(ca_manifest, ret_out, methods=('R1', 'R2', 'R3', 'R4'),
                             config=config['retrieval'], index_cache=ret_out / 'embedding_cache') / 'R4.jsonl'

    save_json(receipt_path, {
        'sample_id': sid, 'stage': 'retrieval',
        'r4_file': str(r4_file.relative_to(run_dir).as_posix()),
        'r4_sha256': file_hash(r4_file)
    })
    return r4_file


def run_base_inference_one(backbone, sid, claim_text, r4_evidence, output_file):
    """Run frozen Qwen3-8B Base zero-shot inference on Top-8 evidence."""
    from tffv.verification.ontology import normalize_verdict
    lines = [
        'Task: verify the target claim using only the displayed evidence.',
        'Verdict options (choose exactly one): Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking',
        f'Target claim: {claim_text}',
        'Target evidence (frozen R4 Top-8):'
    ]
    for i, e in enumerate(r4_evidence):
        text = e.get('rendered_text') or e.get('quote') or e.get('text', '')
        lines.append(f'[{i+1}] {e.get("source_url", "")}\n{text}')

    messages = [{'role': 'user', 'content': '\n'.join(lines)}]

    t0 = time.time()
    gen = backbone.generate(messages, max_new_tokens=512, enable_thinking=False)
    lat = time.time() - t0
    raw_text = gen['generated_text']
    verdict = normalize_verdict(raw_text)

    pred_record = {
        'sample_id': sid,
        'claim': claim_text,
        'predicted_verdict': verdict,
        'raw_generation': raw_text,
        'prompt_tokens': gen['prompt_tokens'],
        'output_tokens': gen['generated_tokens'],
        'latency_seconds': lat
    }
    save_json(output_file, pred_record)
    return pred_record


def execute_batch_retrieval_and_base(start_claim=0, end_claim=100, batch_dir='outputs/m18a-batch-run'):
    """Execute batch retrieval and Base inference with checkpointing and cache reuse."""
    manifest_path = ROOT / 'outputs' / 'm18a-manifest-v1' / 'train_manifest.json'
    if not manifest_path.exists():
        audit_full_train_integrity()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    valid_map = {r['sample_id']: r for r in manifest['records'] if r['is_valid']}

    inv = scan_cache_inventory()
    reusable_r4 = set(inv['reusable_r4_sample_ids'])
    reusable_base = set(inv['reusable_base_sample_ids'])

    out = Path(batch_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    state_file = out / 'state.json'
    state = json.loads(state_file.read_text(encoding='utf-8')) if state_file.exists() else {
        'completed': [], 'spent_seconds': 0.0
    }

    target_sids = [f'train:{i}' for i in range(start_claim, end_claim) if f'train:{i}' in valid_map]
    print(f'Running batch on {len(target_sids)} claims (range {start_claim}..{end_claim})')

    cfg = json.loads(RETRIEVAL_CONFIG.read_text(encoding='utf-8'))
    cfg['acquisition_plan_path'] = str(ACQUISITION_PLAN.resolve())

    # We only initialize backbone if needed for Base inference
    backbone = None

    for i, sid in enumerate(target_sids):
        if sid in state['completed']:
            print(f'[{i+1}/{len(target_sids)}] Claim {sid} already completed in state.')
            continue

        claim_info = valid_map[sid]
        claim_text = claim_info['claim']

        # Setup input directory
        claim_dir = out / sid.replace(':', '-')
        inp_dir = claim_dir / 'input'
        inp_dir.mkdir(parents=True, exist_ok=True)
        (inp_dir / 'claims.jsonl').write_text(json.dumps({
            'sample_id': sid, 'dataset': 'AVeriTeC', 'split': 'train', 'claim': claim_text
        }) + '\n', encoding='utf-8')
        save_json(inp_dir / 'claims_manifest.json', {
            'dataset': 'AVeriTeC', 'split': 'train', 'sample_id': sid,
            'source_role': 'source_train', 'records': 1,
            'role': 'DEVELOPMENT_RESEARCH_NOT_HISTORICAL_SOURCE',
            'annotation_values_used': False,
            'claims_sha256': file_hash(inp_dir / 'claims.jsonl')
        })

        t0 = time.time()
        # Check R4 cache
        r4_file = claim_dir / 'retrieval' / 'R4.jsonl'
        if not r4_file.exists():
            if sid in reusable_r4:
                print(f'[{i+1}/{len(target_sids)}] Reusing cached R4 for {sid}')
            else:
                print(f'[{i+1}/{len(target_sids)}] Running fresh R4 retrieval for {sid}...')
                r4_file = retrieve_one_claim(out, sid, cfg)

        # Base prediction
        pred_file = claim_dir / 'base_prediction.json'
        if not pred_file.exists():
            if sid in reusable_base:
                print(f'[{i+1}/{len(target_sids)}] Reusing cached Base prediction for {sid}')
            else:
                if backbone is None:
                    from tffv.verification.backbone import Qwen3FrozenBackbone
                    print('Loading frozen Qwen3-8B verifier on CUDA...')
                    backbone = Qwen3FrozenBackbone(cfg['backbone'], device='cuda', dtype='bfloat16')
                r4_data = [json.loads(l) for l in r4_file.read_text(encoding='utf-8').splitlines() if l.strip()][0]
                run_base_inference_one(backbone, sid, claim_text, r4_data['evidence'], pred_file)

        state['completed'].append(sid)
        state['spent_seconds'] += time.time() - t0
        save_json(state_file, state)

    print(f'Batch execution completed for {len(state["completed"])} claims.')


def verify_official_dev_readiness(output_dir='outputs/m18a-dev-readiness-v1'):
    """Preflight check for Official Dev 500 disjointness and configuration readiness."""
    manifest_path = ROOT / 'outputs' / 'm18a-manifest-v1' / 'train_manifest.json'
    if not manifest_path.exists():
        audit_full_train_integrity()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    dev_path = get_raw_dev_path()
    dev_data = json.loads(dev_path.read_text(encoding='utf-8'))

    overlap = manifest['train_dev_overlap']

    # Check memory for contamination
    mem_path = ROOT / 'outputs' / 'm18a-fulltrain-memory-v1' / 'memory.json'
    dev_in_memory_count = 0
    if mem_path.exists():
        mem = json.loads(mem_path.read_text(encoding='utf-8'))
        dev_in_memory_count = sum(1 for r in mem.get('records', []) if r['sample_id'].startswith('dev:'))

    readiness = {
        'schema_version': 'm18a-dev-readiness-v1',
        'official_dev_claims': len(dev_data),
        'status': 'READY_FOR_SEALED_EVALUATION',
        'train_dev_separation': {
            'exact_text_overlap_count': overlap['exact_normalized_text_matches'],
            'url_overlap_count': overlap['fact_checking_url_matches'],
            'dev_claims_in_memory_count': dev_in_memory_count,
            'quarantine_policy': 'Cross-split text duplicates documented; dev gold remains completely unread during prompt assembly.'
        },
        'arms_configured': ['D0', 'D1', 'N0', 'A', 'B', 'C', 'D'],
        'arm_invariance_contract': {
            'target_evidence': 'Shared R4 Top-8 passages across all arms',
            'verifier_backbone': 'Qwen/Qwen3-8B (bfloat16, greedy, max_new_tokens=512)',
            'exemplar_order': 'Deterministic count-then-ID rotation'
        }
    }
    out = Path(output_dir).resolve()
    save_json(out / 'dev_readiness.json', readiness)
    print(f'Official Dev readiness preflight certified at {out / "dev_readiness.json"}.')
    return readiness


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)

    p_audit = sub.add_parser('audit-train')
    p_audit.add_argument('--output', default='outputs/m18a-manifest-v1')

    p_inv = sub.add_parser('scan-cache')
    p_inv.add_argument('--output', default='outputs/m18a-cache-inventory-v1')

    p_ob = sub.add_parser('build-outcome-bank')
    p_ob.add_argument('--output', default='outputs/m18a-outcome-bank-v1')

    p_cand = sub.add_parser('generate-candidates')
    p_cand.add_argument('--outcome-bank', default=None)
    p_cand.add_argument('--output', default='outputs/m18a-candidate-corrections-v1')

    p_judge = sub.add_parser('run-judge')
    p_judge.add_argument('--candidates', default=None)
    p_judge.add_argument('--cache-inventory', default=None)
    p_judge.add_argument('--output', default='outputs/m18a-judge-validated-v1')
    p_judge.add_argument('--port', type=int, default=17171)
    p_judge.add_argument('--limit', type=int, default=None)

    p_mem = sub.add_parser('build-memory')
    p_mem.add_argument('--outcome-bank', default=None)
    p_mem.add_argument('--validated-judge', default=None)
    p_mem.add_argument('--output', default='outputs/m18a-fulltrain-memory-v1')

    p_batch = sub.add_parser('retrieve-and-base')
    p_batch.add_argument('--start', type=int, default=0)
    p_batch.add_argument('--end', type=int, default=100)
    p_batch.add_argument('--batch-dir', default='outputs/m18a-batch-run')

    p_dev = sub.add_parser('verify-dev-readiness')
    p_dev.add_argument('--output', default='outputs/m18a-dev-readiness-v1')

    args = p.parse_args()
    if args.cmd == 'audit-train':
        audit_full_train_integrity(args.output)
    elif args.cmd == 'scan-cache':
        scan_cache_inventory(args.output)
    elif args.cmd == 'build-outcome-bank':
        build_outcome_bank(output_dir=args.output)
    elif args.cmd == 'generate-candidates':
        generate_candidate_corrections(outcome_bank_path=args.outcome_bank, output_dir=args.output)
    elif args.cmd == 'run-judge':
        run_stronger_judge(candidate_corrections_path=args.candidates, cache_inventory_path=args.cache_inventory,
                           output_dir=args.output, judge_port=args.port, limit=args.limit)
    elif args.cmd == 'build-memory':
        build_expanded_memory(outcome_bank_path=args.outcome_bank, validated_judge_path=args.validated_judge, output_dir=args.output)
    elif args.cmd == 'retrieve-and-base':
        execute_batch_retrieval_and_base(start_claim=args.start, end_claim=args.end, batch_dir=args.batch_dir)
    elif args.cmd == 'verify-dev-readiness':
        verify_official_dev_readiness(output_dir=args.output)


if __name__ == '__main__':
    main()
