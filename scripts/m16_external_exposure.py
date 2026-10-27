"""Audit ERRORFlow provenance/IDs only; never reuse its predictions or memory."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path('/home/stackops/whale/training-free-fact-verification')
OLD = Path('/home/stackops/whale/ERRORFlow')


def main():
    projection = [json.loads(s) for s in (ROOT / 'outputs/m16-data-audit-v1/claims_metadata.jsonl').read_text().splitlines()]
    claims = {r['sample_id']: r['claim'] for r in projection}
    by_text = {}
    for r in projection:
        if r['claim'].strip():
            by_text.setdefault(r['claim'], []).append(r['sample_id'])
    candidates = [OLD / p for p in (
        'outputs/v3/averitec/b1_train_all.jsonl',
        'outputs/v2/averitec/oracle_train_all_s0.jsonl',
        'outputs/error_memory/train_first1000_counterfactual.jsonl',
        'outputs/v2/averitec/memory/action_outcomes_train.jsonl')]
    result = {'reference_only': True, 'prediction_or_gold_values_exported': False, 'files': [], 'known_external_train_ids': []}
    known = set()
    for p in candidates:
        if not p.exists():
            continue
        data = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
        schemas = sorted({k for r in data for k in r})
        id_fields = [k for k in schemas if k in ('sample_id', 'claim_id', 'example_id', 'id', 'split', 'dataset')]
        ids, matched, ambiguous = set(), set(), 0
        samples = []
        for r in data:
            values = {k: r[k] for k in id_fields if k in r}
            if len(samples) < 3:
                samples.append(values)
            for value in values.values():
                if isinstance(value, str):
                    m = re.fullmatch(r'(?:averitec[:/_-])?train[:/_-](\d+)', value)
                    if m and f'train:{int(m[1])}' in claims:
                        ids.add(f'train:{int(m[1])}')
            text = r.get('claim', r.get('claim_text'))
            if isinstance(text, str) and text in by_text:
                found = by_text[text]
                matched.update(found)
                ambiguous += int(len(found) > 1)
        known.update(ids | matched)
        result['files'].append({'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                                'records': len(data), 'schema_keys': schemas, 'id_samples': samples,
                                'explicit_train_ids': sorted(ids, key=lambda s:int(s.split(':')[1])),
                                'exact_claim_text_matched_ids': sorted(matched, key=lambda s:int(s.split(':')[1])),
                                'ambiguous_exact_text_rows': ambiguous})
    result['known_external_train_ids'] = sorted(known, key=lambda s:int(s.split(':')[1]))
    manifests = []
    for p in (OLD / 'outputs/v3/averitec/b1_train_all.manifest.json',):
        if p.exists():
            raw = json.loads(p.read_text())
            manifests.append({'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                              'schema_keys': list(raw),
                              'metadata': {k:v for k,v in raw.items() if k in ('status','requested_rows','completed_rows','elapsed_seconds') or any(t in k.lower() for t in ('path','split','sha','model','revision','count','dataset','n_examples','n_claims','records','input','source'))}})
    result['manifests'] = manifests
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
