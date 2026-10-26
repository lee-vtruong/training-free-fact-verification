"""CPU/read-only inventory and label-free projection for M1.6.0.

Does not run retrieval/inference, inspect QA/justification values, or construct memory.
Raw JSON is parsed only to project the explicit claim-metadata allowlist.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess


def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--train-json', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root, dest = a.root.resolve(), a.output.resolve()
    if not dest.is_relative_to(root / 'outputs') or dest.exists():
        p.error('Use a new project outputs directory')
    raw = json.loads(a.train_json.read_text(encoding='utf-8-sig'))
    allowed = ('claim', 'claim_date', 'speaker', 'original_claim_url', 'fact_checking_article',
               'fact_checking_strategies', 'claim_types', 'reporting_source')
    # claim_types/strategies can be annotation-derived: inventory their presence,
    # but do NOT export them or use them for grouping/assignment.
    export = ('claim', 'claim_date', 'speaker', 'original_claim_url', 'fact_checking_article', 'reporting_source')
    records = []
    for i, r in enumerate(raw):
        records.append({'sample_id': f'train:{i}', 'official_split': 'train',
                        **{k: r.get(k) for k in export}})
    schemas = {k: sum(k in r for r in raw) for k in allowed}
    del raw
    artifacts = []
    scanned = []
    for f in sorted((root / 'outputs').rglob('*')):
        if not f.is_file() or dest in f.parents:
            continue
        rel = f.relative_to(root).as_posix()
        if any(x in f.parts for x in ('index_cache', '__pycache__')):
            continue
        name = f.name.lower()
        kind = None
        if name.endswith('.jsonl') and ('predictions' in name or 'per_claim' in name):
            kind = 'model_or_evaluation_output'
        elif name.endswith('.csv') and 'per_claim' in name:
            kind = 'evaluation_output'
        elif name in ('claims.jsonl', 'source_claims.jsonl'):
            kind = 'claim_metadata_or_retrieval_input'
        if kind:
            ids, errors = set(), []
            try:
                with f.open(encoding='utf-8-sig', newline='') as stream:
                    rr = csv.DictReader(stream) if f.suffix == '.csv' else (json.loads(line) for line in stream if line.strip())
                    for r in rr:
                        sid = r.get('sample_id', r.get('claim_id'))
                        if isinstance(sid, str) and sid.startswith('train:'):
                            ids.add(sid)
            except (ValueError, OSError) as e:
                errors.append(str(e))
            artifacts.append({'path': rel, 'kind': kind, 'bytes': f.stat().st_size,
                              'sha256': digest_file(f), 'sample_ids': sorted(ids, key=lambda s: int(s.split(':')[1])),
                              'errors': errors})
        elif name.endswith('manifest.json') or name in ('summary.json', 'evaluation_summary.json', 'resources.json'):
            scanned.append({'path': rel, 'bytes': f.stat().st_size, 'sha256': digest_file(f)})
    dest.mkdir(parents=True)
    with (dest / 'claims_metadata.jsonl').open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')
    report = {'dataset': {'path': str(a.train_json.resolve()), 'sha256': digest_file(a.train_json),
                           'official_split': 'train', 'claims': len(records), 'metadata_presence_counts': schemas},
              'projection': {'path': 'claims_metadata.jsonl', 'sha256': digest_file(dest / 'claims_metadata.jsonl'),
                             'allowed_fields': list(records[0])},
              'artifact_exposure': artifacts, 'metadata_inventory': scanned,
              'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
              'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
              'gpu_snapshot': subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip(),
              'limits': ['No inference/retrieval. No labels, QA or justifications exported.',
                         'Artifact inventory cannot certify unrecorded human or external-project exposure.',
                         'Annotation-derived claim_types/strategies excluded from projection and assignment.']}
    write(dest / 'inventory.json', report)
    print(json.dumps({'dataset': report['dataset'], 'artifact_files': len(artifacts),
                      'metadata_files': len(scanned), 'output': str(dest)}, indent=2))


if __name__ == '__main__':
    main()
