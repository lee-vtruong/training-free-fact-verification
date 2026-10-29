"""Read-only CPU resource, archive and historical timing audit; no model loading."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

ROOT = Path('/home/stackops/whale/training-free-fact-verification')
OLD = Path('/home/stackops/whale/ERRORFlow')


def file_sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    result = {'gpu_snapshot': subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version', '--format=csv,noheader'], text=True).strip(),
              'disk': shutil.disk_usage(ROOT)._asdict(), 'timings': [], 'logs': [],
              'archive_inventory': [], 'external_reference_inventory': [], 'index_cache': {}}
    result['external_artifact_files'] = []
    for base in (OLD / 'outputs',):
        if base.is_dir():
            for p in sorted(base.rglob('*')):
                if p.is_file() and p.suffix in ('.json', '.jsonl', '.csv', '.log', '.md'):
                    result['external_artifact_files'].append({'path': str(p), 'bytes': p.stat().st_size})
    for p in sorted((ROOT / 'outputs').rglob('summary.json')):
        if not any(s in p.as_posix() for s in ('m11-store-r14', 'm141-retrieval', 'm142-retrieval', 'm15-folde-r4')):
            continue
        obj = json.loads(p.read_text())
        methods = obj.get('methods', {})
        result['timings'].append({'path': p.relative_to(ROOT).as_posix(), 'sha256': file_sha(p),
                                  'elapsed_seconds': obj.get('elapsed_seconds'),
                                  'methods': {k: {'n_claims': v.get('n_claims'),
                                                  'mean_latency_seconds': v.get('means', {}).get('latency_seconds'),
                                                  'peak_gpu_memory_bytes': v.get('means', {}).get('peak_gpu_memory_bytes')}
                                              for k, v in methods.items()}})
    for p in sorted((ROOT / 'outputs').glob('*.log')):
        refs = []
        with p.open(encoding='utf-8', errors='replace') as f:
            for no, line in enumerate(f, 1):
                ids = sorted({int(s) for s in re.findall(r'\btrain:(\d+)\b', line) if int(s) >= 350})
                if ids:
                    refs.append({'line': no, 'ids': ids, 'line_sha256': hashlib.sha256(line.encode()).hexdigest()})
        result['logs'].append({'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
                               'sha256': file_sha(p), 'train_350_plus_references': refs})
    # Bounded reference inventory; no old-project predictions/labels are exported.
    for base in (OLD, OLD / 'outputs', OLD / 'results', OLD / 'runs', OLD / 'data', ROOT / 'data'):
        if not base.is_dir():
            continue
        result['external_reference_inventory'].append({'path': str(base), 'entries': sorted(p.name for p in base.iterdir())[:300]})
    zip_paths = set()
    for base in (OLD / 'data', ROOT / 'data'):
        if base.is_dir():
            zip_paths.update(base.rglob('*.zip'))
    for p in sorted(zip_paths):
        with zipfile.ZipFile(p) as z:
            names = z.infolist()
            ids = sorted({int(s) for n in names for s in re.findall(r'(?:train[-_/])(\d+)', n.filename)})
            result['archive_inventory'].append({'path': str(p), 'bytes': p.stat().st_size,
                                                 'members': len(names), 'uncompressed_bytes': sum(n.file_size for n in names),
                                                 'train_ids': ids,
                                                 'member_directory_sha256': hashlib.sha256(json.dumps([(n.filename,n.file_size,n.CRC) for n in names]).encode()).hexdigest(),
                                                 'full_archive_sha256': None})
    cache = ROOT / 'outputs/index_cache'
    if cache.exists():
        cc = list(cache.rglob('*'))
        result['index_cache'] = {'files': sum(p.is_file() for p in cc),
                                  'bytes': sum(p.stat().st_size for p in cc if p.is_file()),
                                  'manifest_files': [str(p.relative_to(ROOT)) for p in cc if p.is_file() and p.suffix == '.json'][:25],
                                  'reuse_verified_for_new_partitions': False}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
