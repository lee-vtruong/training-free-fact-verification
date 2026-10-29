"""Read-only CPU inventory for M1.6 recovery provenance; no gold content emitted."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path('/home/stackops/whale/training-free-fact-verification')
out = root / 'outputs'
files = sorted(p for p in out.rglob('*') if p.is_file() and
               ('m15' in str(p.relative_to(out)).lower() or
                any(t in p.name.lower() for t in ('backup', 'failed', 'recovery', 'invalid'))))
inventory = []
for p in files:
    # Avoid huge retrieval payload reads; all other matching provenance is hashed.
    if p.stat().st_size > 20_000_000:
        continue
    inventory.append({'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size,
                      'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
baseline_path = out / 'm16-integrity-audit/audit.json'
frozen_check = None
if baseline_path.exists():
    baseline = json.loads(baseline_path.read_text())['frozen_inventory']
    mismatches = [rel for rel, h in baseline.items()
                  if not (root / rel).is_file() or hashlib.sha256((root / rel).read_bytes()).hexdigest() != h]
    frozen_check = {'files_checked': len(baseline), 'mismatches': mismatches,
                    'all_unchanged': not mismatches}
    assert not mismatches, 'Frozen artifact changed during audit'
recovery = {}
for run in ('m15-evaluation-20261026-142010', 'm15-evaluation-20261026-151554'):
    recovery[run] = {}
    for c in ('M0', 'M1', 'M4'):
        p = out / run / c / 'predictions.jsonl'
        rr = [json.loads(s) for s in p.read_text().splitlines()]
        recovery[run][c] = {'n': len(rr), 'stored_valid': sum(r['is_valid_verdict'] for r in rr),
                            'raw_nonempty': sum(bool(r.get('raw_text', '').strip()) for r in rr),
                            'input_tokens': sum(r.get('input_tokens', r.get('total_input_tokens', 0)) for r in rr),
                            'output_tokens': sum(r.get('output_tokens', r.get('total_output_tokens', 0)) for r in rr)}
        if c == 'M4':
            dr = [json.loads(s) for s in (p.parent / 'permutations_M4_detail.jsonl').read_text().splitlines()]
            recovery[run][c]['permutation_raw_nonempty'] = sum(bool(v.get('raw_text', '').strip()) for r in dr for k, v in r.items() if k.startswith('order_'))
before = out / 'm15-evaluation-20261026-142010'
after = out / 'm15-evaluation-20261026-151554'
recovery['partition_bytes_equal'] = (before / 'partition_manifest.json').read_bytes() == (after / 'partition_manifest.json').read_bytes()
recovery['prompt_hashes_equal'] = {}
for c in ('M0', 'M1'):
    def ph(d):
        return {r['sample_id']: r['prompt_sha256'] for r in map(json.loads, (d / c / 'predictions.jsonl').read_text().splitlines())}
    recovery['prompt_hashes_equal'][c] = ph(before) == ph(after)
print(json.dumps({'frozen_before_after_check': frozen_check, 'recovery': recovery, 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                  'status': subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
                  'runner_diff': subprocess.check_output(['git', 'diff', '--', 'scripts/run_m15_eval.sh'], cwd=root, text=True),
                  'files': inventory}, indent=2))
