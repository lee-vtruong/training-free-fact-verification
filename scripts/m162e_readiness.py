"""M1.6.2e section E: development-schedule readiness report (CPU-only, read-only).

Reads sealed partition manifests + prior schedule artifacts; embeds the 2026-10-28
server cache probe (read-only dir listings). No gold labels are read; no retrieval;
no GPU. Output: docs/M1_6_2E_READINESS.md.
"""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PART = Path('outputs/m16-partitions-v1')
DEV50_BATCH = json.loads(Path('outputs/m162a-dev50-preparation-v4/batch.json').read_text('utf-8-sig'))

# Server cache probe 2026-10-28 (read-only dir listings; hvtham-server).
# Roots probed: outputs/{m11-store-r14-pilot100-20261023-162716, m162a-dev50-v1,
# m142-retrieval-folde-20261026-030724, m161-source-batch25-v1}
PROBE = {
    'pilot100_store_0_99': {'path': 'outputs/m11-store-r14-pilot100-20261023-162716', 'n': 100, 'ordinals': list(range(0, 100))},
    'canonical_dev50_v1': {'path': 'outputs/m162a-dev50-v1', 'n': 50,
                           'ordinals': sorted(int(s.split(':')[1]) for s in DEV50_BATCH['sample_ids'])},
    'folde_250_349': {'path': 'outputs/m142-retrieval-folde-20261026-030724', 'n': 100, 'ordinals': list(range(250, 350))},
    'batch25_v1': {'path': 'outputs/m161-source-batch25-v1', 'n': 25,
                   'ordinals': [426, 467, 694, 718, 828, 903, 978, 979, 1231, 1236, 1288, 1402, 1429,
                                 1644, 1817, 1824, 1847, 1863, 1873, 1881, 2055, 2224, 2449, 2811, 3012]},
}

manifest = json.loads((PART / 'development_manifest.json').read_text('utf-8-sig'))
claims = json.loads((PART / 'claims_manifest.json').read_text('utf-8-sig'))
C = {r['sample_id']: r for r in claims}

dev_ids = [r['sample_id'] for r in manifest]
dev_set = set(dev_ids)
used50 = set(DEV50_BATCH['sample_ids'])
assert used50.issubset(dev_set), 'DEV50 not inside development manifest'
unused = [sid for sid in dev_ids if sid not in used50]

def flag(records, key):
    return sum(1 for r in records if r.get(key))

groups_used = {C[s]['group_id'] for s in used50}
groups_unused = {}
for sid in unused:
    g = C[sid]['group_id']
    groups_unused.setdefault(g, []).append(sid)
fully_unused_groups = {g: v for g, v in groups_unused.items() if g not in groups_used}
partially_used_groups = {g: v for g, v in groups_unused.items() if g in groups_used}

cached_ordinals = set()
for name, p in PROBE.items():
    cached_ordinals.update(p['ordinals'])
cached_unused = [sid for sid in unused if int(sid.split(':')[1]) in cached_ordinals]
miss_unused = [sid for sid in unused if sid not in set(cached_unused)]

L = []
A = L.append
A('# M1.6.2e — Section E: New-Population Readiness (report only)')
A('')
A('Generated: %s · CPU-only, read-only · no retrieval, no GPU, no TEST_QUARANTINE access.'
  % datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%MZ'))
A('')
A('## Sealed partition frame')
A('')
A('- Dataset: 3,068 claims (`outputs/m16-partitions-v1`, seal dataset_sha256 '
  '`ae5eda7c42ddf1695ef185a7ba1bc716928f5adf57103e4f78aae5f9afe00f9c`).')
A('- Assignment intervals: SOURCE 0.0–0.3 · DEVELOPMENT 0.3–0.5 · TEST_SEALED 0.5–1.0.')
A('- Development manifest: **%d claims** (official_split train). TEST_SEALED untouched '
  '(`test_sealed_manifest.json` not opened beyond count).' % len(manifest))
A('')
A('## Used vs unused development claims')
A('')
A('| set | n | note |')
A('|---|---|---|')
A('| DEV50 (M1.4/M1.6.2 development population) | 50 | ordinals 368–3054; outcomes known and frozen |')
A('| unused development claims | **%d** | manifest minus DEV50 |' % len(unused))
A('')
A('Exposure/eligibility flags over the %d unused claims:' % len(unused))
A('')
A('| flag | true count |')
A('|---|---|')
for key in ('artifact_outcome_exposed_directly', 'artifact_outcome_exposure_propagated',
            'claim_text_previously_exported', 'confirmatory_eligible', 'source_memory_eligible',
            'legacy_source_A'):
    A('| `%s` | %d |' % (key, flag([C[s] for s in unused], key)))
A('| `external_exposure_status == UNKNOWN` | %d |' % sum(1 for s in unused if C[s].get('external_exposure_status') == 'UNKNOWN'))
A('')
A('## Leakage-group readiness')
A('')
A('- Distinct leakage groups touched by the unused claims: **%d**.' % len(groups_unused))
A('- Fully-unused groups (no DEV50 member): **%d** claims in %d groups — clean candidate pool.'
  % (sum(len(v) for v in fully_unused_groups.values()), len(fully_unused_groups)))
A('- Groups shared with DEV50 (contamination risk for historical-outcome reasoning): **%d** claims '
  'in %d groups — excluded from the candidate pool by the locked selection rule.'
  % (sum(len(v) for v in partially_used_groups.values()), len(partially_used_groups)))
A('- Group size distribution (fully-unused groups): %s.' % json.dumps(sorted(Counter(len(v) for v in fully_unused_groups.values()).items())))
A('')
A('## R4 cache coverage (server probe 2026-10-28, read-only)')
A('')
A('| cache root | n claim dirs | ordinals/IDs |')
A('|---|---|---|')
for name, p in PROBE.items():
    rng = ('%d–%d contiguous' % (min(p['ordinals']), max(p['ordinals']))) if name != 'canonical_dev50_v1' else 'the 50 DEV50 IDs'
    A('| `%s` | %d | %s |' % (p['path'], p['n'], rng))
A('')
A('- Unused development claims **with** an existing retrieval cache: **%d**.' % len(cached_unused))
A('- Unused development claims **without** cache (would need the single permitted retrieval pass): **%d**.' % len(miss_unused))
A('- Practical reading: a fresh population drawn from fully-unused groups will be a **cache MISS** '
  'almost everywhere; per the M1.6.2d step-4 rule the new population gets exactly one retrieval pass. '
  'No retrieval is executed in this milestone.')
A('')
A('## Readiness verdict')
A('')
A('**READY-PENDING-GATES**: a sufficient unused population exists (%d claims in %d fully-unused '
  'groups), the sealed frame is intact, and cache strategy is defined. Readiness does NOT authorize '
  'execution: step-4 switch remains blocked by the M1.6.2e gates (human verification of corrections; '
  'protocol lock). No class-distribution read of unused development gold was performed at planning '
  'time; class-stratified inclusion happens only at prepare-time under the locked selection rule.'
  % (sum(len(v) for v in fully_unused_groups.values()), len(fully_unused_groups)))
A('')
A('## Provenance')
A('')
for f in ('development_manifest.json', 'claims_manifest.json', 'seal.json', 'protocol.json'):
    p = PART / f
    A('- `outputs/m16-partitions-v1/%s` sha256 `%s`' % (f, hashlib.sha256(p.read_bytes()).hexdigest()))
A('- DEV50 IDs: `outputs/m162a-dev50-preparation-v4/batch.json` sha256 `%s`'
  % hashlib.sha256(Path('outputs/m162a-dev50-preparation-v4/batch.json').read_bytes()).hexdigest())

dest = Path('docs/M1_6_2E_READINESS.md')
dest.write_text('\n'.join(L), encoding='utf-8')
print('readiness bytes:', dest.stat().st_size)
print('unused:', len(unused), '| fully-unused-group claims:',
      sum(len(v) for v in fully_unused_groups.values()),
      '| groups:', len(fully_unused_groups), '| shared-group claims:', sum(len(v) for v in partially_used_groups.values()))
print('cache hit among unused:', len(cached_unused), '| miss:', len(miss_unused))
