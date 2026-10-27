"""Validate the sealed ID-only M1.6 manifests without raw data or labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def validate(folder):
    seal = json.loads((folder / 'seal.json').read_text(encoding='utf-8-sig'))
    for name, expected in seal['files_sha256'].items():
        actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Sealed hash mismatch: ' + name)
    def read(name):
        return json.loads((folder / name).read_text(encoding='utf-8-sig'))
    rows, groups, edges = read('claims_manifest.json'), read('groups_manifest.json'), read('group_edges.json')
    ids = {r['sample_id']: r for r in rows}
    assert len(ids) == len(rows) == seal['claims']
    assert set(ids) == {f'train:{i}' for i in range(seal['claims'])}
    assert Counter(r['partition'] for r in rows) == Counter({k:v['claims'] for k,v in seal['partitions'].items() if v['claims']})
    group_by_id = {g['group_id']: g for g in groups}
    assert len(group_by_id) == len(groups) == seal['leakage_blocks']
    seen = []
    for g in groups:
        for sid in g['sample_ids']:
            assert ids[sid]['group_id'] == g['group_id']
            assert ids[sid]['partition'] == g['partition']
            seen.append(sid)
    assert len(seen) == len(set(seen)) == len(rows)
    for e in edges:
        assert ids[e['a']]['group_id'] == ids[e['b']]['group_id']
        assert ids[e['a']]['partition'] == ids[e['b']]['partition']
    for part, counts in seal['partitions'].items():
        sub = read(part.lower() + '_manifest.json')
        assert sub == [r for r in rows if r['partition'] == part]
        assert len(sub) == counts['claims']
        assert len({r['group_id'] for r in sub}) == counts['groups']
    assert all(r['partition'] == 'SOURCE' and
               (r['legacy_source_A'] or not r['artifact_outcome_exposure_propagated'])
               for r in rows if r['source_memory_eligible'])
    assert not any(r['external_exposure_status'] == 'KNOWN_EXTERNAL_PREDICTION_EXPOSURE'
                   for r in rows if r['partition'] == 'TEST_SEALED')
    assert not any(r['artifact_outcome_exposure_propagated'] for r in rows if r['partition'] in ('DEVELOPMENT','TEST_SEALED'))
    return {'claims': len(rows), 'leakage_blocks': len(groups), 'cross_partition_edges': 0,
            'certified_confirmatory_claims': sum(r['confirmatory_eligible'] for r in rows),
            'all_manifest_hashes_match': True, 'confirmation_ready': seal['confirmation_ready']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    print(json.dumps(validate(p.parse_args().folder), indent=2))
