"""Leakage and sealing contracts; constructed cases are software tests only."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


audit = module('m16_partition', ROOT / 'scripts/m16_partition_audit.py')
validator = module('m16_validate', ROOT / 'scripts/m16_validate_seal.py')


def test_transitive_components_and_url_event_edges_cannot_split():
    config = json.loads((ROOT / 'configs/m16_partition_v2.json').read_text())
    rows = [
        {'sample_id':'train:0', 'claim':'red green blue orange', 'fact_checking_article':'https://example.com/a'},
        {'sample_id':'train:1', 'claim':'red green blue violet', 'fact_checking_article':'https://example.com/b'},
        {'sample_id':'train:2', 'claim':'black white silver', 'fact_checking_article':'http://www.example.com/b?utm_source=x'},
    ]
    groups, edges, dup = audit.build_groups(rows, config)
    assert groups == [['train:0','train:1','train:2']]
    records, _ = audit.assign(rows, groups, dup, {'artifact_exposure': []}, config, ['train:2'])
    assert len({r['partition'] for r in records}) == 1
    assert all(not r['confirmatory_eligible'] for r in records)
    assert records[0]['source_memory_eligible']


def test_day_month_year_event_rule_and_no_homepage_merge():
    config = json.loads((ROOT / 'configs/m16_partition_v2.json').read_text())
    rows = [
        {'sample_id':'train:350', 'claim':'Vaccine distribution needs a refrigerator', 'speaker':'Dr Example', 'claim_date':'24-8-2020'},
        {'sample_id':'train:351', 'claim':'Vaccination programme reached village elders', 'speaker':'Dr Example', 'claim_date':'1-8-2020'},
    ]
    groups, edges, _ = audit.build_groups(rows, config)
    assert len(groups) == 1
    assert any(e['kind'] == 'speaker_month_anchor' for e in edges)
    assert audit.canonical_url('https://example.com/') is None


def test_real_seal_no_exposed_test_and_tamper_detection(tmp_path):
    original = ROOT / 'docs/partitions/m16-v2'
    assert validator.validate(original)['claims'] == 3068
    assert validator.validate(original)['certified_confirmatory_claims'] == 0
    copy = tmp_path / 'seal'
    shutil.copytree(original, copy)
    with (copy / 'test_sealed_manifest.json').open('a') as f:
        f.write(' ')
    with pytest.raises(ValueError, match='hash mismatch'):
        validator.validate(copy)


def test_external_exposed_candidate_is_quarantined_not_reallocated():
    config = json.loads((ROOT / 'configs/m16_partition_v2.json').read_text())
    # Choose a deterministic singleton falling in the sealed-test hash interval.
    rows = [{'sample_id':f'train:{i}', 'claim':f'unrelated claim {i}'} for i in range(350, 400)]
    groups = [[r['sample_id']] for r in rows]
    dup = {r['sample_id']:r['sample_id'] for r in rows}
    before, _ = audit.assign(rows,groups,dup,{'artifact_exposure':[]},config)
    target = next(r['sample_id'] for r in before if r['partition']=='TEST_SEALED')
    after, _ = audit.assign(rows,groups,dup,{'artifact_exposure':[]},config,[target])
    assert next(r for r in after if r['sample_id']==target)['partition']=='TEST_QUARANTINE_EXPOSED'
    assert [(r['sample_id'],r['partition']) for r in before if r['sample_id']!=target] == [(r['sample_id'],r['partition']) for r in after if r['sample_id']!=target]
