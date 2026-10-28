"""M1.6.2c coverage-arm tests: selection rule, prompt contract, config freeze"""
import json
from pathlib import Path

import pytest
from tffv.experiments import m162c


def record(sid, error, gold, group, evidence_count=1):
    evidence = [{'source_url': 'https://example.org/' + sid, 'rendered_text': 'ev ' + sid}
                for _ in range(evidence_count)]
    return {'sample_id': sid, 'is_base_error': error, 'gold_verdict': gold,
            'group_id': group, 'base_prediction': {'reparsed_verdict': 'Refuted'},
            'corrected_explanation': 'DRAFT MUST NOT APPEAR', 'evidence': evidence,
            'citations': []}


def test_addition_selection_is_deterministic_count_then_id():
    config = m162c.load_config()
    records = [record('train:23', False, 'Not Enough Evidence', 'g1', evidence_count=3),
               record('train:2811', False, 'Not Enough Evidence', 'g2', evidence_count=5),
               record('train:21', True, 'Not Enough Evidence', 'g3', evidence_count=99)]
    added = m162c.select_added({r['sample_id']: r for r in records},
                               'Not Enough Evidence', set(), config)
    assert added['sample_id'] == 'train:2811'
    added = m162c.select_added({r['sample_id']: r for r in records},
                               'Not Enough Evidence', {'train:2811'}, config)
    assert added['sample_id'] == 'train:23'
    with pytest.raises(ValueError, match='No usable Base-correct'):
        m162c.select_added({r['sample_id']: r for r in records}, 'Missing class', set(), config)


def test_frozen_coverage_config_contract():
    config = m162c.load_config()
    assert list(config['arms']) == ['COV_PLACEBO_G3', 'COV_NEI_G3', 'COV_CONF_G3']
    assert config['k'] == 3 and config['added_context_excerpts'] == 1
    assert config['order'] == ['pair_correct', 'added', 'pair_wrong']
    assert config['generation'] == {'max_new_tokens': 512, 'enable_thinking': False,
                                    'do_sample': False}
    assert config['rerun_conditions'] == []
    assert set(config['baseline_predictions_reused']) == {'P2', 'N0', 'MIXED_GOLD_ONLY_K2'}


def test_added_header_is_neutral_and_shared_across_arms():
    header = m162c.ADDED_HEADER.strip()
    assert 'COV_' not in header and 'Not Enough' not in header
    assert 'Conflicting' not in header and 'gold-only' in header


def test_arm_validation_rejects_unknown_names():
    with pytest.raises(SystemExit):
        m162c.prepare(canonical_run=None, cache_index=None, memory_path=None,
                      tokenizer_path=None, arm='COV_BOGUS', destination='unused')


def test_locked_result_schema_accepts_versioned_pilot_results():
    result = json.loads((Path(__file__).resolve().parents[1]
                         / 'docs/audits/M1_6_2B_MIXED_RESULTS.json').read_text('utf-8-sig'))
    assert m162c.validate_result(result) == result
    broken = {**result, 'exact_mcnemar_p_missing_removed': None}
    del broken['gpu_active_seconds']
    with pytest.raises(ValueError, match='missing locked key'):
        m162c.validate_result(broken)
    drift = json.loads(json.dumps(result))
    drift['comparisons'][list(drift['comparisons'])[0]]['exact_mcnemar_p'] = 1.0
    del drift['comparisons'][list(drift['comparisons'])[0]]['transitions']
    with pytest.raises(ValueError, match='transitions'):
        m162c.validate_result(drift)


def test_results_file_formally_validates_against_json_schema():
    jsonschema = pytest.importorskip('jsonschema')
    schema = json.loads((Path(__file__).resolve().parents[1]
                         / 'schemas/m162c_results.schema.json').read_text('utf-8-sig'))
    result = json.loads((Path(__file__).resolve().parents[1]
                         / 'docs/audits/M1_6_2B_MIXED_RESULTS.json').read_text('utf-8-sig'))
    jsonschema.validate(result, schema)
