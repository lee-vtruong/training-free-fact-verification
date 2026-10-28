"""M1.6.2e gate tests: arithmetic, review provenance, and fail-closed coverage."""
import json
from pathlib import Path

import pytest
from tffv.experiments import m162e

ROOT = Path(__file__).resolve().parents[1]


def load_records():
    return m162e.load_review_records()


def test_gate_arithmetic_matches_locked_policy():
    report = m162e.gate_report()
    assert report['total_records'] == 33
    assert report['counts']['ORIGINAL_DRAFT'] == 5
    assert report['counts']['E3_CANDIDATE_DRAFT'] == 16
    assert report['counts']['E2_CANDIDATE_DRAFT_QUARANTINED'] == 4
    assert report['counts']['E1_NO_DRAFT'] == 8
    assert report['max_plausible_candidates_pre_rejection'] == 25
    assert report['unchanged_top8_ceiling'] == 21
    assert report['full_memory_gate_verified'] == 30
    assert 25 < 30 and 21 < 30  # the >=30 milestone gate is structurally unsatisfiable
    assert report['verified_now'] == 5
    assert report['feasibility_ok'] is False
    assert report['gate'] == 'TIER1_UNMET_FAIL_CLOSED'


def test_config_validates_against_locked_policy():
    cfg = m162e.load_config()
    with pytest.raises(ValueError):
        m162e.validate_config({**cfg, 'gate': {**cfg['gate'], 'feasibility_min_verified': 3}})
    with pytest.raises(ValueError):
        m162e.validate_config({**cfg, 'decoding': {**cfg['decoding'], 'temperature': 0.7}})
    with pytest.raises(ValueError):
        m162e.validate_config({**cfg, 'stats': {**cfg['stats'], 'contrasts': ['A_vs_B']}})


def test_record_schema_complete_for_every_record():
    required = ('record_kind', 'sample_id', 'claim', 'gold_verdict', 'base_verdict',
                'draft_text', 'mechanism', 'limitations', 'decisive_quotes',
                'evidence_availability_change', 'excluded_from_primary_experiment',
                'triage_class', 'review_status', 'verified', 'reviews')
    for r in load_records():
        for key in required:
            assert key in r, (r['sample_id'], key)
        assert r['sample_id'].startswith('train:')
        if r['sample_id'] in {'train:6', 'train:11', 'train:25', 'train:978', 'train:1824'}:
            assert r['verified'] is True
            assert r['review_status'] == 'VERIFIED'
            assert r['mechanism']
            for slot in r['reviews'].values():
                assert slot['reviewer_type'] == 'human'
                assert slot['independent_review'] is True
                assert slot['submission_ref']
        else:
            assert r['verified'] is False
            assert r['reviews'] == {'reviewer_1': None, 'reviewer_2': None}


def test_e2_quarantined_and_e1_no_draft_invariants():
    for r in load_records():
        if r['triage_class'] == 'E2':
            assert r['evidence_availability_change'] is True
            assert r['excluded_from_primary_experiment'] is True
            assert all(q.get('verified_in') == 'server_store_probe_20261028'
                       for q in r['decisive_quotes'])
        if r['triage_class'] == 'E1':
            assert r['draft_text'] is None
            assert r['review_status'] == 'NOT_DRAFTABLE'
        if r['triage_class'] == 'E3':
            assert r['evidence_availability_change'] is False
            assert all(q['source'] == 'top8_span' for q in r['decisive_quotes'])
            assert r['mechanism'] and r['limitations']


def test_originals_passthrough_untouched_from_memory_v3():
    mem = json.loads((ROOT / 'outputs/m1611-correction-audit-v3/memory75.json')
                     .read_text('utf-8-sig'))
    drafts = {r['sample_id']: r.get('corrected_explanation') for r in mem['records']}
    for r in load_records():
        if r['record_kind'] == 'ORIGINAL_DRAFT':
            assert r['draft_text'] == drafts[r['sample_id']]
            assert r['draft_text_provenance'].startswith('outputs/m1611-correction-audit-v3')


def test_reviewer_sheets_identical_and_blank():
    sheets = [json.loads(p.read_text('utf-8-sig')) for p in m162e.REVIEW_SHEETS]
    assert sheets[0]['records'] == sheets[1]['records']
    blank = ('decision', 'edited_text', 'rationale', 'ontology_signoff', 'reviewed_at')
    for sheet in sheets:
        for r in sheet['records']:
            for who in ('reviewer_1', 'reviewer_2'):
                assert r['reviews'][who] is None
            for key in blank:
                assert key not in r or r.get(key) in (None, ''), (r['sample_id'], key)
        assert 'assistant must not fill' in sheet['instructions'].lower()


def test_verified_pool_still_fails_closed_on_coverage():
    pool = m162e.verified_pool()
    assert len(pool) == 5
    assert m162e.class_coverage(pool, 2) == {}
    with pytest.raises(ValueError, match='Tier-1 feasibility gate not met'):
        m162e.require_feasible(pool, m162e.load_config())


def test_verified_pool_accepts_only_double_reviewed_primary_records():
    base = {'record_kind': 'E3_CANDIDATE_DRAFT', 'sample_id': 'train:9',
            'mechanism': 'selective_reading', 'verified': True,
            'excluded_from_primary_experiment': False,
            'evidence_availability_change': False,
            'reviews': {'reviewer_1': {'decision': 'ACCEPT'},
                        'reviewer_2': {'decision': 'ACCEPT_WITH_EDITS',
                                       'edited_text': 'reviewer-edited text'}}}
    assert len(m162e.verified_pool([base])) == 1
    assert m162e.effective_draft(base) == 'reviewer-edited text'
    # single reviewer -> not verified
    one = {**base, 'reviews': {'reviewer_1': {'decision': 'ACCEPT'}, 'reviewer_2': None}}
    assert m162e.verified_pool([one]) == []
    # reject -> not verified
    rej = {**base, 'reviews': {'reviewer_1': {'decision': 'ACCEPT'},
                               'reviewer_2': {'decision': 'REJECT'}}}
    assert m162e.verified_pool([rej]) == []
    # assistant self-mark without human decisions -> fail closed
    selfmark = {**base, 'reviews': {'reviewer_1': None, 'reviewer_2': None}}
    assert m162e.verified_pool([selfmark]) == []
    # E2 quarantine holds even when fully accepted
    e2 = {**base, 'record_kind': 'E2_CANDIDATE_DRAFT_QUARANTINED',
          'evidence_availability_change': True, 'excluded_from_primary_experiment': True}
    assert m162e.verified_pool([e2]) == []


def test_feasibility_ok_requires_two_classes_two_each():
    cfg = m162e.load_config()

    def rec(sid, mech):
        return {'record_kind': 'E3_CANDIDATE_DRAFT', 'sample_id': sid, 'mechanism': mech,
                'verified': True, 'excluded_from_primary_experiment': False,
                'evidence_availability_change': False,
                'reviews': {'reviewer_1': {'decision': 'ACCEPT'},
                            'reviewer_2': {'decision': 'ACCEPT'}}}
    single_class = [rec('train:8', 'selective_reading'), rec('train:19', 'selective_reading')]
    assert m162e.feasibility_ok(single_class, cfg) is False  # only one class
    two_classes_thin = single_class + [rec('train:21', 'metric_mismatch')]
    assert m162e.feasibility_ok(two_classes_thin, cfg) is False  # 2nd class has only 1
    two_classes_full = two_classes_thin + [rec('train:31', 'metric_mismatch')]
    assert m162e.feasibility_ok(two_classes_full, cfg) is True
    assert m162e.class_coverage(two_classes_full, 2) == {'metric_mismatch': 2,
                                                         'selective_reading': 2}
