"""M1.6.2e protocol tests: gold-blind selection, determinism, arm invariance, length control."""
import pytest
from tffv.experiments import m162e


def exemplar(sid, mech):
    return {'record_kind': 'E3_CANDIDATE_DRAFT', 'sample_id': sid, 'mechanism': mech,
            'claim': 'claim of %s' % sid, 'gold_verdict': 'Not Enough Evidence',
            'base_verdict': 'Supported', 'draft_text': 'draft text for %s' % sid,
            'verified': True, 'excluded_from_primary_experiment': False,
            'evidence_availability_change': False,
            'decisive_quotes': [{'source': 'top8_span', 'quote': 'decisive quote %s' % sid}],
            'reviews': {'reviewer_1': {'decision': 'ACCEPT'},
                        'reviewer_2': {'decision': 'ACCEPT'}}}


def target(sid):
    return {'sample_id': sid, 'claim': 'target claim %s' % sid}


PRED = {'train:100': 'selective_reading', 'train:101': 'selective_reading',
        'train:102': 'selective_reading', 'train:103': 'selective_reading',
        'train:104': 'selective_reading', 'train:105': 'selective_reading',
        'train:200': 'metric_mismatch', 'train:201': 'metric_mismatch',
        'train:202': 'metric_mismatch', 'train:203': 'metric_mismatch',
        'train:204': 'metric_mismatch', 'train:205': 'metric_mismatch',
        'train:106': 'missed_direct_match'}
POOL = [exemplar('train:8', 'selective_reading'), exemplar('train:19', 'selective_reading'),
        exemplar('train:21', 'metric_mismatch'), exemplar('train:31', 'metric_mismatch')]
COV = m162e.class_coverage(POOL, 2)


def candidates():
    return [target(s) for s in PRED]


def test_selection_is_gold_blind_count_then_id_deterministic():
    cfg = m162e.load_config()
    first = m162e.select_targets(candidates(), PRED, COV, cfg)
    again = m162e.select_targets(list(reversed(candidates())), PRED, COV, cfg)
    assert first == again  # deterministic regardless of input order
    assert first == ['train:100', 'train:101', 'train:102', 'train:103', 'train:104', 'train:105',
                     'train:200', 'train:201', 'train:202', 'train:203', 'train:204', 'train:205']


def test_selection_honors_manifest_flags_and_class_coverage():
    cfg = m162e.load_config()
    cand = candidates() + [{'sample_id': 'train:107',  # exposed outcome -> dropped
                            'artifact_outcome_exposed_directly': True},
                           {'sample_id': 'train:108'}]  # uncovered class -> dropped
    got = m162e.select_targets(cand, {**PRED, 'train:107': 'selective_reading',
                                      'train:108': 'missed_direct_match'}, COV, cfg)
    assert 'train:107' not in got and 'train:108' not in got
    with pytest.raises(ValueError, match='target minimum not met'):
        m162e.select_targets(candidates()[:4], PRED, COV, cfg)


def test_leakage_check_blocks_dev50_ids_and_shared_groups():
    dev = ['train:368', 'train:1201']
    groups = {s: 'g-%s' % s for s in list(PRED) + dev}
    groups['train:100'] = 'g-train:1201'  # shares a group with a DEV50 member
    with pytest.raises(ValueError, match='shares a group with DEV50'):
        m162e.leakage_check(['train:100', 'train:101'], dev, groups.get)
    groups['train:100'] = 'g-train:100'
    groups['train:101'] = 'g-train:368'  # now a second collision
    with pytest.raises(ValueError, match='shares a group with DEV50'):
        m162e.leakage_check(['train:101'], dev, groups.get)
    groups['train:101'] = 'g-train:101'
    m162e.leakage_check(['train:101'], dev, groups.get)  # clean frame passes
    with pytest.raises(ValueError, match='is a DEV50 member'):
        m162e.leakage_check(['train:368'], dev, groups.get)


def test_assignment_deterministic_count_then_id_rotation():
    cfg = m162e.load_config()
    targets = m162e.select_targets(candidates(), PRED, COV, cfg)
    a1 = m162e.assign_exemplars(targets, POOL, PRED)
    a2 = m162e.assign_exemplars(list(reversed(targets)), POOL, PRED)
    assert a1 == a2
    assert a1['train:100'] == 'train:8'
    assert a1['train:101'] == 'train:19'
    assert a1['train:102'] == 'train:8'  # rotation wraps
    assert a1['train:200'] == 'train:21'
    assert a1['train:201'] == 'train:31'
    with pytest.raises(ValueError, match='no verified exemplar'):
        m162e.assign_exemplars(['train:106'], POOL, PRED)


def test_all_four_arms_share_the_skeleton():
    cfg = m162e.load_config()
    targets = m162e.select_targets(candidates(), PRED, COV, cfg)
    ex = {r['sample_id']: r for r in POOL}
    assigned = m162e.assign_exemplars(targets, POOL, PRED)
    ev = ['[1] some rendered evidence span', '[2] another rendered span']
    blocks = {arm: m162e.render_target_prompt(arm, ex[assigned[targets[0]]],
                                              target(targets[0]), ev)
              for arm in m162e.ARMS}
    contents = {arm: b['content'] for arm, b in blocks.items()}
    for arm in m162e.ARMS:
        assert contents[arm].startswith('Task: verify the target claim')
        assert m162e.LABEL_MENU in contents[arm]
        assert 'Target claim: target claim train:100' in contents[arm]
        assert '[1] some rendered evidence span' in contents[arm]
    # arm-specific segments
    assert 'Verified corrective explanation:' in contents['B']
    assert 'Verified corrective explanation:' in contents['C']
    assert 'Verified corrective explanation:' not in contents['A']
    assert 'Base verdict (wrong): Supported' in contents['C']
    assert 'Base verdict (wrong)' not in contents['B'] and 'Base verdict (wrong)' not in contents['A']
    assert 'Reference note:' in contents['D']
    # shared exemplar lines identical across arms (before any arm-specific segment)
    def head(c):
        return c.split('Target claim:')[0].split('Gold verdict:')[0].split('Base verdict (wrong):')[0]
    assert head(contents['A']) == head(contents['B']) == head(contents['C']) == head(contents['D'])


def test_length_control_exact_word_count_and_content_free():
    explanation = ('The displayed evidence documents commitments, not striving. Counties made '
                   'promises to allocate budgets, and no displayed passage reports allocations. '
                   'The Base equated the promise with progress and returned Supported, but only '
                   'commitments are displayed, so striving is unevidenced.')
    filler = m162e.neutral_filler(explanation)
    assert len(filler.split()) == len(explanation.split())
    # determinism
    assert filler == m162e.neutral_filler(explanation)
    # no verdict labels or mechanism-relevant words in the placebo bank output
    for label in ('Supported', 'Refuted', 'Not Enough Evidence',
                  'Conflicting Evidence/Cherrypicking', 'verdict is', 'the claim'):
        assert label.lower() not in filler.lower(), label
    # exact empty case
    assert m162e.neutral_filler('') == ''


def test_filler_matches_per_exemplar_for_arm_d():
    ex = POOL[0]
    block_b = m162e.render_target_prompt('B', ex, target('train:100'), ['ev'])
    block_d = m162e.render_target_prompt('D', ex, target('train:100'), ['ev'])
    b_text = [ln for ln in block_b['content'].split('\n')
              if ln.startswith('Verified corrective explanation:')][0]
    d_text = [ln for ln in block_d['content'].split('\n')
              if ln.startswith('Reference note:')][0]
    b_words = b_text[len('Verified corrective explanation: '):].split()
    d_words = d_text[len('Reference note: '):].split()
    assert len(b_words) == len(d_words)
    assert b_words != d_words


def test_unknown_arm_rejected():
    with pytest.raises(ValueError, match='unknown arm'):
        m162e.render_target_prompt('E', POOL[0], target('train:100'), ['ev'])
