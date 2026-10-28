import pytest
from pathlib import Path
import json

from tffv.experiments import m17_eval

def test_select_clean_development_targets():
    targets = m17_eval.select_clean_development_targets(24)
    assert len(targets) == 24
    assert len(set(targets)) == 24
    assert all(t.startswith('train:') for t in targets)

    # Check zero overlap with DEV50
    dev50 = set(json.loads(m17_eval.DEV50_BATCH.read_text('utf-8-sig'))['sample_ids'])
    assert dev50.isdisjoint(targets)

    # Check mutually disjoint groups
    all_claims = {r['sample_id']: r for r in json.loads(m17_eval.CLAIMS_MANIFEST.read_text('utf-8-sig'))}
    groups = [all_claims[t]['group_id'] for t in targets]
    assert len(set(groups)) == 24


def test_render_m17_prompt_arm_invariance():
    exemplar = {
        'sample_id': 'train:12',
        'claim': 'historical claim',
        'gold_verdict': 'Refuted',
        'base_verdict': 'Supported',
        'correction': 'historical correction text',
        'llm_judge': {
            'correction_validation': {
                'grounding_quotes': ['historical quote']
            }
        }
    }
    target_claim = 'target claim 352'
    ev_lines = ['[1] http://example.com\nSome evidence text']

    prompt_a = m17_eval.render_m17_prompt('A', exemplar, target_claim, ev_lines)
    prompt_b = m17_eval.render_m17_prompt('B', exemplar, target_claim, ev_lines)
    prompt_c = m17_eval.render_m17_prompt('C', exemplar, target_claim, ev_lines)
    prompt_d = m17_eval.render_m17_prompt('D', exemplar, target_claim, ev_lines)

    # Target block must be strictly identical across all 4 arms
    target_block = 'Target claim: target claim 352\nTarget evidence (frozen R4 Top-8):\n[1] http://example.com\nSome evidence text'
    for p in [prompt_a, prompt_b, prompt_c, prompt_d]:
        assert target_block in p
        assert 'Task: verify the target claim using only the displayed evidence.' in p

    # Arm-specific assertions
    assert 'Verified corrective explanation:' not in prompt_a
    assert 'Base verdict (wrong):' not in prompt_a

    assert 'Verified corrective explanation: historical correction text' in prompt_b
    assert 'Base verdict (wrong):' not in prompt_b

    assert 'Base verdict (wrong): Supported' in prompt_c
    assert 'Verified corrective explanation: historical correction text' in prompt_c

    assert 'Reference note:' in prompt_d
    assert 'Verified corrective explanation:' not in prompt_d
