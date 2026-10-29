import pytest
from pathlib import Path
from tffv.experiments import m17

def output_for(req):
 return {'request_id':req['request_id'],'stage':req['stage'],'evidence_supported':True,
         'verdict_assessment':'Not Enough Evidence','grounding_quotes':[req['evidence'][0]['quote']],
         'failure_mechanism':'metric_mismatch','decision':'ACCEPT','rationale':'Grounded in displayed excerpt.',
         'confidence':.8,'unsupported_facts':[],
         'provenance':{'model_id':'judge','model_revision':'rev','temperature':0.0,
                       'generation':{'temperature':0.0,'do_sample':False,'max_new_tokens':768},'prompt_sha256':req['prompt_sha256']}}

def test_source_candidate_scope_and_gold_blind_stage_one():
 rows=m17.candidates(); assert len(rows)==21
 q=m17.request(rows[0],'evidence_assessment')
 assert 'source_gold_verdict' not in q and 'candidate_correction' not in q
 assert q['sample_id'].startswith('train:') and q['evidence']

def test_stage_two_and_exact_quote_validation():
 r=m17.candidates()[0]; first=m17.request(r,'evidence_assessment'); prior=output_for(first)
 second=m17.request(r,'correction_validation',prior); out=output_for(second)
 out['verdict_assessment']=r['gold_verdict']
 m17._validate_output(out,second)
 out['grounding_quotes']=['invented fact']
 with pytest.raises(ValueError,match='not exact'):
  m17._validate_output(out,second)

def test_accept_with_unsupported_fact_fails_closed():
 q=m17.request(m17.candidates()[0],'evidence_assessment'); out=output_for(q);out['unsupported_facts']=['outside fact']
 with pytest.raises(ValueError,match='not grounded'):
  m17._validate_output(out,q)

def test_make_requests_reuses_only_empty_preflight_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(m17, 'output_path', lambda _: tmp_path / 'run')
    (tmp_path / 'run').mkdir()
    result=m17.make_requests('ignored')
    assert result.is_file()
    with pytest.raises(FileExistsError, match='not an empty'):
        m17.make_requests('ignored')

def test_build_memory_handles_none_mechanism(tmp_path, monkeypatch):
    monkeypatch.setattr(m17, 'output_path', lambda p: Path(p))
    import json
    candidates = m17.candidates()
    stage1_rows = []
    stage2_rows = []
    for r in candidates:
        q1 = m17.request(r, 'evidence_assessment')
        o1 = output_for(q1)
        o1['verdict_assessment'] = r['gold_verdict']
        stage1_rows.append(o1)
        q2 = m17.request(r, 'correction_validation', o1)
        o2 = output_for(q2)
        o2['verdict_assessment'] = r['gold_verdict']
        stage2_rows.append(o2)

    s1_path = tmp_path / 'stage1.jsonl'
    s2_path = tmp_path / 'stage2.jsonl'
    s1_path.write_text('\n'.join(json.dumps(x) for x in stage1_rows), encoding='utf-8')
    s2_path.write_text('\n'.join(json.dumps(x) for x in stage2_rows), encoding='utf-8')

    out = m17.build_memory(m17.CANDIDATES, s1_path, s2_path, tmp_path / 'mem_out')
    assert (out / 'memory.json').exists()
    assert (out / 'summary.json').exists()
    summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    assert 'UNSPECIFIED' in summary['mechanism_coverage']
    assert summary['source_candidates'] == 21

