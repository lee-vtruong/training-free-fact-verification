import pytest
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
