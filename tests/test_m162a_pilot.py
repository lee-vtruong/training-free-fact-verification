"""Source-only drafting, matched prompting and held-out evaluation guards. CPU only."""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tffv.memory import heal,corrections
from tffv.verification import heal_pilot as hp
from tffv.experiments import m162a,m161
from tffv.provenance import digest
from tffv.verification.prompting import build_verification_prompt
from tffv.verification.m14_prompting import build_d1_matched_prompt


class TokenizerFixture:
    def __call__(self,text):return {'input_ids':text.split()}
    def apply_chat_template(self,messages,**kw):return '\n'.join(m['content'] for m in messages)


def fixture_record(sid,wrong=True):
    text='The independent source establishes the statement.'
    r={'sample_id':sid,'claim':'A factual historical statement','gold_verdict':'Supported',
       'group_id':sid,'record_id':sid,'is_base_error':wrong,'failure_classification':'BASE_INCORRECT_VERDICT_UNDIAGNOSED' if wrong else 'BASE_CORRECT',
       'review_status':'PENDING_REVIEW','base_prediction':{'reparsed_verdict':'Refuted' if wrong else 'Supported'},
       'evidence':[{'passage_id':sid+'p','source_url':'https://example.org','rendered_text':text}],
       'citations':[{'passage_id':sid+'p','start':0,'end':len(text),'quote':text}] if wrong else [],
       'corrected_explanation':'The supplied statement is established, not denied.',
       'draft_assessment':{'status':'PENDING_REVIEW'}}
    return r


def test_matched_views_preserve_native_baselines_ids_and_context():
    e=fixture_record('train:0');c=fixture_record('train:1',False)
    memory={'records':[e,c]};ctx=hp.source_context(e);cc=hp.source_context(c)
    slate={'error_id':'train:0','correct_id':'train:1','error_context':ctx,'correct_context':cc}
    tok=TokenizerFixture();ev=[{'text':'Target evidence differs from the source.','source_url':'https://target.org'}]
    result=hp.render_set('Target claim',ev,slate,memory,tok,hp.CORE)
    assert result['P0']['messages']==build_verification_prompt('Target claim',ev)['messages']
    assert result['P1']['messages']==build_d1_matched_prompt('Target claim',ev)['messages']
    assert result['P2']['exemplar_ids']==result['P3']['exemplar_ids']==['train:0']
    assert result['N0']['exemplar_ids']==['train:1']
    assert max(result[k]['prompt_tokens'] for k in ('P2','P3','N0'))-min(result[k]['prompt_tokens'] for k in ('P2','P3','N0'))<=2
    assert e['corrected_explanation'] not in result['P3']['messages'][1]['content']
    assert 'Refuted (incorrect for this historical example)' in result['P3']['messages'][1]['content']
    assert 'Refuted (incorrect for this historical example)' not in result['P2']['messages'][1]['content']
    with pytest.raises(ValueError,match='P4 BLOCKED'):hp.render_set('Target',ev,slate,memory,tok,[*hp.CORE,'P4'])


def test_p4_cannot_be_enabled_by_changing_a_status_string():
    e=fixture_record('train:0');e['review_status']='VERIFIED';c=fixture_record('train:1',False)
    sl={'error_id':e['sample_id'],'correct_id':c['sample_id'],'error_context':hp.source_context(e),'correct_context':hp.source_context(c)}
    with pytest.raises((ValueError,KeyError)):
        hp.render_set('Target',[],sl,{'records':[e,c]},TokenizerFixture(),[*hp.CORE,'P4'])


def test_pool_excludes_insufficient_and_requires_different_group_matching():
    e=fixture_record('train:0');c=fixture_record('train:1',False)
    class Contract:
        legacy_ids=['train:0'];batch_ids=['train:1']
        def check_memory(self,m):pass
    ctx=Contract();memory={'records':[e,c]}
    pool,match=hp.pool(memory,ctx,TokenizerFixture())
    assert set(pool)=={'train:0'} and match['train:0']['record']['sample_id']=='train:1'
    e['review_status']='INSUFFICIENT_EVIDENCE'
    with pytest.raises(ValueError,match='BLOCKED'):hp.pool(memory,ctx,TokenizerFixture())
    e['review_status']='PENDING_REVIEW';c['group_id']=e['group_id']
    with pytest.raises(ValueError,match='BLOCKED'):hp.pool(memory,ctx,TokenizerFixture())


def test_fixed_dev50_rejects_gold_fields_and_wrong_partition():
    e=fixture_record('train:0');c=fixture_record('train:1',False)
    ids=[f'train:{i}' for i in range(350,400)]
    claims={s:{'sample_id':s,'claim':'A target claim','dataset':'AVeriTeC','split':'train'} for s in ids}
    class Contract:
        legacy_ids=['train:0'];batch_ids=['train:1']
        schedule={'development_pilot_claims':ids,'development_order':ids}
        rows={s:{'partition':'DEVELOPMENT','group_id':s,'claim_sha256':hashlib.sha256(b'A target claim').hexdigest()} for s in ids}
        def check_memory(self,m):pass
    contract=Contract();mem={'records':[e,c]}
    sl,status=hp.prepare_slates(mem,claims,contract,TokenizerFixture())
    assert len(sl)==50 and status['P4']=='BLOCKED'
    claims[ids[0]]['label']='Supported'
    with pytest.raises(ValueError,match='annotations'):hp.prepare_slates(mem,claims,contract,TokenizerFixture())
    del claims[ids[0]]['label'];contract.rows[ids[0]]['partition']='TEST_QUARANTINE_EXPOSED'
    with pytest.raises(ValueError,match='DEVELOPMENT50'):hp.prepare_slates(mem,claims,contract,TokenizerFixture())


def test_pair_metrics_include_invalid_and_exact_mcnemar():
    gold=['Supported','Refuted','Not Enough Evidence','Conflicting Evidence/Cherrypicking']
    base=['Refuted','Supported',None,'Supported'];target=gold
    comp=hp.compare(gold,base,target,n_boot=50,n_perm=99)
    assert comp['transitions']['WC']==4 and comp['exact_mcnemar_p']==.125
    assert comp['accuracy_delta']==1.0 and comp['macro_f1_delta']==1.0
    assert 'UW' not in comp['invalid_transitions'] and comp['invalid_transitions']['UC']==1
    assert hp.metric(gold,base)['valid_rate']==.75
    assert hp.holm({'a':.01,'b':.04})=={'a':.02,'b':.04}


def test_no_gpu_without_execute_and_no_gold_before_completion(tmp_path,monkeypatch):
    monkeypatch.setattr(m162a,'environment',lambda *a:pytest.fail('No model preflight allowed'))
    with pytest.raises(ValueError,match='--execute'):m162a.execute(tmp_path,'prediction')
    monkeypatch.setattr(m161,'output_path',lambda p:p)
    monkeypatch.setattr(m162a,'check',lambda r:{})
    with pytest.raises(FileNotFoundError):m162a.evaluate(tmp_path,tmp_path/'GOLD_MUST_NOT_BE_OPENED',tmp_path/'results')


def test_source_draft_assistant_cannot_verify_or_hide_cases():
    class Contract:
        legacy_ids=['train:0'];batch_ids=[]
        rows={'train:0':{'group_id':'train:0','claim_sha256':hashlib.sha256(b'claim').hexdigest()}}
        def require(self,ids):assert ids==['train:0']
        check_memory=heal.SourceContract.check_memory
    contract=Contract()
    r=heal.make_record('train:0','claim','Refuted',{'predicted_verdict':'Supported','raw_generation':'Verdict: Supported'},
        [{'passage_id':'p','text':'A related topic only.','text_sha256':digest('A related topic only.'),'source_url':'https://example.org'}],
        {'origin':'SOFTWARE_TEST_ONLY','artifact_checks_passed':True},contract)
    memory={'partition_seal_sha256':heal.PARTITION_SEAL,'records':[r]}
    d={'errors':{'train:0':{'record_id':r['record_id'],'status':'INSUFFICIENT_EVIDENCE','reason':'Related topic does not establish gold.','limitations':'Missing decisive fact.','explanation':None}}}
    out=corrections.complete(memory,d,contract)
    assert out['records'][0]['review_status']=='INSUFFICIENT_EVIDENCE' and out['records'][0]['reviews']==[]
    d['errors']['train:0']['status']='VERIFIED'
    with pytest.raises(ValueError,match='cannot assign'):corrections.complete(memory,d,contract)
    with pytest.raises(ValueError,match='every incorrect'):corrections.complete(memory,{'errors':{}},contract)


def test_receipt_reuse_rejects_mutation_and_unhashed_output(tmp_path):
    from tffv.provenance import file_hash
    heal.save(tmp_path/'batch.json',{'fixture':'SOFTWARE_TEST_ONLY'})
    folder=tmp_path/'train-350';folder.mkdir()
    output=folder/'retrieval.json';heal.save(output,{'evidence':'SOFTWARE_TEST_ONLY'})
    rel=output.relative_to(tmp_path).as_posix();receipt=folder/'retrieval.receipt.json'
    obj={'sample_id':'train:350','stage':'retrieval','batch_sha256':file_hash(tmp_path/'batch.json'),
         'output':rel,'files_sha256':{rel:file_hash(output)}}
    heal.save(receipt,obj)
    assert m162a.verified_step(tmp_path,receipt)==output
    m161.atomic(output,{'evidence':'modified'})
    with pytest.raises(ValueError,match='hash/path mismatch'):m162a.verified_step(tmp_path,receipt)
    obj['files_sha256'][rel]=file_hash(output);obj['output']='batch.json';m161.atomic(receipt,obj)
    with pytest.raises(ValueError,match='not covered'):m162a.verified_step(tmp_path,receipt)


def test_context_overflow_fails_without_truncating_target_evidence():
    e=fixture_record('train:0');c=fixture_record('train:1',False)
    sl={'error_id':e['sample_id'],'correct_id':c['sample_id'],'error_context':hp.source_context(e),'correct_context':hp.source_context(c)}
    with pytest.raises(ValueError,match='Context reservation overflow'):
        hp.render_set('Target '*9000,[{'text':'Target evidence','source_url':'https://target.org'}],sl,
                      {'records':[e,c]},TokenizerFixture(),hp.CORE)
