"""Cache-first contracts: fixture artifacts only, no model or retrieval invocation."""
import json
from pathlib import Path

import pytest

from tffv.experiments import m16_cache as cache
from tffv.provenance import digest, file_hash


def put(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+"\n",encoding='utf-8')


def line(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(x,sort_keys=True)+"\n" for x in rows),encoding='utf-8')


def fixture_run(tmp_path):
    root=tmp_path/'project'; (root/'outputs').mkdir(parents=True,exist_ok=True)
    cache.ROOT=root
    run=root/'outputs'/'canonical'; sid='train:7'; claim=' A   frozen claim '
    batch={'sample_ids':[sid],'claims':[{'sample_id':sid,'dataset':'AVeriTeC','split':'train','claim':claim}],
           'gold_accessed':False,'configuration':{'retrieval':{'embedding':{'model_id':'embed'},'reranker':{'model_id':'rank'}},
           'candidate_policy':'fixed','backbone':{'model_id':'frozen'},'generation':{'do_sample':False}},
           'status':{'active_conditions':['P0']},'source_memory_sha256':'m'*64,'source_memory_path':'SOURCE_ONLY',
           'slates':[{'sample_id':sid,'fixture':'source-only'}]}
    put(run/'batch.json',batch); bh=file_hash(run/'batch.json'); d=run/'train-7'
    put(d/'input/claims_manifest.json',{'annotation_values_used':False,'source_sha256':'d'*64})
    outputs={}
    for stage,name,value in [('acquire','store.json',{'store':'fixed'}),('candidates','pool.json',{'pool':'fixed'})]:
        p=d/(stage+'-attempt-001')/name;put(p,value);outputs[stage]=p
    r4=d/'retrieval-attempt-001'/'R4.jsonl';line(r4,[{'sample_id':sid,'claim_sha256':digest(claim),'evidence':[{'text':'one'}]*8}]);outputs['retrieval']=r4
    pred=d/'predictions-attempt-001'/'predictions.jsonl';line(pred,[{'sample_id':sid,'condition':'P0','raw_generation':'Verdict: Supported','predicted_verdict':'Supported','input_ids_sha256':'i'*64,'gold_accessed':False}]);outputs['predictions']=pred
    for stage,p in outputs.items():
        rel=p.relative_to(run).as_posix();put(d/(stage+'.receipt.json'),{'sample_id':sid,'stage':stage,'batch_sha256':bh,'output':rel,'files_sha256':{rel:file_hash(p)},'elapsed_seconds':1.})
    prompts={'batch_sha256':bh,'gold_accessed':False,'r4_sha256':{sid:file_hash(r4)},'inputs':{sid:{'P0':{'messages_sha256':'p'*64,'input_ids_sha256':'i'*64}}}}
    put(run/'sealed_prompts.json',prompts)
    return root,run,sid


def test_same_inputs_have_same_contract_keys_and_full_cache_hit(tmp_path):
    root,run,sid=fixture_run(tmp_path)
    a=cache.index(run,root/'outputs'/'index-a.json')
    b=cache.index(run,root/'outputs'/'index-b.json')
    assert a['records'][0]['evidence_cache_key']==b['records'][0]['evidence_cache_key']
    assert a['records'][0]['prediction_cache_keys']==b['records'][0]['prediction_cache_keys']
    assert cache.select(root/'outputs'/'index-a.json',[sid],'P0')['reusable_ids']==[sid]


def test_changed_retrieval_or_prompt_rejects_cache_admission(tmp_path):
    root,run,_=fixture_run(tmp_path)
    batch=json.loads((run/'batch.json').read_text());batch['configuration']['retrieval']['embedding']['model_id']='different';put(run/'batch.json',batch)
    with pytest.raises(ValueError,match='Prompt seal mismatch'):
        cache.index(run,root/'outputs'/'bad.json')
    root,run,_=fixture_run(tmp_path)
    prompts=json.loads((run/'sealed_prompts.json').read_text());prompts['inputs']['train:7']['P0']['input_ids_sha256']='changed';put(run/'sealed_prompts.json',prompts)
    with pytest.raises(ValueError,match='Prediction/prompt'):
        cache.index(run,root/'outputs'/'bad2.json')


def test_missing_evidence_fails_cleanly_and_resume_does_not_duplicate(tmp_path):
    root,run,sid=fixture_run(tmp_path); index=root/'outputs'/'index.json';cache.index(run,index)
    plan=cache.select(index,[sid,'train:8'],'P0')
    assert plan['reusable_ids']==[sid] and plan['missing_ids']==['train:8'] and plan['retrieval_called'] is False
    assert cache.select(index,[sid],'P0')==cache.select(index,[sid],'P0')
    (run/'train-7'/'retrieval-attempt-001'/'R4.jsonl').unlink()
    with pytest.raises(ValueError,match='Receipt output missing'):
        cache.index(run,root/'outputs'/'missing.json')
    assert not (root/'outputs'/'missing.json').exists()


def test_evaluation_only_reuses_predictions_without_model(tmp_path):
    root,run,_=fixture_run(tmp_path); index=root/'outputs'/'index.json';cache.index(run,index)
    gold=root/'gold.json';put(gold,[{}, {}, {}, {}, {}, {}, {}, {'label':'Supported'}])
    result=cache.evaluate_only(index,gold,root/'outputs'/'score.json')
    assert result['metrics']['P0']['accuracy']==1. and result['metrics']['P0']['macro_f1']==.25


def test_cross_partition_source_development_group_leakage_is_blocked():
    cache.require_no_source_target_leakage({'source-a'},{'development-b'})
    with pytest.raises(ValueError,match='group leakage'):
        cache.require_no_source_target_leakage({'shared-near-duplicate'},{'shared-near-duplicate'})
