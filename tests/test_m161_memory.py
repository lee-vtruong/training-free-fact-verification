"""CPU contracts only; constructed cases never measure model quality."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from tffv.memory import heal
from tffv.experiments import m161
from tffv.provenance import digest,file_hash

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def memory():
    class Contract:
        rows={'train:0':{'group_id':'g0','claim_sha256':hashlib.sha256(b'A source claim').hexdigest()}}
        def require(self,ids):
            if any(s!='train:0' for s in ids): raise ValueError('Not SOURCE')
        check_memory=heal.SourceContract.check_memory
    c=Contract()
    r=heal.make_record('train:0','A source claim','Refuted',
        {'predicted_verdict':'Supported','raw_generation':'Verdict: Supported','prompt_sha256':'a'*64},
        [{'passage_id':'p0','text':'The premise is false.','text_sha256':digest('The premise is false.'),'source_url':'https://example.org/evidence'}],
        {'origin':'CPU_TEST_FIXTURE','artifact_checks_passed':True},c)
    return {'schema_version':'heal-bank-v1','origin':'CPU_TEST_FIXTURE','partition_seal_sha256':heal.PARTITION_SEAL,'records':[r]},c


def candidate(memory):
    m,c=memory
    r=m['records'][0]
    return heal.apply_candidates(m,[{'sample_id':'train:0','record_sha256':r['content_sha256'],
        'corrected_explanation':'The evidence explicitly denies the premise.',
        'citations':[{'passage_id':'p0','start':0,'end':21,'quote':'The premise is false.'}],
        'candidate_mechanisms':['negation_polarity']}],c),c


def sheets(memory,tmp_path):
    m,c=candidate(memory)
    heal.export_reviews(m,tmp_path/'sheets')
    a,b=(heal.read(tmp_path/'sheets'/f'reviewer_{i}.json') for i in (1,2))
    for name,sheet in [('human-A',a),('human-B',b)]:
        sheet[0].update(reviewer_id=name,reviewer_type='human',independent_review=True,
            reviewed_at='2026-10-27T10:00:00+07:00',decision='VERIFIED',reason='CPU validation fixture only',
            evidence_sufficient=True,correction_grounded=True,gold_relation_supported=True,
            failure_category='REASONING_FAILURE_SUPPORTED_BY_EVIDENCE',confirmed_mechanisms=['negation_polarity'])
    return m,c,a,b


def test_sealed_source_and_fixed_25_exclude_all_other_partitions():
    c=heal.SourceContract(ROOT)
    assert len(c.legacy_ids)==50 and len(c.batch_ids)==25
    c.require(c.legacy_ids+c.batch_ids)
    for part in ('DEVELOPMENT','TEST_QUARANTINE_EXPOSED','LEGACY_EXPLORATORY'):
        sid=next(s for s,r in c.rows.items() if r['partition']==part)
        with pytest.raises(ValueError,match='SOURCE'): c.require([sid])


def test_gold_only_and_wrong_gold_hold_ids_and_hide_rationale(memory):
    m,c=memory
    assert heal.summary(m)['verified_corrective_rationales']==0
    g=heal.example_view(m,['train:0'],'gold_only',c)[0]
    w=heal.example_view(m,['train:0'],'wrong_gold',c)[0]
    assert g['record_id']==w['record_id'] and g['evidence']==w['evidence']
    assert 'base_verdict' not in g and w['base_verdict']=='Supported'
    assert 'corrected_explanation' not in w
    with pytest.raises(ValueError,match='two completed'): heal.example_view(m,['train:0'],'wrong_gold_verified',c)


def test_two_reviews_same_candidate_enable_only_verified_view(memory,tmp_path):
    m,c,a,b=sheets(memory,tmp_path)
    result=heal.import_reviews(m,a,b,c)
    assert result['records'][0]['review_status']=='VERIFIED'
    assert heal.summary(result)['verified_mechanism_support']=={'negation_polarity':1}
    assert heal.example_view(result,['train:0'],'wrong_gold_verified',c)[0]['citations']


@pytest.mark.parametrize('mutation', ['same_person','stale','false_grounding','edited_context','nonhuman'])
def test_review_forgery_and_staleness_rejected(memory,tmp_path,mutation):
    m,c,a,b=sheets(memory,tmp_path)
    if mutation=='same_person': b[0]['reviewer_id']='human-A'
    elif mutation=='stale': b[0]['candidate_sha256']='0'*64
    elif mutation=='false_grounding': b[0]['correction_grounded']=False
    elif mutation=='edited_context': b[0]['gold_verdict']='Supported'
    else: b[0]['reviewer_type']='assistant'
    with pytest.raises(ValueError): heal.import_reviews(m,a,b,c)


@pytest.mark.parametrize('decision',['REJECTED','INSUFFICIENT_EVIDENCE'])
def test_negative_review_is_never_verified(memory,tmp_path,decision):
    m,c,a,b=sheets(memory,tmp_path)
    b[0]['decision']=decision
    result=heal.import_reviews(m,a,b,c)
    assert result['records'][0]['review_status']==decision
    with pytest.raises(ValueError): heal.example_view(result,['train:0'],'wrong_gold_verified',c)


def test_parser_error_is_distinct_and_qa_cannot_enter_memory(memory):
    m,c=memory
    old=m['records'][0]
    r=heal.make_record('train:0',old['claim'],old['gold_verdict'],
        {'predicted_verdict':'Refuted','raw_generation':'Verdict: Supported','prompt_sha256':'a'*64},
        old['evidence'],old['provenance'],c)
    m['records']=[r]
    c.check_memory(m)
    assert heal.summary(m)['parser_errors']==1
    with pytest.raises(ValueError): heal.example_view(m,['train:0'],'gold_only',c)
    r['gold_justification']='FORBIDDEN'
    heal.refresh(r)
    with pytest.raises(ValueError,match='QA/justification'): c.check_memory(m)


def test_out_of_display_citation_rejected(memory):
    m,c=memory
    r=m['records'][0]
    with pytest.raises(ValueError,match='span/quote'):
        heal.apply_candidates(m,[{'sample_id':'train:0','record_sha256':r['content_sha256'],
            'corrected_explanation':'unsupported','citations':[{'passage_id':'p0','start':0,'end':999,'quote':'invented'}]}],c)


def test_receipt_hash_corruption_rejected_without_recompute(tmp_path):
    (tmp_path/'batch.json').write_text('{}')
    p=tmp_path/'done.txt';p.write_text('complete')
    receipt=tmp_path/'receipt.json'
    heal.save(receipt,{'batch_sha256':file_hash(tmp_path/'batch.json'),'output':'done.txt','files_sha256':{'done.txt':file_hash(p)}})
    assert m161.verify_receipt(tmp_path,receipt)==p
    p.write_text('changed')
    with pytest.raises(ValueError,match='hash/path'): m161.verify_receipt(tmp_path,receipt)


def test_unsealed_attempt_gets_new_directory_and_sealed_attempt_reused(tmp_path,monkeypatch):
    (tmp_path/'batch.json').write_text('{}')
    (tmp_path/'train-0/acquire-attempt-001').mkdir(parents=True)
    monkeypatch.setattr(m161,'check_batch',lambda run:{})
    monkeypatch.setattr(m161,'resources',lambda run,b:0)
    called=[]
    def action(out):
        called.append(out.name);out.mkdir();p=out/'ok.json';p.write_text('{}');return p
    first=m161.step(tmp_path,'train:0','acquire',action)
    second=m161.step(tmp_path,'train:0','acquire',action)
    assert called==['acquire-attempt-002'] and first==second


def test_run_without_execute_never_calls_preflight(monkeypatch,tmp_path):
    monkeypatch.setattr(m161,'preflight',lambda *a,**k:pytest.fail('Must not load environment/model paths'))
    with pytest.raises(ValueError,match='--execute'): m161.run_batch(tmp_path)


def test_shard_boundaries_and_actual_schedule():
    from tffv.retrieval.acquire_subset import archive_for
    from tffv.retrieval.remote_zip import validate_plan
    plan=heal.read(ROOT/'outputs/m11-corpus-plan-20261021-042129/acquisition_plan.json') if (ROOT/'outputs/m11-corpus-plan-20261021-042129/acquisition_plan.json').exists() else None
    if plan is None:
        # Boundary algorithm fixture, not remote-data or performance evidence.
        archives=[{'path':f'train_{a}_{b}.zip'} for a,b in ((0,999),(1000,1999),(2000,3067))]
    else: archives=validate_plan(plan)
    assert archive_for(999,archives)['path'].endswith('train_0_999.zip')
    assert archive_for(1000,archives)['path'].endswith('train_1000_1999.zip')
    assert archive_for(2000,archives)['path'].endswith('train_2000_3067.zip')
    c=heal.SourceContract(ROOT)
    assert len({archive_for(int(s.split(':')[1]),archives)['path'] for s in c.batch_ids})==3


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    ids=[f'train:{i}' for i in range(350,375)]
    rows=[{'sample_id':s,'claim':'Input source '+s,'dataset':'AVeriTeC','split':'train'} for s in ids]
    class Contract:
        batch_ids=ids
        seal={'dataset_sha256':'d'*64}
    Contract.rows={r['sample_id']:{'claim_sha256':hashlib.sha256(r['claim'].encode()).hexdigest()} for r in rows}
    monkeypatch.setattr(m161,'SourceContract',lambda root:Contract())
    monkeypatch.setattr(m161,'output_path',lambda path:Path(path).resolve())
    monkeypatch.setattr(m161,'code_hash',lambda:'c'*64)
    source=tmp_path/'input';source.mkdir()
    m161.write_lines(source/'claims.jsonl',rows)
    heal.save(source/'claims_manifest.json',{'source_sha256':'d'*64,'annotation_values_used':False,
        'split':'train','claims_sha256':file_hash(source/'claims.jsonl')})
    # Real pinned plan contains metadata only; supply a format-equivalent fixture if absent.
    plan=ROOT/'configs/m161_acquisition_plan.json'
    if not plan.exists():
        rev='a'*40
        a=[]
        for low,high in ((0,999),(1000,1999),(2000,3067)):
            path=f'data_store/knowledge_store/train/train_{low}_{high}.zip'
            a.append({'path':path,'url':f'https://huggingface.co/chenxwh/AVeriTeC/resolve/{rev}/{path}',
                      'compressed_bytes':1,'sha256':'0'*64})
        plan=tmp_path/'plan.json';heal.save(plan,{'repo_id':'chenxwh/AVeriTeC','split':'train','resolved_revision':rev,'archives':a})
    run=m161.prepare(tmp_path/'run',plan,source/'claims_manifest.json',ROOT/'configs/m161_source_batch25.json')
    return run


def test_prepare_and_structural_preflight_never_acquire_or_load_models(prepared,monkeypatch):
    import tffv.retrieval.acquire_subset as acquisition
    monkeypatch.setattr(acquisition,'acquire',lambda *a,**k:pytest.fail('Network forbidden in preparation'))
    report=m161.preflight(prepared,structural=True)
    assert report['source_count']==25 and report['ready_to_execute'] is False
    assert report['hardware_checks']=='NOT_RUN'
    assert not (prepared/'environment.json').exists()


def test_resume_refuses_changed_code_and_per_claim_inputs(prepared,monkeypatch):
    monkeypatch.setattr(m161,'code_hash',lambda:'0'*64)
    with pytest.raises(ValueError,match='Code changed'): m161.check_batch(prepared)
    monkeypatch.setattr(m161,'code_hash',lambda:'c'*64)
    p=prepared/'train-350/input/claims.jsonl'
    p.write_text('{}\n')
    with pytest.raises(ValueError,match='Per-claim'): m161.check_batch(prepared)


def test_resource_gates_and_live_lock_refusal(prepared,monkeypatch):
    b=m161.check_batch(prepared)
    monkeypatch.setattr(m161.shutil,'disk_usage',lambda p:type('Disk',(),{'free':0})())
    with pytest.raises(ValueError,match='Free disk'): m161.resources(prepared,b)
    with m161.run_lock(prepared):
        with pytest.raises(FileExistsError):
            with m161.run_lock(prepared): pass
        with pytest.raises(ValueError,match='live/foreign'):
            with m161.run_lock(prepared,recover=True): pass


def test_gold_join_rejects_development_ids_before_opening_file(tmp_path):
    c=heal.SourceContract(ROOT)
    sid=next(s for s,r in c.rows.items() if r['partition']=='DEVELOPMENT')
    with pytest.raises(ValueError,match='SOURCE'): heal.source_gold(tmp_path/'does-not-exist.json',[sid],c)


def test_immutable_identity_and_duplicate_merge_fail_closed(memory):
    m,c=memory
    with pytest.raises(ValueError,match='Duplicate'): heal.merge_memories([m,m],c)
    r=m['records'][0]
    r['gold_verdict']='Supported'
    r['is_base_error']=False
    heal.refresh(r)
    with pytest.raises(ValueError,match='identity'): c.check_memory(m)


def test_provenance_unknown_and_missing_evidence_are_distinct(memory):
    m,c=memory;r=m['records'][0]
    unknown=heal.make_record('train:0',r['claim'],r['gold_verdict'],
        {'predicted_verdict':'Supported','raw_generation':'Verdict: Supported'},r['evidence'],
        {'origin':'CPU_TEST_FIXTURE','artifact_checks_passed':False},c)
    insufficient=heal.make_record('train:0',r['claim'],r['gold_verdict'],
        {'predicted_verdict':'Supported','raw_generation':'Verdict: Supported'},[],
        {'origin':'CPU_TEST_FIXTURE','artifact_checks_passed':True},c)
    assert unknown['review_status']=='PROVENANCE_UNKNOWN'
    assert insufficient['review_status']=='INSUFFICIENT_EVIDENCE'
    with pytest.raises(ValueError): heal.example_view({**m,'records':[unknown]},['train:0'],'gold_only',c)


def test_formal_schemas_and_unfilled_review_exports(memory,tmp_path):
    import jsonschema
    m,c=memory
    schema=heal.read(ROOT/'schemas/heal_memory_v1.schema.json')
    review_schema=heal.read(ROOT/'schemas/heal_review_v1.schema.json')
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator.check_schema(review_schema)
    # Same schema shape, explicitly constructed software fixture only.
    m=deepcopy(m);m['origin']='OWN_PROJECT_LEGACY_FOLD_A'
    jsonschema.validate(m,schema)
    heal.export_reviews(m,tmp_path/'blank')
    for i in (1,2): jsonschema.validate(heal.read(tmp_path/'blank'/f'reviewer_{i}.json'),review_schema)
