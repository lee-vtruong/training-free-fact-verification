"""Minimal exploratory HEAL pilot: CPU prepare/render/score; GPU only via explicit execute."""
from pathlib import Path
import argparse
from collections import Counter
import json
import os
import signal
import subprocess
import sys
import time

from tffv.memory.heal import SourceContract,PARTITION_SEAL,SCHEDULE_HASH,read,save,jsonl,indexed
from tffv.provenance import digest,file_hash
from tffv.experiments import m161
from tffv.verification import heal_pilot as hp

ROOT=Path(__file__).resolve().parents[3]


def prepare(output,memory,claims_manifest,tokenizer_path):
    contract=SourceContract(ROOT);dest=m161.output_path(output)
    from tffv.retrieval.acquire_subset import archive_for
    from tffv.retrieval.remote_zip import validate_plan
    plan=ROOT/'configs/m161_acquisition_plan.json'
    if file_hash(plan)!=m161.ACQUISITION_PLAN_SHA:raise ValueError('Pinned shard-aware acquisition plan changed')
    archives=validate_plan(read(plan))
    shards={sid:archive_for(int(sid.split(':')[1]),archives)['path'] for sid in contract.schedule['development_pilot_claims']}
    cm=read(claims_manifest);cf=Path(claims_manifest).resolve().parent/'claims.jsonl'
    if cm['annotation_values_used'] is not False or cm['source_sha256']!=contract.seal['dataset_sha256'] or file_hash(cf)!=cm['claims_sha256']:
        raise ValueError('Label-free claim export mismatch')
    mem=read(memory);claims=indexed(jsonl(cf));tok=hp.tokenizer(Path(tokenizer_path))
    slates,status=hp.prepare_slates(mem,claims,contract,tok)
    cfg=read(ROOT/'configs/m161_source_batch25.json')
    if digest(cfg['retrieval'])!=m161.FROZEN_RETRIEVAL_CONFIG or cfg['backbone']['revision']!=m161.MODEL_REVISION or cfg['generation']!={'max_new_tokens':512,'enable_thinking':False,'do_sample':False,'dtype':'bfloat16','max_context_length':8192}:
        raise ValueError('Frozen model/retrieval/generation config changed')
    cfg['limits']['wall_seconds']=86400
    protocol=read(ROOT/'configs/m162a_protocol.json')
    if protocol['k']!=1 or protocol['seed']!=42 or protocol['output_tokens']!=512:raise ValueError('Initial pilot protocol changed')
    condition_files=[ROOT/p for p in protocol['condition_files']]
    expected={'P0':'native_D0','P1':'native_D1_matched','P2':'gold_only','P3':'wrong_gold','P4':'wrong_gold_verified','N0':'non_error_gold_only'}
    if set(protocol['condition_files'])!={f'configs/m162a_{k}.json' for k in expected} or protocol['n_bootstrap']!=10000 or protocol['n_paired_permutations']!=19999 or protocol['metrics']!=['Accuracy','Macro-F1']:
        raise ValueError('Initial analysis/condition family changed')
    for path in condition_files:
        cc=read(path)
        if cc!={'condition':path.stem.split('_')[-1],'template':expected[path.stem.split('_')[-1]],'max_new_tokens':512,'enable_thinking':False,'do_sample':False}:
            raise ValueError('Condition inference configuration changed')
    data={'schema_version':'m162a-dev50-v1','scope':'EXPLORATORY','partition_seal_sha256':PARTITION_SEAL,
        'schedule_sha256':SCHEDULE_HASH,'sample_ids':contract.schedule['development_pilot_claims'],
        'claims':[claims[s] for s in contract.schedule['development_pilot_claims']],
        'configuration':cfg,'protocol':protocol,'source_memory_path':str(Path(memory).resolve()),
        'source_memory_sha256':file_hash(memory),'slates':slates,'status':status,
        'tokenizer_path':str(Path(tokenizer_path).resolve()),'code_sha256':m161.code_hash(),
        'input_hashes':{str(Path(p).resolve()):file_hash(p) for p in [claims_manifest,cf,ROOT/'configs/m161_source_batch25.json',ROOT/'configs/m161_acquisition_plan.json',ROOT/'configs/m162a_protocol.json',*condition_files]},
        'acquisition_plan_path':str(plan),'shards':shards,'gold_accessed':False}
    if dest.exists():
        if read(dest/'batch.json')!=data:raise ValueError('Existing pilot input/protocol differs; no overwrite/reselection')
        check(dest);return dest
    dest.mkdir(parents=True,exist_ok=False);save(dest/'batch.json',data)
    for r in data['claims']:
        d=dest/r['sample_id'].replace(':','-')/'input';d.mkdir(parents=True)
        m161.write_lines(d/'claims.jsonl',[r])
        save(d/'claims_manifest.json',{'dataset':'AVeriTeC','split':'train','source_role':'source_train',
          'annotation_values_used':False,'source_sha256':contract.seal['dataset_sha256'],'records':1,
          'claims_sha256':file_hash(d/'claims.jsonl'),'role':'DEVELOPMENT_RESEARCH_NOT_HISTORICAL_SOURCE'})
    save(dest/'state.json',{'spent_seconds':0.,'active_started_epoch':None,'active_child_pid':None,
                           'host':m161.socket.gethostname(),'completed_ids':[],'retrieval_ids':[],'prediction_ids':[]})
    print(json.dumps({'prepared':str(dest),'development_claims':50,**status,'model_calls':0},indent=2))
    return dest


def check(run):
    run=m161.output_path(run);b=read(run/'batch.json');c=SourceContract(ROOT)
    if b['sample_ids']!=c.schedule['development_pilot_claims'] or len(set(b['sample_ids']))!=50 or b['partition_seal_sha256']!=PARTITION_SEAL or b['schedule_sha256']!=SCHEDULE_HASH:
        raise ValueError('Locked DEV50 identity changed')
    if b['code_sha256']!=m161.code_hash():raise ValueError('Implementation changed after pilot freeze')
    for p,h in b['input_hashes'].items():
        if file_hash(p)!=h:raise ValueError('Pilot frozen input changed: '+p)
    if file_hash(b['source_memory_path'])!=b['source_memory_sha256']:raise ValueError('Source bank changed')
    mem=read(b['source_memory_path']);c.check_memory(mem)
    expected=read(ROOT/'configs/m161_source_batch25.json');expected['limits']['wall_seconds']=86400
    if expected!=b['configuration'] or b['protocol']!=read(ROOT/'configs/m162a_protocol.json'):raise ValueError('Embedded protocol changed')
    tok=hp.tokenizer(Path(b['tokenizer_path']))
    claims=indexed(b['claims'])
    if set(claims)!=set(b['sample_ids']):raise ValueError('Extra/missing DEV claims')
    slates,status=hp.prepare_slates(mem,claims,c,tok)
    if slates!=b['slates'] or status!=b['status']:raise ValueError('Exemplar slates/eligibility changed')
    original=indexed(jsonl(next(p for p in b['input_hashes'] if Path(p).name=='claims.jsonl')))
    for r in b['claims']:
        sid=r['sample_id'];d=run/sid.replace(':','-')/'input'
        if c.rows[sid]['partition']!='DEVELOPMENT' or r!=original[sid] or jsonl(d/'claims.jsonl')!=[r]:raise ValueError('DEV input leakage/change')
        if read(d/'claims_manifest.json')['claims_sha256']!=file_hash(d/'claims.jsonl'):raise ValueError('DEV input manifest changed')
    return b


def verified_step(run,path):
    result=m161.verify_receipt(run,path).resolve();receipt=read(path)
    if not result.is_relative_to(run.resolve()) or not result.is_file():
        raise ValueError('Receipt output escaped run or is missing')
    if result.relative_to(run.resolve()).as_posix() not in receipt['files_sha256']:
        raise ValueError('Receipt output is not covered by its file hashes')
    if receipt['sample_id']!=path.parent.name.replace('-',':',1) or receipt['stage']!=path.name.removesuffix('.receipt.json'):
        raise ValueError('Receipt stage/source identity mismatch')
    return result


def step(run,sid,name,action):
    folder=run/sid.replace(':','-');receipt=folder/(name+'.receipt.json')
    if receipt.exists():return verified_step(run,receipt)
    attempt=1
    while (folder/f'{name}-attempt-{attempt:03d}').exists():attempt+=1
    out=folder/f'{name}-attempt-{attempt:03d}';b=check(run);m161.resources(run,b)
    print(f'START {sid} {name} attempt={attempt} output={out}',flush=True);t=time.time()
    result=Path(action(out)).resolve()
    if not result.is_relative_to(out.resolve()):raise ValueError('Step output escaped attempt')
    m161.resources(run,b)
    save(receipt,{'sample_id':sid,'stage':name,'batch_sha256':file_hash(run/'batch.json'),'output':result.relative_to(run).as_posix(),
                  'files_sha256':{p.relative_to(run).as_posix():file_hash(p) for p in out.rglob('*') if p.is_file()},'elapsed_seconds':time.time()-t})
    return result


def worker(run,sid,phase):
    b=check(run)
    if sid not in b['sample_ids']:raise ValueError('Worker not DEV50')
    if phase=='retrieval':
        from tffv.retrieval.acquire_subset import acquire
        from tffv.retrieval.audit_subset import audit
        from tffv.retrieval.prepare_store_candidates import prepare_candidates
        from tffv.retrieval.store_retrieval import run as retrieve
        acq=step(run,sid,'acquire',lambda out:acquire(b['acquisition_plan_path'],out,[int(sid.split(':')[1])],execute=True))
        aud=step(run,sid,'audit',lambda out:audit(acq,out,max_record_mib=64))
        cm=run/sid.replace(':','-')/'input/claims_manifest.json'
        ca=step(run,sid,'candidates',lambda out:prepare_candidates(aud,cm,out,policy=b['configuration']['candidate_policy']))
        step(run,sid,'retrieval',lambda out:retrieve(ca,out,methods=('R1','R2','R3','R4'),config=b['configuration']['retrieval'],index_cache=out/'embedding_cache')/'R4.jsonl')
    else:
        step(run,sid,'predictions',lambda out:predict_one(run,sid,out))


def seal_prompts(run):
    run=m161.output_path(run);b=check(run);mem=read(b['source_memory_path']);tok=hp.tokenizer(Path(b['tokenizer_path']))
    all_inputs={};rawhashes={};slates=indexed(b['slates'])
    for r in b['claims']:
        sid=r['sample_id'];folder=run/sid.replace(':','-')
        for stage in ('acquire','audit','candidates','retrieval'):verified_step(run,folder/(stage+'.receipt.json'))
        path=verified_step(run,folder/'retrieval.receipt.json');rr=jsonl(path)
        if len(rr)!=1 or rr[0]['sample_id']!=sid or len(rr[0]['evidence'])!=8 or rr[0]['claim_sha256']!=digest(r['claim']):raise ValueError('Target Top8 identity/coverage mismatch')
        rawhashes[sid]=file_hash(path)
        all_inputs[sid]=hp.render_set(r['claim'],rr[0]['evidence'],slates[sid],mem,tok,b['status']['active_conditions'])
    obj={'batch_sha256':file_hash(run/'batch.json'),'inputs':all_inputs,'r4_sha256':rawhashes,'gold_accessed':False}
    if (run/'sealed_prompts.json').exists():
        if read(run/'sealed_prompts.json')!=obj:raise ValueError('Sealed prompts/evidence changed')
    else:save(run/'sealed_prompts.json',obj)
    print('READY: all50 prompts/evidence sealed; matched P2/P3/N0 actual token lengths within2; P4='+b['status']['P4'])


def environment(run):
    b=check(run)
    import torch,importlib.metadata,platform
    from tffv.retrieval.frozen import validate_spec,snapshot_fingerprint
    from tffv.verification.backbone import validate_backbone_spec
    if platform.system()!='Linux' or not torch.cuda.is_available():raise ValueError('Linux/CUDA required; no model fallback')
    if torch.cuda.mem_get_info()[0]<24*1024**3:raise ValueError('Need24 GiB free VRAM')
    m161.resources(run,b);cfg=b['configuration'];fps={}
    for k in ('embedding','reranker'):
        p=validate_spec(cfg['retrieval'][k],cfg['retrieval'][k]['model_id'],verify_bytes=True);fps[k]=snapshot_fingerprint(p)['snapshot_sha256']
    backbone_path=validate_backbone_spec(cfg['backbone'])
    if backbone_path.name!=m161.MODEL_REVISION:raise ValueError('Backbone revision/directory mismatch')
    hp.tokenizer(backbone_path)
    fps['backbone']=snapshot_fingerprint(backbone_path)['snapshot_sha256']
    obj={'batch_sha256':file_hash(run/'batch.json'),'model_fingerprints':fps,
         'dependencies':{k:importlib.metadata.version(k) for k in ('torch','transformers','numpy')},
         'gpu_name':torch.cuda.get_device_name(0),'torch_cuda_version':torch.version.cuda,'new_model_calls':0}
    if (run/'environment.json').exists():
        if read(run/'environment.json')!=obj:raise ValueError('Model/runtime changed')
    else:save(run/'environment.json',obj)
    return obj


def predict_one(run,sid,out):
    from tffv.verification.backbone import Qwen3FrozenBackbone
    from tffv.verification.ontology import normalize_verdict
    b=check(run);sealed=read(run/'sealed_prompts.json')
    if sealed['batch_sha256']!=file_hash(run/'batch.json'):raise ValueError('Sealed prompt parent changed')
    prompts=sealed['inputs'][sid]
    model=Qwen3FrozenBackbone(b['configuration']['backbone'],device='cuda',dtype='bfloat16')
    order=list(b['status']['active_conditions']);hp.random.Random(42+int(sid.split(':')[1])).shuffle(order)
    out.mkdir(parents=True,exist_ok=False);records=[]
    try:
        for mode in order:
            p=prompts[mode];ids=hp.tokens(model.tokenizer,p['messages'])
            if digest(ids)!=p['input_ids_sha256'] or len(ids)+512>8192:raise ValueError('Tokenized effective input mismatch')
            model.reset_peak();model.torch.cuda.synchronize();t=time.perf_counter()
            gen=model.generate(p['messages'],max_new_tokens=512,enable_thinking=False)
            model.torch.cuda.synchronize();lat=time.perf_counter()-t
            if gen['prompt_tokens']!=p['prompt_tokens']:raise ValueError('Input was truncated')
            row={'sample_id':sid,'condition':mode,'raw_generation':gen['generated_text'],'predicted_verdict':normalize_verdict(gen['generated_text']),
                 'prompt_tokens':gen['prompt_tokens'],'output_tokens':gen['generated_tokens'],'latency_seconds':lat,
                 'peak_gpu_memory_bytes':model.peak_bytes(),'input_ids_sha256':p['input_ids_sha256'],
                 'exemplar_ids':p['exemplar_ids'],'gold_accessed':False,'max_new_tokens':512,'enable_thinking':False}
            save(out/(mode+'.json'),row);records.append(row)
    finally:model.close()
    m161.write_lines(out/'predictions.jsonl',records)
    return out/'predictions.jsonl'


def execute(run,phase,allowed=False,recover=False):
    if not allowed:raise ValueError('GPU execution requires --execute')
    run=m161.output_path(run)
    with m161.run_lock(run,recover):
        environment(run);b=check(run);state=read(run/'state.json')
        if phase=='prediction':
            # Independently regenerate/equality-check sealed prompts before a model call.
            seal_prompts(run)
        if state['active_started_epoch'] is not None:
            state['spent_seconds']+=max(0,time.time()-state['active_started_epoch']);state['active_started_epoch']=None;state['active_child_pid']=None
            m161.atomic(run/'state.json',state)
        for sid in b['sample_ids']:
            folder=run/sid.replace(':','-');name='retrieval' if phase=='retrieval' else 'predictions'
            if (folder/(name+'.receipt.json')).exists():
                for stage in (('acquire','audit','candidates','retrieval') if phase=='retrieval' else ('acquire','audit','candidates','retrieval','predictions')):
                    verified_step(run,folder/(stage+'.receipt.json'))
                if sid not in state[phase+'_ids']:state[phase+'_ids'].append(sid)
                if phase=='prediction':state['completed_ids']=list(state['prediction_ids'])
                continue
            m161.resources(run,b);remaining=b['configuration']['limits']['wall_seconds']-state['spent_seconds']
            if remaining<=0:raise ValueError('24-hour cumulative envelope exhausted')
            started=time.time();state['active_started_epoch']=started;m161.atomic(run/'state.json',state);child=None
            try:
                child=subprocess.Popen([sys.executable,'-m','tffv.experiments.m162a','_worker','--run',str(run),'--sample-id',sid,'--phase',phase],start_new_session=True)
                state['active_child_pid']=child.pid;m161.atomic(run/'state.json',state);deadline=time.monotonic()+remaining
                while child.poll() is None:
                    m161.resources(run,b)
                    if time.monotonic()>deadline:raise subprocess.TimeoutExpired(child.args,remaining)
                    time.sleep(2)
                if child.returncode:raise RuntimeError('Worker failed; retain receipts and resume same run/version')
            except BaseException:
                if child and child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
            finally:
                state['spent_seconds']+=time.time()-started;state['active_started_epoch']=None;state['active_child_pid']=None;m161.atomic(run/'state.json',state)
            print(f'CHECKPOINT {phase} {sid} elapsed={state["spent_seconds"]:.1f}s',flush=True)
            state[phase+'_ids'].append(sid)
            if phase=='prediction':state['completed_ids']=list(state['prediction_ids'])
            m161.atomic(run/'state.json',state)
        m161.atomic(run/'state.json',state)
        if phase=='prediction':
            hashes={sid:file_hash(run/sid.replace(':','-')/'predictions.receipt.json') for sid in b['sample_ids']}
            obj={'batch_sha256':file_hash(run/'batch.json'),'sealed_prompts_sha256':file_hash(run/'sealed_prompts.json'),
                 'prediction_receipt_sha256':hashes,'conditions':b['status']['active_conditions'],'claims':50,'gold_accessed':False}
            if (run/'completed.json').exists():
                if read(run/'completed.json')!=obj:raise ValueError('Completion identity changed')
            else:save(run/'completed.json',obj)


def evaluate(run,gold_source,output):
    run=m161.output_path(run);b=check(run);done=read(run/'completed.json');c=SourceContract(ROOT)
    if done['batch_sha256']!=file_hash(run/'batch.json') or done['sealed_prompts_sha256']!=file_hash(run/'sealed_prompts.json') or done['conditions']!=b['status']['active_conditions']:
        raise ValueError('Cannot score incomplete or changed conditions')
    predictions={mode:{} for mode in done['conditions']}
    from tffv.verification.ontology import normalize_verdict
    for sid in b['sample_ids']:
        receipt=run/sid.replace(':','-')/'predictions.receipt.json'
        if file_hash(receipt)!=done['prediction_receipt_sha256'][sid]:raise ValueError('Prediction seal changed')
        for stage in ('acquire','audit','candidates','retrieval'):verified_step(run,receipt.parent/(stage+'.receipt.json'))
        rows=jsonl(verified_step(run,receipt))
        if len(rows)!=len(predictions) or {r['condition'] for r in rows}!=set(predictions):raise ValueError('Condition population mismatch')
        for r in rows:
            if r['sample_id']!=sid or normalize_verdict(r['raw_generation'])!=r['predicted_verdict'] or r['gold_accessed'] is not False:raise ValueError('Stored/raw/source identity mismatch')
            predictions[r['condition']][sid]=r
    # Gold is opened ONLY after every prediction condition/population seal passed.
    if file_hash(gold_source)!=c.seal['dataset_sha256']:raise ValueError('Evaluator gold release mismatch')
    raw=read(gold_source);gold=[raw[int(s.split(':')[1])]['label'] for s in b['sample_ids']]
    result={'scope':'EXPLORATORY_NOT_CONFIRMATORY','claims':50,'metrics':{},'comparisons':{},'P4_status':b['status']['P4'],
            'batch_sha256':file_hash(run/'batch.json'),'completed_sha256':file_hash(run/'completed.json')}
    for mode,rows in predictions.items():
        pred=[rows[s]['predicted_verdict'] for s in b['sample_ids']];met=hp.metric(gold,pred)
        met.update(prompt_tokens=sum(r['prompt_tokens'] for r in rows.values()),output_tokens=sum(r['output_tokens'] for r in rows.values()),
                   generation_wall_seconds=sum(r['latency_seconds'] for r in rows.values()),gpu_active_seconds=None)
        result['metrics'][mode]=met
    if 'P4' not in predictions:result['metrics']['P4']=None
    contrasts=[('P2','P3'),('P1','P3')]+([('P3','P4')] if 'P4' in predictions else [])
    pvals={}
    for base,target in contrasts+[('P0','P3'),('N0','P3')]:
        key=target+'_vs_'+base;a=[predictions[base][s]['predicted_verdict'] for s in b['sample_ids']];bb=[predictions[target][s]['predicted_verdict'] for s in b['sample_ids']]
        comp=hp.compare(gold,a,bb,seed=b['protocol']['seed'],n_boot=b['protocol']['n_bootstrap'],n_perm=b['protocol']['n_paired_permutations']);comp['changed_ids']=[s for s,x,y in zip(b['sample_ids'],a,bb) if x!=y]
        comp['correction_ids']=[s for s,g,x,y in zip(b['sample_ids'],gold,a,bb) if x!=g and y==g]
        comp['regression_ids']=[s for s,g,x,y in zip(b['sample_ids'],gold,a,bb) if x==g and y!=g]
        result['comparisons'][key]=comp
        if (base,target) in contrasts:
            pvals[key+'_accuracy']=comp['exact_mcnemar_p'];pvals[key+'_macro_f1']=comp['macro_f1_paired_swap_p']
    result['holm_primary_family']=hp.holm(pvals)
    result['pilot_signal']='POSITIVE_DIRECTION_ONLY_REQUIRES_VALIDITY_COST_AND_CLASS_REVIEW' if all(result['comparisons'][k]['accuracy_delta']>0 and result['comparisons'][k]['macro_f1_delta']>0 for k in ('P3_vs_P2','P3_vs_P1')) else 'MODIFY_OR_NO_GO; no automatic CoT/seed/k expansion'
    lookup=indexed(read(b['source_memory_path'])['records']);sl=indexed(b['slates'])
    result['diagnostics']={'source_exemplar_counts':b['status']['selected_error_counts'],
       'demo_gold_copy_count':sum(predictions['P3'][s]['predicted_verdict']==lookup[sl[s]['error_id']]['gold_verdict'] for s in b['sample_ids']),
       'demo_wrong_copy_count':sum(predictions['P3'][s]['predicted_verdict']==lookup[sl[s]['error_id']]['base_prediction']['reparsed_verdict'] for s in b['sample_ids']),
       'prediction_label_counts':{mode:dict(Counter(r['predicted_verdict'] or 'Unparseable' for r in rr.values())) for mode,rr in predictions.items()},
       'demonstration_gold_label_counts':dict(Counter(lookup[s['error_id']]['gold_verdict'] for s in b['slates'])),
       'per_claim_prompt_tokens':{mode:{s:r['prompt_tokens'] for s,r in rr.items()} for mode,rr in predictions.items()},
       'target_labels_used_in_selection':False,'scope':'Post-score descriptions, not automatic method selection/tuning'}
    result['total_session_wall_seconds']=read(run/'state.json')['spent_seconds'];result['gpu_active_seconds']=None
    out=m161.output_path(output);out.mkdir(parents=True,exist_ok=False);save(out/'results.json',result)
    m161.write_lines(out/'per_claim.jsonl',[{'sample_id':sid,'gold_verdict':g,'slate':sl[sid],
        'conditions':{mode:predictions[mode][sid] for mode in predictions}} for sid,g in zip(b['sample_ids'],gold)])
    print(json.dumps({'scored':str(out),'pilot_signal':result['pilot_signal'],'P4':b['status']['P4']},indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    for k in ('output','memory','claims-manifest','tokenizer'):q.add_argument('--'+k,type=Path,required=True)
    for command in ('check','retrieve','seal-prompts','run-pilot','_worker','evaluate'):
        q=sub.add_parser(command);q.add_argument('--run',type=Path,required=True)
        if command in ('retrieve','run-pilot'):q.add_argument('--execute',action='store_true');q.add_argument('--recover-stale-lock',action='store_true')
        if command=='_worker':q.add_argument('--sample-id',required=True);q.add_argument('--phase',choices=['retrieval','prediction'],required=True)
        if command=='evaluate':q.add_argument('--gold-source',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.output,a.memory,a.claims_manifest,a.tokenizer)
    elif a.command=='check':print(json.dumps(check(a.run)['status'],indent=2))
    elif a.command=='seal-prompts':seal_prompts(a.run)
    elif a.command in ('retrieve','run-pilot'):execute(a.run,'retrieval' if a.command=='retrieve' else 'prediction',a.execute,a.recover_stale_lock)
    elif a.command=='evaluate':evaluate(a.run,a.gold_source,a.output)
    else:
        if not (a.run/'execution.lock').exists() or read(a.run/'execution.lock')['pid']!=os.getppid():raise ValueError('Private worker requires locked parent')
        worker(a.run,a.sample_id,a.phase)


if __name__=='__main__':main()
