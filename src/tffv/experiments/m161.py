"""M1.6.1 CPU memory tools and explicit user-run fixed batch-25 execution.

No command runs a model except run-batch --execute (and its private worker).
Inference workers have no gold input. Memory gold join is a separate command.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time

from tffv.memory.heal import (SourceContract, MODEL_REVISION, PARTITION_SEAL, SCHEDULE_HASH,
    read, jsonl, save, indexed, build_legacy, summary, export_reviews, apply_candidates,
    import_reviews, example_view, source_gold, make_record)
from tffv.memory.heal import merge_memories
from tffv.provenance import digest, file_hash

ROOT = Path(__file__).resolve().parents[3]
FROZEN_RETRIEVAL_CONFIG = '7f716c4fe2888ba311a4edba9ca8dd618c368de43707264dde1d5cf129e907ec'
ACQUISITION_PLAN_SHA = '833f46c5f8ebefa28d9a171e909a8f9500319eaf539ce2bf290e262245604f23'


def atomic(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp,path)


def code_hash():
    return digest({p.relative_to(ROOT).as_posix():file_hash(p) for p in sorted((ROOT/'src/tffv').rglob('*.py'))})


def output_path(path):
    p = Path(path).resolve()
    if not p.is_relative_to((ROOT/'outputs').resolve()) or p == (ROOT/'outputs').resolve():
        raise ValueError('Use a new subdirectory of this project outputs/')
    return p


def write_lines(path, rows):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')


def prepare(output, plan, claims_manifest, config):
    from tffv.retrieval.acquire_subset import archive_for
    from tffv.retrieval.remote_zip import validate_plan
    contract = SourceContract(ROOT)
    dest = output_path(output)
    cm, cfg, acquisition = read(claims_manifest), read(config), read(plan)
    if file_hash(plan)!=ACQUISITION_PLAN_SHA:
        raise ValueError('Use the byte-pinned historical acquisition plan')
    archives = validate_plan(acquisition)
    claims_file = Path(claims_manifest).resolve().parent/'claims.jsonl'
    if cm['source_sha256']!=contract.seal['dataset_sha256'] or cm['annotation_values_used'] is not False or cm['split']!='train':
        raise ValueError('Require sealed own-project label-free official-train claims export')
    if file_hash(claims_file)!=cm['claims_sha256']:
        raise ValueError('Source claims hash mismatch')
    all_claims = indexed(jsonl(claims_file))
    selected = []
    for sid in contract.batch_ids:
        r = all_claims[sid]
        if set(r)!={'sample_id','dataset','split','claim'} or r['dataset']!='AVeriTeC' or r['split']!='train':
            raise ValueError('Only scrubbed claim fields may reach batch workers')
        if digest(r['claim']) != digest(r['claim'].strip()):
            raise ValueError('Unexpected source claim normalization; do not silently change the claim')
        if hashlib.sha256(r['claim'].encode()).hexdigest()!=contract.rows[sid]['claim_sha256']:
            raise ValueError('Selected source claim differs from sealed metadata hash')
        selected.append(r)
    # Exact semantic retrieval config from the historical protocol is pinned in Git.
    if digest(cfg['retrieval'])!=cfg['legacy_retrieval_config_sha256'] or cfg['legacy_retrieval_config_sha256']!=FROZEN_RETRIEVAL_CONFIG:
        raise ValueError('Frozen retrieval config changed')
    if cfg['generation']!={'max_new_tokens':512,'enable_thinking':False,'do_sample':False,'dtype':'bfloat16','max_context_length':8192}:
        raise ValueError('Frozen D0 decoder/context changed')
    if cfg['backbone']['revision']!=MODEL_REVISION:
        raise ValueError('Frozen backbone revision changed')
    if cfg['limits']!={'wall_seconds':43200,'scratch_bytes':30*1024**3,'free_disk_bytes':100*1024**3} or cfg['candidate_policy']!='quarantine_review_flags_v1':
        raise ValueError('Registered resource/candidate policy changed')
    inputs = {str(Path(p).resolve()):file_hash(p) for p in (plan,claims_manifest,claims_file,config)}
    data = {'schema_version':'m161-batch25-v1','partition_seal_sha256':PARTITION_SEAL,
            'schedule_sha256':SCHEDULE_HASH,'sample_ids':contract.batch_ids,'configuration':cfg,
            'input_hashes':inputs,'code_sha256':code_hash(),'acquisition_plan_path':str(Path(plan).resolve()),
            'config_path':str(Path(config).resolve()),'claims_input_path':str(claims_file.resolve()),
            'claims_manifest_path':str(Path(claims_manifest).resolve()),
            'shards':{sid:archive_for(int(sid.split(':')[1]),archives)['path'] for sid in contract.batch_ids},
            'gold_available_to_worker':False,'claims':selected,'exploratory_only':True}
    if dest.exists():
        if read(dest/'batch.json') != data:
            raise ValueError('Existing run has different configuration/inputs/code; no overwrite')
        check_batch(dest)
        return dest
    dest.mkdir(parents=True,exist_ok=False)
    save(dest/'batch.json',data)
    for r in selected:
        d=dest/r['sample_id'].replace(':','-')/'input'
        d.mkdir(parents=True)
        write_lines(d/'claims.jsonl',[r])
        save(d/'claims_manifest.json',{'dataset':'AVeriTeC','split':'train','source_role':'source_train',
            'annotation_values_used':False,'source_sha256':cm['source_sha256'],'records':1,
            'claims_sha256':file_hash(d/'claims.jsonl'),'parent_claims_manifest_sha256':file_hash(claims_manifest)})
    save(dest/'state.json',{'spent_seconds':0.0,'active_started_epoch':None,'active_child_pid':None,
                           'host':socket.gethostname(),'completed_ids':[]})
    print(json.dumps({'prepared':str(dest),'source_count':25,'shard_counts':{k:list(data['shards'].values()).count(k) for k in sorted(set(data['shards'].values()))},'gpu_calls':0},indent=2))
    return dest


def check_batch(run):
    run=output_path(run)
    b=read(run/'batch.json')
    contract=SourceContract(ROOT)
    if b['sample_ids']!=contract.batch_ids or b['partition_seal_sha256']!=PARTITION_SEAL or b['schedule_sha256']!=SCHEDULE_HASH:
        raise ValueError('Fixed batch/schedule/partition changed')
    if b['code_sha256']!=code_hash():
        raise ValueError('Code changed since preparation; do not resume a different implementation')
    for path,h in b['input_hashes'].items():
        if file_hash(path)!=h:
            raise ValueError('Batch input changed: '+path)
    from tffv.retrieval.acquire_subset import archive_for
    from tffv.retrieval.remote_zip import validate_plan
    if b['acquisition_plan_path'] not in b['input_hashes'] or file_hash(b['acquisition_plan_path'])!=ACQUISITION_PLAN_SHA:
        raise ValueError('Acquisition plan identity changed')
    archives=validate_plan(read(b['acquisition_plan_path']))
    if b['shards']!={s:archive_for(int(s.split(':')[1]),archives)['path'] for s in contract.batch_ids}:
        raise ValueError('Shard routing changed')
    if b['configuration'] != read(b['config_path']):
        raise ValueError('Embedded run configuration changed')
    if indexed(b['claims']) != {r['sample_id']:r for r in b['claims']} or set(indexed(b['claims']))!=set(contract.batch_ids):
        raise ValueError('Claim slate changed')
    # Bind scrubbed text to the immutable original label-free export.
    original=b['claims_input_path']
    all_claims=indexed(jsonl(original))
    for r in b['claims']:
        if r!=all_claims[r['sample_id']]:
            raise ValueError('Batch claim text changed')
        d=run/r['sample_id'].replace(':','-')/'input'
        expected_cm={'dataset':'AVeriTeC','split':'train','source_role':'source_train',
            'annotation_values_used':False,'source_sha256':contract.seal['dataset_sha256'],'records':1,
            'claims_sha256':file_hash(d/'claims.jsonl'),'parent_claims_manifest_sha256':file_hash(b['claims_manifest_path'])}
        if jsonl(d/'claims.jsonl') != [r] or read(d/'claims_manifest.json')!=expected_cm:
            raise ValueError('Per-claim input changed')
    return b


def resources(run,b):
    lim=b['configuration']['limits']
    if shutil.disk_usage(run).free<lim['free_disk_bytes']:
        raise ValueError('Free disk below 100 GiB floor')
    size=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
    if size>lim['scratch_bytes']:
        raise ValueError('Run footprint exceeds 30 GiB; stop without deleting old artifacts')
    return size


def preflight(run, structural=False):
    run=output_path(run)
    b=check_batch(run)
    report={'batch_sha256':file_hash(run/'batch.json'),'source_count':25,'configuration':b['configuration'],
            'structural_checks':'PASS','hardware_checks':'NOT_RUN','model_fingerprints':None,
            'new_gpu_model_calls':0,'new_retrieval_calls':0,'ready_to_execute':False}
    if not structural:
        resources(run,b)
        import torch
        from tffv.retrieval.frozen import validate_spec,snapshot_fingerprint
        from tffv.verification.backbone import validate_backbone_spec
        if platform.system()!='Linux' or not torch.cuda.is_available():
            raise ValueError('User execution requires Linux/CUDA; no CPU model fallback')
        free,total=torch.cuda.mem_get_info()
        if free<24*1024**3:
            raise ValueError('At least 24 GiB free VRAM required for this RTX 5090 session')
        if tuple(map(int,importlib.metadata.version('transformers').split('.')[:2]))<(4,51):
            raise ValueError('transformers>=4.51 required')
        cfg=b['configuration']
        fps={}
        for name in ('embedding','reranker'):
            spec=cfg['retrieval'][name]
            path=validate_spec(spec,spec['model_id'],verify_bytes=True)
            fps[name]=snapshot_fingerprint(path)['snapshot_sha256']
        path=validate_backbone_spec(cfg['backbone'])
        if path.name!=MODEL_REVISION:
            raise ValueError('Verification snapshot directory/revision mismatch')
        fps['backbone']=snapshot_fingerprint(path)['snapshot_sha256']
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
        report.update(hardware_checks='PASS',model_fingerprints=fps,gpu_snapshot=gpu,
                      dependencies={k:importlib.metadata.version(k) for k in ('torch','transformers','numpy')},ready_to_execute=True)
        if (run/'environment.json').exists():
            old=read(run/'environment.json')
            for key in ('batch_sha256','model_fingerprints','dependencies'):
                if old[key]!=report[key]:
                    raise ValueError('Runtime/model identity changed; refusing resume')
        else:
            save(run/'environment.json',report)
    print(json.dumps(report,indent=2))
    return report


def verify_receipt(run, receipt):
    r=read(receipt)
    if r['batch_sha256']!=file_hash(run/'batch.json'):
        raise ValueError('Step belongs to another batch')
    if not r['files_sha256']:
        raise ValueError('Empty step seal')
    for relative,h in r['files_sha256'].items():
        p=(run/relative).resolve()
        if not p.is_relative_to(run.resolve()) or file_hash(p)!=h:
            raise ValueError('Step hash/path mismatch: '+relative)
    return run/r['output']


def step(run,sid,name,action):
    folder=run/sid.replace(':','-')
    receipt=folder/(name+'.receipt.json')
    if receipt.exists():
        print(f'RESUME verified {sid} {name}',flush=True)
        return verify_receipt(run,receipt)
    attempt=1
    while (folder/f'{name}-attempt-{attempt:03d}').exists():
        attempt+=1
    out=folder/f'{name}-attempt-{attempt:03d}'
    b=check_batch(run)
    resources(run,b)
    started=time.time()
    print(f'START {sid} {name} attempt={attempt} output={out}',flush=True)
    result=Path(action(out)).resolve()
    if not out.is_dir() or not result.is_relative_to(out.resolve()):
        raise ValueError('Step output escaped its fresh attempt')
    resources(run,b)
    files={p.relative_to(run).as_posix():file_hash(p) for p in sorted(out.rglob('*')) if p.is_file()}
    save(receipt,{'sample_id':sid,'stage':name,'batch_sha256':file_hash(run/'batch.json'),
                  'output':result.relative_to(run).as_posix(),'files_sha256':files,'elapsed_seconds':time.time()-started})
    return result


def infer_single(run,sid,r4_path,out):
    from tffv.verification.backbone import Qwen3FrozenBackbone
    from tffv.verification.prompting import build_verification_prompt
    from tffv.verification.ontology import normalize_verdict
    b=check_batch(run)
    claim=indexed(b['claims'])[sid]['claim']
    r4=jsonl(r4_path)
    if len(r4)!=1 or r4[0]['sample_id']!=sid or r4[0]['claim_sha256']!=digest(claim):
        raise ValueError('Retrieval/inference claim mismatch')
    ev=r4[0]['evidence']
    if len(ev)!=8:
        raise ValueError('Frozen Top-8 evidence incomplete; do not silently change the protocol')
    cfg=b['configuration']
    model=Qwen3FrozenBackbone(cfg['backbone'],device='cuda',dtype=cfg['generation']['dtype'])
    prompt=build_verification_prompt(claim,ev)
    try:
        text=model.tokenizer.apply_chat_template(prompt['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        tokens=model.tokenizer(text)['input_ids']
        if len(tokens)+512>8192:
            raise ValueError('Context/output reservation overflow; evidence may not be truncated')
        started=time.perf_counter()
        result=model.generate(prompt['messages'],max_new_tokens=512,enable_thinking=False)
        elapsed=time.perf_counter()-started
        if result['prompt_tokens']!=len(tokens):
            raise ValueError('Effective input changed or was truncated')
    finally:
        model.close()
    out.mkdir(parents=True,exist_ok=False)
    pred={'sample_id':sid,'claim_sha256':digest(claim),'predicted_verdict':normalize_verdict(result['generated_text']),
          'raw_generation':result['generated_text'],'prompt_sha256':prompt['prompt_sha256'],
          'effective_input_ids_sha256':digest(tokens),'effective_chat_text_sha256':digest(text),
          'prompt_tokens':result['prompt_tokens'],'generated_tokens':result['generated_tokens'],
          'max_new_tokens':512,'enable_thinking':False,'latency_seconds':elapsed,'gold_accessed':False}
    write_lines(out/'predictions.jsonl',[pred])
    save(out/'prompt.json',prompt)
    save(out/'provenance.json',{'r4_sha256':file_hash(r4_path),'environment_sha256':file_hash(run/'environment.json'),
                               'batch_sha256':file_hash(run/'batch.json'),'sample_id':sid})
    return out/'predictions.jsonl'


def claim_worker(run,sid):
    from tffv.retrieval.acquire_subset import acquire
    from tffv.retrieval.audit_subset import audit
    from tffv.retrieval.prepare_store_candidates import prepare_candidates
    from tffv.retrieval.store_retrieval import run as retrieve
    b=check_batch(run)
    if sid not in b['sample_ids']:
        raise ValueError('Worker ID outside fixed batch')
    cfg=b['configuration']
    acq=step(run,sid,'acquire',lambda out:acquire(b['acquisition_plan_path'],out,[int(sid.split(':')[1])],execute=True))
    aud=step(run,sid,'audit',lambda out:audit(acq,out,max_record_mib=64))
    cm=run/sid.replace(':','-')/'input/claims_manifest.json'
    cand=step(run,sid,'candidates',lambda out:prepare_candidates(aud,cm,out,policy=cfg['candidate_policy']))
    # Never point the old bare-npy cache loader at an unverified external cache.
    # A complete sealed retrieval is reused as a whole; incomplete attempts are fresh.
    r4=step(run,sid,'retrieval',lambda out:retrieve(cand,out,methods=('R1','R2','R3','R4'),
             config=cfg['retrieval'],index_cache=out/'embedding_cache')/'R4.jsonl')
    step(run,sid,'prediction',lambda out:infer_single(run,sid,r4,out))


def alive(pid):
    if not pid:
        return False
    if platform.system()!='Linux':
        return True  # Refuse recovery; os.kill(pid, 0) is not a safe Windows probe.
    try:
        os.kill(pid,0)
        return True
    except ProcessLookupError:
        return False


@contextmanager
def run_lock(run,recover=False):
    lock=run/'execution.lock'
    if lock.exists() and recover:
        old=read(lock)
        state=read(run/'state.json')
        if old['host']!=socket.gethostname() or alive(old['pid']) or alive(state.get('active_child_pid')):
            raise ValueError('Cannot recover live/foreign-host execution lock')
        lock.unlink()
    with lock.open('x',encoding='utf-8') as f:
        json.dump({'pid':os.getpid(),'host':socket.gethostname()},f)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def run_batch(run,execute=False,recover=False,resume=False):
    if not execute:
        raise ValueError('GPU/network execution requires explicit run-batch --execute')
    run=output_path(run)
    initial=read(run/'state.json')
    if not resume and (initial['spent_seconds'] or initial['completed_ids'] or initial['active_started_epoch']):
        raise ValueError('Existing progress requires explicit --resume')
    with run_lock(run,recover):
        preflight(run)
        b=check_batch(run)
        state=read(run/'state.json')
        if state['active_started_epoch'] is not None:
            state['spent_seconds']+=max(0,time.time()-state['active_started_epoch'])
            state['active_started_epoch']=None
            state['active_child_pid']=None
            atomic(run/'state.json',state)
        for sid in b['sample_ids']:
            receipt=run/sid.replace(':','-')/'prediction.receipt.json'
            if receipt.exists():
                # Check every upstream receipt too, even when prediction is complete.
                for name in ('acquire','audit','candidates','retrieval','prediction'):
                    verify_receipt(run,receipt.parent/(name+'.receipt.json'))
                if sid not in state['completed_ids']:
                    state['completed_ids'].append(sid)
                continue
            resources(run,b)
            remaining=b['configuration']['limits']['wall_seconds']-state['spent_seconds']
            if remaining<=0:
                raise ValueError('12-hour cumulative budget exhausted; do not reset on resume')
            started=time.time()
            state['active_started_epoch']=started
            atomic(run/'state.json',state)
            child=None
            try:
                child=subprocess.Popen([sys.executable,'-m','tffv.experiments.m161','_claim','--run',str(run),'--sample-id',sid],start_new_session=True)
                state['active_child_pid']=child.pid
                atomic(run/'state.json',state)
                deadline=time.monotonic()+remaining
                while child.poll() is None:
                    resources(run,b)
                    if time.monotonic()>=deadline:
                        raise subprocess.TimeoutExpired(child.args,remaining)
                    time.sleep(min(2,max(0,deadline-time.monotonic())))
                rc=child.returncode
                if rc:
                    raise RuntimeError(f'Claim worker failed rc={rc}; resume retains sealed steps, new attempts for incomplete steps')
            except BaseException:
                if child is not None and child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL)
                        child.wait()
                raise
            finally:
                state['spent_seconds']+=time.time()-started
                state['active_started_epoch']=None
                state['active_child_pid']=None
                atomic(run/'state.json',state)
            state['completed_ids'].append(sid)
            atomic(run/'state.json',state)
            print(f'CHECKPOINT source_completed={len(state["completed_ids"])}/25 elapsed={state["spent_seconds"]:.1f}s run={run}',flush=True)
        atomic(run/'state.json',state)
        if not (run/'completed.json').exists():
            save(run/'completed.json',{'sample_ids':b['sample_ids'],'batch_sha256':file_hash(run/'batch.json'),
                                      'state_sha256':file_hash(run/'state.json'),'source_count':25,'gold_accessed':False})
        print(f'COMPLETE fixed_source_count=25 run={run}; memory join must run separately',flush=True)


def build_batch(run,gold_path):
    from tffv.verification.prompting import build_verification_prompt
    run=output_path(run)
    b=check_batch(run)
    complete=read(run/'completed.json')
    if complete['batch_sha256']!=file_hash(run/'batch.json') or complete['sample_ids']!=b['sample_ids']:
        raise ValueError('Batch completion contract mismatch')
    if complete['state_sha256']!=file_hash(run/'state.json') or set(read(run/'state.json')['completed_ids'])!=set(b['sample_ids']):
        raise ValueError('Batch completion state mismatch')
    contract=SourceContract(ROOT)
    gold=source_gold(gold_path,b['sample_ids'],contract)
    records=[]
    for sid in b['sample_ids']:
        d=run/sid.replace(':','-')
        for name in ('acquire','audit','candidates','retrieval','prediction'):
            verify_receipt(run,d/(name+'.receipt.json'))
        pp=verify_receipt(run,d/'prediction.receipt.json')
        ep=verify_receipt(run,d/'retrieval.receipt.json')
        p=jsonl(pp)[0]
        ev=jsonl(ep)[0]['evidence']
        g=gold[sid]
        if p['prompt_sha256']!=build_verification_prompt(g['claim'],ev)['prompt_sha256'] or p['claim_sha256']!=digest(g['claim']):
            raise ValueError('New source prompt/claim provenance mismatch')
        if p['max_new_tokens']!=512 or p['enable_thinking'] is not False or p['gold_accessed'] is not False:
            raise ValueError('Source inference decoding/gold-isolation mismatch')
        prov={'origin':'OWN_PROJECT_M161_BATCH25','artifact_checks_passed':True,
              'batch_sha256':file_hash(run/'batch.json'),'prediction_sha256':file_hash(pp),'evidence_sha256':file_hash(ep),
              'runtime_environment':read(run/'environment.json'),'gold_source_sha256':file_hash(gold_path),
              'effective_input_token_ids_sha256':p['effective_input_ids_sha256']}
        records.append(make_record(sid,g['claim'],g['label'],p,ev,prov,contract))
    memory={'schema_version':'heal-bank-v1','partition_seal_sha256':PARTITION_SEAL,'origin':'OWN_PROJECT_M161_BATCH25','records':records}
    contract.check_memory(memory)
    return memory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('legacy-audit')
    for name in ('predictions','prediction-manifest','evidence','retrieval-manifest','gold-source','output'):
        q.add_argument('--'+name,required=True,type=Path)
    q=sub.add_parser('prepare-batch')
    for name in ('output','plan','claims-manifest'):
        q.add_argument('--'+name,required=True,type=Path)
    q.add_argument('--config',type=Path,default=ROOT/'configs/m161_source_batch25.json')
    q=sub.add_parser('preflight')
    q.add_argument('--run',type=Path,required=True)
    q.add_argument('--structural-only',action='store_true')
    q=sub.add_parser('run-batch')
    q.add_argument('--run',type=Path,required=True)
    q.add_argument('--execute',action='store_true')
    q.add_argument('--resume',action='store_true',help='Explicit intent; all sealed steps are always hash-checked')
    q.add_argument('--recover-stale-lock',action='store_true')
    q=sub.add_parser('_claim',help=argparse.SUPPRESS)
    q.add_argument('--run',type=Path,required=True)
    q.add_argument('--sample-id',required=True)
    for name in ('export-reviews','validate-memory','apply-candidates','import-reviews','view'):
        q=sub.add_parser(name)
        q.add_argument('--memory',type=Path,required=True)
        if name!='validate-memory':
            q.add_argument('--output',type=Path,required=True)
        if name=='apply-candidates': q.add_argument('--annotations',type=Path,required=True)
        if name=='import-reviews':
            q.add_argument('--review-1',type=Path,required=True)
            q.add_argument('--review-2',type=Path,required=True)
        if name=='view':
            q.add_argument('--ids',nargs='+',required=True)
            q.add_argument('--mode',choices=['gold_only','wrong_gold','wrong_gold_verified','contrastive','mechanism'],required=True)
            q.add_argument('--mechanism')
    q=sub.add_parser('build-batch-memory')
    for name in ('run','gold-source','output'): q.add_argument('--'+name,type=Path,required=True)
    q=sub.add_parser('merge-memory')
    q.add_argument('--memory',type=Path,nargs='+',required=True)
    q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    contract=SourceContract(ROOT)
    if a.command=='prepare-batch': prepare(a.output,a.plan,a.claims_manifest,a.config)
    elif a.command=='preflight': preflight(a.run,a.structural_only)
    elif a.command=='run-batch': run_batch(a.run,a.execute,a.recover_stale_lock,a.resume)
    elif a.command=='_claim':
        if not (a.run/'execution.lock').exists() or read(a.run/'execution.lock')['pid']!=os.getppid() or not (a.run/'environment.json').exists():
            raise ValueError('Worker must be launched by preflighted/locked run-batch parent')
        claim_worker(a.run,a.sample_id)
    elif a.command in ('legacy-audit','build-batch-memory'):
        mem=build_legacy(contract,a.predictions,a.prediction_manifest,a.evidence,a.retrieval_manifest,a.gold_source) if a.command=='legacy-audit' else build_batch(a.run,a.gold_source)
        dest=output_path(a.output)
        dest.mkdir(parents=True,exist_ok=False)
        save(dest/'memory.json',mem)
        save(dest/'summary.json',summary(mem))
        print(json.dumps(summary(mem),indent=2))
    elif a.command=='merge-memory':
        save(output_path(a.output),merge_memories([read(p) for p in a.memory],contract))
    else:
        mem=read(a.memory)
        contract.check_memory(mem)
        if a.command=='validate-memory': print(json.dumps(summary(mem),indent=2))
        elif a.command=='export-reviews': export_reviews(mem,output_path(a.output))
        elif a.command=='apply-candidates':
            result=apply_candidates(mem,read(a.annotations),contract)
            result['candidate_annotations_file_sha256']=file_hash(a.annotations)
            save(output_path(a.output),result)
        elif a.command=='import-reviews':
            result=import_reviews(mem,read(a.review_1),read(a.review_2),contract)
            result['review_source_files_sha256']={str(p.resolve()):file_hash(p) for p in (a.review_1,a.review_2)}
            save(output_path(a.output),result)
        elif a.command=='view': save(output_path(a.output),example_view(mem,a.ids,a.mode,contract,a.mechanism))


if __name__=='__main__':
    main()
