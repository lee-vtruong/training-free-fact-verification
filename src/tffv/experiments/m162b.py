"""M1.6.2b: cache-only mixed gold-only-memory arm. GPU runs require --execute."""
from pathlib import Path
import argparse, json, os, signal, subprocess, sys, time
from collections import Counter

from tffv.provenance import digest,file_hash,write_json
from tffv.memory.heal import read,save,update,jsonl,indexed,SourceContract
from tffv.experiments import m16_cache,m161
from tffv.verification import heal_pilot as hp

ROOT=Path(__file__).resolve().parents[3]
CONDITION='MIXED_GOLD_ONLY_K2'
ORDER=('base_correct_gold_only','base_wrong_gold_only')

def out(path): return m16_cache.output(path)

def mixed_messages(claim,evidence,slate,memory):
    records=indexed(memory['records']); good=records[slate['correct_id']]; bad=records[slate['error_id']]
    native=hp.build_d1_matched_prompt(claim,evidence)['messages']
    blocks=[hp.block(good,'P2',slate['correct_context']),hp.block(bad,'P2',slate['error_context'])]
    messages=[dict(native[0]),dict(native[1])]
    messages[1]['content']=('Historical task references (independent of the target; decide the target from its own evidence).\n'
        'Reference 1 — Base-correct source example, gold-only:\n'+blocks[0]+'\n'
        'Reference 2 — Base-wrong source example, gold-only:\n'+blocks[1]+'\n'
        '--- Target ---\n'+messages[1]['content'])
    return messages

def validate_pair(slate,records,contract):
    good,bad=records[slate['correct_id']],records[slate['error_id']]
    if good['is_base_error'] or not bad['is_base_error']: raise ValueError('Mixed pair must be correct then Base-wrong')
    if good['gold_verdict']!=bad['gold_verdict']: raise ValueError('Mixed pair must share source gold class')
    if good['group_id']==bad['group_id']: raise ValueError('Mixed pair source groups overlap')
    target=contract.rows[slate['sample_id']]['group_id']
    m16_cache.require_no_source_target_leakage([good['group_id'],bad['group_id']],[target])

def prepare(canonical_run,cache_index,memory_path,tokenizer_path,destination):
    run=Path(canonical_run).resolve(); b=read(run/'batch.json'); index=m16_cache.load_index(cache_index)
    if file_hash(run/'batch.json')!=index['batch_sha256'] or index['active_conditions']!=b['status']['active_conditions']: raise ValueError('Canonical cache identity/condition mismatch')
    if index['source_memory_sha256']!=file_hash(memory_path) or b['source_memory_sha256']!=file_hash(memory_path): raise ValueError('Source memory cache mismatch')
    config=read(ROOT/'configs/m162b_mixed_gold_only.json')
    if config['condition']!=CONDITION or config['k']!=2 or tuple(config['order'])!=ORDER or config['generation']!={'max_new_tokens':512,'enable_thinking':False,'do_sample':False}: raise ValueError('Frozen mixed condition config changed')
    c=SourceContract(ROOT); mem=read(memory_path); c.check_memory(mem); records=indexed(mem['records'])
    tok=hp.tokenizer(Path(tokenizer_path)); r4={x['sample_id']:x for x in index['records']}; claims=indexed(b['claims']); prompts={}
    for slate in b['slates']:
        sid=slate['sample_id']; validate_pair(slate,records,c)
        if sid not in r4 or r4[sid]['r4_sha256']!=file_hash(r4[sid]['r4_path']): raise ValueError('Missing/mismatched immutable R4 cache: '+sid)
        evidence=jsonl(r4[sid]['r4_path'])
        if len(evidence)!=1 or evidence[0]['sample_id']!=sid or len(evidence[0]['evidence'])!=8: raise ValueError('Incomplete Top-8 cache: '+sid)
        messages=mixed_messages(claims[sid]['claim'],evidence[0]['evidence'],slate,mem); ids=hp.tokens(tok,messages)
        if len(ids)+512>8192: raise ValueError('Mixed prompt context overflow: '+sid)
        prompts[sid]={'messages':messages,'messages_sha256':digest(messages),'input_ids_sha256':digest(ids),'prompt_tokens':len(ids),
          'evidence_cache_key':r4[sid]['evidence_cache_key'],'exemplar_ids':[slate['correct_id'],slate['error_id']],
          'exemplar_order':list(ORDER),'gold_accessed':False}
    obj={'schema_version':'m162b-mixed-v1','scope':'EXPLORATORY_CACHE_ONLY','condition':CONDITION,'canonical_run':str(run),
      'canonical_batch_sha256':file_hash(run/'batch.json'),'cache_index_sha256':file_hash(cache_index),'source_memory_sha256':file_hash(memory_path),
      'cache_index_path':str(Path(cache_index).resolve()),
      'sample_ids':b['sample_ids'],'slates':b['slates'],'prompts':prompts,
      'condition_config_sha256':file_hash(ROOT/'configs/m162b_mixed_gold_only.json'),
      'generation':{'max_new_tokens':512,'enable_thinking':False,'do_sample':False},'backbone':b['configuration']['backbone'],
      'selection':'reuse canonical fixed source-BM25 error slate; pair its matched same-gold, different-group Base-correct control; order correct then wrong; gold-only blocks',
      'gold_accessed':False}
    dest=out(destination); dest.mkdir(parents=True,exist_ok=False); save(dest/'mixed_batch.json',obj); save(dest/'state.json',{'completed_ids':[],'spent_seconds':0.,'active_child_pid':None});
    print(json.dumps({'prepared':str(dest),'claims':len(prompts),'new_inference_calls':len(prompts),'retrieval_calls':0,
      'error_ids':dict(Counter(s['error_id'] for s in b['slates'])),'correct_ids':dict(Counter(s['correct_id'] for s in b['slates'])),'gold_accessed':False},indent=2)); return obj

def check(run):
    run=out(run); b=read(run/'mixed_batch.json')
    if b['condition']!=CONDITION or len(b['sample_ids'])!=50 or b['gold_accessed'] is not False: raise ValueError('Mixed protocol identity mismatch')
    if len(set(b['sample_ids']))!=50 or set(b['prompts'])!=set(b['sample_ids']): raise ValueError('Mixed prompt coverage mismatch')
    if b['generation']!={'max_new_tokens':512,'enable_thinking':False,'do_sample':False}: raise ValueError('Mixed decoding changed')
    if b['condition_config_sha256']!=file_hash(ROOT/'configs/m162b_mixed_gold_only.json'): raise ValueError('Mixed condition config changed')
    if file_hash(b['cache_index_path'])!=b['cache_index_sha256']: raise ValueError('Cache index changed after mixed preparation')
    cache=m16_cache.load_index(b['cache_index_path']); byid={r['sample_id']:r for r in cache['records']}
    for sid,p in b['prompts'].items():
        if p['exemplar_order']!=list(ORDER) or len(p['exemplar_ids'])!=2 or p['prompt_tokens']+512>8192: raise ValueError('Mixed prompt contract mismatch')
        e=byid.get(sid)
        if e is None or e['evidence_cache_key']!=p['evidence_cache_key'] or file_hash(e['r4_path'])!=e['r4_sha256']:
            raise ValueError('Missing/mismatched immutable R4 cache: '+sid)
    return b

def receipt(run,sid):
    p=run/sid.replace(':','-')/'prediction.receipt.json'
    if not p.exists(): return None
    r=read(p); q=(run/r['output']).resolve()
    if r['sample_id']!=sid or r['batch_sha256']!=file_hash(run/'mixed_batch.json') or not q.is_file() or file_hash(q)!=r['output_sha256']: raise ValueError('Mixed prediction receipt mismatch')
    return r,q

def worker(run,sid):
    from tffv.verification.backbone import Qwen3FrozenBackbone
    from tffv.verification.ontology import normalize_verdict
    b=check(run); p=b['prompts'][sid]; model=Qwen3FrozenBackbone(b['backbone'],device='cuda',dtype='bfloat16')
    try:
        ids=hp.tokens(model.tokenizer,p['messages'])
        if digest(ids)!=p['input_ids_sha256']: raise ValueError('Effective mixed prompt changed')
        model.reset_peak(); model.torch.cuda.synchronize(); started=time.perf_counter()
        g=model.generate(p['messages'],max_new_tokens=512,enable_thinking=False); model.torch.cuda.synchronize()
        peak=model.peak_bytes()
    finally: model.close()
    d=run/sid.replace(':','-'); attempt=1
    while (d/f'prediction-attempt-{attempt:03d}').exists(): attempt+=1
    target=d/f'prediction-attempt-{attempt:03d}'; target.mkdir(parents=True)
    row={'sample_id':sid,'condition':CONDITION,'raw_generation':g['generated_text'],'predicted_verdict':normalize_verdict(g['generated_text']),
      'prompt_tokens':g['prompt_tokens'],'output_tokens':g['generated_tokens'],'latency_seconds':time.perf_counter()-started,
      'peak_gpu_memory_bytes':peak,'input_ids_sha256':p['input_ids_sha256'],'exemplar_ids':p['exemplar_ids'],
      'gold_accessed':False,'max_new_tokens':512,'enable_thinking':False}
    save(target/'prediction.json',row); save(d/'prediction.receipt.json',{'sample_id':sid,'batch_sha256':file_hash(run/'mixed_batch.json'),'output':(target/'prediction.json').relative_to(run).as_posix(),'output_sha256':file_hash(target/'prediction.json')}); return row

def execute(run,allowed=False):
    if not allowed: raise ValueError('New GPU inference requires explicit --execute')
    run=out(run)
    with m161.run_lock(run,False):
      b=check(run); import torch
      if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<24*1024**3: raise ValueError('Need Linux CUDA with >=24GiB free VRAM')
      state=read(run/'state.json')
      for sid in b['sample_ids']:
        if receipt(run,sid):
          if sid not in state['completed_ids']: state['completed_ids'].append(sid)
          continue
        started=time.time(); row=worker(run,sid); state['spent_seconds']+=time.time()-started; state['completed_ids'].append(sid); update(run/'state.json',state)
        print(f'CHECKPOINT mixed={len(state["completed_ids"])}/50 sid={sid}',flush=True)
      if len(state['completed_ids'])!=50: raise ValueError('Incomplete mixed prediction population')
      save(run/'completed.json',{'batch_sha256':file_hash(run/'mixed_batch.json'),'sample_ids':b['sample_ids'],'condition':CONDITION,'gold_accessed':False})

def evaluate(run,gold_source,destination):
    run=out(run); b=check(run); done=read(run/'completed.json')
    if done['batch_sha256']!=file_hash(run/'mixed_batch.json') or done['sample_ids']!=b['sample_ids']: raise ValueError('Mixed run incomplete')
    raw=read(gold_source); pred=[]; gold=[]; rows=[]; baselines={'P2':[],'N0':[]}
    for sid in b['sample_ids']:
      _,p=receipt(run,sid); x=read(p); pred.append(x['predicted_verdict']); gold.append(raw[int(sid.split(':')[1])]['label']); rows.append(x)
      canonical=Path(b['canonical_run'])/sid.replace(':','-')/'predictions-attempt-001'/'predictions.jsonl'
      prior={v['condition']:v for v in jsonl(canonical)}
      if not {'P2','N0'}.issubset(prior): raise ValueError('Canonical matched controls missing: '+sid)
      baselines['P2'].append(prior['P2']['predicted_verdict']);baselines['N0'].append(prior['N0']['predicted_verdict'])
    result={'scope':'EXPLORATORY_NOT_CONFIRMATORY','condition':CONDITION,'claims':50,'metrics':hp.metric(gold,pred),
      'prompt_tokens':sum(x['prompt_tokens'] for x in rows),'output_tokens':sum(x['output_tokens'] for x in rows),'generation_wall_seconds':sum(x['latency_seconds'] for x in rows),'gpu_active_seconds':None}
    result['comparisons']={}
    for name,base in baselines.items():
      z=hp.compare(gold,base,pred,seed=42,n_boot=10000,n_perm=19999)
      z['correction_ids']=[sid for sid,g,a,y in zip(b['sample_ids'],gold,base,pred) if a!=g and y==g]
      z['regression_ids']=[sid for sid,g,a,y in zip(b['sample_ids'],gold,base,pred) if a==g and y!=g]
      result['comparisons'][CONDITION+'_vs_'+name]=z
    dest=out(destination); dest.mkdir(parents=True,exist_ok=False); save(dest/'results.json',result); print(json.dumps(result,indent=2)); return result

def main():
 p=argparse.ArgumentParser(); s=p.add_subparsers(dest='cmd',required=True)
 q=s.add_parser('prepare'); q.add_argument('--canonical-run',type=Path,required=True);q.add_argument('--cache-index',type=Path,required=True);q.add_argument('--memory',type=Path,required=True);q.add_argument('--tokenizer',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
 for n in ('check','run','evaluate'):
  q=s.add_parser(n);q.add_argument('--run',type=Path,required=True)
  if n=='run':q.add_argument('--execute',action='store_true')
  if n=='evaluate':q.add_argument('--gold-source',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.cmd=='prepare':prepare(a.canonical_run,a.cache_index,a.memory,a.tokenizer,a.output)
 elif a.cmd=='check':print(json.dumps(check(a.run),indent=2))
 elif a.cmd=='run':execute(a.run,a.execute)
 else:evaluate(a.run,a.gold_source,a.output)
if __name__=='__main__':main()
