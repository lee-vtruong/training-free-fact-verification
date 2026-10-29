"""Read-only cache contracts for sealed M1.6 artifacts; never retrieves or loads a model.

This tool indexes an existing sealed run.  It deliberately does not treat an
ordinal, an unsealed .npy file, or an ERRORFlow output as reusable evidence.
"""
from pathlib import Path
import argparse
from collections import Counter
import json
import sys
import time

from tffv.provenance import digest, file_hash, write_json
from tffv.memory.heal import jsonl, read
from tffv.experiments import m161
from tffv.verification import heal_pilot as hp

ROOT = Path(__file__).resolve().parents[3]


def output(path):
    p=Path(path).resolve(); root=(ROOT/'outputs').resolve()
    if not p.is_relative_to(root) or p==root: raise ValueError('New cache artifacts must be under outputs/')
    return p


def normalized_claim_hash(claim):
    return digest(' '.join(claim.split()))


def require_no_source_target_leakage(source_groups, target_groups):
    overlap=set(source_groups)&set(target_groups)
    if overlap: raise ValueError('Cross-partition SOURCE/DEVELOPMENT group leakage: '+','.join(sorted(overlap)))


def receipt(run, path, stage, sid):
    path=Path(path)
    if not path.is_file(): raise ValueError(f'Missing {stage} receipt: {sid}')
    r=read(path)
    if r.get('sample_id')!=sid or r.get('stage')!=stage or r.get('batch_sha256')!=file_hash(run/'batch.json'):
        raise ValueError(f'Receipt identity mismatch: {stage} {sid}')
    result=(run/r['output']).resolve()
    if not result.is_relative_to(run.resolve()) or not result.is_file(): raise ValueError(f'Receipt output missing/outside run: {stage} {sid}')
    if result.relative_to(run.resolve()).as_posix() not in r.get('files_sha256',{}): raise ValueError(f'Receipt does not seal output: {stage} {sid}')
    for rel, h in r['files_sha256'].items():
        f=(run/rel).resolve()
        if not f.is_relative_to(run.resolve()) or not f.is_file() or file_hash(f)!=h: raise ValueError(f'Receipt file hash mismatch: {stage} {sid}')
    return r, result


def evidence_contract(batch, claim, dataset_sha256, acquire, candidates, r4):
    cfg=batch['configuration']; retrieval=cfg['retrieval']
    return {
        'dataset_release_sha256':dataset_sha256,
        'claim_id':claim['sample_id'], 'claim_sha256':digest(claim['claim']),
        'normalized_claim_sha256':normalized_claim_hash(claim['claim']),
        'store_snapshot_receipt_sha256':file_hash(acquire),
        'candidate_pool_receipt_sha256':file_hash(candidates),
        'retrieval_r1_r4_config_sha256':digest(retrieval),
        'embedding_checkpoint':retrieval['embedding'], 'reranker_checkpoint':retrieval['reranker'],
        'candidate_policy':cfg['candidate_policy'], 'top_k':8,
        'rendered_top8_evidence_sha256':file_hash(r4),
    }


def prediction_contract(batch, evidence_key, prompt):
    cfg=batch['configuration']
    return {
        'evidence_cache_key':evidence_key,
        'verifier_snapshot':cfg['backbone'],
        'effective_rendered_prompt_sha256':prompt['messages_sha256'],
        'effective_input_ids_sha256':prompt['input_ids_sha256'],
        'chat_template_thinking_mode':'enable_thinking=false',
        'generation':cfg['generation'],
        'output_parser':'tffv.verification.ontology.normalize_verdict/v1',
    }


def index(canonical_run, destination):
    run=Path(canonical_run).resolve(); batch=read(run/'batch.json')
    if batch.get('gold_accessed') is not False or len(batch['sample_ids'])!=len(set(batch['sample_ids'])): raise ValueError('Invalid canonical batch')
    prompts=read(run/'sealed_prompts.json')
    if prompts['batch_sha256']!=file_hash(run/'batch.json') or prompts.get('gold_accessed') is not False: raise ValueError('Prompt seal mismatch')
    claims={r['sample_id']:r for r in batch['claims']}; rows=[]
    # Canonical M1.6 batches retain SOURCE memory separately.  Check groups when
    # the sealed project contract is present; fixture/unit indexes remain useful
    # for byte-contract testing without a full project partition tree.
    memory_path=Path(batch['source_memory_path'])
    contract_root=ROOT/'docs/partitions/m16-v2'
    if memory_path.is_file() and contract_root.is_dir():
        from tffv.memory.heal import SourceContract
        source=read(memory_path); c=SourceContract(ROOT); c.check_memory(source)
        require_no_source_target_leakage((r['group_id'] for r in source['records']),
                                         (c.rows[s]['group_id'] for s in batch['sample_ids']))
    started=time.monotonic(); total=len(batch['sample_ids'])
    print(f'INDEX start: validating {total} sealed claims; no output index is written until all hashes pass.',file=sys.stderr,flush=True)
    for number,sid in enumerate(batch['sample_ids'],1):
        claim=claims.get(sid)
        if not claim or set(claim)!={'sample_id','dataset','split','claim'}: raise ValueError('Target labels/extra fields in cache indexing')
        d=run/sid.replace(':','-')
        a,_=receipt(run,d/'acquire.receipt.json','acquire',sid)
        c,_=receipt(run,d/'candidates.receipt.json','candidates',sid)
        rr,r4=receipt(run,d/'retrieval.receipt.json','retrieval',sid)
        pr,preds=receipt(run,d/'predictions.receipt.json','predictions',sid)
        evidence=jsonl(r4)
        if len(evidence)!=1 or evidence[0]['sample_id']!=sid or evidence[0]['claim_sha256']!=digest(claim['claim']) or len(evidence[0]['evidence'])!=8: raise ValueError('Top-8 cache identity/coverage failure')
        if prompts['r4_sha256'].get(sid)!=file_hash(r4): raise ValueError('Prompt/evidence cache mismatch')
        im=read(d/'input/claims_manifest.json')
        if im.get('annotation_values_used') is not False or not im.get('source_sha256'): raise ValueError('Input is not a scrubbed, hashed claim manifest')
        ec=evidence_contract(batch,claim,im['source_sha256'],d/'acquire.receipt.json',d/'candidates.receipt.json',r4); ek=digest(ec)
        rendered=prompts['inputs'][sid]; predrows=jsonl(preds)
        if {x['condition'] for x in predrows}!=set(batch['status']['active_conditions']): raise ValueError('Canonical prediction condition mismatch')
        pks={}
        for p in predrows:
            prompt=rendered[p['condition']]
            if p['input_ids_sha256']!=prompt['input_ids_sha256'] or p['gold_accessed'] is not False: raise ValueError('Prediction/prompt cache mismatch')
            pks[p['condition']]=digest(prediction_contract(batch,ek,prompt))
        rows.append({'sample_id':sid,'evidence_cache_key':ek,'evidence_contract':ec,
                     'r4_path':str(r4),'r4_sha256':file_hash(r4),'retrieval_receipt_sha256':file_hash(d/'retrieval.receipt.json'),
                     'prediction_receipt_sha256':file_hash(d/'predictions.receipt.json'),'prediction_cache_keys':pks,
                     'conditions':sorted(pks),'prediction_path':str(preds),'prompt_paths_hash':digest(rendered)})
        print(f'INDEX {number}/{total}: {sid} verified; elapsed={time.monotonic()-started:.1f}s',file=sys.stderr,flush=True)
    obj={'schema_version':'m16-cache-index-v1','scope':'READ_ONLY_CANONICAL_REUSE','canonical_run':str(run),
         'batch_sha256':file_hash(run/'batch.json'),'sealed_prompts_sha256':file_hash(run/'sealed_prompts.json'),
         'source_memory_sha256':batch['source_memory_sha256'],'source_memory_path':batch['source_memory_path'],
         'source_memory_contract':{'source_partitions':'SOURCE only','source_base_run':'bound in memory record provenance',
             'gold_provenance':'source-only record field/provenance','selection_order_sha256':digest(batch['slates']),
             'correction_review_version':'unreviewed; P4 BLOCKED'},
         'active_conditions':batch['status']['active_conditions'],'records':rows,'gold_accessed':False}
    dest=output(destination); dest.parent.mkdir(parents=True,exist_ok=True); write_json(dest,obj)
    print(f'INDEX complete: {total}/{total}; output={dest}',file=sys.stderr,flush=True)
    return obj


def load_index(path):
    x=read(path)
    if x['schema_version']!='m16-cache-index-v1' or x['gold_accessed'] is not False: raise ValueError('Invalid cache index')
    if len({r['sample_id'] for r in x['records']})!=len(x['records']): raise ValueError('Duplicate cache IDs')
    return x


def select(index_path, ids, condition=None):
    x=load_index(index_path); lookup={r['sample_id']:r for r in x['records']}; want=ids or sorted(lookup)
    missing=[sid for sid in want if sid not in lookup]; hits=[]
    for sid in want:
        if sid in lookup and (condition is None or condition in lookup[sid]['prediction_cache_keys']): hits.append(sid)
    return {'requested_ids':want,'reusable_ids':hits,'missing_ids':missing,
            'missing_prediction_ids':[] if condition is None else [sid for sid in want if sid in lookup and condition not in lookup[sid]['prediction_cache_keys']],
            'retrieval_called':False,'model_loaded':False,'gold_accessed':False}


def evaluate_only(index_path, gold_source, destination):
    x=load_index(index_path); run=Path(x['canonical_run']); batch=read(run/'batch.json')
    if file_hash(run/'batch.json')!=x['batch_sha256'] or file_hash(run/'sealed_prompts.json')!=x['sealed_prompts_sha256']: raise ValueError('Canonical identity changed')
    # Evaluation-only is allowed to open gold only after all cache records/receipts were admitted.
    from tffv.verification.ontology import normalize_verdict
    raw=read(gold_source); out={'scope':'EXPLORATORY_EVALUATION_ONLY','claims':len(x['records']),'metrics':{},'gold_accessed':True}
    for cond in x['active_conditions']:
        pred=[]; gold=[]
        for r in x['records']:
            found=[z for z in jsonl(r['prediction_path']) if z['condition']==cond]
            if len(found)!=1: raise ValueError('Prediction cache incomplete')
            pred.append(normalize_verdict(found[0]['raw_generation'])); gold.append(raw[int(r['sample_id'].split(':')[1])]['label'])
        out['metrics'][cond]=hp.metric(gold,pred)
    write_json(output(destination),out); return out


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('index'); q.add_argument('--canonical-run',type=Path,required=True); q.add_argument('--output',type=Path,required=True)
    for name in ('prepare-evidence','inference-only','resume'):
        q=sub.add_parser(name); q.add_argument('--index',type=Path,required=True); q.add_argument('--ids',nargs='*');
        if name!='prepare-evidence': q.add_argument('--condition',required=True)
    q=sub.add_parser('evaluate-only'); q.add_argument('--index',type=Path,required=True); q.add_argument('--gold-source',type=Path,required=True); q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='index': result=index(a.canonical_run,a.output)
    elif a.command=='evaluate-only': result=evaluate_only(a.index,a.gold_source,a.output)
    else: result=select(a.index,a.ids,a.condition if a.command!='prepare-evidence' else None)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
