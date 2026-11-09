"""Frozen minimal HEAL pilot preparation/scoring. Never reads target gold to select examples."""
from collections import Counter
from copy import deepcopy
import hashlib
import math
import random
import re

from tffv.memory.heal import SourceContract,read,indexed,check_citations,validate_record
from tffv.provenance import digest,file_hash
from tffv.verification.prompting import build_verification_prompt
from tffv.verification.m14_prompting import build_d1_matched_prompt
from tffv.verification.ontology import VERDICTS

TOKENIZER_HASHES={'tokenizer.json':'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
                 'tokenizer_config.json':'d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101'}
CORE=('P0','P1','P2','P3','N0')


def tokenizer(path):
    from transformers import AutoTokenizer
    for name,h in TOKENIZER_HASHES.items():
        if file_hash(path/name)!=h: raise ValueError('Pinned tokenizer mismatch: '+name)
    return AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)


def tokens(tok,messages):
    text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    return tok(text)['input_ids']


def source_context(record):
    """Source-only excerpt selection is fixed before target retrieval/outcomes."""
    citations=record['citations']
    if citations:
        result=[]
        for e in record['evidence']:
            cc=[c for c in citations if c['passage_id']==e['passage_id']]
            if cc:
                lo=max(0,min(c['start'] for c in cc)-180)
                hi=min(len(e['rendered_text']),max(c['end'] for c in cc)+180)
                result.append({'passage_id':e['passage_id'],'source_url':e['source_url'],
                               'start':lo,'end':hi,'text':e['rendered_text'][lo:hi]})
        return result
    e=record['evidence'][0]
    return [{'passage_id':e['passage_id'],'source_url':e['source_url'],'start':0,'end':min(600,len(e['rendered_text'])),
             'text':e['rendered_text'][:600]}]


def block(record,mode,context):
    evidence='\n'.join(f"[{i+1}] {e['source_url']}\n{e['text']}" for i,e in enumerate(context))
    previous=record['base_prediction']['reparsed_verdict']+' (incorrect for this historical example)' if mode in ('P3','P4') else '[not provided]'
    correction=record['corrected_explanation'] if mode=='P4' else '[not provided]'
    return ('Historical task reference (independent of the target; decide the target from its own evidence):\n'
            f"Claim: {record['claim']}\nHistorical retrieved evidence:\n{evidence}\n"
            f"Gold verdict: {record['gold_verdict']}\nPrevious Base verdict: {previous}\n"
            f"Verified correction: {correction}\n")


def demo_length(tok,record,mode,context):
    return len(tok(block(record,mode,context))['input_ids'])


def pool(memory,contract,tok):
    contract.check_memory(memory)
    if set(indexed(memory['records']))!=set(contract.legacy_ids+contract.batch_ids):
        raise ValueError('Initial pilot source cohort must remain exactly75')
    correct=[r for r in memory['records'] if not r['is_base_error'] and r['failure_classification']=='BASE_CORRECT'
             and r['review_status'] not in ('REJECTED','INSUFFICIENT_EVIDENCE','PROVENANCE_UNKNOWN') and r['evidence']]
    eligible={};matching={}
    for r in memory['records']:
        if not r['is_base_error'] or r['review_status'] not in ('PENDING_REVIEW','VERIFIED'):continue
        if r.get('draft_assessment',{}).get('status')!='PENDING_REVIEW':continue
        check_citations(r)
        context=source_context(r)
        n=demo_length(tok,r,'P3',context)
        if n>768:continue
        candidates=[]
        for c in correct:
            if c['group_id']==r['group_id'] or c['gold_verdict']!=r['gold_verdict']:continue
            cc=source_context(c);cn=demo_length(tok,c,'P2',cc)
            if max(n,cn)<=768 and abs(cn-n)/max(1,n)<=.20:
                candidates.append((abs(cn-n),c['sample_id'],c,cc,cn))
        if candidates:
            _,_,c,cc,cn=min(candidates,key=lambda x:(x[0],x[1]))
            eligible[r['sample_id']]={'record':r,'context':context,'demo_tokens':n}
            matching[r['sample_id']]={'record':c,'context':cc,'demo_tokens':cn}
    if not eligible:raise ValueError('BLOCKED: no sufficient source errors with length-matched different-group correct controls')
    return eligible,matching


def bm25_choice(query,eligible):
    # Fixed lexical parameters; no fit, labels, outcomes, seed or k search.
    docs={sid:re.findall(r'\w+',v['record']['claim'].casefold()) for sid,v in eligible.items()}
    avg=sum(map(len,docs.values()))/len(docs) or 1
    q=set(re.findall(r'\w+',query.casefold()));scores={}
    for sid,words in docs.items():
        counts=Counter(words);s=0.0
        for term in sorted(q):
            df=sum(term in ww for ww in docs.values());tf=counts[term]
            if tf:s+=math.log(1+(len(docs)-df+.5)/(df+.5))*tf*2.2/(tf+1.2*(.25+.75*len(words)/avg))
        scores[sid]=s
    return min(scores,key=lambda sid:(-scores[sid],sid))


def prepare_slates(memory,claims,contract,tok):
    ids=contract.schedule['development_pilot_claims']
    if ids!=contract.schedule['development_order'][:50] or len(set(ids))!=50:raise ValueError('DEV50 schedule changed')
    for sid in ids:
        if contract.rows[sid]['partition']!='DEVELOPMENT':raise ValueError('Only locked DEVELOPMENT50 allowed')
    if len({contract.rows[s]['group_id'] for s in ids})!=50:raise ValueError('Planned paired units require50 distinct sealed blocks')
    eligible,matching=pool(memory,contract,tok)
    slates=[]
    for sid in ids:
        target=claims[sid]
        if set(target)!={'sample_id','dataset','split','claim'}:raise ValueError('Target annotations/extra fields prohibited')
        if target['dataset']!='AVeriTeC' or target['split']!='train':raise ValueError('Official split/dataset changed')
        if hashlib.sha256(target['claim'].encode()).hexdigest()!=contract.rows[sid]['claim_sha256']:
            raise ValueError('DEV claim changed')
        chosen=bm25_choice(target['claim'],eligible)
        e=eligible[chosen];c=matching[chosen]
        if e['record']['group_id']==contract.rows[sid]['group_id'] or c['record']['group_id']==contract.rows[sid]['group_id']:
            raise ValueError('Source-target group leakage')
        slates.append({'sample_id':sid,'error_id':chosen,'correct_id':c['record']['sample_id'],
                       'error_record_id':e['record']['record_id'],'correct_record_id':c['record']['record_id'],
                       'error_context':e['context'],'correct_context':c['context'],
                       'unpadded_demo_tokens':{'error':e['demo_tokens'],'control':c['demo_tokens']}})
    lookup=indexed(memory['records']);selected={s['error_id'] for s in slates}
    p4=all(lookup[s]['review_status']=='VERIFIED' for s in selected)
    return slates,{'k':1,'seed':42,'source_eligible_ids':sorted(eligible),
        'selected_error_counts':dict(Counter(s['error_id'] for s in slates)),
        'selected_error_groups':len({lookup[s]['group_id'] for s in selected}),
        'matched_control_coverage':1.0,'control_human_review_pending':sum(lookup[s['correct_id']]['review_status']!='VERIFIED' for s in slates),
        'P4':'READY' if p4 else 'BLOCKED','P4_reason':None if p4 else 'Selected same-ID exemplars lack two real human approvals',
        'active_conditions':list(CORE)+(['P4'] if p4 else [])}


def render_set(claim,evidence,slate,memory,tok,active):
    lookup=indexed(memory['records']);e=lookup[slate['error_id']];c=lookup[slate['correct_id']]
    native=build_d1_matched_prompt(claim,evidence)['messages']
    result={'P0':build_verification_prompt(claim,evidence)['messages'],'P1':native}
    for mode in active:
        if mode in result:continue
        if mode=='P4' and e['review_status']!='VERIFIED':raise ValueError('P4 BLOCKED: no verified same-ID correction')
        if mode=='P4':validate_record(e)
        r=c if mode=='N0' else e;context=slate['correct_context'] if mode=='N0' else slate['error_context']
        msgs=deepcopy(native)
        msgs[1]['content']=block(r,mode,context)+'\nLength-control filler (no factual content):\n\n--- Target ---\n'+msgs[1]['content']
        result[mode]=msgs
    # Actual pinned-tokenizer equality for P2/P3/N0; P0/P1 stay native.
    matched=('P2','P3','N0');lengths={k:len(tokens(tok,result[k])) for k in matched}
    target=max(lengths.values())
    for mode in matched:
        for _ in range(800):
            n=len(tokens(tok,result[mode]))
            if abs(target-n)<=1:break
            if n>target:raise ValueError('Padding overshot its prespecified matching target')
            result[mode][1]['content']=result[mode][1]['content'].replace('\n\n--- Target ---',' .\n\n--- Target ---',1)
        else:raise ValueError('Cannot meet matched-length tolerance')
    if max(len(tokens(tok,result[k])) for k in matched)-min(len(tokens(tok,result[k])) for k in matched)>2:
        raise ValueError('Matched actual prompt lengths differ by >2 tokens')
    output={}
    for mode,msgs in result.items():
        ids=tokens(tok,msgs)
        if len(ids)+512>8192:raise ValueError('Context reservation overflow; no target evidence truncation allowed')
        output[mode]={'messages':msgs,'prompt_tokens':len(ids),'input_ids_sha256':digest(ids),'messages_sha256':digest(msgs),
                      'exemplar_ids':[] if mode in ('P0','P1') else [slate['correct_id'] if mode=='N0' else slate['error_id']]}
    if 'P4' in active and output['P4']['prompt_tokens']-output['P1']['prompt_tokens']>768:
        raise ValueError('P4 exceeds preregistered example overhead; no silent rationale truncation')
    return output


def metric(gold,pred):
    from tffv.verification.evaluate import compute_metrics
    out=compute_metrics([{'gold_label':g,'predicted_verdict':p,'is_correct':g==p} for g,p in zip(gold,pred)])
    out['valid_rate']=sum(p in VERDICTS for p in pred)/len(gold)
    return out


def compare(gold,base,target,seed=42,n_boot=10000,n_perm=19999):
    import numpy as np
    n=len(gold);c={v:i for i,v in enumerate(VERDICTS)}
    g=np.array([c[x] for x in gold]);a=np.array([c.get(x,-1) for x in base]);b=np.array([c.get(x,-1) for x in target])
    def f1(y,p):
        return sum(2*np.sum((y==k)&(p==k))/max(1,np.sum(y==k)+np.sum(p==k)) for k in range(4))/4
    trans=Counter(('C' if x==z else 'W')+('C' if y==z else 'W') for z,x,y in zip(gold,base,target))
    discord=trans['CW']+trans['WC'];small=min(trans['CW'],trans['WC'])
    p=min(1.0,2*sum(math.comb(discord,k) for k in range(small+1))/2**discord) if discord else 1.0
    rng=np.random.default_rng(seed);acc=[];mf=[];missing=0
    for _ in range(n_boot):
        ix=rng.integers(0,n,n);acc.append(float(np.mean(b[ix]==g[ix])-np.mean(a[ix]==g[ix])))
        missing+=len(np.unique(g[ix]))<4
        mf.append(float(f1(g[ix],b[ix])-f1(g[ix],a[ix])))
    observed=float(f1(g,b)-f1(g,a));extreme=0
    for _ in range(n_perm):
        swap=rng.integers(0,2,n).astype(bool);aa=np.where(swap,b,a);bb=np.where(swap,a,b)
        extreme+=abs(float(f1(g,bb)-f1(g,aa)))>=abs(observed)-1e-12
    return {'transitions':{k:trans[k] for k in ('CC','CW','WC','WW')},'accuracy_delta':float(np.mean(b==g)-np.mean(a==g)),
        'accuracy_bootstrap_ci95':list(map(float,np.quantile(acc,[.025,.975]))),
        'macro_f1_delta':observed,'macro_f1_bootstrap_ci95':list(map(float,np.quantile(mf,[.025,.975]))),
        'exact_mcnemar_p':p,'macro_f1_paired_swap_p':(1+extreme)/(n_perm+1),'bootstrap_samples':n_boot,'permutations':n_perm,
        'missing_class_bootstrap_frequency':missing/n_boot,
        'invalid_transitions':dict(Counter(('U' if x not in VERDICTS else ('C' if x==z else 'W'))+('U' if y not in VERDICTS else ('C' if y==z else 'W')) for z,x,y in zip(gold,base,target)))}


def holm(values):
    result={};maximum=0
    for i,(k,p) in enumerate(sorted(values.items(),key=lambda kv:kv[1])):
        maximum=max(maximum,min(1.,p*(len(values)-i)));result[k]=maximum
    return result
