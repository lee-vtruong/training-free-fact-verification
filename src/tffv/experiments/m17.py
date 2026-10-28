"""M1.7 LLM-judged corrective-memory preparation; CPU-only and source-only.

The external judge is deliberately an adapter: this module creates signed JSONL
requests and validates returned JSONL. It never imports an API SDK, loads a
model, calls a server, or modifies M1.6 human-review artifacts.
"""
import argparse, hashlib, json
from pathlib import Path

from tffv.experiments.m161 import output_path
from tffv.memory.heal import SourceContract, read, save
from tffv.provenance import digest, file_hash
from tffv.verification.ontology import VERDICTS

ROOT = Path(__file__).resolve().parents[3]
CFG = ROOT / 'configs/m17_protocol.json'
CANDIDATES = ROOT / 'docs/audits/M1_6_2E_CANDIDATE_DRAFTS.json'
# The server's canonical M1.6.2a input bank is first. The older combined75
# copy exists only in some local worktrees and is a read-only compatibility aid.
BANK_CANDIDATES = (ROOT / 'outputs/m162a-inputs-v1/memory75.json',
                   ROOT / 'outputs/m161-review-preparation-v1/combined75.json')

def _canon(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
def _hash(x): return hashlib.sha256(_canon(x).encode()).hexdigest()
def _lines(path): return [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def _write_lines(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        for row in rows: f.write(_canon(row)+'\n')

def config():
    c=read(CFG)
    if c['version']!='m17-judge-feasibility-v1' or c['human_gate_unchanged'] is not True: raise ValueError('M1.7 protocol drift')
    if c['candidate_kinds']!=['ORIGINAL_DRAFT','E3_CANDIDATE_DRAFT'] or c['excluded_triage_classes']!=['E1','E2']: raise ValueError('source scope drift')
    if c['judge_generation']!={'temperature':0.0,'do_sample':False,'max_new_tokens':768}: raise ValueError('judge decoding drift')
    return c

def evidence_bank(path=None):
    if path is not None:
        p=Path(path)
        if not p.is_file(): raise FileNotFoundError('M1.7 evidence bank missing: '+str(p))
        return p
    for p in BANK_CANDIDATES:
        if p.is_file(): return p
    raise FileNotFoundError('M1.7 needs canonical evidence bank; checked: '+', '.join(map(str,BANK_CANDIDATES)))

def candidates(path=CANDIDATES, bank_path=None):
    c=config(); data=read(path); rows=[]
    bank_file=evidence_bank(bank_path)
    bank={r['sample_id']:r for r in read(bank_file)['records']}
    for r in data['records']:
        if r['record_kind'] not in c['candidate_kinds']: continue
        if r['triage_class'] in c['excluded_triage_classes']: continue
        if r['evidence_availability_change']: raise ValueError('E2 cannot enter M1.7')
        if not r['draft_text'] or r['sample_id'] not in bank: raise ValueError('candidate lacks frozen correction/evidence')
        top8=bank[r['sample_id']]['evidence']
        if len(top8)!=8 or any(not e.get('rendered_text') for e in top8): raise ValueError('full rendered Top-8 unavailable')
        rr=dict(r); rr['_m17_top8']=top8; rr['_m17_bank_path']=str(bank_file.resolve()); rows.append(rr)
    if len(rows)!=21 or len({r['sample_id'] for r in rows})!=21: raise ValueError('expected exactly 5 original + 16 E3 SOURCE records')
    return sorted(rows,key=lambda r:int(r['sample_id'].split(':')[1]))

def evidence_text(r):
    # Judge receives every immutable rendered Top-8 passage, never an inferred summary.
    return [{'rank':q.get('rank'),'url':q.get('source_url'),'quote':q['rendered_text']}
            for q in r['_m17_top8']]

def request(r, stage, prior=None):
    base={'schema_version':'m17-judge-request-v1','stage':stage,'sample_id':r['sample_id'],
          'claim':r['claim'],'evidence':evidence_text(r),'base_verdict':r['base_verdict'],
          'protocol_sha256':file_hash(CFG),'candidate_record_sha256':_hash(r)}
    if stage=='evidence_assessment':
        base['instructions']='Assess only displayed SOURCE claim and evidence. Gold verdict and correction are withheld. Return the required structured JSON; quote only exact displayed excerpts.'
    elif stage=='correction_validation':
        if prior is None: raise ValueError('stage two requires stage-one output')
        base.update({'source_gold_verdict':r['gold_verdict'],'candidate_correction':r['draft_text'],'candidate_mechanism':r['mechanism'],
                     'stage_one_output':prior,
                     'instructions':'Validate whether the candidate correction is grounded only in displayed evidence and supports the SOURCE gold verdict. Reject facts outside displayed excerpts. Return required structured JSON.'})
    else: raise ValueError('unknown stage')
    base['request_id']=_hash(base)
    base['prompt_sha256']=_hash({k:v for k,v in base.items() if k not in ('request_id','prompt_sha256')})
    return base

def make_requests(output, candidate_path=CANDIDATES, stage_one_outputs=None, bank_path=None):
    rows=candidates(candidate_path,bank_path); reqs=[]
    if stage_one_outputs is None:
        reqs=[request(r,'evidence_assessment') for r in rows]; name='evidence_assessment_requests.jsonl'
    else:
        got={x['request_id']:x for x in _lines(stage_one_outputs)}
        for r in rows:
            first=request(r,'evidence_assessment')
            if first['request_id'] not in got: raise ValueError('missing stage-one output '+r['sample_id'])
            reqs.append(request(r,'correction_validation',got[first['request_id']]))
        name='correction_validation_requests.jsonl'
    out=output_path(output)
    # Earlier versions made this directory before validating the evidence bank.
    # An interrupted/failed preflight therefore left an empty directory. Reuse
    # only that demonstrably empty shell; never overwrite a partial run.
    if out.exists():
        if not out.is_dir() or any(out.iterdir()):
            raise FileExistsError('Output exists and is not an empty failed-preflight directory: '+str(out))
    else:
        out.mkdir(parents=True,exist_ok=False)
    _write_lines(out/name,reqs)
    save(out/'manifest.json',{'schema_version':'m17-request-manifest-v1','stage':reqs[0]['stage'],'count':len(reqs),'requests_sha256':file_hash(out/name),'candidate_source_sha256':file_hash(candidate_path),'evidence_bank_path':rows[0]['_m17_bank_path'],'evidence_bank_sha256':file_hash(rows[0]['_m17_bank_path']),'protocol_sha256':file_hash(CFG),'source_only':True,'development_targets_present':False})
    return out/name

def _validate_output(out, req):
    required={'request_id','stage','evidence_supported','verdict_assessment','grounding_quotes','failure_mechanism','decision','rationale','confidence','unsupported_facts','provenance'}
    if set(out)!=required or out['request_id']!=req['request_id'] or out['stage']!=req['stage']: raise ValueError('judge output/request identity mismatch')
    if out['verdict_assessment'] not in set(VERDICTS)|{'ABSTAIN'} or out['decision'] not in {'ACCEPT','REVISE','REJECT','ABSTAIN'}: raise ValueError('judge ontology/decision invalid')
    if not isinstance(out['confidence'],(int,float)) or not 0<=out['confidence']<=1: raise ValueError('confidence invalid')
    allowed='\n'.join(q['quote'] for q in req['evidence'])
    for q in out['grounding_quotes']:
        if not isinstance(q,str) or q not in allowed: raise ValueError('judge quote is not exact displayed evidence')
    p=out['provenance']; g=config()['judge_generation']
    if not p['model_id'] or not p['model_revision'] or p['temperature']!=g['temperature'] or p['generation']!=g or p['prompt_sha256']!=req['prompt_sha256']: raise ValueError('judge provenance/decoding mismatch')
    if out['decision']=='ACCEPT' and (not out['evidence_supported'] or not out['grounding_quotes'] or out['unsupported_facts']): raise ValueError('accepted output is not grounded')

def validate(stage_requests, judge_outputs, output):
    reqs={r['request_id']:r for r in _lines(stage_requests)}; outs=_lines(judge_outputs)
    if len(reqs)!=21 or len(outs)!=21 or len({x['request_id'] for x in outs})!=21: raise ValueError('must validate exactly one output per 21 source request')
    for x in outs:
        if x['request_id'] not in reqs: raise ValueError('unknown judge request')
        _validate_output(x,reqs[x['request_id']])
    out=output_path(output); out.mkdir(parents=True,exist_ok=False)
    _write_lines(out/'validated_outputs.jsonl',outs)
    save(out/'receipt.json',{'schema_version':'m17-judge-validation-v1','requests_sha256':file_hash(stage_requests),'raw_outputs_sha256':file_hash(judge_outputs),'validated_outputs_sha256':file_hash(out/'validated_outputs.jsonl'),'count':21,'stage':outs[0]['stage'],'source_only':True})
    return out

def build_memory(candidate_path, stage_one, stage_two, output, bank_path=None):
    rows={r['sample_id']:r for r in candidates(candidate_path,bank_path)}
    one={x['request_id']:x for x in _lines(stage_one)}; two={x['request_id']:x for x in _lines(stage_two)}
    selected=[]
    for r in rows.values():
        q1=request(r,'evidence_assessment'); q2=request(r,'correction_validation',one[q1['request_id']])
        a,b=one[q1['request_id']],two.get(q2['request_id'])
        if b and b['decision']=='ACCEPT' and b['evidence_supported'] and b['verdict_assessment']==r['gold_verdict'] and not b['unsupported_facts']:
            selected.append({'sample_id':r['sample_id'],'claim':r['claim'],'evidence':evidence_text(r),'gold_verdict':r['gold_verdict'],'base_verdict':r['base_verdict'],'correction':r['draft_text'],'mechanism':r['mechanism'],'memory_status':'LLM_VALIDATED','llm_judge':{'evidence_assessment':a,'correction_validation':b}})
    out=output_path(output); out.mkdir(parents=True,exist_ok=False)
    payload={'schema_version':'m17-llm-validated-memory-v1','scope':'EXPLORATORY_SOURCE_ONLY','human_review_status':'UNCHANGED_NOT_HUMAN_VERIFIED','records':selected,'input_hashes':{'candidates':file_hash(candidate_path),'stage_one':file_hash(stage_one),'stage_two':file_hash(stage_two),'protocol':file_hash(CFG)}}
    save(out/'memory.json',payload)
    save(out/'summary.json',{'source_candidates':21,'llm_validated':len(selected),'gold_coverage':sorted(set(r['gold_verdict'] for r in selected)),'mechanism_coverage':sorted(set(r['mechanism'] for r in selected)),'leakage_groups':'not emitted by candidate artifact; no DEVELOPMENT target used','human_verified_count':0})
    return out

def main():
 p=argparse.ArgumentParser(); s=p.add_subparsers(dest='cmd',required=True)
 for name in ('make-requests','validate','build-memory'):
  q=s.add_parser(name)
  if name=='make-requests': q.add_argument('--output',required=True);q.add_argument('--candidates',default=str(CANDIDATES));q.add_argument('--evidence-bank');q.add_argument('--stage-one-outputs')
  elif name=='validate': q.add_argument('--requests',required=True);q.add_argument('--outputs',required=True);q.add_argument('--output',required=True)
  else: q.add_argument('--candidates',default=str(CANDIDATES));q.add_argument('--evidence-bank');q.add_argument('--stage-one',required=True);q.add_argument('--stage-two',required=True);q.add_argument('--output',required=True)
 a=p.parse_args()
 if a.cmd=='make-requests': print(make_requests(a.output,a.candidates,a.stage_one_outputs,a.evidence_bank))
 elif a.cmd=='validate': print(validate(a.requests,a.outputs,a.output))
 else: print(build_memory(a.candidates,a.stage_one,a.stage_two,a.output,a.evidence_bank))
if __name__=='__main__': main()
