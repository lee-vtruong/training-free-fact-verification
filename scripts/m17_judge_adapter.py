"""Run a user-chosen judge command once per M1.7 request, with receipts.

The command receives one request JSON on stdin and must write one JSON object
containing every judge field except `provenance` to stdout. This adapter adds the
frozen model/decode provenance, writes raw JSONL, and makes no API decision.
"""
import argparse, json, shlex, subprocess
from pathlib import Path
from tffv.provenance import file_hash

def rows(path): return [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--requests',required=True);p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true')
 p.add_argument('--command',required=True);p.add_argument('--model-id',required=True);p.add_argument('--model-revision',required=True)
 p.add_argument('--temperature',type=float,required=True);p.add_argument('--max-new-tokens',type=int,required=True);a=p.parse_args()
 reqs=rows(a.requests); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
 generation={'temperature':a.temperature,'do_sample':False,'max_new_tokens':a.max_new_tokens}
 existing=[]
 if out.exists():
  if not a.resume: raise ValueError('refuse overwrite raw judge output; use --resume only after verifying pinned command/model settings')
  existing=rows(out)
  expected={r['request_id']:r for r in reqs}
  if len({x.get('request_id') for x in existing})!=len(existing): raise ValueError('duplicate request in existing raw output')
  for x in existing:
   if x.get('request_id') not in expected: raise ValueError('existing output contains unknown request')
   p=x.get('provenance',{})
   if p.get('model_id')!=a.model_id or p.get('model_revision')!=a.model_revision or p.get('temperature')!=a.temperature or p.get('generation')!=generation or p.get('prompt_sha256')!=expected[x['request_id']]['prompt_sha256']:
    raise ValueError('existing raw output provenance differs from requested resume settings')
 done={x['request_id'] for x in existing}
 mode='a' if out.exists() else 'x'
 with out.open(mode,encoding='utf-8',newline='\n') as f:
  for r in reqs:
   if r['request_id'] in done: continue
   proc=subprocess.run(shlex.split(a.command),input=json.dumps(r,ensure_ascii=False),text=True,capture_output=True,check=True)
   obj=json.loads(proc.stdout); obj['provenance']={'model_id':a.model_id,'model_revision':a.model_revision,'temperature':a.temperature,'generation':generation,'prompt_sha256':r['prompt_sha256']}
   f.write(json.dumps(obj,ensure_ascii=False,sort_keys=True)+'\n')
 print(json.dumps({'requests':len(reqs),'reused':len(existing),'executed':len(reqs)-len(existing),'output':str(out),'sha256':file_hash(out),'model_id':a.model_id,'model_revision':a.model_revision},indent=2))
if __name__=='__main__': main()
