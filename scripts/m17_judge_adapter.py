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
 p=argparse.ArgumentParser(); p.add_argument('--requests',required=True);p.add_argument('--output',required=True)
 p.add_argument('--command',required=True);p.add_argument('--model-id',required=True);p.add_argument('--model-revision',required=True)
 p.add_argument('--temperature',type=float,required=True);p.add_argument('--max-new-tokens',type=int,required=True);a=p.parse_args()
 reqs=rows(a.requests); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists(): raise ValueError('refuse overwrite raw judge output')
 generation={'temperature':a.temperature,'do_sample':False,'max_new_tokens':a.max_new_tokens}
 with out.open('x',encoding='utf-8',newline='\n') as f:
  for r in reqs:
   proc=subprocess.run(shlex.split(a.command),input=json.dumps(r,ensure_ascii=False),text=True,capture_output=True,check=True)
   obj=json.loads(proc.stdout); obj['provenance']={'model_id':a.model_id,'model_revision':a.model_revision,'temperature':a.temperature,'generation':generation,'prompt_sha256':r['prompt_sha256']}
   f.write(json.dumps(obj,ensure_ascii=False,sort_keys=True)+'\n')
 print(json.dumps({'requests':len(reqs),'output':str(out),'sha256':file_hash(out),'model_id':a.model_id,'model_revision':a.model_revision},indent=2))
if __name__=='__main__': main()
