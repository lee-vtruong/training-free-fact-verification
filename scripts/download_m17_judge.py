"""Download and receipt-pin the approved M1.7 frozen judge snapshot; no model load."""
import argparse, json
from pathlib import Path
from huggingface_hub import snapshot_download
from tffv.provenance import file_hash

REPO='Qwen/Qwen3-14B-AWQ'
REVISION='55182c306332ddb9f47ade5de8d8fdb53b8bcb03'

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache-dir',required=True);p.add_argument('--receipt',required=True);a=p.parse_args()
 receipt=Path(a.receipt)
 if receipt.exists(): raise FileExistsError('Refuse overwrite judge download receipt: '+str(receipt))
 snapshot=Path(snapshot_download(repo_id=REPO,revision=REVISION,cache_dir=a.cache_dir)).resolve()
 if snapshot.name!=REVISION: raise ValueError('Hub returned an unexpected snapshot revision')
 files={x.relative_to(snapshot).as_posix():file_hash(x) for x in sorted(snapshot.rglob('*')) if x.is_file()}
 receipt.parent.mkdir(parents=True,exist_ok=True)
 with receipt.open('x',encoding='utf-8') as f:
  json.dump({'schema_version':'m17-judge-download-v1','repo_id':REPO,'revision':REVISION,
             'quantization':'AWQ 4-bit','snapshot_path':str(snapshot),'files_sha256':files,
             'file_count':len(files)},f,sort_keys=True,indent=2);f.write('\n')
 print(json.dumps({'repo_id':REPO,'revision':REVISION,'snapshot':str(snapshot),'file_count':len(files),'receipt':str(receipt)},indent=2))
if __name__=='__main__':main()
