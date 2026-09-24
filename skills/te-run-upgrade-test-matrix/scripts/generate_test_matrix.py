#!/usr/bin/env python3
import argparse,json,subprocess
from pathlib import Path
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--repo',type=Path,required=True); ap.add_argument('--ref',required=True); ap.add_argument('--output',type=Path,required=True)
 a=ap.parse_args(); repo=a.repo.resolve(); a.output.mkdir(parents=True,exist_ok=True)
 def run(*x): return subprocess.run(['git','-C',str(repo),*x],text=True,errors='replace',capture_output=True)
 ref=run('rev-parse','--verify',a.ref+'^{commit}').stdout.strip(); files=run('ls-tree','-r','--name-only',ref).stdout.splitlines()
 groups=[]
 groups.append({'name':'python-static','hardware':'none','status':'runnable','command':'python -m compileall transformer_engine'})
 groups.append({'name':'plugin-tests','hardware':'NVIDIA-or-CPU','status':'runnable','command':'pytest tests/plugin -q'})
 groups.append({'name':'cuda-qa','hardware':'NVIDIA-CUDA','status':'runnable','command':'bash qa/L0_pytorch_unittest/test.sh'})
 groups.append({'name':'flagscale-smoke','hardware':'NVIDIA-CUDA','status':'runnable','command':'FlagScale single-config smoke'})
 vendors=sorted({p.split('/')[5] for p in files if p.startswith('transformer_engine/plugin/core/backends/vendor/') and len(p.split('/'))>5 and p.split('/')[5]!='__init__.py'})
 native={'cuda'}
 for v in vendors:
  if v not in native: groups.append({'name':'vendor-'+v,'hardware':v,'status':'blocked','reason':'Only NVIDIA GPU host authorized','owner':'user','command':''})
 data={'ref':ref,'groups':groups,'counts':{'total':len(groups),'blocked':sum(x['status']=='blocked' for x in groups)}}
 (a.output/'test-matrix.json').write_text(json.dumps(data,indent=2)+chr(10))
 (a.output/'test-matrix.tsv').write_text('name\thardware\tstatus\tcommand\treason\towner\n'+''.join(x['name']+'\t'+x['hardware']+'\t'+x['status']+'\t'+x.get('command','')+'\t'+x.get('reason','')+'\t'+x.get('owner','')+'\n' for x in groups))
 (a.output/'test-report.md').write_text('# Test Matrix\n\n'+'- Total: '+str(len(groups))+'\n- Blocked: '+str(data['counts']['blocked'])+'\n')
 print(json.dumps({'output':str(a.output),'total':len(groups),'blocked':data['counts']['blocked']}))
if __name__=='__main__': main()