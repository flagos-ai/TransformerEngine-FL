#!/usr/bin/env python3
import argparse,json,re,subprocess
from pathlib import Path
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--repo',type=Path,required=True); ap.add_argument('--ref',required=True); ap.add_argument('--output',type=Path,required=True)
 a=ap.parse_args(); repo=a.repo.resolve(); a.output.mkdir(parents=True,exist_ok=True)
 def run(*x): return subprocess.run(['git','-C',str(repo),*x],text=True,errors='replace',capture_output=True)
 ref=run('rev-parse','--verify',a.ref+'^{commit}').stdout.strip()
 files=run('ls-tree','-r','--name-only',ref).stdout.splitlines()
 wf=[p for p in files if p.startswith('.github/workflows/') and p.endswith(('.yml','.yaml'))]
 cfg=[p for p in files if p.startswith('.github/configs/') and p.endswith(('.yml','.yaml'))]
 rows=[]; missing=[]; masks=[]
 for path in wf+cfg:
  text=run('show',ref+':'+path).stdout; rows.append({'path':path,'nonempty':bool(text.strip()),'syntax':'unverified'})
  for local in re.findall(r'uses:\s*\./([^\s#]+)',text):
   local=local.rstrip('/')
   if local not in files and not any(x.startswith(local+'/') for x in files): missing.append({'source':path,'reference':local})
  masks += re.findall(r'\|\|\s*true|continue-on-error:\s*true',text,re.I)
 vendors=sorted({p.split('/')[5] for p in files if p.startswith('transformer_engine/plugin/core/backends/vendor/') and len(p.split('/'))>5 and p.split('/')[5]!='__init__.py'})
 alltext='\n'.join(run('show',ref+':'+p).stdout for p in wf)
 coverage=[{'backend':v,'covered':v.lower() in alltext.lower(),'reason':''} for v in vendors]
 data={'ref':ref,'workflows':rows,'configs':cfg,'missing_references':missing,'backend_coverage':coverage,'failure_masks':sorted(set(masks))}
 (a.output/'cicd-audit.json').write_text(json.dumps(data,indent=2)+chr(10))
 (a.output/'workflow-matrix.tsv').write_text('backend\tcovered\treason\n'+''.join(x['backend']+'\t'+str(x['covered']).lower()+'\t\n' for x in coverage))
 (a.output/'missing-references.tsv').write_text('source\treference\n'+''.join(x['source']+'\t'+x['reference']+'\n' for x in missing))
 (a.output/'cicd-audit.md').write_text('# CI/CD Audit\n\n'+'- Workflows/configs: '+str(len(rows))+'\n- Missing local references: '+str(len(missing))+'\n- Failure masks: '+str(len(set(masks)))+'\n')
 print(json.dumps({'output':str(a.output),'workflows':len(rows),'missing':len(missing),'failure_masks':len(set(masks))}))
if __name__=='__main__': main()