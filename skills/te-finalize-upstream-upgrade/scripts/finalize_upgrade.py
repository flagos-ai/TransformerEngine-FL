#!/usr/bin/env python3
import argparse,json,subprocess
from pathlib import Path
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--repo',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--artifacts',type=Path,required=True)
 a=ap.parse_args(); repo=a.repo.resolve(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
 def run(*x): return subprocess.run(['git','-C',str(repo),*x],text=True,errors='replace',capture_output=True)
 expected=['inventory.json','conflict-inventory.json','api-inventory.json','runtime-patch-ledger.json','build-audit.json','cicd-audit.json','test-matrix.json']
 found={n:(a.artifacts/n).exists() for n in expected}
 status=run('status','--short').stdout.splitlines(); branch=run('branch','--show-current').stdout.strip()
 data={'repo':str(repo),'branch':branch,'dirty_worktree':status,'artifacts':found,'all_artifacts_present':all(found.values()),'push_performed':False,'pr_created':False}
 (out/'evidence-index.json').write_text(json.dumps(data,indent=2)+chr(10))
 (out/'blockers.tsv').write_text('item\tstatus\towner\tevidence\n'+('missing phase artifacts\tblocked\tuser\t'+str(a.artifacts)+'\n' if not all(found.values()) else ''))
 (out/'rollback-plan.md').write_text('# Rollback Plan\n\nRecord the final merge commit before applying any revert. Use git revert -m 1 <merge-commit> after review. Do not reset or delete branches.\n')
 (out/'final-report.md').write_text('# Upgrade Handoff\n\n'+'- Branch: '+branch+'\n- Dirty worktree entries: '+str(len(status))+'\n- All phase artifacts present: '+str(all(found.values()))+'\n- Push/PR performed: false\n')
 print(json.dumps({'output':str(out),'all_artifacts_present':all(found.values()),'dirty_entries':len(status)}))
if __name__=='__main__': main()