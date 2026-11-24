#!/usr/bin/env python3
import argparse,json,subprocess
from pathlib import Path
def run(repo,*args,check=True):
 p=subprocess.run(["git","-C",str(repo),*args],text=True,errors="replace",capture_output=True)
 if check and p.returncode: raise SystemExit(p.stderr.strip() or "git failed")
 return p.stdout
def main():
 ap=argparse.ArgumentParser()
 for n in ("base","fork","target"): ap.add_argument("--"+n,required=True)
 ap.add_argument("--repo",type=Path,required=True); ap.add_argument("--output",type=Path,required=True)
 a=ap.parse_args(); repo=a.repo.resolve(); a.output.mkdir(parents=True,exist_ok=True)
 refs={n:run(repo,"rev-parse","--verify",getattr(a,n)+"^{commit}").strip() for n in ("base","fork","target")}
 changed=run(repo,"diff","--name-only",refs["base"]+".."+refs["fork"]).splitlines()
 runtime=[p for p in changed if p=="transformer_engine/__init__.py" or p.startswith(("transformer_engine/pytorch/","transformer_engine/common/","transformer_engine/debug/"))]
 targetfiles=set(run(repo,"diff","--name-only",refs["base"]+".."+refs["target"]).splitlines())
 both=[p for p in runtime if p in targetfiles]
 rows=[]
 for p in sorted(runtime):
  text=run(repo,"show",refs["fork"]+":"+p,check=False)
  markers=[m for m in ("TE_DEVICE_TYPE","transformer_engine.plugin","plugin.ops","cuda","device_type","register_ops") if m in text]
  rows.append({"path":p,"both_changed":p in both,"markers":markers,"status":"proposed","invariant":"","preservation":"","test":""})
 data={"resolved_refs":refs,"runtime_file_count":len(rows),"both_changed_count":len(both),"paths":rows}
 (a.output/"runtime-patch-ledger.json").write_text(json.dumps(data,indent=2)+chr(10))
 (a.output/"runtime-patch-ledger.tsv").write_text("path"+chr(9)+"both_changed"+chr(9)+"markers"+chr(9)+"invariant"+chr(9)+"preservation"+chr(9)+"test"+chr(9)+"status"+chr(10)+"".join(f"{r['path']}"+chr(9)+str(r['both_changed']).lower()+chr(9)+",".join(r['markers'])+chr(9)+chr(9)+chr(9)+chr(9)+"proposed"+chr(10) for r in rows))
 (a.output/"runtime-patch-audit.md").write_text("# Runtime Patch Audit"+chr(10)+chr(10)+f"- Runtime files: {len(rows)}"+chr(10)+f"- Both changed: {len(both)}"+chr(10)+f"- Decisions required: {len(rows)}"+chr(10))
 (a.output/"raw-runtime-files.txt").write_text(chr(10).join(sorted(runtime))+chr(10))
 print(json.dumps({"output":str(a.output),"runtime_files":len(rows),"both_changed":len(both)}))
if __name__=="__main__": main()
