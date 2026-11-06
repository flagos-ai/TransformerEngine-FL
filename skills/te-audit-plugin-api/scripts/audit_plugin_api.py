#!/usr/bin/env python3
import argparse,json,re,subprocess
from pathlib import Path
def git(repo,*args):
 p=subprocess.run(["git","-C",str(repo),*args],text=True,errors="replace",capture_output=True)
 if p.returncode: raise SystemExit(p.stderr.strip() or "git failed")
 return p.stdout
def names(text,pattern):
 return sorted(set(re.findall(pattern,text,re.M)))
def main():
 ap=argparse.ArgumentParser()
 for n in ("base","fork","target"): ap.add_argument("--"+n,required=True)
 ap.add_argument("--repo",type=Path,required=True); ap.add_argument("--output",type=Path,required=True)
 a=ap.parse_args(); repo=a.repo.resolve(); a.output.mkdir(parents=True,exist_ok=True)
 refs={n:git(repo,"rev-parse","--verify",getattr(a,n)+"^{commit}").strip() for n in ("base","fork","target")}
 def tree(ref,glob):
  return git(repo,"grep","-h","-E",glob,ref,"--","transformer_engine/pytorch/csrc",check=False) if False else ""
 # Use git grep per ref and retain output even if no match.
 def grep(ref,pat,paths):
  p=subprocess.run(["git","-C",str(repo),"grep","-h","-E",pat,ref,"--",*paths],text=True,errors="replace",capture_output=True)
  return p.stdout
 bind=names(grep(refs["target"], r"\.def\(\s*\"[A-Za-z0-9_]+", ["transformer_engine/pytorch/csrc/"]), r"\.def\(\s*\"([A-Za-z0-9_]+)")
 pyops=names(grep(refs["fork"],r"^\s*def\s+\w+",["transformer_engine/plugin/core/ops.py"]),r"^\s*def\s+(\w+)")
 reg=names(grep(refs["fork"],r'op_name[[:space:]]*=[[:space:]]*"[^"]+',["transformer_engine/plugin/core/backends"]),r'op_name\s*=\s*"([^"]+)"')
 files=git(repo,"ls-tree","-r","--name-only",refs["fork"]).splitlines()
 pref="transformer_engine/plugin/core/backends/"
 backend=sorted({("vendor/"+q[1]) if q[0]=="vendor" and len(q)>1 else q[0] for p in files if p.startswith(pref) for q in [p[len(pref):].split("/")] if q[0] in {"vendor","flagos","reference"} and (q[0]!="vendor" or (len(q)>1 and q[1]!="__init__.py"))})
 rows=[]
 for n in sorted(set(bind)|set(pyops)|set(reg)):
  rows.append({"symbol":n,"binding":n in bind,"plugin_base":n in pyops,"registered":n in reg,"backends":backend,"disposition":"proposed"})
 data={"resolved_refs":refs,"counts":{"bindings":len(bind),"plugin_base":len(pyops),"registered":len(reg),"matrix":len(rows)},"backends":backend,"symbols":rows}
 (a.output/"api-inventory.json").write_text(json.dumps(data,indent=2)+chr(10))
 (a.output/"api-matrix.tsv").write_text("symbol"+chr(9)+"binding"+chr(9)+"plugin_base"+chr(9)+"registered"+chr(9)+"disposition"+chr(9)+"owner"+chr(9)+"test"+chr(10)+"".join(f"{r['symbol']}"+chr(9)+str(r['binding']).lower()+chr(9)+str(r['plugin_base']).lower()+chr(9)+str(r['registered']).lower()+chr(9)+"proposed"+chr(9)+chr(9)+chr(10) for r in rows))
 for fn,items in [("upstream-bindings.txt",bind),("plugin-base-methods.txt",pyops),("plugin-registered-ops.txt",reg)]: (a.output/fn).write_text(chr(10).join(items)+chr(10))
 (a.output/"decisions.tsv").write_text("symbol"+chr(9)+"disposition"+chr(9)+"reason"+chr(9)+"owner"+chr(9)+"test"+chr(9)+"status"+chr(9)+"evidence"+chr(10))
 (a.output/"api-audit.md").write_text("# Plugin API Audit"+chr(10)+chr(10)+f"- Matrix symbols: {len(rows)}"+chr(10)+f"- Backends: {', '.join(backend)}"+chr(10)+f"- Decisions required: {len(rows)}"+chr(10))
 print(json.dumps({"output":str(a.output),"matrix":len(rows),"bindings":len(bind),"backends":backend}))
if __name__=="__main__": main()
