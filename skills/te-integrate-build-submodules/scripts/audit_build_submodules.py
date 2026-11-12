#!/usr/bin/env python3
import argparse,json,subprocess
from pathlib import Path
def run(repo,*args,check=True):
 p=subprocess.run(["git","-C",str(repo),*args],text=True,errors="replace",capture_output=True)
 if check and p.returncode: raise SystemExit(p.stderr.strip() or "git failed")
 return p.stdout
def link(repo,ref,path):
 s=run(repo,"ls-tree",ref,"--",path,check=False)
 return s.split()[2] if s.startswith("160000 commit ") else None
def main():
 ap=argparse.ArgumentParser()
 for n in ("base","fork","target"): ap.add_argument("--"+n,required=True)
 ap.add_argument("--repo",type=Path,required=True); ap.add_argument("--output",type=Path,required=True)
 a=ap.parse_args(); repo=a.repo.resolve(); a.output.mkdir(parents=True,exist_ok=True)
 refs={n:run(repo,"rev-parse","--verify",getattr(a,n)+"^{commit}").strip() for n in ("base","fork","target")}
 changed=set(run(repo,"diff","--name-only",refs["base"]+".."+refs["fork"]).splitlines())
 build=sorted(p for p in changed if p in {"setup.py","pyproject.toml","MANIFEST.in","CMakeLists.txt"} or p.startswith("build_tools/") or p.startswith("transformer_engine/pytorch/setup.py"))
 tree=set(run(repo,"ls-tree","-r","--name-only",refs["fork"]).splitlines())
 smpaths=sorted(p for p in set(tree)|changed if p==".gitmodules" or p.startswith("3rdparty/") if any(link(repo,r,p) for r in refs.values()) or p==".gitmodules")
 sm=[{"path":p,**{n:link(repo,r,p) for n,r in refs.items()},"status":"proposed"} for p in smpaths]
 plugins=[p for p in tree if p.startswith("transformer_engine/plugin/")]
 data={"resolved_refs":refs,"build_files":build,"build_file_count":len(build),"plugin_file_count":len(plugins),"submodules":sm}
 (a.output/"build-audit.json").write_text(json.dumps(data,indent=2)+chr(10))
 (a.output/"build-matrix.tsv").write_text("path\tplugin_target\tinclude_paths\tpackage_data\tversion\tstatus\n"+"".join(f"{p}\tproposed\tproposed\tproposed\tproposed\tproposed\n" for p in build))
 (a.output/"submodule-matrix.tsv").write_text("path\tbase\tfork\ttarget\tselection\treason\tstatus\n"+"".join(f"{x['path']}\t{x['base']}\t{x['fork']}\t{x['target']}\t\t\tproposed\n" for x in sm))
 (a.output/"build-audit.md").write_text("# Build and Submodule Audit"+chr(10)+chr(10)+f"- Build files: {len(build)}"+chr(10)+f"- Plugin files: {len(plugins)}"+chr(10)+f"- Submodule paths: {len(sm)}"+chr(10))
 print(json.dumps({"output":str(a.output),"build_files":len(build),"plugin_files":len(plugins),"submodules":len(sm)}))
if __name__=="__main__": main()
