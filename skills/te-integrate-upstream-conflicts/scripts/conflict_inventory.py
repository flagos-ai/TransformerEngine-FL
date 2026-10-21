#!/usr/bin/env python3
import argparse,json,subprocess
from pathlib import Path
def run(repo,*args):
 p=subprocess.run(["git","-C",str(repo),*args],text=True,errors="replace",capture_output=True)
 if p.returncode: raise SystemExit(p.stderr.strip() or "git command failed")
 return p.stdout
def main():
 ap=argparse.ArgumentParser()
 for n in ("base","fork","target"): ap.add_argument("--"+n,required=True)
 ap.add_argument("--repo",type=Path,required=True); ap.add_argument("--output",type=Path,required=True)
 a=ap.parse_args(); repo=a.repo.resolve(); a.output.mkdir(parents=True,exist_ok=True)
 refs={n:run(repo,"rev-parse","--verify",getattr(a,n)+"^{commit}").strip() for n in ("base","fork","target")}
 def paths(right): return {x for x in run(repo,"diff","--name-only",refs["base"]+".."+right).splitlines() if x}
 both=sorted(paths(refs["fork"]) & paths(refs["target"]))
 merge=run(repo,"merge-tree",refs["base"],refs["fork"],refs["target"])
 (a.output/"merge-tree.txt").write_text(merge)
 conflict_paths={line.rsplit(" ",1)[-1] for line in merge.splitlines() if "CONFLICT" in line and " " in line}
 def priority(p):
  if p.startswith(("transformer_engine/plugin/","transformer_engine/pytorch/","transformer_engine/common/","transformer_engine/debug/")) or p=="transformer_engine/__init__.py": return "P0"
  if p.startswith(("setup.py","build_tools/","3rdparty/","tests/","qa/")): return "P1"
  return "P2"
 rows=[{"path":p,"priority":priority(p),"textual_conflict":p in conflict_paths,"semantic_risk":True,"status":"proposed"} for p in both]
 data={"resolved_refs":refs,"both_changed_count":len(rows),"textual_conflict_count":len(conflict_paths),"paths":rows}
 (a.output/"conflict-inventory.json").write_text(json.dumps(data,indent=2)+"\n")
 (a.output/"conflict-inventory.md").write_text("# Conflict Inventory\n\n"+f"- Both-changed paths: {len(rows)}\n\n"+"\n".join(f"- [{r['priority']}] {r['path']} — proposed" for r in rows)+"\n")
 (a.output/"decisions.tsv").write_text("path\tpriority\tfork_invariant\tupstream_change\tstrategy\towner\tacceptance\tstatus\tevidence\n")
 print(json.dumps({"output":str(a.output),"both_changed":len(rows)}))
if __name__=="__main__": main()
