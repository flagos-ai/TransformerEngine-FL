#!/usr/bin/env python3
import argparse, json, subprocess
from collections import Counter
from pathlib import Path

def git(repo,*args,check=True):
    p=subprocess.run(["git","-C",str(repo),*args],text=True,capture_output=True)
    if check and p.returncode: raise SystemExit(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout.rstrip("\n")

def changes(repo,a,b):
    rows=[]
    for line in git(repo,"diff","--name-status","--find-renames",f"{a}..{b}").splitlines():
        f=line.split("\t"); rows.append({"status":f[0],"path":f[-1]})
    return rows

def category(p):
    rules=[("plugin-backend",lambda: p.startswith("transformer_engine/plugin/core/backends/")),
      ("plugin-core",lambda: p.startswith("transformer_engine/plugin/")),
      ("submodule",lambda: p==".gitmodules" or p.startswith("3rdparty/")),
      ("cicd",lambda: p.startswith(".github/")),("qa",lambda: p.startswith("qa/")),
      ("tests",lambda: p.startswith("tests/")),
      ("build-packaging",lambda: p in {"setup.py","pyproject.toml","MANIFEST.in","CMakeLists.txt"} or p.startswith("build_tools/")),
      ("device-abstraction",lambda: p=="transformer_engine/__init__.py" or "patches.py" in p),
      ("invasive-runtime",lambda: p.startswith(("transformer_engine/pytorch/","transformer_engine/common/","transformer_engine/debug/"))),
      ("docs-examples-benchmarks",lambda: p.startswith(("docs/","examples/","benchmarks/"))),
      ("repository-metadata",lambda: p in {".gitignore",".pre-commit-config.yaml","README.rst","CONTRIBUTING.rst","CPPLINT.cfg"})]
    return next((n for n,test in rules if test()),"unclassified")

def priority(c,both):
    if both or c in {"plugin-core","plugin-backend","invasive-runtime","device-abstraction"}: return "P0"
    if c in {"build-packaging","submodule","tests"}: return "P1"
    return "P2" if c!="unclassified" else "P3"

def gitlink(repo,ref,path):
    s=git(repo,"ls-tree",ref,"--",path,check=False)
    return s.split()[2] if s.startswith("160000 commit ") else None

def main():
    ap=argparse.ArgumentParser()
    for x in ("base","fork","target"): ap.add_argument(f"--{x}",required=True)
    ap.add_argument("--repo",type=Path,required=True); ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--allow-divergent-upstream",action="store_true")
    a=ap.parse_args(); repo=a.repo.resolve()
    refs={x:git(repo,"rev-parse","--verify",f"{getattr(a,x)}^{{commit}}") for x in ("base","fork","target")}
    linear=subprocess.run(["git","-C",str(repo),"merge-base","--is-ancestor",refs["base"],refs["target"]]).returncode==0
    common=git(repo,"merge-base",refs["base"],refs["target"])
    if not linear and not a.allow_divergent_upstream:
        raise SystemExit(f"base is not an ancestor of target; merge-base={common}; rerun with --allow-divergent-upstream after review")
    rows=changes(repo,refs["base"],refs["fork"]); target={r["path"] for r in changes(repo,refs["base"],refs["target"])}
    for r in rows:
        r.update(category=category(r["path"]),both_changed=r["path"] in target)
        r["priority"]=priority(r["category"],r["both_changed"])
    tree=git(repo,"ls-tree","-r","--name-only",refs["fork"]).splitlines()
    pref="transformer_engine/plugin/core/backends/"
    def backend_id(path):
        parts=path[len(pref):].split("/")
        if not parts or parts[0] not in {"flagos","reference","vendor"}: return None
        return f"vendor/{parts[1]}" if parts[0]=="vendor" and len(parts)>2 and parts[1]!="__init__.py" else (parts[0] if parts[0]!="vendor" else None)
    backends=sorted({b for p in tree if p.startswith(pref) for b in [backend_id(p)] if b})
    target_rows=changes(repo,refs["base"],refs["target"])
    smpaths=sorted({r["path"] for r in rows+target_rows if r["path"].startswith("3rdparty/") and any(gitlink(repo,v,r["path"]) for v in refs.values())})
    surfaces={"plugin_backends":backends,
      "workflows":sorted(p for p in tree if p.startswith(".github/workflows/")),
      "ci_configs":sorted(p for p in tree if p.startswith(".github/configs/")),
      "qa":sorted(p for p in tree if p.startswith("qa/")),
      "tests":sorted(p for p in tree if p.startswith("tests/")),
      "build_packaging":sorted(p for p in tree if category(p)=="build-packaging"),
      "submodules":[{"path":p,**{k:gitlink(repo,v,p) for k,v in refs.items()}} for p in smpaths]}
    cats=Counter(r["category"] for r in rows); pris=Counter(r["priority"] for r in rows)
    a.output.mkdir(parents=True,exist_ok=True)
    data={"repo":str(repo),"resolved_refs":refs,"upstream_history":{"linear":linear,"merge_base":common,"divergence_explicitly_allowed":a.allow_divergent_upstream},"dirty_worktree":git(repo,"status","--short").splitlines(),
      "fork_change_count":len(rows),"both_changed_count":sum(r["both_changed"] for r in rows),
      "category_counts":dict(cats),"priority_counts":dict(pris),"fork_changes":rows,"surfaces":surfaces}
    (a.output/"inventory.json").write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
    head="status\tpriority\tcategory\tboth_changed\tpath\n"
    lines=[f"{r['status']}\t{r['priority']}\t{r['category']}\t{str(r['both_changed']).lower()}\t{r['path']}\n" for r in rows]
    (a.output/"fork-changes.tsv").write_text(head+"".join(lines))
    (a.output/"both-changed.tsv").write_text(head+"".join(x for x,r in zip(lines,rows) if r["both_changed"]))
    md=["# TransformerEngine Fork Delta Inventory","",*[f"- {k.title()}: `{v}`" for k,v in refs.items()],
      f"- Fork-owned paths: {len(rows)}",f"- Both-changed paths: {data['both_changed_count']}","",
      "## Categories","","| Category | Count |","|---|---:|",*[f"| {k} | {v} |" for k,v in sorted(cats.items())],
      "","## Acceptance blockers","",f"- Unclassified paths: {cats.get('unclassified',0)}"]
    (a.output/"inventory.md").write_text("\n".join(md)+"\n")
    print(json.dumps({"output":str(a.output),"fork_changes":len(rows),"both_changed":data["both_changed_count"],"unclassified":cats.get("unclassified",0)}))
    if not rows or cats.get("unclassified"): raise SystemExit(2)

if __name__=="__main__": main()
