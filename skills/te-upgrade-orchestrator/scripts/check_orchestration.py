#!/usr/bin/env python3
import argparse,json
from pathlib import Path
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--evidence",type=Path,required=True); ap.add_argument("--output",type=Path,required=True)
 a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
 names=["inventory.json","conflict-inventory.json","api-inventory.json","runtime-patch-ledger.json","build-audit.json","cicd-audit.json","test-matrix.json","evidence-index.json"]
 present={n:(a.evidence/n).exists() for n in names}
 data={"phases":present,"complete":all(present.values()),"external_actions_authorized":False}
 (a.output/"orchestration-status.json").write_text(json.dumps(data,indent=2)+chr(10))
 (a.output/"orchestration-status.md").write_text("# Orchestration Status"+chr(10)+chr(10)+"- Complete: "+str(data["complete"])+chr(10)+"- External actions authorized: false"+chr(10))
 print(json.dumps({"output":str(a.output),"complete":data["complete"],"missing":[n for n,v in present.items() if not v]}))
if __name__=="__main__": main()
