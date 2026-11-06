"""Run an archived .ipynb in original cell order through IPython.

The orchestrator only invokes heavy notebooks when the author explicitly opts in.
This runner writes to the external WORK directory, not to the notebook source.
"""
import argparse, json, os, sys, time, traceback
from pathlib import Path
from IPython.core.interactiveshell import InteractiveShell

def run(path):
    if not path.is_file():raise FileNotFoundError(path)
    private_guard=["02_internal_conditioning_seam_ablation","03_external_dfgan_minilm"]
    if any(n in path.name for n in private_guard):
        if os.environ.get("ENDEKGAN_ALLOW_TRAINING")!="YES_RETRAIN_EXPLICITLY":
            raise PermissionError("Refusing expensive notebook execution without explicit env opt-in")
    nb=json.loads(path.read_text(encoding="utf-8"))
    shell=InteractiveShell.instance()
    cells=[c for c in nb["cells"] if c.get("cell_type")=="code"]
    patch_counts={"RUN_TRAINING":0,"RUN_DFGAN_ADAPTED":0,"RUN_METRICS":0}
    for idx,c in enumerate(cells,1):
        source="".join(c.get("source",[]))
        if os.environ.get("ENDEKGAN_ACTIVATE_EXTERNAL_TRAINING")=="YES":
            if path.name!="03_external_dfgan_minilm.ipynb":
                raise ValueError("External training activation may only apply to the adapted baseline source")
            for key in patch_counts:
                src=key+" = False"
                dst=key+" = True"
                if src in source:
                    source=source.replace(src,dst,1)
                    patch_counts[key]+=1
            if "RUN_STYLEGAN2_ADA = True" in source:
                raise RuntimeError("Unexpected StyleGAN2-ADA training activation: not supported evidence.")
        if not source.strip():continue
        print(f"[CELL {idx}/{len(cells)}] "+source.strip().splitlines()[0][:110],flush=True)
        result=shell.run_cell(source,store_history=False,silent=False)
        if result.error_before_exec or result.error_in_exec:
            err=result.error_before_exec or result.error_in_exec
            raise RuntimeError("Notebook "+path.name+" failed in code cell "+str(idx)+": "+str(err)) from err
    if os.environ.get("ENDEKGAN_ACTIVATE_EXTERNAL_TRAINING")=="YES":
        assert all(v>=1 for v in patch_counts.values()),patch_counts
    print("ALL_NOTEBOOK_CELLS_EXECUTED",path.name,len(cells),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--notebook",type=Path,required=True)
    parser.add_argument("--log-prefix",default="run")
    args=parser.parse_args()
    run(args.notebook.resolve())
