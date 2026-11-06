"""Stage dispatcher for the EndekGAN-T2I master notebook.

Default path recomputes paper evidence only (no datasets, checkpoints, or CUDA).
Full training is opt-in, requires an isolated output root and confirmation token.
"""
from pathlib import Path
import json, os, subprocess, sys, shutil, time, hashlib
from typing import Optional

REPO=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get("ENDEKGAN_PROJECT_ROOT",str(REPO.parent))).resolve()
DATASET=Path(os.environ.get("ENDEKGAN_DATASET_ROOT",str(SOURCE.parent/"EndekGAN_Dataset"))).resolve()
WORK=Path(os.environ.get("ENDEKGAN_WORK_ROOT",str(REPO/"local_runs"))).resolve()
WORK.mkdir(parents=True,exist_ok=True)
STAGE_MAP={
 "dataset":REPO/"notebooks"/"01_dataset_and_prompt_audit.ipynb",
 "internal":REPO/"notebooks"/"02_internal_conditioning_seam_ablation.ipynb",
 "external":REPO/"notebooks"/"03_external_dfgan_minilm.ipynb",
 "recap":REPO/"notebooks"/"05_master_recap_original.ipynb",
}
def summarize_paths():
    status={"repository":str(REPO),"source_archive_exists":SOURCE.is_dir(),
            "original_dataset_exists":DATASET.is_dir(),"isolated_work_root":str(WORK),
            "source_is_same_as_work":SOURCE==WORK,
            "archived_internal_results":(SOURCE/"EndekGAN_IEEE_V5_OUTPUT"/"E1_E7_NEW_RESULTS_BY_RUN.csv").is_file(),
            "archived_external_results":(SOURCE/"EndekGAN_V6_EXTERNAL_BASELINES"/"E8_EXTERNAL_MACRO_BY_RUN.csv").is_file(),
            "archived_visual_diagnostics":(SOURCE/"EVALUATION_EXTENSION_20261105"/"RESULTS"/"PER_SEED_SUMMARY.csv").is_file(),
            "public_summary":(REPO/"results"/"three_seed_quantitative_summary.csv").is_file()}
    return status

def execute_python(script,args=(),env=None,label="stage"):
    logs=WORK/"logs";logs.mkdir(parents=True,exist_ok=True)
    log=logs/(label+".log")
    command=[sys.executable,str(script),*list(args)]
    print("LAUNCH",label,"log:",log,flush=True)
    with log.open("w",encoding="utf-8") as fp:
        cp=subprocess.run(command,cwd=str(REPO),env=env or os.environ.copy(),
                          stdout=fp,stderr=subprocess.STDOUT,check=False)
    text=log.read_text(encoding="utf-8",errors="replace")
    print("RETURN",label,cp.returncode,"TAIL",text[-1100:],flush=True)
    if cp.returncode!=0:
        raise RuntimeError("Stage failed: "+label+"; inspect "+str(log))
    return {"stage":label,"status":"completed","log":str(log)}

def execute_ipynb(path,label,env):
    # Intentional notebook-to-code execution: preserves cell order and IPython
    # semantics, but does not claim to produce a saved executed .ipynb.
    assert path.is_file(),path
    return execute_python(REPO/"scripts"/"execute_ipynb.py",
                          ["--notebook",str(path),"--log-prefix",label],env,label)

def guard_training():
    if os.environ.get("ENDEKGAN_ALLOW_TRAINING")!="YES_RETRAIN_EXPLICITLY":
        raise PermissionError("Set ENDEKGAN_ALLOW_TRAINING=YES_RETRAIN_EXPLICITLY to enable costly training.")
    if SOURCE==WORK:raise RuntimeError("Training WORK root must differ from original source archive.")
    if not DATASET.is_dir():raise FileNotFoundError(DATASET)
    print("Training explicitly authorized, source archive will remain separate.")

def stage(name, enabled=False):
    if not enabled:
        print("SKIP",name,"(disabled by master notebook)")
        return {"stage":name,"status":"skipped"}
    common=os.environ.copy()
    common["ENDEKGAN_DATASET_ROOT"]=str(DATASET)
    common["ENDEKGAN_WORK_ROOT"]=str(WORK)
    if name=="published_summary":
        return execute_python(REPO/"scripts"/"reanalyze_evidence.py",
            ["--output",str(WORK/"reanalysis")],common,"published_summary")
    if name=="dataset":
        if not DATASET.is_dir():raise FileNotFoundError("Authorized dataset directory is missing: "+str(DATASET))
        common["ENDEKGAN_AUDIT_OUTPUT"]=str(WORK/"dataset_audit")
        return execute_ipynb(STAGE_MAP[name],"dataset",common)
    if name=="internal":
        guard_training()
        common["ENDEKGAN_V5_OUTPUT"]=str(WORK/"EndekGAN_IEEE_V5_OUTPUT")
        common["ENDEKGAN_PROJECT_ROOT"]=str(WORK)
        return execute_ipynb(STAGE_MAP[name],"internal_18_training_runs",common)
    if name=="external":
        guard_training()
        if not (WORK/"EndekGAN_IEEE_V5_OUTPUT"/"EndekGAN_IEEE_V5_OUTPUT.zip").is_file():
            raise FileNotFoundError("The isolated work root requires the privately authorized fixed-split bundle before DF-GAN training.")
        common["ENDEKGAN_PROJECT_ROOT"]=str(WORK)
        common["ENDEKGAN_ACTIVATE_EXTERNAL_TRAINING"]="YES"
        return execute_ipynb(STAGE_MAP[name],"external_3_training_runs",common)
    if name=="image_evaluation":
        if not (SOURCE/"EndekGAN_IEEE_V5_OUTPUT").is_dir() or not (SOURCE/"EndekGAN_V6_EXTERNAL_BASELINES").is_dir():
            raise FileNotFoundError("Existing generated image archive absent. Required for local-only image diagnostics.")
        common["ENDEKGAN_PROJECT_ROOT"]=str(SOURCE)
        common["ENDEKGAN_EVAL_OUTPUT"]=str(WORK/"image_evaluation")
        return execute_python(REPO/"scripts"/"evaluate_existing_outputs.py",
            ["--seeds","42","123","2025","--metric-per-class","128"],common,"image_evaluation")
    if name=="recap":
        if not (SOURCE/"EndekGAN_IEEE_V5_OUTPUT").is_dir():raise FileNotFoundError("Archived runs required for recap")
        common["ENDEKGAN_PROJECT_ROOT"]=str(SOURCE)
        common["ENDEKGAN_RECAP_OUTPUT"]=str(WORK/"master_recap")
        return execute_ipynb(STAGE_MAP[name],"master_recap",common)
    raise ValueError("Unknown stage: "+str(name))
