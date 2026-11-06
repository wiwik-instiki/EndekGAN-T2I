"""Reanalyse the published aggregate evidence without original Endek images or GPU."""
from pathlib import Path
import argparse, csv, json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[1]
RES=REPO/"results"
MODELS=["Category GAN","Hash-Text","Transformer","Transformer (no class)",
        "Proposed Seam-Aware","Proposed (no class)","DF-GAN-MiniLM (adapted)"]

def compute(out):
    out.mkdir(parents=True,exist_ok=True)
    internal=pd.read_csv(RES/"internal_18_training_runs.csv")
    external=pd.read_csv(RES/"adapted_dfgan_3_training_runs.csv")
    seam=pd.read_csv(RES/"three_seed_boundary_diagnostics.csv")
    precomputed=pd.read_csv(RES/"three_seed_quantitative_summary.csv")
    assert len(internal)==18 and len(external)==3 and len(seam)==21
    assert sorted(internal.seed.unique())==[42,123,2025]
    assert sorted(external.seed.unique())==[42,123,2025]
    # Underlying configuration identifiers and values, not output-stage labels.
    def classify(r):
        if r.encoder=="cat":return "Category GAN"
        if r.encoder=="hash_text":return "Hash-Text"
        se=float(r.seam_lambda)>.1
        no=str(r.disable_class).lower()=="true"
        return {(False,False):"Transformer",(False,True):"Transformer (no class)",
                (True,False):"Proposed Seam-Aware",(True,True):"Proposed (no class)"}[(se,no)]
    internal["model"]=internal.apply(classify,axis=1)
    external["model"]="DF-GAN-MiniLM (adapted)"
    a=internal.rename(columns={"macro_FID_exploratory":"FID","macro_KID_exploratory":"KID"})
    b=external.rename(columns={"macro_FID_exploratory":"FID","macro_KID_exploratory":"KID"})
    runs=pd.concat([a[["model","seed","FID","KID"]],b[["model","seed","FID","KID"]]],ignore_index=True)
    seam["model"]=seam.model.str.replace("DF-GAN–MiniLM","DF-GAN-MiniLM (adapted)",regex=False)
    merged=runs.merge(seam,on=["model","seed"],validate="one_to_one")
    assert len(merged)==21
    means=merged.groupby("model").agg(
        FID=("FID","mean"),FID_SD=("FID","std"),
        KID=("KID","mean"),KID_SD=("KID","std"),
        seam=("seam_gap","mean"),seam_SD=("seam_gap","std"),
        detail=("detail_energy","mean")).reindex(MODELS)
    assert means.notna().all().all()
    ref=precomputed.set_index("model")
    for col,original in [("FID","FID_mean"),("KID","KID_mean"),("seam","seam_mean")]:
        assert np.allclose(means[col].values,ref.reindex(MODELS)[original].values,atol=1e-8)
    for m1,m2 in [("Transformer","Proposed Seam-Aware"),
                  ("Transformer (no class)","Proposed (no class)")]:
        assert means.loc[m2,"seam"]<means.loc[m1,"seam"]
        assert means.loc[m2,"FID"]>means.loc[m1,"FID"]
    with_class=100*(means.loc["Transformer","seam"]-means.loc["Proposed Seam-Aware","seam"])/means.loc["Transformer","seam"]
    no_class=100*(means.loc["Transformer (no class)","seam"]-means.loc["Proposed (no class)","seam"])/means.loc["Transformer (no class)","seam"]
    assert abs(with_class-39.75213)<.005 and abs(no_class-45.57179)<.005
    means.to_csv(out/"VERIFIED_SEVEN_MODEL_SUMMARY.csv",index_label="model")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9})
    for metric,save_name in [("FID","FIG_RECALCULATED_FID.png"),("KID","FIG_RECALCULATED_KID.png"),("seam","FIG_RECALCULATED_SEAM_GAP.png")]:
        plot=means.sort_values(metric)
        fig,ax=plt.subplots(figsize=(9,4.5))
        bars=ax.barh(range(len(plot)),plot[metric],xerr=plot[metric+"_SD"],
            facecolor="#e8e8e8",edgecolor="black",capsize=3)
        for bar,label in zip(bars,plot.index):
            if "Proposed" in label:bar.set_hatch("////")
        ax.set_yticks(range(len(plot)),plot.index)
        ax.invert_yaxis()
        ax.set_xlabel(metric+" (lower is better; three-seed mean ± SD)")
        if metric in ("FID","KID"):ax.set_title("Exploratory due to 37 real test images")
        else:ax.set_title("Outermost-pixel seam gap; not motif quality")
        fig.tight_layout();fig.savefig(out/save_name,dpi=160,bbox_inches="tight");plt.close(fig)
    report=[
    "# EndekGAN-T2I: Recomputed Evidence (No Retraining)",
    "",
    "This file is generated from 21 archived run-level records (18 internal + 3 adapted DF-GAN), "
    "and cross-checked against the independently archived published aggregate table.",
    "",
    "| Configuration | Exploratory FID (mean) | Exploratory KID (mean) | Seam gap (mean) |",
    "|---|---:|---:|---:|",
    ]
    for name,row in means.iterrows():
        report.append(f"| {name} | {row.FID:.2f} | {row.KID:.5f} | {row.seam:.5f} |")
    report += [
    "",
    "## Main observation",
    f"With class embedding, the border discontinuity decreases {with_class:.1f}%; without class embedding, {no_class:.1f}%.",
    "In both matched pairs, FID worsens. The class-conditioned proposed setting does not outperform the adapted external baseline on FID/KID.",
    "",
    "## Research question mapping",
    "- RQ1: Hash-Text has lower FID/KID than category and class-conditioned Transformer on this test.",
    "- RQ2: Seam-aware penalty improves the edge-gap diagnostic but introduces fidelity/diversity costs.",
    "- RQ3: Adapted DF-GAN–MiniLM gives better exploratory fidelity than the class-conditioned proposed setting, whereas the proposed setting has lower edge gap.",
    "",
    "## Explicit limitations",
    "The held-out test set contains only 37 real images: FID/KID are exploratory.",
    "A smaller seam gap measures edge agreement, not culturally recognized Endek ornamentation or weaving feasibility.",
    "LPIPS was not estimated for the external baseline under the same experiment, and multilingual CLIP diagnostic scores exist only for seed 42.",
    "No human expert ratings or trained StyleGAN2-ADA score are claimed.",
    ]
    (out/"RQ1_RQ2_RQ3_EVIDENCE_REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    status={"runs":len(merged),"seeds":[42,123,2025],"models":len(means),
         "seam_reduction_pct_with_class":with_class,"seam_reduction_pct_without_class":no_class,
         "trained_now":False,"uses_private_images":False,"all_crosschecks_passed":True}
    (out/"REANALYSIS_VALIDATION.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    print("PASS",json.dumps(status,indent=2))
    return status

if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("--output",type=Path,default=REPO/"local_runs"/"reanalysis")
    args=arg.parse_args()
    compute(args.output.resolve())
