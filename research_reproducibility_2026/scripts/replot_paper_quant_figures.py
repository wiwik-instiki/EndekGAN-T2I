"""Recreate the paper's quantitative Figures 4–7 from public, archival run summaries.

Figure 1–3 are manuscript design/data illustrations and are distributed as exact
manuscript images. Figure 8–10 are outputs from archived image-level experiments
and can be regenerated through evaluate_existing_outputs.py if authorized images
are available. This script only regenerates numerical graphics, never trains.
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
REPO=HERE.parent
ROOT=REPO.parent if REPO.name=="research_reproducibility_2026" else REPO
DATA=REPO/"results"

def build(out):
 out.mkdir(parents=True,exist_ok=True)
 d=pd.read_csv(DATA/"three_seed_quantitative_summary.csv")
 assert len(d)==7 and set(["model","FID_mean","FID_sd","KID_mean","KID_sd","seam_mean","seam_sd"]).issubset(d.columns)
 lp=pd.read_csv(DATA/"internal_lpips_variability.csv")
 assert len(lp)==6
 plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"savefig.facecolor":"white"})
 figlist=[]
 for metric,n in (("FID",4),("KID",5)):
  ds=d.sort_values(f"{metric}_mean")
  fig,ax=plt.subplots(figsize=(9.4,4.8))
  ax.barh(np.arange(len(ds)),ds[f"{metric}_mean"],xerr=ds[f"{metric}_sd"],
          color="#e4e4e4",edgecolor="#111111",capsize=3)
  ax.set_yticks(np.arange(len(ds)),ds["model"])
  ax.invert_yaxis()
  ax.set_title(f"Figure {n}. Exploratory class-macro {metric} across three seeds (mean ± SD)")
  ax.set_xlabel(f"{metric}: lower is better (n=3 seeds)")
  ax.grid(axis="x",alpha=.16)
  fig.tight_layout()
  dst=out/f"figure_{n:02d}_{metric.lower()}_regenerated.png"
  fig.savefig(dst,dpi=180,bbox_inches="tight");plt.close(fig);figlist.append(dst)
 fig,ax=plt.subplots(figsize=(8.2,5.2))
 for _,r in d.iterrows():
  x,y=r["seam_mean"],r["FID_mean"]
  ax.scatter(x,y,s=85,facecolor="white",edgecolor="#111111",
             marker="D" if "Proposed" in r["model"] else "o")
  ax.annotate(r["model"],(x,y),xytext=(4,7),textcoords="offset points",fontsize=7.7)
 ax.set_xlabel("Mean outermost seam gap ↓ (RGB normalized [0,1])")
 ax.set_ylabel("Exploratory class-macro FID ↓")
 ax.set_title("Figure 6. Boundary continuity vs distributional fidelity")
 ax.grid(alpha=.16)
 fig.tight_layout()
 dst=out/"figure_06_tradeoff_regenerated.png";fig.savefig(dst,dpi=180,bbox_inches="tight");plt.close(fig);figlist.append(dst)
 assert {"model","LPIPS_mean","LPIPS_sd"}.issubset(lp.columns),list(lp.columns)
 fig,ax=plt.subplots(figsize=(9.0,4.5))
 ax.bar(np.arange(len(lp)),lp["LPIPS_mean"],yerr=lp["LPIPS_sd"],color="#dddddd",edgecolor="#111111",capsize=3)
 ax.set_xticks(np.arange(len(lp)),lp["model"],rotation=26,ha="right")
 ax.set_ylabel("Within-model LPIPS variation (higher = greater variation)")
 ax.set_title("Figure 7. Perceptual variation across six internal configurations")
 ax.grid(axis="y",alpha=.16)
 fig.tight_layout()
 dst=out/"figure_07_lpips_regenerated.png";fig.savefig(dst,dpi=180,bbox_inches="tight");plt.close(fig);figlist.append(dst)
 for p in figlist:
  assert p.is_file() and p.stat().st_size>20_000,p
 print("REGENERATED",len(figlist),"QUANTITATIVE_FIGURES",out)
 return figlist
if __name__=="__main__":
 pa=argparse.ArgumentParser()
 pa.add_argument("--output",type=Path,default=REPO/"local_runs"/"paper_quant_figures")
 args=pa.parse_args()
 build(args.output.resolve())
