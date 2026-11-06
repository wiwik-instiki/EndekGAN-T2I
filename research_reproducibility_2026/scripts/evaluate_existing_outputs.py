# EndekGAN-T2I: evidence-only qualitative and structural evaluation
# Runs on existing exported images. No model weights are changed or trained.
# Python >=3.9; packages: numpy pandas Pillow matplotlib
import argparse, json, random, hashlib, warnings, csv, os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("ENDEKGAN_PROJECT_ROOT","../EndekGAN-T2I")).resolve()
RUNS = ROOT / "EndekGAN_IEEE_V5_OUTPUT" / "FULL_runs_CPU_bs4"
SPLIT = ROOT / "EndekGAN_IEEE_V5_OUTPUT" / "fixed_split" / "test_split.csv"
GPU = ROOT / "EndekGAN_V6_EXTERNAL_BASELINES"
OUT = Path(os.environ.get("ENDEKGAN_EVAL_OUTPUT", str(ROOT / "EVALUATION_EXTENSION_20261105" / "RESULTS"))).resolve()
CLASSES = ["flora", "fauna", "dekoratif", "geometris"]
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}
METHODS = {
    "Category GAN": "cat",
    "Hash-Text": "hash_text",
    "Transformer": "transformer_text",
    "Transformer (no class)": "transformer_text_noclass",
    "Proposed Seam-Aware": "transformer_text_seam",
    "Proposed (no class)": "transformer_text_seam_noclass",
    "DF-GAN–MiniLM": "DFGAN_MiniLM",
}
COLORS = {"Proposed Seam-Aware": "#206e8a", "Transformer": "#444444"}

def imread(p):
    with Image.open(p) as src:
        return np.asarray(src.convert("RGB").resize((256, 256)), dtype=np.uint8)

def edge_gap(x):
    f = x.astype(np.float32) / 255.
    return float((np.abs(f[:, 0] - f[:, -1]).mean() + np.abs(f[0] - f[-1]).mean()) / 2)

def detail_energy(x):
    f = x.astype(np.float32).mean(axis=2) / 255.
    return float((np.abs(np.diff(f, axis=0)).mean() + np.abs(np.diff(f, axis=1)).mean()) / 2)

def periodicity_proxy(x):
    # Mean-shifted luminance autocorrelation at predefined offsets.
    # High values can be obtained by blur/flat regions. This is NOT motif regularity.
    g = x.astype(np.float32).mean(axis=2) / 255.
    if g.std() < .005:
        return np.nan
    vals = []
    for offset in (16, 32, 64):
        for a, b in ((g[:, :-offset], g[:, offset:]), (g[:-offset], g[offset:])):
            a, b = (a - a.mean()).ravel(), (b - b.mean()).ravel()
            den = float(np.linalg.norm(a) * np.linalg.norm(b))
            if den > 1e-9:
                vals.append(float(np.dot(a, b) / den))
    return float(np.max(vals)) if vals else np.nan

def paths_for_internal(method, seed):
    return RUNS / f"{method}_{seed}" / "generated" / "generated_eval_index.csv"

def build_indices(seed, test):
    """
    DF-GAN image j in class k uses test embedding test[k][j % n_k].
    Internal images use their own generated_eval_index.csv. Find the same
    source_filename AND exact prompt, not the same file number.
    """
    mapping = {}
    for nice, tag in METHODS.items():
        if tag == "DFGAN_MiniLM":
            continue
        p = paths_for_internal(tag, seed)
        if not p.is_file():
            raise FileNotFoundError(p)
        tab = pd.read_csv(p).copy()
        need = {"generated_path", "label", "source_filename", "prompt", "generated_index"}
        if not need.issubset(tab.columns):
            raise ValueError(f"Index {p} missing {need-set(tab.columns)}")
        mapping[nice] = tab
    dfroot = GPU / "external_generated_test" / f"DFGAN_MiniLM_seed_{seed}"
    if not dfroot.is_dir():
        raise FileNotFoundError(dfroot)
    return mapping, dfroot

def selected_rows(seed, test, indices, dfroot, per_class=1):
    rows = []
    for label in CLASSES:
        part = test.loc[test.label == label].reset_index(drop=True)
        if part.empty:
            raise ValueError(f"Missing test category {label}")
        for j in range(per_class):
            testrow = part.iloc[j % len(part)]
            filename = str(testrow.filename)
            prompt = str(testrow.prompt)
            source = Path(str(testrow.image_path))
            if not source.is_file():
                raise FileNotFoundError(source)
            item = {"class": label, "test_source": filename, "prompt": prompt,
                    "seed": seed, "prompt_id": j, "Real test image": source}
            for nice, idx in indices.items():
                matches = idx.loc[(idx.label == label) & (idx.source_filename == filename)
                                  & (idx.prompt == prompt)]
                if matches.empty:
                    raise ValueError(f"Unmatched internal prompt: {nice}/{label}/{filename}")
                # First stable observation with this text/source, not visually selected.
                rr = matches.sort_values("generated_index").iloc[0]
                im = Path(str(rr.generated_path))
                if not im.is_file():
                    raise FileNotFoundError(im)
                item[nice] = im
                item[nice + "_internal_idx"] = int(rr.generated_index)
            external = dfroot / str(CLASS_ID[label]) / f"{j:04d}.png"
            if not external.is_file():
                raise FileNotFoundError(external)
            item["DF-GAN–MiniLM"] = external
            rows.append(item)
    return rows

def grid_figure(rows, seed, output):
    names = ["Real test image"] + list(METHODS)
    fig, axes = plt.subplots(len(rows), len(names), figsize=(2.55 * len(names), 2.95 * len(rows)), squeeze=False)
    for i, rec in enumerate(rows):
        for k, model in enumerate(names):
            ax = axes[i, k]
            ax.imshow(imread(rec[model]))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(model.replace(" (no class)", "\n(no class)").replace("Proposed ", "Proposed\n"), fontsize=9, fontweight="bold")
            if k == 0:
                ax.set_ylabel(f'{rec["class"].capitalize()}\nSeed {seed}', fontsize=9, fontweight="bold")
            if model == "Proposed Seam-Aware":
                for sp in ax.spines.values():
                    sp.set_color("#177090"); sp.set_linewidth(2.5)
    fig.suptitle(f'EndekGAN-T2I: class/prompt-matched visual comparison | seed={seed}\n'
                 'Internal runs share indexed inputs; DF-GAN uses the same text but independent noise', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .965])
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)

def tiled_figure(rows, seed, output):
    names = ["Transformer", "Proposed Seam-Aware", "DF-GAN–MiniLM"]
    fig, axes = plt.subplots(len(rows), len(names), figsize=(4.2 * 3, 4.2 * len(rows)), squeeze=False)
    for i, rec in enumerate(rows):
        for k, nice in enumerate(names):
            img = imread(rec[nice])
            mosaic = np.tile(img, (2, 2, 1))
            ax = axes[i, k]
            ax.imshow(mosaic)
            ax.axvline(255.5, color="red", linewidth=.9, alpha=.7)
            ax.axhline(255.5, color="red", linewidth=.9, alpha=.7)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(nice, fontsize=12, fontweight="bold")
            if k == 0: ax.set_ylabel(rec["class"].capitalize(), fontsize=11)
            ax.text(4, 498, f'Seam-gap={edge_gap(img):.4f}',
                    fontsize=10, color="white", bbox={"facecolor":"black","alpha":.75,"pad":2})
    fig.suptitle(f'2×2 repeated tiles (red lines = seams); seed={seed}\n'
                 'Tiling visualization is diagnostic, not evidence of weaving feasibility', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .965])
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

def export_expert_sheet(records, target, seed):
    # Model identity stays in a separate mapping file. Do not send mapping to raters.
    rng = random.Random(54321 + seed)
    records = list(records); rng.shuffle(records)
    img_dir = target / "BLINDED_IMAGES"; img_dir.mkdir(parents=True, exist_ok=True)
    form, secret = [], []
    for i, row in enumerate(records, 1):
        uid = f"E{seed}-{i:03d}"
        dest = img_dir / f"{uid}.jpg"
        Image.fromarray(imread(row["image_path"])).save(dest, quality=92)
        form.append(dict(expert_id="", image_id=uid, image_file=str(dest),
                         motif_category=row["class"], requested_prompt=row["prompt"],
                         ornament_clarity_1to5="", repeat_regular_1to5="",
                         color_prompt_match_1to5="", visual_cultural_plausibility_1to5="",
                         weaving_feasibility_1to5="", overall_1to5="", notes=""))
        secret.append(dict(image_id=uid, model=row["model"], seed=seed,
                           source_file=str(row["image_path"]), test_source=row["test_source"]))
    pd.DataFrame(form).to_csv(target / "EXPERT_BLIND_RATING_BLANK.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(secret).to_csv(target / "EXPERT_PRIVATE_MAPPING_DO_NOT_SHARE.csv", index=False, encoding="utf-8-sig")

def optional_semantic(records, target, max_images):
    """
    Diagnostic only: multilingual CLIP text encoder distills the same CLIP
    image space. Prompts are HASH-TEMPLATED and not human-verified labels.
    Images scored against supplied prompts, no causal semantic evidence.
    """
    import os
    os.environ["HF_HOME"] = str(ROOT / "MODEL_CACHE" / "huggingface")
    os.environ["TORCH_HOME"] = str(ROOT / "MODEL_CACHE" / "torch")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError("Install sentence-transformers before --semantic") from e
    print("Loading CLIP image and multilingual text encoders; downloads may be needed.")
    img_encoder = SentenceTransformer("sentence-transformers/clip-ViT-B-32")
    txt_encoder = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1")
    work = records[:max_images]
    texts = [str(r["prompt"]) for r in work]
    ims = [Image.fromarray(imread(r["image_path"])) for r in work]
    imgs = img_encoder.encode(ims, batch_size=12, normalize_embeddings=True, show_progress_bar=True)
    txts = txt_encoder.encode(texts, batch_size=12, normalize_embeddings=True, show_progress_bar=True)
    if imgs.shape != txts.shape:
        raise ValueError(f"CLIP encoders incompatible dimensions: {imgs.shape} vs {txts.shape}")
    sim = np.sum(imgs * txts, axis=1)
    out = [{**{k: r[k] for k in ("model", "seed", "motif_class", "prompt", "image_path")},
            "multilingual_clip_cosine_diagnostic": float(z)} for r, z in zip(work, sim)]
    pd.DataFrame(out).to_csv(target / "EVAL_MULTILINGUAL_CLIP_DIAGNOSTIC.csv", index=False, encoding="utf-8-sig")
    print("CLIP diagnostic complete:", len(out), "images; human verification is still required.")

def run(seed, per_class, metric_per_class, semantic=False):
    target = OUT / f"SEED_{seed}"; target.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(SPLIT)
    assert len(test) == 37 and test["class_idx"].nunique() == 4
    idx, dfroot = build_indices(seed, test)
    figures = selected_rows(seed, test, idx, dfroot, per_class)
    visual_rows = [figures[i * per_class] for i in range(len(CLASSES))]
    grid_figure(visual_rows, seed, target/"FIG_MODEL_COMPARISON.png")
    tiled_figure(visual_rows, seed, target/"FIG_SEAM_TILE_COMPARISON.png")
    manifest, sampled, records = [], [], []
    for rec in figures:
        for nice in METHODS:
            manifest.append({"model": nice, "class":rec["class"],"seed":seed,
                 "test_source":rec["test_source"], "prompt":rec["prompt"],
                 "image_path":str(rec[nice]),
                 "internal_index":rec.get(nice+"_internal_idx",""),
                 "pairing":"same source+prompt; internal common protocol" if nice!="DF-GAN–MiniLM"
                         else "same source+prompt; independent DF-GAN latent"})
            records.append({"model":nice, "class":rec["class"],"seed":seed,
                            "test_source":rec["test_source"],"prompt":rec["prompt"],
                            "image_path":str(rec[nice])})
    pd.DataFrame(manifest).to_csv(target/"FIG_INPUT_MANIFEST.csv",index=False,encoding="utf-8-sig")
    # Preselected/ordered per-class images to avoid cherry-picking.
    for nice, tag in METHODS.items():
        for c in CLASSES:
            if tag == "DFGAN_MiniLM":
                part = test.loc[test.label == c].reset_index(drop=True)
                paths = [dfroot/str(CLASS_ID[c])/f"{i:04d}.png" for i in range(metric_per_class)]
                prompts = [str(part.iloc[i % len(part)].prompt) for i in range(metric_per_class)]
            else:
                frame = idx[nice].loc[idx[nice].label == c].sort_values("generated_index").head(metric_per_class)
                paths = [Path(str(v)) for v in frame.generated_path]
                prompts = [str(v) for v in frame.prompt]
            for path,prompt in zip(paths,prompts):
                if not path.is_file(): raise FileNotFoundError(path)
                img=imread(path)
                sampled.append(dict(model=nice,seed=seed,motif_class=c,
                    image_path=str(path),prompt=prompt,
                    seam_gap=edge_gap(img),detail_energy=detail_energy(img),
                    periodicity_proxy=periodicity_proxy(img)))
    pd.DataFrame(sampled).to_csv(target/"EVAL_IMAGE_LEVEL.csv",index=False,encoding="utf-8-sig")
    summary=pd.DataFrame(sampled).groupby(["model","motif_class"],as_index=False).agg(
        n_images=("seam_gap","size"), seam_gap_mean=("seam_gap","mean"),
        seam_gap_sd=("seam_gap","std"), detail_energy_mean=("detail_energy","mean"),
        periodicity_proxy_mean=("periodicity_proxy","mean"))
    summary.to_csv(target/"EVAL_CLASS_SUMMARY.csv",index=False,encoding="utf-8-sig")
    export_expert_sheet(records,target,seed)
    if semantic: optional_semantic(sampled,target,max_images=len(sampled))
    (target/"METHOD_LIMITATIONS.txt").write_text(
        "Images are real outputs, not simulated. FIG_MODEL_COMPARISON pairs models by "
        "category/source prompt; DF-GAN has distinct latent noise and architecture.\n"
        "Seam-gap is opposite-edge pixel disagreement. Detail-energy is pixel gradient proxy. "
        "Periodicity proxy is not ornament regularity and can reward blur; never use alone.\n"
        "Multilingual CLIP diagnostic is NOT validated prompt correctness: original prompts "
        "were algorithmically constructed and expert_verified_rows=0.\n"
        "Expert rating file remains blank until human raters independently score all images. "
        "Do not release the private identity mapping to raters.\n"
        "FID/KID and LPIPS already exist in the experiment archive; this program does "
        "NOT invent/recalculate them.\n",encoding="utf-8")
    print(f"Seed {seed}: wrote {target}; quantitative images={len(sampled)}, expert images={len(records)}")
    return summary

def main():
    parser=argparse.ArgumentParser(description="EndekGAN image comparison and evidence-first diagnostics")
    parser.add_argument("--seeds",nargs="+",type=int,default=[42],choices=[42,123,2025])
    parser.add_argument("--panel-per-class",type=int,default=1)
    parser.add_argument("--metric-per-class",type=int,default=16)
    parser.add_argument("--semantic",action="store_true",
        help="Optional multilingual CLIP diagnostic (may download models; not semantic ground truth)")
    args=parser.parse_args()
    if not 1 <= args.panel_per_class <= 4: parser.error("panel-per-class 1..4")
    if not 1 <= args.metric_per_class <= 128: parser.error("metric-per-class 1..128")
    OUT.mkdir(parents=True,exist_ok=True)
    if args.semantic:
        # Semantic diagnostics must not overwrite results from the three-seed full evaluation.
        for seed in args.seeds:
            target = OUT / f"SEED_{seed}"
            src = target / "EVAL_IMAGE_LEVEL.csv"
            if not src.is_file():
                raise FileNotFoundError("First run Step 4 and verify this file: " + str(src))
            saved = pd.read_csv(src)
            work = saved.groupby(["model", "motif_class"], sort=False).head(args.metric_per_class)
            print(f"Semantic-only diagnostic: seed={seed}, images={len(work)}; saved quantitative files untouched")
            optional_semantic(work.to_dict("records"), target, max_images=len(work))
        return
    results=[]
    for seed in args.seeds:
        results.append(run(seed,args.panel_per_class,args.metric_per_class,args.semantic))
    if results:
        allrows=[]
        for seed in args.seeds:
            p=OUT/f"SEED_{seed}"/"EVAL_IMAGE_LEVEL.csv"
            allrows.append(pd.read_csv(p))
        agg=pd.concat(allrows,ignore_index=True)
        pd.DataFrame(agg).to_csv(OUT/"ALL_IMAGE_LEVEL.csv",index=False,encoding="utf-8-sig")
        pooled=agg.groupby(["model","seed"],as_index=False).agg(
            seam_gap=("seam_gap","mean"),detail_energy=("detail_energy","mean"),
            periodicity_proxy=("periodicity_proxy","mean"))
        pooled.to_csv(OUT/"PER_SEED_SUMMARY.csv",index=False,encoding="utf-8-sig")
        print("\nPer-seed summary:\n",pooled.to_string(index=False))
        (OUT/"RUN_MANIFEST.json").write_text(json.dumps(dict(seeds=args.seeds,
             metric_per_class=args.metric_per_class,panel_per_class=args.panel_per_class,
             semantic=args.semantic,models=list(METHODS),data_split="fixed 296/37/37",
             interpretation="exploratory; no expert ratings yet"),indent=2),encoding="utf-8")

if __name__ == "__main__":
    main()
