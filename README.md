# EndekGAN-T2I: Seam-Aware Text-to-Image GAN for Balinese Endek Motif Synthesis under Limited Data

**Research repository for an ongoing doctoral research study on limited-data conditional image synthesis and Balinese Endek textile motifs.** This page presents the final controlled experimental protocol and **all Figure 1–10 images embedded in the paper**, together with the research code used to train and evaluate internal conditioning variants, seam-aware configurations, and the independently adapted DF-GAN–MiniLM baseline.

> **Research status:** 21 completed model–seed runs (six internal configurations plus an adapted external comparator, each across seeds 42, 123 and 2025). No additional or fabricated experiment is implied by repository regeneration. The class-macro FID/KID scores are exploratory because the held-out reference population has only 37 real images.


## Scientific motivation and cultural context

Balinese Endek is a traditional handwoven textile and a culturally situated design domain. The Indonesian Directorate General of Intellectual Property records Balinese Endek as communal intellectual property / traditional knowledge. This supplies a documented heritage context for studying design-oriented motif synthesis; it does **not** prove that a generated motif is culturally authentic, artistically acceptable, or suitable for weaving. Source: [Indonesia Directorate General of Intellectual Property — Balinese Endek and Songket communal IP registration](https://dgip.go.id/index.php/artikel/detail-artikel-berita/bali-catatkan-kain-endek-dan-songket-sebagai-kekayaan-intelektual-komunal?kategori=ki-komunal).

The curated 370-image research collection (296 training / 37 validation / 37 test) represents the **currently available, access-controlled samples**, not the full diversity of Endek designs. Data scarcity is the experimental setting and motivation, not evidence that the model can generalize across undocumented motif families. Research claims are restricted to the controlled computational comparisons actually completed.

The study shows that an explicit four-pixel opposing-border L1 penalty can lower a measured pixel-level seam gap by **39.8% with class embedding** and **45.6% without class embedding**, while exploratory FID/KID and measured perceptual variability may worsen. It therefore reports a **boundary-continuity versus image-fidelity/diversity trade-off**, not universal visual-quality superiority or independently validated text-to-image semantic accuracy.

**Research status:** Manuscript in preparation for journal submission. This repository makes no claim of journal acceptance, peer-review endorsement or published DOI. Legacy materials using the earlier IEEE formatting are retained only as part of the research history.

**Prospective next stages (not completed or claimed as present results):** rights-cleared dataset expansion covering more motif families; independent Endek expert annotation and prompt validation; culturally appropriate generative evaluation; additional data-efficient baselines and seam/fidelity multi-objective learning; and expert-led assessment of possible textile-design or weaving implications. See [Research roadmap](docs/RESEARCH_ROADMAP.md).

### One notebook: complete experiment entry point

**[Open 00_MASTER_ENDekGAN_EXPERIMENT_PIPELINE.ipynb](research_reproducibility_2026/00_MASTER_ENDekGAN_EXPERIMENT_PIPELINE.ipynb)**

The notebook calls the code for:
1. **Dataset/Endek design descriptions** — image validation, duplicate checks, prompt and class information.
2. **Internal conditional-GAN experiments** — Category, Hash-Text and Transformer controls; standard and four-pixel seam-aware paired ablations, with or without class conditioning; seeds 42, 123 and 2025.
3. **Adapted DF-GAN–MiniLM** — independently trained external comparison, not the internal backbone.
4. **Image evaluation** — class- and prompt-matched generated panels, opposite-edge seam gap, qualitative tile repetitions, and saved diagnostic outputs.
5. **Three-seed numerical and paper-level figures** — FID, KID, within-model LPIPS, trade-off, real-vs-generated examples, and RQ1–RQ3 interpretation.

**Re-running the notebook with default settings does not retrain models.** It verifies the archived 21-run evidence and shows the manuscript figure gallery. Full CPU/CUDA retraining is optional and gated behind separately authorized dataset files and explicit configuration. See [Execution Modes](research_reproducibility_2026/docs/EXECUTION_MODES.md), [Paper-to-code traceability](research_reproducibility_2026/docs/EXPERIMENT_TO_PAPER.md), and [Data/Image rights](research_reproducibility_2026/docs/DATA_RIGHTS.md).

## Results overview

The dataset comprises 370 unique images with a **296/37/37** fixed train/validation/test partition. The proposed seam-aware objective reduces opposing image-edge discontinuity but does not outperform every baseline on distributional fidelity or measured perceptual variety.

| Experiment | Exploratory FID ↓ | Exploratory KID ↓ | Opposite-edge gap ↓ |
|---|---:|---:|---:|
| Category GAN | 304.05 | 0.22170 | 0.08613 |
| Hash-Text | 284.33 | 0.18525 | 0.08125 |
| Transformer | 346.76 | 0.29815 | 0.07910 |
| Transformer (no class) | 276.19 | 0.16098 | 0.08171 |
| **Proposed Seam-Aware** | 360.34 | 0.33627 | 0.04766 |
| Proposed (no class) | 336.83 | 0.26171 | 0.04447 |
| Adapted DF-GAN–MiniLM | 278.77 | 0.16988 | 0.13176 |

These are **three-seed means, not newly trained scores**. Standard deviations, per-seed values and matched comparisons are provided in [numerical results](research_reproducibility_2026/results/) and are regenerated by the master notebook. The seam score measures opposite outer-edge pixels and should never be interpreted as motif semantics, cultural validity, or weaving feasibility.

## Full paper figure gallery — Figures 1–10

Every numbered image below is taken from the figure embedded in the research manuscript. **The displayed gallery mirrors the research manuscript's Figure 1–10 numbering.** Figure 1 combines the four category panels (a–d) extracted from Word; Figure 8–10 reproduce actual archived image comparisons for seed 42. Supplementary panels for seeds 123 and 2025 appear further down.

### Figure 1. Dataset sampling (Flora, Fauna, Decorative, Geometric)

![Figure 1: Dataset sampling (Flora, Fauna, Decorative, Geometric)](paper_figures/figure_01_endek_dataset_samples.png)

Four Endek-category panels show the original training-domain samples, not synthesized outputs. Code: dataset audit and preprocessing.

### Figure 2. Research framework

![Figure 2: Research framework](paper_figures/figure_02_research_framework.png)

Research sequence from the Endek dataset and structured prompts to baseline conditioning, proposed seam-aware training, adapted external comparison, and quantitative/visual evaluation.

### Figure 3. Proposed Seam-Aware EndekGAN-T2I architecture

![Figure 3: Proposed Seam-Aware EndekGAN-T2I architecture](paper_figures/figure_03_proposed_seam_aware_architecture.png)

Internal conditional generator/discriminator, class/text embeddings and 4-pixel opposite-border objective. The adapted DF-GAN is an external comparator, not this internal architecture.

### Figure 4. Exploratory class-macro FID

![Figure 4: Exploratory class-macro FID](paper_figures/figure_04_fid_comparison.png)

Lower is better; average and sample SD across three seeds. Model-comparison evidence for RQ1–RQ3.

### Figure 5. Exploratory class-macro KID

![Figure 5: Exploratory class-macro KID](paper_figures/figure_05_kid_comparison.png)

Lower is better; dataset-size limitations apply. Conditioning and adapted-comparator evidence for RQ1 and RQ3.

### Figure 6. FID–seam-gap fidelity/continuity trade-off

![Figure 6: FID–seam-gap fidelity/continuity trade-off](paper_figures/figure_06_fid_seam_tradeoff.png)

The seam-aware model obtains smaller boundary discrepancy but does not dominate image-distribution metrics. Primary RQ2 trade-off.

### Figure 7. Perceptual variability (LPIPS)

![Figure 7: Perceptual variability (LPIPS)](paper_figures/figure_07_lpips_variability.png)

Internal model variability only; no directly matched LPIPS score is claimed for the adapted external baseline.

### Figure 8. Real Endek vs. internal conditioning alternatives

![Figure 8: Real Endek vs. internal conditioning alternatives](paper_figures/figure_08_conditioning_image_comparison.png)

Actual source-matched examples: real motifs, Category GAN, Hash-Text, and Transformer. Seed 42; RQ1 visual counterpart.

### Figure 9. Real Endek vs. proposed and external comparators

![Figure 9: Real Endek vs. proposed and external comparators](paper_figures/figure_09_proposed_vs_baselines_images.png)

Actual examples: real motifs, standard Transformer, proposed seam-aware Transformer, and adapted DF-GAN–MiniLM. Seed 42; DF-GAN uses independent noise.

### Figure 10. Repeated-tile border comparison

![Figure 10: Repeated-tile border comparison](paper_figures/figure_10_repeated_tile_visualization.jpg)

Archived 2×2 tiling comparison for seed 42. Border agreement cannot prove cultural validity or physical weaving feasibility.


## Supplementary three-seed qualitative inspections

The following figure sets were produced from the same archived experiment outputs; they represent other **training seeds**, not additional models. Expand to view real-vs-generated and repeated tile comparisons. Do not treat them as an expert-blinded motif acceptability score.

<details>
<summary>View seed 123 qualitative comparisons</summary>

**Category / Hash-Text / Transformer — seed 123**

![Category / Hash-Text / Transformer seed 123](supplementary_figures/FIG_08A_RQ1_CONDITIONING_SEED_123.png)

**Proposed vs standard Transformer and adapted DF-GAN — seed 123**

![Proposed vs standard Transformer and adapted DF-GAN seed 123](supplementary_figures/FIG_08B_PROPOSED_VS_COMPARATORS_SEED_123.png)

**2×2 motif repetition — seed 123**

![2×2 motif repetition seed 123](supplementary_figures/FIG_09_REPEAT_TILES_SEED_123.jpg)

</details>

<details>
<summary>View seed 2025 qualitative comparisons</summary>

**Category / Hash-Text / Transformer — seed 2025**

![Category / Hash-Text / Transformer seed 2025](supplementary_figures/FIG_08A_RQ1_CONDITIONING_SEED_2025.png)

**Proposed vs standard Transformer and adapted DF-GAN — seed 2025**

![Proposed vs standard Transformer and adapted DF-GAN seed 2025](supplementary_figures/FIG_08B_PROPOSED_VS_COMPARATORS_SEED_2025.png)

**2×2 motif repetition — seed 2025**

![2×2 motif repetition seed 2025](supplementary_figures/FIG_09_REPEAT_TILES_SEED_2025.jpg)

</details>

## Reproduction code and experimental scope

| Paper component | Source code | Evidence |
|---|---|---|
| Dataset, duplicate removal, motif prompts | [01_dataset_and_prompt_audit.ipynb](research_reproducibility_2026/notebooks/01_dataset_and_prompt_audit.ipynb) | Figure 1 |
| Category/Hash/Transformer internal ablations | [02_internal_conditioning_seam_ablation.ipynb](research_reproducibility_2026/notebooks/02_internal_conditioning_seam_ablation.ipynb) | Figures 4, 5, 7, 8 |
| Seam-aware generator, 4-pixel boundary loss | [same internal experiment notebook](research_reproducibility_2026/notebooks/02_internal_conditioning_seam_ablation.ipynb) | Figures 3, 6, 9, 10 |
| Adapted DF-GAN–MiniLM | [03_external_dfgan_minilm.ipynb](research_reproducibility_2026/notebooks/03_external_dfgan_minilm.ipynb) | Figures 4, 5, 6, 9 |
| Image-level evaluation | [evaluate_existing_outputs.py](research_reproducibility_2026/scripts/evaluate_existing_outputs.py) | Figures 8, 9, 10; supplemental panels |
| Three-seed summaries and RQ1–RQ3 checks | [replot_paper_quant_figures.py](research_reproducibility_2026/scripts/replot_paper_quant_figures.py) and [reanalyze_evidence.py](research_reproducibility_2026/scripts/reanalyze_evidence.py) | Figures 4–7 and numerical CSV |
| Single launcher for complete workflow | [Master experiment notebook](research_reproducibility_2026/00_MASTER_ENDekGAN_EXPERIMENT_PIPELINE.ipynb) | Figures 1–10; all experimental stages |

**Read first:** [Repository README for the full workflow](research_reproducibility_2026/README.md) · [Raw-image access and ethics](research_reproducibility_2026/docs/DATA_RIGHTS.md) · [Source provenance](research_reproducibility_2026/docs/SOURCE_MANIFEST.json).

The public repository includes the **research manuscript's embedded images**, but does not include the full raw Endek corpus, trained checkpoints or private expert-rater mappings. Recreating image experiments from scratch requires authorized access to these source materials. Some panels depict example real Endek images; confirm image-release rights before pushing to a public repository.

**Research limitations:** No completed blinded expert rating, motif-specific semantic compliance validation, or weaving trial is claimed. The multilingual CLIP diagnostic is available for seed 42 only. The original study did not produce a trained StyleGAN2-ADA comparator. The demonstrated improvement is opposing-border continuity; FID, KID, and perceptual variability can deteriorate.

## Earlier repository materials

The previous baseline research README is archived without overwriting its original text at [docs/README_ORIGINAL_BASELINE_2026.md](docs/README_ORIGINAL_BASELINE_2026.md). Pre-existing notebooks, older model outputs, and experimental history are retained in this Git repository but are **not** counted as new runs in the revised paper.
