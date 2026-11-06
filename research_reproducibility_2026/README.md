# EndekGAN-T2I: End-to-End Experiment Repository

**One integrated study. One master notebook. Source-backed results.**

The research investigates Endek-specific structured textual conditioning, a four-pixel opposing-border generator regularizer, and the fidelity/diversity/continuity trade-off under limited data. This is a reproducibility and code-provenance package, not a claim that the proposed model wins every metric.

## First step: VS Code

1. Open this entire repository folder in VS Code.
2. Open **00_MASTER_ENDekGAN_EXPERIMENT_PIPELINE.ipynb**.
3. Select the experiment Python kernel.
4. Click **Run All**. The default switches reanalyse the 21 archived model/seed results without training or downloading weights.
5. See local_runs/reanalysis for generated tables, figures and the RQ1–RQ3 report. This folder is ignored by Git.

The [GitHub root README](../README.md) displays all ten exact manuscript figures (Figure 1–10); the [paper gallery](../paper_figures/) and [supplemental visual evidence](../supplementary_figures/) contain the image files. Use `scripts/replot_paper_quant_figures.py` to recompute figures 4–7 from archived public numerical evidence, or `scripts/export_figures_from_ieee_manuscript.py` to export exact figure images from an authorized local Word manuscript.

The master notebook calls the actual source notebooks and scripts in chronological research order. It does not combine or alter their GAN layers. See docs/EXECUTION_MODES.md for enabling the full local pipeline.

## Study and workflow

| Step | Executable in this repository | Role |
|---|---|---|
| 0 | 00_MASTER_ENDekGAN_EXPERIMENT_PIPELINE.ipynb | Sequential orchestrator |
| 1 | notebooks/01_dataset_and_prompt_audit.ipynb | Data curation; source images; text metadata |
| 2 | notebooks/02_internal_conditioning_seam_ablation.ipynb | Category, lexical hash, Transformer and seam variants; six settings × three seeds |
| 3 | notebooks/03_external_dfgan_minilm.ipynb | Independent adapted DF-GAN–MiniLM comparator; three seeds |
| 4 | scripts/evaluate_existing_outputs.py | Image-level seam, motif/tile comparisons, optional diagnostics |
| 5 | notebooks/05_master_recap_original.ipynb | Original evidence and provenance reconstruction |
| 6 | scripts/reanalyze_evidence.py | Fully public three-seed numerical recalculation |
| 7 | results/ and figures/ | Auditable public summary of the reported experiment |

The original CPU and CUDA execution stages belong to the **same study**, not separately reported methods. Historical prototype/recovery notebooks are intentionally excluded from this new-paper publication package. Seed-2025 external recovery was an independent restart following a damaged optimizer state; it is not a fourth seed or a resumed epoch-125 checkpoint.

## Verified experimental population

- Endek source images: 372 raw, 370 unique after two exact duplicates.
- Fixed split: 296 train, 37 validation, 37 test (seed 3407).
- Class-wise split, train/validation/test: Flora 118/14/15; Fauna 42/6/5; Decorative 90/11/11; Geometric 46/6/6.
- Internal runs: six configurations × three seeds (42, 123, 2025) = 18.
- External adapted DF-GAN–MiniLM: three completed runs, total 21.
- Saved image-level diagnostics: 7 configurations × 4 classes × 128 images × 3 seeds = 10,752 observations.

## Main experimentally supported result

With class embedding, the seam-aware regularizer reduced mean outer-edge discontinuity from 0.07910 to 0.04766 (39.8%). Without class embedding, the reduction was 45.6%. Under the class-conditioned comparison, FID increased from 346.76 to 360.34, KID from 0.29815 to 0.33627, and within-model LPIPS variability fell from 0.26436 to 0.17686. The adapted external comparator yielded FID 278.77 and KID 0.16988 but a higher mean edge gap of 0.13176. Thus, the outcome is a measurable fidelity–diversity–continuity trade-off.

## Directory

- notebooks/: copied experiment notebook sources, with large saved outputs cleared.
- scripts/: runnable evaluation, sequential dispatch and evidence regeneration.
- results/: selected aggregate numerical CSVs. Source-image file paths and original imagery are excluded.
- figures/: grayscale quantitative publication figures; not original textile images.
- docs/: source hashes, experimental mapping, execution prerequisites, data-rights warning, and limitations.
- local_runs/: ignored local output for a new execution; created by default in the first Run All.

## Explicit limitations and publication checks

FID/KID estimates are exploratory because only 37 real test images are available. A reduced outer-edge gap does not establish Endek motif recognizability, cultural authenticity, semantic prompt adherence, or weaving feasibility. Multilingual CLIP diagnostics only exist for seed 42; no completed blinded expert assessments exist. StyleGAN2-ADA was **not trained** in this final protocol and must not appear as a completed comparator.

Original image files, checkpoints, private mappings, paper Word files, trained model weights, and third-party source trees are deliberately not included. This folder may be pushed only after the authors have verified image-data rights, code ownership, collaborator agreement, publisher rules, and the choice of repository visibility. **No open-source license is granted by this export.**

See docs/EXECUTION_MODES.md and docs/DATA_RIGHTS.md.
