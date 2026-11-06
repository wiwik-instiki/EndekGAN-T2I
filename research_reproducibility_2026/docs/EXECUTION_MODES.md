# Execution modes and prerequisites

## Evidence-only (default; executed and validated)

Run all cells of 00_MASTER_ENDekGAN_EXPERIMENT_PIPELINE.ipynb. Only RUN_PUBLIC_REANALYSIS=True; all other stage flags=False. The notebook reads public CSVs and reconstructs the numerical tables and plots in local_runs/reanalysis. No private dataset or GPU is needed. The 21-run aggregation and ablation effect checks have been executed successfully.

## Full archived-image evaluation on the original research computer

Set ENDEKGAN_PROJECT_ROOT to the authorized source archive (for the existing Windows project, the parent of this repository). Set ENDEKGAN_DATASET_ROOT to the authorized local EndekGAN_Dataset folder. Keep ENDEKGAN_WORK_ROOT set to an isolated folder, not the original research archive. In the master notebook, enable RUN_IMAGE_LEVEL_EVALUATION=True and/or RUN_MASTER_ARCHIVE_RECAP=True; then Run All. These tasks read saved model-generated PNGs and write only into the isolated local_runs directory. IMAGE RESULTS ARE PRIVATE unless explicit image rights exist.

The image evaluator processes seeds 42, 123 and 2025, up to 128 saved generated images per class per model. It does not recompute the original FID/KID or train weights. Its optional CLIP diagnostic is not part of the default pipeline and may download models.

## Re-training from scratch (NOT executed as part of this GitHub packaging)

Requirements:
1. Obtain permission to use/release the source images and the frozen-model downloads.
2. Use Python/PyTorch/PyTorch-CUDA and packages from requirements-training.txt; validate CUDA separately.
3. Set ENDEKGAN_PROJECT_ROOT to the original read-only source archive for reading legacy outputs, ENDEKGAN_DATASET_ROOT to the actual image dataset, and ENDEKGAN_WORK_ROOT to a DIFFERENT writable destination.
4. Set ENDEKGAN_ALLOW_TRAINING=YES_RETRAIN_EXPLICITLY **in the notebook kernel environment before launching** (for example by assigning os.environ in the master first code cell). Only then enable RUN_INTERNAL_18_TRAINING or RUN_EXTERNAL_3_TRAINING.
5. The adapted external source requires a privately authorized fixed-split bundle inside the isolated work root at EndekGAN_IEEE_V5_OUTPUT/EndekGAN_IEEE_V5_OUTPUT.zip. It depends on a pinned external implementation, CUDA and its own training schedule. A fresh full rerun has not been validated in this GitHub export; verify dependencies and source-code licenses before enabling.
6. The notebook executor respects original source-cell order. It does NOT pretend that the archived optimizer-state recovery is an additional model result. Model training can take substantial time and produce large output files; never enable this mode when seeking only the published evidence.

Notebook variables:
- RUN_DATASET_AUDIT
- RUN_INTERNAL_18_TRAINING
- RUN_EXTERNAL_3_TRAINING
- RUN_IMAGE_LEVEL_EVALUATION
- RUN_MASTER_ARCHIVE_RECAP
- RUN_PUBLIC_REANALYSIS

The default authoring workflow is evidence-only, not a seamless one-button training reproduction from a public clone. Generated private image metadata and human-expert blind mappings stay out of public Git.
