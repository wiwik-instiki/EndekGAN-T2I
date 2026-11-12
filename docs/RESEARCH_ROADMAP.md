# EndekGAN-T2I — Planned Research Roadmap

**Status:** Future research directions only. None of the following stages is claimed as completed in the existing 21-run study.

## Current evidence baseline
- 370 curated unique RGB images, split into 296 training, 37 validation and 37 test samples (split seed 3407).
- Six internal variants and one independently trained adapted DF-GAN–MiniLM comparator, each evaluated with seeds 42, 123, and 2025 (21 training runs).
- A four-pixel opposing-border penalty reduces the mean pixel-edge gap by 39.8% with class embeddings and 45.6% without them, but exploratory FID/KID are worse in the matched comparisons.
- No completed blinded expert assessment, independently validated prompt semantics, or verified weaving feasibility.

## Phase 2 (proposed): Dataset expansion and rights
1. Collaborate with Endek artisans, cultural institutions and rights holders to document provenance and permitted uses.
2. Add legally shareable and representative motif samples; audit duplicates, near-duplicates, imbalance and source leakage.
3. Reassess data coverage before making generalization claims.

## Phase 3 (proposed): Domain-expert assessment
1. Develop expert-validated motif descriptions and evaluation rubrics with Endek practitioners.
2. Measure recognizability, prompt adherence, cultural interpretation and inter-rater agreement.
3. Distinguish visual continuity from cultural authenticity and physical weaving feasibility.

## Phase 4 (proposed): Computational extension
1. Evaluate stronger limited-data generative baselines and data augmentation.
2. Optimize a seam-versus-fidelity/diversity multiobjective formulation rather than seam-gap alone.
3. Repeat multi-seed and held-out evaluation after expanding data and descriptions.

## Interpretation
The present study is a controlled computational characterization, not a completed cultural digitization system or a replacement for artisans.

**Cultural context reference:** [Indonesian Directorate General of Intellectual Property: Balinese Endek and Songket communal intellectual property registration](https://dgip.go.id/index.php/artikel/detail-artikel-berita/bali-catatkan-kain-endek-dan-songket-sebagai-kekayaan-intelektual-komunal?kategori=ki-komunal).
