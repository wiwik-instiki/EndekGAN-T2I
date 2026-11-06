# Paper-to-experiment traceability

## Section III: Dataset and proposed method

- Data audit: source notebook 01; fixed split 296/37/37 and four classes.
- Internal conditional-GAN comparisons: notebook 02, sharing the base generator/discriminator.
- Proposed components: multilingual MiniLM conditioning (384-D) and separate learned 64-D class embeddings for G and D; 448-D combined conditions projected to 256-D; border-strip loss width b=4, lambda=0.5 in seam configurations; fixed feature-matching coefficient=5.
- External adapted DF-GAN: notebook 03, independent architecture and training. It is NOT the backbone of the internal EndekGAN-T2I models.

## Section IV: Answering RQ1, RQ2 and RQ3

| RQ | Experimental comparison | Actual evidence |
|---|---|---|
| RQ1 | Category-only vs Hash-Text vs Transformer, all with class embedding and no seam penalty | internal_18_training_runs.csv, three_seed_quantitative_summary.csv, published FID/KID charts |
| RQ2 | Transformer standard vs seam-aware with class; Transformer without class vs seam-aware without class | matched_seam_ablations.csv, three_seed_boundary_diagnostics.csv, internal_lpips_variability.csv |
| RQ3 | Proposed and other internal settings vs independent adapted DF-GAN–MiniLM | adapted_dfgan_3_training_runs.csv, boundary_by_motif_class.csv; image panels regenerated locally |

### Key numbers: 3 seeds (mean)

- Category GAN: FID 304.05 / KID 0.22170.
- Hash-Text: 284.33 / 0.18525.
- Standard Transformer with class: 346.76 / 0.29815.
- Proposed class-conditioned seam-aware: 360.34 / 0.33627; seam gap 0.04766.
- Transformer without class: 276.19 / 0.16098; seam gap 0.08171.
- Seam-aware without class: 336.83 / 0.26171; seam gap 0.04447.
- Adapted DF-GAN–MiniLM: 278.77 / 0.16988; seam gap 0.13176.

### Figures and evidence boundaries

Repository figures provide aggregate grayscale FID, KID, LPIPS variability, and seam/FID trade-off. In the original private research archive, further real test images, generated outputs and repeated-tile comparisons support visual discussions but MUST NOT be uploaded without clearance.

FID/KID are exploratory and not a verified semantic evaluation. LPIPS measured variation among generated samples for internal conditions; its absence for the external adapted baseline must be stated. CLIP cosine scores are a partial seed-42 diagnostic, not expert-verified image-prompt correctness. StyleGAN2-ADA was not trained in this final experiment.
