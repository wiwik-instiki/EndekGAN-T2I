from pathlib import Path
import unittest,json,hashlib
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
WORKFLOW=ROOT/"research_reproducibility_2026"
class PaperGalleryTests(unittest.TestCase):
    def test_exact_figure_1_to_10_present(self):
        m=json.loads((ROOT/"paper_figures"/"FIGURE_MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual([x["figure"] for x in m],list(range(1,11)))
        self.assertEqual(len(set(x["file"] for x in m)),10)
        for x in m:
            p=ROOT/x["file"]
            self.assertTrue(p.is_file(),str(p))
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),x["sha256"])
            with Image.open(p) as im:
                im.verify()
            self.assertIn(x["file"],(ROOT/"README.md").read_text(encoding="utf-8"))
    def test_gallery_visible_in_root_readme(self):
        readme=(ROOT/"README.md").read_text(encoding="utf-8")
        self.assertEqual(readme.count("### Figure "),10)
        self.assertEqual(readme.count("![Figure "),10)
        self.assertIn("research_reproducibility_2026/00_MASTER_ENDekGAN_EXPERIMENT_PIPELINE.ipynb",readme)
        self.assertIn("research_reproducibility_2026/scripts/replot_paper_quant_figures.py",readme)
    def test_auxiliary_seed_panels_are_present(self):
        files=list((ROOT/"supplementary_figures").glob("*.png"))+list((ROOT/"supplementary_figures").glob("*.jpg"))
        self.assertEqual(len(files),6)
        for p in files:
            with Image.open(p) as im:im.verify()
    def test_experiment_pipeline_and_figure_scripts_present(self):
        scripts=["reanalyze_evidence.py","evaluate_existing_outputs.py",
                 "replot_paper_quant_figures.py","export_figures_from_ieee_manuscript.py",
                 "workflow.py","execute_ipynb.py"]
        for x in scripts:self.assertTrue((WORKFLOW/"scripts"/x).is_file(),x)
        notebooks=["02_internal_conditioning_seam_ablation.ipynb",
                   "03_external_dfgan_minilm.ipynb"]
        for x in notebooks:self.assertTrue((WORKFLOW/"notebooks"/x).is_file(),x)
        master=(WORKFLOW/"00_MASTER_ENDekGAN_EXPERIMENT_PIPELINE.ipynb").read_text(encoding="utf-8")
        self.assertIn("ALL TEN MANUSCRIPT FIGURES VERIFIED",master)
        self.assertIn("RUN_INTERNAL_18_TRAINING = False",master)
        self.assertIn("RUN_EXTERNAL_3_TRAINING = False",master)
    def test_no_full_original_raw_dataset_added(self):
        # Only manuscript-excerpt images and authorized synthesized figure panels.
        self.assertEqual(len(list((ROOT/"paper_figures").glob("figure_*.png"))),9)
        self.assertTrue((ROOT/"paper_figures"/"figure_10_repeated_tile_visualization.jpg").is_file())
if __name__=="__main__":
    unittest.main()
