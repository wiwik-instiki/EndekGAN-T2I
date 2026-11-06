import json,unittest,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from reanalyze_evidence import compute

class PublishedEvidenceTests(unittest.TestCase):
    def test_notebook_files_are_valid_json(self):
        files=list((ROOT/"notebooks").rglob("*.ipynb"))+[ROOT/"00_MASTER_ENDekGAN_EXPERIMENT_PIPELINE.ipynb"]
        self.assertEqual(len(files),6)
        for f in files:
            nb=json.loads(f.read_text(encoding="utf-8"))
            self.assertEqual(nb["nbformat"],4)
            self.assertGreater(len(nb["cells"]),2)
            for cell in nb["cells"]:
                if cell["cell_type"]=="code":
                    self.assertEqual(cell.get("outputs",[]),[])
                    self.assertIsNone(cell.get("execution_count"))
    def test_training_disabled_in_master(self):
        p=ROOT/"00_MASTER_ENDekGAN_EXPERIMENT_PIPELINE.ipynb"
        nb=json.loads(p.read_text(encoding="utf-8"))
        code="\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
        for var in ["RUN_DATASET_AUDIT","RUN_INTERNAL_18_TRAINING","RUN_EXTERNAL_3_TRAINING","RUN_IMAGE_LEVEL_EVALUATION","RUN_MASTER_ARCHIVE_RECAP"]:
            self.assertIn(var+" = False",code)
        self.assertIn("RUN_PUBLIC_REANALYSIS = True",code)
    def test_run_level_completeness(self):
        a=pd.read_csv(ROOT/"results"/"internal_18_training_runs.csv")
        b=pd.read_csv(ROOT/"results"/"adapted_dfgan_3_training_runs.csv")
        c=pd.read_csv(ROOT/"results"/"three_seed_boundary_diagnostics.csv")
        self.assertEqual((len(a),len(b),len(c)),(18,3,21))
        self.assertEqual(set(a.seed),{42,123,2025})
        self.assertEqual(set(b.seed),{42,123,2025})
        self.assertEqual(len(a.run.unique()),18)
    def test_numeric_recalculation(self):
        d=ROOT/"local_runs"/"test_validation"
        out=compute(d)
        self.assertEqual(out["runs"],21)
        self.assertEqual(out["models"],7)
        self.assertAlmostEqual(out["seam_reduction_pct_with_class"],39.752126,places=3)
        self.assertAlmostEqual(out["seam_reduction_pct_without_class"],45.571794,places=3)
    def test_no_private_paths_in_public_results(self):
        for p in (ROOT/"results").glob("*.csv"):
            df=pd.read_csv(p)
            forbidden=["path","filename","prompt","email","private","md5"]
            self.assertFalse([x for x in df.columns if any(k in x.lower() for k in forbidden)],p.name)
    def test_no_raw_images_or_weights(self):
        files=list(ROOT.rglob("*"))
        self.assertFalse([x for x in files if x.is_file() and x.suffix.lower() in (".pt",".pth",".ckpt",".safetensors",".docx",".pdf")])
        self.assertFalse([x for x in files if x.is_file() and x.name.startswith("EXPERT_PRIVATE_MAPPING")])

if __name__=="__main__":
    unittest.main()
