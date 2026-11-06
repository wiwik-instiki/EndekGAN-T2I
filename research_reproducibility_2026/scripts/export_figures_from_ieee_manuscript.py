"""Export the exact images embedded in an author's IEEE manuscript.

All outputs are the same archived manuscript figures; this tool does not train
or improve a model and cannot substitute for the original experiment code.
Requires a locally authorized Word document (not distributed with GitHub).
"""
from pathlib import Path
from io import BytesIO
import argparse,json,hashlib
from docx import Document
from PIL import Image,ImageDraw,ImageFont

FILES={
 1:"figure_01_endek_dataset_samples.png",
 2:"figure_02_research_framework.png",
 3:"figure_03_proposed_seam_aware_architecture.png",
 4:"figure_04_fid_comparison.png",
 5:"figure_05_kid_comparison.png",
 6:"figure_06_fid_seam_tradeoff.png",
 7:"figure_07_lpips_variability.png",
 8:"figure_08_conditioning_image_comparison.png",
 9:"figure_09_proposed_vs_baselines_images.png",
 10:"figure_10_repeated_tile_visualization.jpg",
}
def images_preceding(doc,index):
    out=[]
    for paragraph in doc.paragraphs[max(0,index-5):index]:
        for b in paragraph._p.xpath(".//a:blip"):
            rid=b.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
            out.append(doc.part.related_parts[rid].blob)
    return out
def export(docx_path,target):
    target.mkdir(parents=True,exist_ok=True)
    doc=Document(docx_path)
    captions=[(idx,p) for idx,p in enumerate(doc.paragraphs) if p.style.name=="Fig Caption"]
    if len(captions)!=10:
        raise ValueError(f"Expected exactly ten Figure captions, found {len(captions)}. Check the paper revision.")
    manifest=[]
    for number,(paragraph_index,caption) in enumerate(captions,1):
        blobs=images_preceding(doc,paragraph_index)
        if number==1:
            if len(blobs)!=4:raise ValueError("Figure 1 must contain the four Endek categories")
            parts=[Image.open(BytesIO(blob)).convert("RGB") for blob in blobs]
            target_w=max(im.width for im in parts)
            result=Image.new("RGB",(target_w+14,sum(im.height+50 for im in parts)+10),"white")
            d=ImageDraw.Draw(result); ypos=4
            try:font=ImageFont.truetype("arial.ttf",25)
            except OSError:font=ImageFont.load_default()
            for i,(label,im) in enumerate(zip(("Flora","Fauna","Decorative","Geometric"),parts)):
                d.text((7,ypos),f"({chr(97+i)}) {label}",font=font,fill="#222222")
                ypos+=36;result.paste(im,(7,ypos));ypos+=im.height+14
        else:
            if not blobs:raise ValueError("No preceding image for Figure "+str(number))
            result=Image.open(BytesIO(blobs[-1])).convert("RGB")
        filename=FILES[number];dest=target/filename
        if filename.endswith(".jpg"):result.save(dest,quality=93,optimize=True)
        else:result.save(dest,optimize=True)
        manifest.append({"figure":number,"file":dest.name,"caption":caption.text,
                         "width":result.width,"height":result.height,
                         "sha256":hashlib.sha256(dest.read_bytes()).hexdigest()})
    (target/"EXPORTED_FIGURES_MANIFEST.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    print("EXPORTED_IEEE_PAPER_FIGURES",len(manifest),target)
    return manifest
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--manuscript",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    export(args.manuscript.resolve(),args.output.resolve())
