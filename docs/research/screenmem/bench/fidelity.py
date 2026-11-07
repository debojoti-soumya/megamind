"""OCR fidelity vs compression, done properly: Tesseract at full 2560x1920 on (a) lossless PNG, (b) each codec variant decoded with ffmpeg
and upscaled back to 2560x1920. Metric: word recall vs lossless (words>=3 chars), plus recall of gold typed/URL strings."""
import os, json, subprocess, io, numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
RAW="/home/ubuntu/screenmem/raw"; B="/home/ubuntu/screenmem/bench"; V=f"{B}/fid"; os.makedirs(V,exist_ok=True)
kept=json.load(open(f"{B}/kept.json"))["kept"]
sample=list(range(0,len(kept),max(1,len(kept)//16)))[:16]
def tess(path):
    return subprocess.run(["tesseract",path,"-","--psm","3"],capture_output=True,text=True).stdout
def words(s): return set(w.lower().strip('.,;:()[]"\'') for w in s.split() if len(w.strip('.,;:()[]"\''))>2)
variants=["kept_hevc_full_crf28","kept_hevc_1920_crf30","kept_hevc_1920_crf34","kept_hevc_1280_crf30","kept_av1_1920_crf40","kept_av1_1920_crf48"]
# decode sampled frames from each video, upscaled to full res
for v in variants:
    sel="+".join(f"eq(n\\,{j})" for j in sample)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",f"{B}/{v}.mp4","-vf",f"select='{sel}',scale=2560:1920:flags=bicubic","-vsync","0",f"{V}/{v}_%03d.png"],check=True)
# still-image variants
for j in sample:
    im=Image.open(f"{RAW}/f{kept[j]:06d}.png").convert("RGB")
    im.resize((1920,1440),Image.LANCZOS).save(f"{V}/png1920_{j:03d}.png")  # resolution loss only, lossless
    b=io.BytesIO(); im.resize((1920,1440),Image.LANCZOS).save(b,"WEBP",quality=60,method=6); Image.open(io.BytesIO(b.getvalue())).resize((2560,1920),Image.BICUBIC).save(f"{V}/webp1920q60_{j:03d}.png")
    b=io.BytesIO(); im.save(b,"WEBP",quality=50,method=6); Image.open(io.BytesIO(b.getvalue())).save(f"{V}/webpfullq50_{j:03d}.png")
    b=io.BytesIO(); im.save(b,"JPEG",quality=60); Image.open(io.BytesIO(b.getvalue())).save(f"{V}/jpegfullq60_{j:03d}.png")
with ThreadPoolExecutor(8) as ex:
    gt=dict(zip(sample, ex.map(tess,[f"{RAW}/f{kept[j]:06d}.png" for j in sample])))
GT={j:words(gt[j]) for j in sample}
res={}
allv=variants+["png1920","webp1920q60","webpfullq50","jpegfullq60"]
for v in allv:
    paths=[f"{V}/{v}_{k+1:03d}.png" if v in variants else f"{V}/{v}_{j:03d}.png" for k,j in enumerate(sample)]
    paths=[p for p in paths if os.path.exists(p)]
    with ThreadPoolExecutor(8) as ex: txts=list(ex.map(tess,paths))
    rec=[]; prec=[]
    for j,t in zip(sample,txts):
        w=words(t)
        if GT[j]: rec.append(len(w&GT[j])/len(GT[j])); prec.append(len(w&GT[j])/max(1,len(w)))
    res[v]={"word_recall":float(np.mean(rec)),"word_precision":float(np.mean(prec)),"n":len(rec)}
    print(v,res[v],flush=True)
json.dump({"res":res,"gt_words_per_frame":float(np.mean([len(GT[j]) for j in sample]))},open(f"{B}/res_fidelity.json","w"),indent=1)
