"""Benchmark storage strategies on the captured 1-fps frame set. Outputs bytes per frame-second for each strategy."""
import os, json, subprocess, sys, time, glob, io
import numpy as np, cv2
from PIL import Image
import zstandard as zstd
RAW="/home/ubuntu/screenmem/raw"; OUT="/home/ubuntu/screenmem/bench"; os.makedirs(OUT,exist_ok=True)
frames=[json.loads(l) for l in open(f"{RAW}/frames.jsonl")]
frames=[f for f in frames if os.path.exists(f"{RAW}/{f['file']}")]
N=len(frames); dur=frames[-1]["t"]-frames[0]["t"]
print(f"{N} frames over {dur/60:.1f} min")
res={}
def sz(p): return os.path.getsize(p)
def dirsz(pattern): return sum(os.path.getsize(p) for p in glob.glob(pattern))
res["raw_png"]=dirsz(f"{RAW}/*.png")

# --- A. straight video encodes at 1 fps (concat list) ---
with open(f"{OUT}/list.txt","w") as f:
    for fr in frames: f.write(f"file '{RAW}/{fr['file']}'\nduration 1\n")
def enc(name, scale, codec_args, ext="mp4"):
    out=f"{OUT}/{name}.{ext}"
    if not os.path.exists(out):
        vf=f"scale={scale}:-2" if scale else "null"
        t=time.time()
        subprocess.run(["ffmpeg","-y","-loglevel","error","-f","concat","-safe","0","-r","1","-i",f"{OUT}/list.txt","-vf",vf,"-pix_fmt","yuv420p",*codec_args,"-r","1",out],check=True)
        print(f"  {name}: {time.time()-t:.0f}s")
    res[name]=sz(out); return out
# x264 (what Rewind uses)
enc("h264_full_crf28",None,["-c:v","libx264","-preset","medium","-crf","28","-g","300","-tune","stillimage"])
enc("h264_1280_crf28",1280,["-c:v","libx264","-preset","medium","-crf","28","-g","300","-tune","stillimage"])
# x265
enc("hevc_full_crf28",None,["-c:v","libx265","-preset","medium","-crf","28","-x265-params","keyint=600:log-level=error"])
enc("hevc_full_crf32",None,["-c:v","libx265","-preset","medium","-crf","32","-x265-params","keyint=600:log-level=error"])
enc("hevc_1920_crf30",1920,["-c:v","libx265","-preset","medium","-crf","30","-x265-params","keyint=600:log-level=error"])
enc("hevc_1280_crf30",1280,["-c:v","libx265","-preset","medium","-crf","30","-x265-params","keyint=600:log-level=error"])
# AV1 (libaom, slow but best ratio)
enc("av1_1920_crf40",1920,["-c:v","libaom-av1","-cpu-used","6","-crf","40","-b:v","0","-g","600","-tune","ssim"])
enc("av1_full_crf40",None,["-c:v","libaom-av1","-cpu-used","6","-crf","40","-b:v","0","-g","600","-tune","ssim"])
json.dump(res,open(f"{OUT}/res_video.json","w"),indent=1)

# --- B. change detection: keep frame only if it differs from last kept ---
def small(p): 
    im=cv2.imread(p,cv2.IMREAD_GRAYSCALE); return cv2.resize(im,(64,48),interpolation=cv2.INTER_AREA).astype(np.float32)
kept=[]; last=None; diffs=[]
for fr in frames:
    s=small(f"{RAW}/{fr['file']}")
    d=0 if last is None else float(np.mean(np.abs(s-last)))/255
    diffs.append(d)
    if last is None or d>0.02:
        kept.append(fr); last=s
print(f"change-detect kept {len(kept)}/{N} frames ({100*len(kept)/N:.0f}%)")
json.dump({"kept":[k["i"] for k in kept],"diffs":diffs},open(f"{OUT}/kept.json","w"))
with open(f"{OUT}/list_kept.txt","w") as f:
    for fr in kept: f.write(f"file '{RAW}/{fr['file']}'\nduration 1\n")
def enck(name, scale, codec_args):
    out=f"{OUT}/{name}.mp4"
    if not os.path.exists(out):
        vf=f"scale={scale}:-2" if scale else "null"
        subprocess.run(["ffmpeg","-y","-loglevel","error","-f","concat","-safe","0","-r","1","-i",f"{OUT}/list_kept.txt","-vf",vf,"-pix_fmt","yuv420p",*codec_args,"-r","1",out],check=True)
    res[name]=sz(out)
enck("kept_hevc_full_crf28",None,["-c:v","libx265","-preset","medium","-crf","28","-x265-params","keyint=600:log-level=error"])
enck("kept_hevc_1920_crf30",1920,["-c:v","libx265","-preset","medium","-crf","30","-x265-params","keyint=600:log-level=error"])
enck("kept_hevc_1920_crf34",1920,["-c:v","libx265","-preset","medium","-crf","34","-x265-params","keyint=600:log-level=error"])
enck("kept_hevc_1280_crf30",1280,["-c:v","libx265","-preset","medium","-crf","30","-x265-params","keyint=600:log-level=error"])
enck("kept_av1_1920_crf40",1920,["-c:v","libaom-av1","-cpu-used","6","-crf","40","-b:v","0","-g","600","-tune","ssim"])
enck("kept_av1_1920_crf48",1920,["-c:v","libaom-av1","-cpu-used","6","-crf","48","-b:v","0","-g","600","-tune","ssim"])
# per-keyframe stills (random access, no video decode): webp q60 @1920
tot=0
for fr in kept:
    im=Image.open(f"{RAW}/{fr['file']}").convert("RGB"); im=im.resize((1920,1440),Image.LANCZOS)
    b=io.BytesIO(); im.save(b,"WEBP",quality=60,method=6); tot+=b.tell()
res["kept_webp_1920_q60"]=tot
json.dump(res,open(f"{OUT}/res_all.json","w"),indent=1)
for k,v in res.items():
    print(f"{k:28s} {v/1e6:8.2f} MB  {v/N:9.0f} B/s  -> {v/N*86400/1e9:6.2f} GB/day(24h)  {v/N*3600*16/1e9:6.2f} GB/day(16h active)")
