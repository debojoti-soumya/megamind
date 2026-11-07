"""Text tier: dedup -> OCR -> SQLite FTS5 + embeddings. Measures text-tier bytes and OCR fidelity on degraded (video-decoded) frames."""
import os, json, sqlite3, time, io, sys, subprocess
import numpy as np, cv2
from PIL import Image
import zstandard as zstd
from rapidocr_onnxruntime import RapidOCR
from sentence_transformers import SentenceTransformer
RAW="/home/ubuntu/screenmem/raw"; B="/home/ubuntu/screenmem/bench"
frames={f["i"]:f for f in (json.loads(l) for l in open(f"{RAW}/frames.jsonl"))}
kept=json.load(open(f"{B}/kept.json"))["kept"]
ocr=RapidOCR(); emb=SentenceTransformer("all-MiniLM-L6-v2")
db=sqlite3.connect(f"{B}/memory.sqlite"); c=db.cursor()
c.executescript("""
DROP TABLE IF EXISTS frames; DROP TABLE IF EXISTS fts;
CREATE TABLE frames(i INTEGER PRIMARY KEY, t REAL, title TEXT, text TEXT, emb BLOB);
CREATE VIRTUAL TABLE fts USING fts5(text, title, content='frames', content_rowid='i');
""")
def do_ocr(img):
    r,_=ocr(img); r=r or []
    return " ".join(x[1] for x in r), r
t0=time.time(); texts={}
for n,i in enumerate(kept):
    fr=frames[i]; img=cv2.imread(f"{RAW}/{fr['file']}")
    txt,_=do_ocr(img); texts[i]=txt
    e=emb.encode(txt[:2000] if txt else fr["title"]).astype(np.float16).tobytes()
    c.execute("INSERT INTO frames VALUES(?,?,?,?,?)",(i,fr["t"],fr["title"],txt,e))
    if n%20==0: print(f"ocr {n}/{len(kept)} {time.time()-t0:.0f}s", flush=True)
c.execute("INSERT INTO fts(fts) VALUES('rebuild')"); db.commit()
print(f"OCR+embed {len(kept)} frames in {time.time()-t0:.0f}s ({(time.time()-t0)/len(kept):.2f}s/frame)")
# text-tier sizes
alltext="\n".join(f"{frames[i]['t']:.0f}\t{frames[i]['title']}\t{texts[i]}" for i in kept).encode()
cz=zstd.ZstdCompressor(level=19).compress(alltext)
# dictionary-style: compress consecutive frames' text as a stream (screen text is highly repetitive across adjacent frames)
embs=len(kept)*384*2
embs_int8=len(kept)*384
res={"n_kept":len(kept),"n_total":len(frames),"text_raw":len(alltext),"text_zstd19":len(cz),"emb_fp16":embs,"emb_int8":embs_int8,
     "sqlite_db":os.path.getsize(f"{B}/memory.sqlite")}
# --- OCR fidelity vs compression: decode same frames from HEVC/AV1 files, compare OCR word recall ---
def words(s): return set(w.lower() for w in s.split() if len(w)>2)
def frames_from_video(path, idxs, scale_back=(2560,1920)):
    out={}
    cap=cv2.VideoCapture(path); pos={k:j for j,k in enumerate(kept)}
    j=0
    while True:
        ok,f=cap.read()
        if not ok: break
        if j in idxs:
            out[j]=f
        j+=1
    return out
sample=list(range(0,len(kept),max(1,len(kept)//40)))[:40]
gt={j:words(texts[kept[j]]) for j in sample}
fid={}
for name in ["kept_hevc_full_crf28","kept_hevc_1920_crf30","kept_hevc_1920_crf34","kept_hevc_1280_crf30","kept_av1_1920_crf40","kept_av1_1920_crf48"]:
    p=f"{B}/{name}.mp4"
    if not os.path.exists(p): continue
    fs=frames_from_video(p,set(sample))
    rec=[];
    for j,f in fs.items():
        txt,_=do_ocr(f); w=words(txt)
        if gt[j]: rec.append(len(w&gt[j])/len(gt[j]))
    fid[name]=float(np.mean(rec)) if rec else None
    print(name,"OCR word recall vs lossless:",fid[name], flush=True)
res["ocr_recall"]=fid
json.dump(res,open(f"{B}/res_text.json","w"),indent=1)
print(json.dumps(res,indent=1))
