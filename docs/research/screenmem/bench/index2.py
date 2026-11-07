"""Text tier v2: typing-safe change detection (thumbnail MAD > 0.002), full-res Tesseract OCR (parallel), MiniLM embeddings, SQLite FTS5.
Also: delta-text encoding (store only lines not present in previous frame) + zstd to size the text tier honestly."""
import os, json, sqlite3, time, subprocess, numpy as np, cv2, zstandard as zstd
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
RAW="/home/ubuntu/screenmem/raw"; B="/home/ubuntu/screenmem/bench"
frames=[json.loads(l) for l in open(f"{RAW}/frames.jsonl")]; frames=[f for f in frames if os.path.exists(f"{RAW}/{f['file']}")]
def small(p): return cv2.resize(cv2.imread(p,cv2.IMREAD_GRAYSCALE),(64,48),interpolation=cv2.INTER_AREA).astype(np.float32)
kept=[]; last=None
for fr in frames:
    s=small(f"{RAW}/{fr['file']}")
    if last is None or float(np.mean(np.abs(s-last)))/255>0.002: kept.append(fr); last=s
print(f"kept {len(kept)}/{len(frames)}",flush=True)
json.dump({"kept":[k["i"] for k in kept]},open(f"{B}/kept2.json","w"))
def tess(fr): return subprocess.run(["tesseract",f"{RAW}/{fr['file']}","-","--psm","3"],capture_output=True,text=True).stdout
t0=time.time()
with ThreadPoolExecutor(8) as ex: texts=list(ex.map(tess,kept))
print(f"tesseract {len(kept)} frames in {time.time()-t0:.0f}s ({(time.time()-t0)/len(kept):.2f}s/frame wall, 8 threads)",flush=True)
emb=SentenceTransformer("all-MiniLM-L6-v2")
E=emb.encode([ (t[:2000] if t.strip() else fr["title"]) for t,fr in zip(texts,kept)],batch_size=32)
db=sqlite3.connect(f"{B}/memory.sqlite"); c=db.cursor()
c.executescript("DROP TABLE IF EXISTS frames; DROP TABLE IF EXISTS fts; CREATE TABLE frames(i INTEGER PRIMARY KEY,t REAL,title TEXT,text TEXT,emb BLOB); CREATE VIRTUAL TABLE fts USING fts5(text,title,content='frames',content_rowid='i');")
for fr,t,e in zip(kept,texts,E): c.execute("INSERT INTO frames VALUES(?,?,?,?,?)",(fr["i"],fr["t"],fr["title"],t,e.astype(np.float16).tobytes()))
c.execute("INSERT INTO fts(fts) VALUES('rebuild')"); db.commit(); db.execute("VACUUM"); db.close()
full="\n".join(f"{fr['t']:.0f}\t{fr['title']}\t{t}" for fr,t in zip(kept,texts)).encode()
# delta lines vs previous frame
prev=set(); delta=[]
for fr,t in zip(kept,texts):
    lines=[l.strip() for l in t.splitlines() if l.strip()]
    new=[l for l in lines if l not in prev]; prev=set(lines)
    delta.append(f"{fr['t']:.0f}\t{fr['title']}\t"+"\n".join(new))
delta="\n".join(delta).encode()
cz=zstd.ZstdCompressor(level=19)
N=len(frames); dur=frames[-1]["t"]-frames[0]["t"]
res={"n_kept":len(kept),"n_total":N,"text_full_raw":len(full),"text_full_zstd19":len(cz.compress(full)),"text_delta_raw":len(delta),"text_delta_zstd19":len(cz.compress(delta)),
     "emb_fp16":len(kept)*384*2,"emb_int8":len(kept)*384,"emb_binary":len(kept)*384//8,"sqlite_db":os.path.getsize(f"{B}/memory.sqlite"),
     "words_per_kept_frame":float(np.mean([len(t.split()) for t in texts]))}
for k in ["text_full_zstd19","text_delta_zstd19","emb_int8","sqlite_db"]: res[k+"_MB_per_16h"]=res[k]/dur*3600*16/1e6
json.dump(res,open(f"{B}/res_text2.json","w"),indent=1); print(json.dumps(res,indent=1))
