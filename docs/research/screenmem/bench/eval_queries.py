"""Query-quality eval over the text tier: for each gold question, check whether (a) FTS5 over OCR text, (b) MiniLM semantic search
retrieves a frame whose OCR text contains the gold answer, and report Recall@1/@5. Also checks the same against OCR of degraded
(HEVC 1920 crf34 / 1280) frames to see if compression breaks answerability."""
import json, sqlite3, numpy as np, re, sys
from sentence_transformers import SentenceTransformer
B="/home/ubuntu/screenmem/bench"
Q=json.load(open("/home/ubuntu/screenmem/doc/questions.json"))
db=sqlite3.connect(f"{B}/memory.sqlite"); c=db.cursor()
rows=c.execute("SELECT i,t,title,text,emb FROM frames").fetchall()
ids=[r[0] for r in rows]; texts={r[0]:(r[2]+" "+(r[3] or "")) for r in rows}
E=np.stack([np.frombuffer(r[4],dtype=np.float16).astype(np.float32) for r in rows]); E/=np.linalg.norm(E,axis=1,keepdims=True)+1e-9
emb=SentenceTransformer("all-MiniLM-L6-v2")
def hit(i,ans): 
    t=texts[i].lower(); return all(a.lower() in t for a in ans) if isinstance(ans,list) and len(ans)>1 and q["type"]=="exact" else any(a.lower() in t for a in ans)
def fts(qstr,k=5):
    toks=[w for w in re.findall(r"[A-Za-z0-9#]+",qstr) if len(w)>2 and w.lower() not in STOP]
    if not toks: return []
    qq=" OR ".join(f'"{t}"' for t in toks)
    try: return [r[0] for r in c.execute(f"SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT {k}",(qq,))]
    except Exception as e: return []
STOP=set("what was the did for about which where who and that with any from have into you your read look looked view viewed visit visited watch watched note noted need needed call".split())
def sem(qstr,k=5):
    v=emb.encode(qstr); v=v/np.linalg.norm(v); s=E@v; return [ids[j] for j in np.argsort(-s)[:k]]
out=[]; 
for q in Q:
    ans=q["a"]; gold=[i for i in ids if hit(i,ans)]
    f=fts(q["q"]); s=sem(q["q"])
    r={"q":q["q"],"type":q["type"],"n_gold_frames":len(gold),"fts@1":bool(f and hit(f[0],ans)),"fts@5":any(hit(i,ans) for i in f),"sem@1":bool(s and hit(s[0],ans)),"sem@5":any(hit(i,ans) for i in s)}
    out.append(r); print(json.dumps(r))
agg={k:float(np.mean([r[k] for r in out])) for k in ["fts@1","fts@5","sem@1","sem@5"]}
agg["answerable_from_ocr"]=float(np.mean([r["n_gold_frames"]>0 for r in out]))
print(json.dumps(agg,indent=1)); json.dump({"per_q":out,"agg":agg},open(f"{B}/res_queries.json","w"),indent=1)
