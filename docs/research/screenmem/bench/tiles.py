"""Remote-desktop-style lossless tier: 64x64 tiles, store only tiles whose hash changed vs previous frame, zstd-compress tile stream.
Also measures: % frames identical, % tiles changed per frame, bytes/frame for lossless delta."""
import os, json, numpy as np, cv2, zstandard as zstd, hashlib, sys
RAW="/home/ubuntu/screenmem/raw"; B="/home/ubuntu/screenmem/bench"
frames=[json.loads(l) for l in open(f"{RAW}/frames.jsonl")]
frames=[f for f in frames if os.path.exists(f"{RAW}/{f['file']}")]
T=64
cz=zstd.ZstdCompressor(level=3)
prev=None; tot=0; tot_png=0; identical=0; tile_frac=[]; per_frame=[]
for k,fr in enumerate(frames):
    im=cv2.imread(f"{RAW}/{fr['file']}")  # BGR 1920x2560
    H,W,_=im.shape
    if prev is None:
        b=cz.compress(im.tobytes()); tot+=len(b); per_frame.append(len(b)); prev=im; tile_frac.append(1.0); continue
    diff=(im!=prev).any(axis=2)
    if not diff.any():
        identical+=1; per_frame.append(8); tot+=8; tile_frac.append(0); continue
    # tiles with any change
    th,tw=H//T,W//T
    d=diff[:th*T,:tw*T].reshape(th,T,tw,T).any(axis=(1,3))
    idx=np.argwhere(d)
    tiles=np.stack([im[y*T:(y+1)*T,x*T:(x+1)*T] for y,x in idx]) if len(idx) else np.zeros((0,T,T,3),np.uint8)
    # XOR with previous tile content helps zstd on small changes
    ptiles=np.stack([prev[y*T:(y+1)*T,x*T:(x+1)*T] for y,x in idx]) if len(idx) else tiles
    payload=np.bitwise_xor(tiles,ptiles).tobytes()+idx.astype(np.uint16).tobytes()
    b=cz.compress(payload); tot+=len(b); per_frame.append(len(b)); tile_frac.append(len(idx)/(th*tw)); prev=im
    if k%100==0: print(k,len(b),f"{len(idx)/(th*tw):.2f}",flush=True)
N=len(frames)
res={"n":N,"identical_frames":identical,"identical_pct":100*identical/N,"mean_tile_changed_pct":100*float(np.mean(tile_frac)),
     "median_tile_changed_pct":100*float(np.median(tile_frac)),"total_bytes":tot,"bytes_per_s":tot/N,"gb_per_16h":tot/N*3600*16/1e9,
     "p50_frame_bytes":float(np.percentile(per_frame,50)),"p90_frame_bytes":float(np.percentile(per_frame,90)),"p99":float(np.percentile(per_frame,99))}
json.dump({"res":res,"per_frame":per_frame,"tile_frac":tile_frac},open(f"{B}/res_tiles.json","w"))
print(json.dumps(res,indent=1))
