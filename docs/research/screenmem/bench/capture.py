"""1 fps lossless screen capture with window-title metadata. Ground-truth source for all compression tests."""
import subprocess, time, json, os, sys, signal
OUT = "/home/ubuntu/screenmem/raw"
meta = open(os.path.join(OUT, "frames.jsonl"), "a")
run = True
def stop(*a):
    global run; run = False
signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
i = len([f for f in os.listdir(OUT) if f.endswith(".png")])
while run:
    t0 = time.time()
    fn = os.path.join(OUT, f"f{i:06d}.png")
    subprocess.run(["ffmpeg","-y","-loglevel","error","-f","x11grab","-video_size","2560x1920","-i",":0","-frames:v","1","-compression_level","1",fn])
    try:
        title = subprocess.run(["xdotool","getactivewindow","getwindowname"],capture_output=True,text=True,timeout=1).stdout.strip()
    except Exception:
        title = ""
    meta.write(json.dumps({"i":i,"t":t0,"file":os.path.basename(fn),"title":title})+"\n"); meta.flush()
    i += 1
    dt = time.time()-t0
    if dt < 1.0: time.sleep(1.0-dt)
