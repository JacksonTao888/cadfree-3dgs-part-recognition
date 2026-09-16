import os, csv, glob, shutil, numpy as np
from PIL import Image
from recompose_cropped import square_crop, save_pair
ROOT=os.environ.get("CADFREE_ROOT", os.path.expanduser("~/cadfree_work"))
for r in csv.DictReader(open(f"{ROOT}/clip_to_class.csv")):
    clip,cls=r["clip"],r["class"]
    dw=f"{ROOT}/data_rawframes_source/{cls}"; db=f"{ROOT}/data_rawframes_source_bg/{cls}"
    shutil.rmtree(dw,ignore_errors=True); shutil.rmtree(db,ignore_errors=True); os.makedirs(dw); os.makedirs(db)
    n=0
    for ip in sorted(glob.glob(f"{ROOT}/{clip}/images_hi/frame_*.jpg")):
        mp=f"{ROOT}/{clip}/masks_screw_hi/{os.path.basename(ip)}.png"
        if not os.path.exists(mp): continue
        a=np.array(Image.open(ip).convert("RGB")); m=np.array(Image.open(mp).convert("L"))>127
        n+=save_pair(a,m,dw,db,os.path.basename(ip)[:-4])
    print(f"  rawframes {cls:8s}: {n}")
