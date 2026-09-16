"""Test sets WITH the real background kept: same object-bbox square crop (margin 0.18) as
data_real_test, but the background is NOT whitened. Uses the existing SAM2 masks only to
locate the crop. Builds data_real_test_rawbg/<cls>/ (video B) and data_home_test_rawbg/<cls>/."""
import os, glob, csv, numpy as np
from PIL import Image
from recompose_cropped import square_crop
ROOT = os.environ.get("CADFREE_ROOT", os.path.expanduser("~/cadfree_work"))
def build(src_build, out_root, classes):
    for cls in classes:
        W = f"{src_build}/{cls}"; out = f"{out_root}/{cls}"; os.makedirs(out, exist_ok=True); n = 0
        for ip in sorted(glob.glob(f"{W}/images/frame_*.jpg")):
            mp = f"{W}/masks_screw/{os.path.basename(ip)}.png"
            if not os.path.exists(mp): continue
            a = np.array(Image.open(ip).convert("RGB")); m = np.array(Image.open(mp).convert("L")) > 127
            if m.sum() < 50: continue
            c, cm = square_crop(a, m)
            if c is None: continue
            Image.fromarray(c).save(f"{out}/{os.path.splitext(os.path.basename(ip))[0]}.jpg", quality=92); n += 1
        print(f"{out_root.split('/')[-1]:22s} {cls:8s} {n}")
classes = sorted(r["class"] for r in csv.DictReader(open(f"{ROOT}/clip_to_class.csv")))
build(f"{ROOT}/data_build/realtest", f"{ROOT}/data_real_test_rawbg", classes)
build(f"{ROOT}/data_build/home", f"{ROOT}/data_home_test_rawbg", [d for d in os.listdir(f"{ROOT}/data_build/home") if os.path.isdir(f"{ROOT}/data_build/home/{d}")])
