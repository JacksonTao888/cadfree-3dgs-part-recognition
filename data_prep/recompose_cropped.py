"""
Re-build source + real-test datasets with OBJECT-BBOX SQUARE CROP so the object
is centered and fills the frame consistently for BOTH domains (fixes the
source(big/centered) vs real(small/off-center) scale mismatch, and the DTD-bg
swamping). Reuses existing SAM2 masks for real (no re-extraction / no re-SAM2).
CPU only.
"""
import os, csv, glob, random, shutil
import numpy as np
from PIL import Image

ROOT = os.environ.get("CADFREE_ROOT", os.path.expanduser("~/cadfree_work"))
DTD = os.environ.get("DTD_DIR", os.path.expanduser("~/datasets/dtd/images"))
MARGIN = 0.18
random.seed(0)
bgs = glob.glob(os.path.join(DTD, "**", "*.jpg"), recursive=True)


def square_crop(arr, m, margin=MARGIN):
    ys, xs = np.where(m)
    if len(xs) < 30:
        return None, None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0, x1 - x0
    py, px = int(h * margin), int(w * margin)
    y0, y1, x0, x1 = y0 - py, y1 + py, x0 - px, x1 + px
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    s = max(y1 - y0, x1 - x0) / 2
    Y0, Y1, X0, X1 = int(cy - s), int(cy + s), int(cx - s), int(cx + s)
    H, W = arr.shape[:2]
    pad_t, pad_l = max(0, -Y0), max(0, -X0)
    pad_b, pad_r = max(0, Y1 - H), max(0, X1 - W)
    a = np.pad(arr, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), constant_values=255)
    mm = np.pad(m, ((pad_t, pad_b), (pad_l, pad_r)), constant_values=False)
    Y0 += pad_t; Y1 += pad_t; X0 += pad_l; X1 += pad_l
    return a[Y0:Y1, X0:X1], mm[Y0:Y1, X0:X1]


def save_pair(arr, m, dw, db, stem):
    c, cm = square_crop(arr, m)
    if c is None:
        return 0
    w = c.copy(); w[~cm] = 255
    Image.fromarray(w).save(f"{dw}/{stem}.jpg", quality=92)
    bg = np.array(Image.open(random.choice(bgs)).convert("RGB").resize((c.shape[1], c.shape[0]))).astype(np.float32)
    mm = cm[..., None]
    comp = (c.astype(np.float32) * mm + bg * (1 - mm)).clip(0, 255).astype(np.uint8)
    Image.fromarray(comp).save(f"{db}/{stem}_bg.jpg", quality=92)
    return 1


def fresh(d):
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d, exist_ok=True)


def build_source():
    for r in csv.DictReader(open(f"{ROOT}/clip_to_class.csv")):
        clip, cls = r["clip"], r["class"]
        dw, db = f"{ROOT}/data_3dgs_source/{cls}", f"{ROOT}/data_3dgs_source_bg/{cls}"
        fresh(dw); fresh(db); n = 0
        for ip in sorted(glob.glob(f"{ROOT}/{clip}/renders_final/elev_*/*.png")):
            a = np.array(Image.open(ip).convert("RGB")); m = a.max(2) > 18
            stem = f"{os.path.basename(os.path.dirname(ip))}_{os.path.basename(ip)[:-4]}"
            n += save_pair(a, m, dw, db, stem)
        print(f"  src {cls:8s}: {n}")


def build_real():
    for r in csv.DictReader(open(f"{ROOT}/test_clip_to_class.csv")):
        cls = r["class"]; W = f"{ROOT}/data_build/realtest/{cls}"
        dw, db = f"{ROOT}/data_real_test/{cls}", f"{ROOT}/data_real_test_bg/{cls}"
        fresh(dw); fresh(db); n = 0
        for ip in sorted(glob.glob(f"{W}/images/frame_*.jpg")):
            mp = f"{W}/masks_screw/{os.path.basename(ip)}.png"
            if not os.path.exists(mp):
                continue
            a = np.array(Image.open(ip).convert("RGB"))
            m = np.array(Image.open(mp).convert("L")) > 127
            n += save_pair(a, m, dw, db, os.path.basename(ip)[:-4])
        print(f"  real {cls:8s}: {n}")


if __name__ == "__main__":
    print("== source =="); build_source()
    print("== real =="); build_real()
    print("done")
