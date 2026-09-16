"""
Assemble the REAL TEST domain from the SEPARATE test videos (video B).
Per class in test_clip_to_class.csv:
  ffmpeg extract frames -> SAM2 object mask (sam2_screw.py) -> composite
    object-on-WHITE  -> data_real_test/<class>/
    object-on-DTD-bg -> data_real_test_bg/<class>/
Run with the sam3d-objects env (SAM2) AFTER the GPU is free.
"""
import os, csv, glob, subprocess, random
import numpy as np
from PIL import Image

ROOT = os.environ.get("CADFREE_ROOT", os.path.expanduser("~/cadfree_work"))
TESTVID = os.environ.get("CADFREE_TESTVID", os.path.expanduser("~/turntable_videos_test/extracted"))
BUILD = f"{ROOT}/data_build/realtest"
OUT_W = f"{ROOT}/data_real_test"
OUT_BG = f"{ROOT}/data_real_test_bg"
DTD = os.environ.get("DTD_DIR", os.path.expanduser("~/datasets/dtd/images"))
FF = os.path.expanduser("~/miniconda3/envs/sibr_viewer_build/bin/ffmpeg")
PY_SAM = os.path.expanduser("~/miniconda3/envs/sam3d-objects/bin/python")
SAM_SCRIPT = os.path.expanduser("~/screw_recon/sam2_screw.py")
FPS = "3"        # ~100 frames from a 35s clip
random.seed(0)
bgs = glob.glob(os.path.join(DTD, "**", "*.jpg"), recursive=True)


def run(cmd, **kw):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)


def main():
    rows = list(csv.DictReader(open(f"{ROOT}/test_clip_to_class.csv")))
    for r in rows:
        cls, vid = r["class"], r["test_video"]
        W = f"{BUILD}/{cls}"; os.makedirs(f"{W}/images", exist_ok=True)
        # 1. extract frames
        if len(glob.glob(f"{W}/images/frame_*.jpg")) == 0:
            run([FF, "-i", f"{TESTVID}/{vid}", "-vf", f"fps={FPS},scale=1920:-1",
                 "-q:v", "2", "-y", f"{W}/images/frame_%04d.jpg", "-loglevel", "error"])
        # 2. SAM2 object mask
        if len(glob.glob(f"{W}/masks_screw/*.png")) == 0:
            os.makedirs(f"{W}/sam2_frames", exist_ok=True)
            for f in glob.glob(f"{W}/sam2_frames/*"):
                os.remove(f)
            for i, f in enumerate(sorted(glob.glob(f"{W}/images/frame_*.jpg"))):
                os.symlink(f, f"{W}/sam2_frames/{i}.jpg")
            env = dict(os.environ, RECON_WORK=W)
            run([PY_SAM, SAM_SCRIPT], env=env)
        # 3. composite white + DTD
        dw = f"{OUT_W}/{cls}"; db = f"{OUT_BG}/{cls}"
        os.makedirs(dw, exist_ok=True); os.makedirs(db, exist_ok=True)
        n = 0
        for ip in sorted(glob.glob(f"{W}/images/frame_*.jpg")):
            stem = os.path.splitext(os.path.basename(ip))[0]
            mp = f"{W}/masks_screw/{os.path.basename(ip)}.png"   # SAM2: <image>.png = frame_xxxx.jpg.png
            if not os.path.exists(mp):
                continue
            a = np.array(Image.open(ip).convert("RGB"))
            m = np.array(Image.open(mp).convert("L")) > 127
            if m.sum() < 50:
                continue
            w = a.copy(); w[~m] = 255
            Image.fromarray(w).save(f"{dw}/{stem}.jpg", quality=92)
            bg = np.array(Image.open(random.choice(bgs)).convert("RGB").resize((a.shape[1], a.shape[0]))).astype(np.float32)
            mm = m[..., None]
            comp = (a.astype(np.float32) * mm + bg * (1 - mm)).clip(0, 255).astype(np.uint8)
            Image.fromarray(comp).save(f"{db}/{stem}_bg.jpg", quality=92)
            n += 1
        print(f"  {cls:8s} <- {vid[:24]}: {n} real-test images")


if __name__ == "__main__":
    main()
