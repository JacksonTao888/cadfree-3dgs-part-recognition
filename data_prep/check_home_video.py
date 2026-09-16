"""Check one HOME capture: extract frames (3 fps, 1920 wide, auto-rotated), SAM2-mask it,
and report whether it is usable as a cross-environment test video.
usage: python check_home_video.py <video.mp4> <class e.g. screw_a>
Writes ~/recon_runs/data_build/home/<class>/ (images, masks_screw, screw_preview),
a preview grid  ~/recon_runs/data_build/home/<class>_check.jpg  and  <class>_check.json
"""
import os, sys, glob, json, subprocess, numpy as np, cv2
SCALE = int(os.environ.get("SCALE", "1920")); video, cls = sys.argv[1], sys.argv[2]; tag = sys.argv[3] if len(sys.argv) > 3 else cls
ROOT = os.path.expanduser("~/recon_runs"); W = f"{ROOT}/data_build/home/{tag}"
FF = os.path.expanduser("~/miniconda3/envs/sibr_viewer_build/bin/ffmpeg")
PY_SAM = os.path.expanduser("~/miniconda3/envs/sam3d-objects/bin/python")
REF = f"{ROOT}/data_real_test/{cls}"
os.makedirs(f"{W}/images", exist_ok=True)
if not glob.glob(f"{W}/images/frame_*.jpg"):
    subprocess.run([FF, "-i", video, "-vf", f"fps=3,scale=w='if(gt(iw,ih),{SCALE},-2)':h='if(gt(iw,ih),-2,{SCALE})'", "-q:v", "2", "-y",
                    f"{W}/images/frame_%04d.jpg", "-loglevel", "error"], check=True)
imgs = sorted(glob.glob(f"{W}/images/frame_*.jpg"))
if not glob.glob(f"{W}/masks_screw/*.png"):
    os.makedirs(f"{W}/sam2_frames", exist_ok=True)
    for f in glob.glob(f"{W}/sam2_frames/*"): os.remove(f)
    for i, f in enumerate(imgs): os.symlink(f, f"{W}/sam2_frames/{i}.jpg")
    subprocess.run([PY_SAM, os.path.expanduser("~/screw_recon/sam2_home.py")],
                   env=dict(os.environ, RECON_WORK=W), check=True)

def cutout224(img, m):
    ys, xs = np.where(m)
    if len(ys) == 0: return None, 0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    c = img[y0:y1, x0:x1].copy(); c[~m[y0:y1, x0:x1]] = 255
    s = max(c.shape[:2]); sq = np.full((s, s, 3), 255, np.uint8)
    sq[(s - c.shape[0]) // 2:(s - c.shape[0]) // 2 + c.shape[0], (s - c.shape[1]) // 2:(s - c.shape[1]) // 2 + c.shape[1]] = c
    return cv2.resize(sq, (224, 224), interpolation=cv2.INTER_AREA), max(y1 - y0, x1 - x0)
def lapvar(im224): return float(cv2.Laplacian(cv2.cvtColor(im224, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

rows = []
for ip in imgs:
    mp = f"{W}/masks_screw/{os.path.basename(ip)}.png"
    if not os.path.exists(mp): continue
    img = cv2.imread(ip); m = cv2.imread(mp, 0) > 127
    c, ext = cutout224(img, m)
    if c is None: rows.append(dict(f=ip, area=0)); continue
    ys, xs = np.where(m); H, Wd = m.shape
    touch = int(ys.min() == 0 or xs.min() == 0 or ys.max() == H - 1 or xs.max() == Wd - 1)
    rows.append(dict(f=ip, area=int(m.sum()), ext=int(ext), sharp=lapvar(c), touch=touch,
                     bright=float(img[m].mean()), cut=c))
ok = [r for r in rows if r["area"] > 0]
areas = np.array([r["area"] for r in ok]); exts = np.array([r["ext"] for r in ok])
ref = sorted(glob.glob(f"{REF}/*.jpg"))[::5]
ref_sharp = [lapvar(cv2.resize(cv2.imread(f), (224, 224), interpolation=cv2.INTER_AREA)) for f in ref]
rep = dict(cls=cls, video=os.path.basename(video), frames=len(imgs), masked=len(ok),
           empty_masks=len(rows) - len(ok), area_min=int(areas.min()), area_med=int(np.median(areas)),
           area_max=int(areas.max()), area_ratio=float(areas.max() / max(areas.min(), 1)),
           obj_px_min=int(exts.min()), obj_px_med=int(np.median(exts)), obj_px_max=int(exts.max()),
           frames_touching_border=int(sum(r["touch"] for r in ok)),
           sharp_med=float(np.median([r["sharp"] for r in ok])), ref_sharp_med=float(np.median(ref_sharp)),
           bright_med=float(np.median([r["bright"] for r in ok])))
# preview grid: 8 overlays (top, height 270, aspect kept) + 8 cut-outs (bottom)
prev = sorted(glob.glob(f"{W}/screw_preview/*.jpg"))
def h270(im): return cv2.resize(im, (int(270 * im.shape[1] / im.shape[0]), 270))
top = np.hstack([h270(cv2.imread(p)) for p in prev[:8]]) if prev else None
sel = ok[:: max(1, len(ok) // 8)][:8]
bot = np.hstack([cv2.resize(r["cut"], (270, 270)) for r in sel])
if top is not None:
    wmax = max(top.shape[1], bot.shape[1])
    pad = lambda a: np.hstack([a, np.full((270, wmax - a.shape[1], 3), 255, np.uint8)]) if a.shape[1] < wmax else a
    grid = np.vstack([pad(top), pad(bot)])
else:
    grid = bot
cv2.imwrite(f"{ROOT}/data_build/home/{tag}_check.jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
json.dump(rep, open(f"{ROOT}/data_build/home/{tag}_check.json", "w"), indent=1)
verdict = []
if rep["frames"] < 60: verdict.append(f"too short: {rep['frames']} frames (<60)")
if rep["empty_masks"] > 0: verdict.append(f"{rep['empty_masks']} frames lost the object")
if rep["area_ratio"] > 4: verdict.append(f"mask area unstable (max/min={rep['area_ratio']:.1f})")
if rep["obj_px_min"] < 180: verdict.append(f"object too small in some frames ({rep['obj_px_min']} px @1920)")
if rep["frames_touching_border"] > 0.1 * len(ok): verdict.append(f"object cut by frame border in {rep['frames_touching_border']} frames")
if rep["sharp_med"] < 0.3 * rep["ref_sharp_med"]: verdict.append(f"blurry: sharpness {rep['sharp_med']:.0f} vs ref {rep['ref_sharp_med']:.0f}")
rep["verdict"] = "OK" if not verdict else "; ".join(verdict)
print(json.dumps({k: v for k, v in rep.items()}, indent=1))
