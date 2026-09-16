"""Reconstruction fidelity of every 3DGS twin at its OWN training cameras (fit quality, not held-out).
For each clip: render iteration_98 (pruned twin) at the training cameras via gaussian_splatting/render.py,
then compare render vs the masked training frame inside the object's (dilated) mask bounding box.
Reports per-part PSNR / SSIM (object-box) and mean mask-IoU-free silhouette agreement.
Run with the sugar env:  ~/miniconda3/envs/sugar/bin/python render_quality.py
"""
import os, csv, glob, subprocess, sys, math, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

ROOT = os.path.expanduser("~/recon_runs")
GS = os.path.expanduser("~/tools/SuGaR/gaussian_splatting")
PY = os.path.expanduser("~/miniconda3/envs/sugar/bin/python")
DEV = "cuda"
OUT_CSV = f"{ROOT}/logs/render_quality.csv"

def gauss(win=11, sigma=1.5):
    g = torch.tensor([math.exp(-(x - win // 2) ** 2 / (2 * sigma ** 2)) for x in range(win)])
    g = (g / g.sum()).unsqueeze(1); w = (g @ g.t()).float()
    return w.expand(3, 1, win, win).contiguous().to(DEV)
W = gauss()

def ssim(a, b):  # a,b: 3xHxW in [0,1]
    a, b = a.unsqueeze(0), b.unsqueeze(0)
    mu_a = F.conv2d(a, W, padding=5, groups=3); mu_b = F.conv2d(b, W, padding=5, groups=3)
    sa = F.conv2d(a * a, W, padding=5, groups=3) - mu_a ** 2
    sb = F.conv2d(b * b, W, padding=5, groups=3) - mu_b ** 2
    sab = F.conv2d(a * b, W, padding=5, groups=3) - mu_a * mu_b
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (sa + sb + C2))
    return m.mean().item()

def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.).permute(2, 0, 1).to(DEV)

def bbox_from(gt, pad=0.05):
    m = (gt.sum(0) > 0.02)
    ys, xs = torch.where(m)
    if len(ys) == 0: return None
    y0, y1, x0, x1 = ys.min().item(), ys.max().item(), xs.min().item(), xs.max().item()
    H, Wd = m.shape; ph, pw = int(pad * (y1 - y0)), int(pad * (x1 - x0))
    return max(0, y0 - ph), min(H, y1 + ph + 1), max(0, x0 - pw), min(Wd, x1 + pw + 1)

def eval_clip(clip):
    model = f"{ROOT}/{clip}/gs_obj_hi/output"
    rdir = f"{model}/train/ours_98"
    if not os.path.isdir(f"{rdir}/renders"):
        subprocess.run([PY, "render.py", "-m", model, "--iteration", "98", "--skip_test"], cwd=GS,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    rs = sorted(glob.glob(f"{rdir}/renders/*.png")); gs_ = sorted(glob.glob(f"{rdir}/gt/*.png"))
    assert len(rs) == len(gs_) and len(rs) > 0, clip
    ps, ss, sil = [], [], []
    for r, g in zip(rs, gs_):
        R, G = load(r), load(g)
        bb = bbox_from(G)
        if bb is None: continue
        y0, y1, x0, x1 = bb
        Rc, Gc = R[:, y0:y1, x0:x1], G[:, y0:y1, x0:x1]
        mse = ((Rc - Gc) ** 2).mean().item()
        ps.append(10 * math.log10(1.0 / max(mse, 1e-10))); ss.append(ssim(Rc, Gc))
        mr, mg = (Rc.sum(0) > 0.02), (Gc.sum(0) > 0.02)
        sil.append(((mr & mg).sum() / (mr | mg).sum().clamp(min=1)).item())
    return len(ps), float(np.mean(ps)), float(np.mean(ss)), float(np.mean(sil))

if __name__ == "__main__":
    rows = [r for r in csv.DictReader(open(f"{ROOT}/clip_to_class.csv"))]
    only = sys.argv[1:]  # optional subset of clips
    out = []
    for r in rows:
        if only and r["clip"] not in only: continue
        n, p, s, iou = eval_clip(r["clip"])
        out.append(dict(clip=r["clip"], cls=r["class"], n_views=n, psnr_box=round(p, 2), ssim_box=round(s, 4), sil_iou=round(iou, 4)))
        print(out[-1], flush=True)
    with open(OUT_CSV, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print("saved", OUT_CSV)
