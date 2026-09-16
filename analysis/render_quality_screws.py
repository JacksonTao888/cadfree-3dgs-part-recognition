"""Per-render quality metrics for the screw classes of data_3dgs_source (training side only):
sharp = Laplacian variance at 224 px; stray = fraction of mask area outside the largest component;
sim = max cosine similarity (DINOv2) to the same part's masked video-A frames; band = elevation band.
Writes logs/home/render_quality_screws.csv and a contact sheet per class of the worst renders."""
import os, sys, glob, csv, numpy as np, torch, torch.nn.functional as F
from scipy import ndimage
sys.path.insert(0, os.path.expanduser("~/recon_runs"))
from run_meta import feats_for_dir, DEV, ROOT
from PIL import Image, ImageDraw, ImageFont
S = "/tmp/claude-3340513/-home-ctao40/53a271d8-933b-47db-a0c8-32462697cd9f/scratchpad"
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).to(DEV).eval()
screws = [f"screw_{k}" for k in "abcdefgh"]; rows = []
Fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
for cls in screws:
    paths = sorted(glob.glob(f"{ROOT}/data_3dgs_source/{cls}/*.jpg"))
    rf = feats_for_dir(model, f"{ROOT}/data_3dgs_source/{cls}", cap=None); ff = feats_for_dir(model, f"{ROOT}/data_rawframes_source/{cls}", cap=None)
    sim = (rf @ ff.t()).max(1).values.numpy()
    for p, s in zip(paths, sim):
        pim = Image.open(p).convert("RGB"); g = np.array(pim.resize((224, 224), Image.BOX).convert("L"), dtype=np.float64)
        lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]; sharp = lap.var()
        m = np.array(pim).min(2) < 245; lbl, n = ndimage.label(m, structure=np.ones((3, 3)))
        areas = sorted(ndimage.sum(m, lbl, range(1, n + 1)), reverse=True) if n > 0 else [1]
        stray = 1 - areas[0] / max(1, sum(areas))
        rows.append(dict(cls=cls, file=os.path.basename(p), band=os.path.basename(p).split("_0")[0], sharp=float(sharp), stray=float(stray), sim=float(s)))
    R = [r for r in rows if r["cls"] == cls]
    # contact sheet: row 1 = 10 lowest sim, row 2 = 10 lowest sharpness, row 3 = 10 highest stray, row 4 = 10 BEST (highest sim) for reference
    T = 150; sheet = Image.new("RGB", (10 * T, 4 * (T + 30)), "white"); d = ImageDraw.Draw(sheet)
    for ri, (lab, sel) in enumerate((("lowest similarity to frames", sorted(R, key=lambda r: r["sim"])[:10]), ("lowest sharpness", sorted(R, key=lambda r: r["sharp"])[:10]),
                                     ("most stray splats", sorted(R, key=lambda r: -r["stray"])[:10]), ("BEST (highest similarity)", sorted(R, key=lambda r: -r["sim"])[:10]))):
        for k, r in enumerate(sel):
            sheet.paste(Image.open(f"{ROOT}/data_3dgs_source/{cls}/{r['file']}").convert("RGB").resize((T, T)), (k * T, ri * (T + 30)))
            d.text((k * T + 3, ri * (T + 30) + T + 2), f"{r['file'][:-4]} s{r['sim']:.2f} L{r['sharp']:.0f} x{r['stray']:.2f}", fill="black", font=Fnt)
        d.text((3, ri * (T + 30) + T + 15), lab, fill="#c00", font=Fnt)
    sheet.save(f"{S}/rq_{cls}.jpg", quality=88)
    a = np.array([r["sim"] for r in R]); b = np.array([r["sharp"] for r in R]); c = np.array([r["stray"] for r in R])
    print(f"{cls}: sim mean {a.mean():.3f} min {a.min():.3f} p10 {np.percentile(a,10):.3f} | sharp mean {b.mean():.0f} min {b.min():.0f} | stray>0.02: {(c>0.02).sum()} imgs | by band sim: " +
          ", ".join(f"{bd} {np.mean([r['sim'] for r in R if r['band']==bd]):.3f}" for bd in ("elev_base","elev_+8","elev_+15")), flush=True)
with open(f"{ROOT}/logs/home/render_quality_screws.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("RQ DONE")
