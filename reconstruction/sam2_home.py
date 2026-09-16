"""SAM2 video masking for HOME (hand-held, textured background) captures.
Same outputs as sam2_screw.py (masks_screw/<frame>.jpg.png, screw_preview/), but the
frame-0 prompt is chosen robustly: among low-saturation/high-value blobs, take the one
closest to the image centre (area-weighted); if none is plausible, prompt at the centre.
"""
import os, glob, numpy as np, cv2, torch
from sam2.build_sam import build_sam2_video_predictor

HOME = os.environ["RECON_WORK"]
SCRIPTS = os.path.expanduser("~/screw_recon")
FRAMES = f"{HOME}/sam2_frames"
CKPT = f"{SCRIPTS}/sam2_ckpt/sam2.1_hiera_large.pt"
CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
MASK_OUT = f"{HOME}/masks_screw"; PREV_OUT = f"{HOME}/screw_preview"
os.makedirs(MASK_OUT, exist_ok=True); os.makedirs(PREV_OUT, exist_ok=True)

imgs = sorted(glob.glob(f"{HOME}/images/frame_*.jpg"))
names = [os.path.splitext(os.path.basename(f))[0] for f in imgs]
f0 = cv2.imread(imgs[0]); H, W = f0.shape[:2]
hsv = cv2.cvtColor(f0, cv2.COLOR_BGR2HSV); S, V = hsv[..., 1], hsv[..., 2]
# object = clearly BRIGHTER than the frame's median (the desk) and unsaturated
v_bg, s_bg = np.median(V), np.median(S)
def find(s_max, v_gain):
    cand = ((V > v_bg + v_gain) & (S < s_max)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(cand, 8)
    best, best_score = None, -1
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < 0.0005 * H * W or a > 0.3 * H * W:
            continue
        d = (((cent[i][0] - W / 2) / W) ** 2 + ((cent[i][1] - H / 2) / H) ** 2) ** 0.5
        if d > 0.35:
            continue
        score = a * (1.0 - d / 0.35)
        if score > best_score:
            best, best_score = i, score
    return best, lbl, stats
# run all three stages, keep the LARGEST plausible blob (warm light makes the part yellowish -> strict stage only finds highlights)
cands = []
for s_max, v_gain in ((max(60, 0.8 * s_bg), 35), (120, 50), (160, 70)):
    best, lbl, stats = find(s_max, v_gain)
    if best is not None:
        cands.append((int(stats[best, cv2.CC_STAT_AREA]), s_max, v_gain, best, lbl, stats))
if cands:
    area, s_max, v_gain, best, lbl, stats = max(cands, key=lambda t: t[0])
    x, y, w, h = stats[best, cv2.CC_STAT_LEFT], stats[best, cv2.CC_STAT_TOP], stats[best, cv2.CC_STAT_WIDTH], stats[best, cv2.CC_STAT_HEIGHT]
    mx, my = 0.25 * w, 0.25 * h
    box = np.array([max(0, x - mx), max(0, y - my), min(W - 1, x + w + mx), min(H - 1, y + h + my)], np.float32)
    ys, xs = np.where(lbl == best); pts = np.array([[xs.mean(), ys.mean()]], np.float32); labels = np.array([1], np.int32)
    print(f"finder: S<{s_max:.0f} V>bg+{v_gain} (bg V={v_bg:.0f} S={s_bg:.0f}) blob area={area} box={box.astype(int).tolist()}")
else:
    box = None; pts = np.array([[W / 2, H / 2]], np.float32); labels = np.array([1], np.int32)
    print("finder: nothing plausible -> single centre point")

predictor = build_sam2_video_predictor(CFG, CKPT, device="cuda")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(video_path=FRAMES)
    predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, points=pts, labels=labels, box=box)
    areas = []
    prev_idx = set(np.linspace(0, len(imgs) - 1, 8).astype(int).tolist())
    for fidx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        m = (mask_logits[0] > 0.0).cpu().numpy()[0]
        cv2.imwrite(f"{MASK_OUT}/{names[fidx]}.jpg.png", m.astype(np.uint8) * 255)
        areas.append(int(m.sum()))
        if fidx in prev_idx:
            img = cv2.imread(imgs[fidx]); ov = img.copy(); ov[m] = (0, 0, 255)
            blend = cv2.addWeighted(img, 0.55, ov, 0.45, 0)
            cv2.imwrite(f"{PREV_OUT}/screw_{fidx:04d}.jpg", cv2.resize(blend, (960, int(960 * H / W))))
print(f"masks written: {len(areas)}  area px min/med/max = {min(areas)}/{int(np.median(areas))}/{max(areas)}")
