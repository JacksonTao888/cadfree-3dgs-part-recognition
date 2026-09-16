import os, glob, numpy as np, cv2, torch
from sam2.build_sam import build_sam2_video_predictor

HOME = os.environ.get("RECON_WORK") or os.path.expanduser("~/screw_recon")
SCRIPTS = os.path.expanduser("~/screw_recon")
FRAMES = f"{HOME}/sam2_frames_hi"
CKPT = f"{SCRIPTS}/sam2_ckpt/sam2.1_hiera_large.pt"
CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
MASK_OUT = f"{HOME}/masks_screw_hi"
PREV_OUT = f"{HOME}/screw_preview_hi"
os.makedirs(MASK_OUT, exist_ok=True); os.makedirs(PREV_OUT, exist_ok=True)

imgs = sorted(glob.glob(f"{HOME}/images_hi/frame_*.jpg"))
names = [os.path.splitext(os.path.basename(f))[0] for f in imgs]

# locate screw in frame 0: white/gray = low saturation, high value
f0 = cv2.imread(imgs[0]); hsv = cv2.cvtColor(f0, cv2.COLOR_BGR2HSV)
S, V = hsv[..., 1], hsv[..., 2]
cand = ((S < 50) & (V > 140)).astype(np.uint8)
cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
n, lbl, stats, cent = cv2.connectedComponentsWithStats(cand, 8)
big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) if n > 1 else 0
ys, xs = np.where(lbl == big)
cxs, cys = xs.mean(), ys.mean()
pts = np.array([[cxs, cys],
                [xs.min() + 0.15 * (xs.max() - xs.min()), ys[np.argmin(xs)]],
                [xs.max() - 0.15 * (xs.max() - xs.min()), ys[np.argmax(xs)]]], np.float32)
labels = np.array([1, 1, 1], np.int32)
print(f"frame0 {f0.shape[1]}x{f0.shape[0]}  screw blob area={int(stats[big,cv2.CC_STAT_AREA])} pts={np.round(pts).tolist()}")

predictor = build_sam2_video_predictor(CFG, CKPT, device="cuda")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(video_path=FRAMES)
    predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, points=pts, labels=labels)
    areas = []
    for fidx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        m = (mask_logits[0] > 0.0).cpu().numpy()[0]
        cv2.imwrite(f"{MASK_OUT}/{names[fidx]}.jpg.png", m.astype(np.uint8) * 255)
        areas.append(int(m.sum()))
        if fidx in (0, len(imgs)//2, len(imgs)-1):
            img = cv2.imread(imgs[fidx]); ov = img.copy(); ov[m] = (0, 0, 255)
            cv2.imwrite(f"{PREV_OUT}/screw_{fidx:04d}.jpg",
                        cv2.resize(cv2.addWeighted(img, 0.55, ov, 0.45, 0), (900, 600)))
print(f"masks written: {len(areas)}  area px min/med/max = "
      f"{min(areas)}/{int(np.median(areas))}/{max(areas)}")
