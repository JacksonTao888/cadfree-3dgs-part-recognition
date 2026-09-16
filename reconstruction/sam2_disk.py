import os, glob, numpy as np, cv2, torch
from sam2.build_sam import build_sam2_video_predictor

HOME = os.environ.get("RECON_WORK") or os.path.expanduser("~/screw_recon")
SCRIPTS = os.path.expanduser("~/screw_recon")
FRAMES = f"{HOME}/sam2_frames"          # reuse the 1920px integer-named symlinks
CKPT = f"{SCRIPTS}/sam2_ckpt/sam2.1_hiera_large.pt"
CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
MASK_OUT = f"{HOME}/masks_disk"
PREV_OUT = f"{HOME}/disk_preview"
os.makedirs(MASK_OUT, exist_ok=True); os.makedirs(PREV_OUT, exist_ok=True)

imgs = sorted(glob.glob(f"{HOME}/images/frame_*.jpg"))
names = [os.path.splitext(os.path.basename(f))[0] for f in imgs]
H, W = cv2.imread(imgs[0]).shape[:2]
print(f"frames {len(imgs)}  {W}x{H}")

# COLOR-AGNOSTIC geometric prompt: positives on the disk (lower-center spread),
# negatives on the backdrop (top center + top corners). No HSV / no color assumption.
P = lambda fx, fy: [W*fx, H*fy]
# NOTE: segment the DISK only (color-agnostic geometric prompt). The screw is a
# spatially separate object — don't force it into this mask; segment it on its own
# (sam2_screw.py) and OR the two masks if you need disk+screw together.
pts = np.array([P(.5,.55), P(.5,.65), P(.3,.72), P(.7,.72), P(.5,.9),
                P(.2,.62), P(.8,.62),                       # disk (positive)
                P(.5,.03), P(.12,.05), P(.88,.05)], np.float32)  # backdrop (negative)
labels = np.array([1,1,1,1,1,1,1, 0,0,0], np.int32)

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
            img = cv2.imread(imgs[fidx]); ov = img.copy(); ov[~m] = (0, 0, 255)  # red = removed
            cv2.imwrite(f"{PREV_OUT}/disk_{fidx:04d}.jpg",
                        cv2.resize(cv2.addWeighted(img, 0.55, ov, 0.45, 0), (960, 506)))
print(f"masks written: {len(areas)}  disk-area % of frame min/med/max = "
      f"{100*min(areas)/(W*H):.0f}/{100*np.median(areas)/(W*H):.0f}/{100*max(areas)/(W*H):.0f}")
