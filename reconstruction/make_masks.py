import cv2, numpy as np, os, sys, glob

_WORK = os.environ.get("RECON_WORK") or os.path.expanduser("~/screw_recon")
IMG_DIR = os.path.join(_WORK, "images")
MASK_DIR = os.path.join(_WORK, "masks")
PREVIEW_DIR = os.path.join(_WORK, "mask_preview")
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

PREVIEW_ONLY = "--preview" in sys.argv

T = 20  # consecutive non-green rows that confirm "we've reached the disk"

def disk_start_cols(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = ((H >= 30) & (H <= 95) & (S >= 25) & (V >= 20))
    nong = (~green).astype(np.int32)
    h, w = nong.shape
    csum = np.vstack([np.zeros((1, w), np.int32), np.cumsum(nong, axis=0)])
    win = csum[T:, :] - csum[:h - T + 1, :]
    is_start = (win == T)
    return np.where(is_start.any(axis=0), is_start.argmax(axis=0), h), (h, w)

files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
preview_idx = {0, len(files)//4, len(files)//2, 3*len(files)//4, len(files)-1}

# pass 1: per-column disk-start over all frames -> robust static horizon
starts = []
shape = None
for f in files:
    s, shape = disk_start_cols(cv2.imread(f))
    starts.append(s)
starts = np.array(starts)                    # (N, w)
horizon = np.median(starts, axis=0)          # (w,) static rim, robust to per-frame stripes
horizon = cv2.medianBlur(horizon.astype(np.float32).reshape(1, -1), 1).ravel()
# smooth across columns (moving average) to kill residual spikes
k = 31
horizon = np.convolve(horizon, np.ones(k)/k, mode="same")
horizon = np.clip(horizon + 8, 0, shape[0]).astype(np.int32)  # push 8px below rim

h, w = shape
rows = np.arange(h)[:, None]
wall = (rows < horizon[None, :]).astype(np.uint8)   # one static mask for all frames
keep = ((1 - wall) * 255).astype(np.uint8)

# pass 2: write identical mask per frame + previews
for i, f in enumerate(files):
    name = os.path.splitext(os.path.basename(f))[0]
    if not PREVIEW_ONLY:
        cv2.imwrite(os.path.join(MASK_DIR, name + ".jpg.png"), keep)
    if i in preview_idx:
        bgr = cv2.imread(f)
        ov = bgr.copy()
        ov[wall > 0] = (0, 0, 255)
        blend = cv2.addWeighted(bgr, 0.5, ov, 0.5, 0)
        cv2.imwrite(os.path.join(PREVIEW_DIR, f"prev_{i:04d}.jpg"), cv2.resize(blend, (960, 540)))

print("done. masks:", 0 if PREVIEW_ONLY else len(files), "previews:", len(preview_idx))
