import cv2, numpy as np, glob, os

_WORK = os.environ.get("RECON_WORK") or os.path.expanduser("~/screw_recon")
files = sorted(glob.glob(os.path.join(_WORK, "images/*.jpg")))
# use frames where board is clearly oblique (spread across the ring)
pick = [files[i] for i in range(0, len(files), max(1, len(files)//20))]

def lines_of(gray):
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        segs = lsd.detect(gray)[0]
        if segs is None: return None
        return segs.reshape(-1, 4)
    except Exception:
        edges = cv2.Canny(gray, 50, 150)
        hl = cv2.HoughLinesP(edges, 1, np.pi/180, 60, minLineLength=40, maxLineGap=5)
        return None if hl is None else hl.reshape(-1, 4)

def vp(lines):
    # homogeneous line through 2 pts: l = p1 x p2
    L = []
    for x1, y1, x2, y2 in lines:
        p1 = np.array([x1, y1, 1.]); p2 = np.array([x2, y2, 1.])
        L.append(np.cross(p1, p2))
    L = np.array(L)
    # least-squares VP = null space of stacked lines (min eigvec of L^T L)
    _, _, Vt = np.linalg.svd(L)
    v = Vt[-1]
    return v / v[2] if abs(v[2]) > 1e-9 else None

focals = []
for f in pick:
    g = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    cx, cy = w/2, h/2
    segs = lines_of(g)
    if segs is None or len(segs) < 20:
        continue
    # filter short segments, compute angles
    dirs = segs[:, 2:4] - segs[:, 0:2]
    lens = np.linalg.norm(dirs, axis=1)
    keep = lens > 25
    segs, dirs = segs[keep], dirs[keep]
    ang = (np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0])) % 180)
    # cluster into 2 dominant orientations via histogram peaks
    hist, edges = np.histogram(ang, bins=36, range=(0, 180))
    order = np.argsort(hist)[::-1]
    peaks = []
    for o in order:
        c = (edges[o] + edges[o+1]) / 2
        if all(min(abs(c-p), 180-abs(c-p)) > 25 for p in peaks):
            peaks.append(c)
        if len(peaks) == 2: break
    if len(peaks) < 2: continue
    vps = []
    for p in peaks:
        d = np.minimum(np.abs(ang-p), 180-np.abs(ang-p))
        grp = segs[d < 15]
        if len(grp) < 8: break
        v = vp(grp)
        if v is None: break
        vps.append(v[:2])
    if len(vps) < 2: continue
    v1 = np.array(vps[0]) - [cx, cy]; v2 = np.array(vps[1]) - [cx, cy]
    val = -np.dot(v1, v2)               # f^2 = -(v1-p).(v2-p) for orthogonal dirs
    if val > 0:
        focals.append(np.sqrt(val))

focals = np.array(focals)
if len(focals):
    print(f"n={len(focals)}  focal(px) median={np.median(focals):.1f}  "
          f"mean={focals.mean():.1f}  std={focals.std():.1f}")
    print("samples:", np.round(np.sort(focals)).astype(int))
else:
    print("VP focal estimation failed (insufficient line structure)")
