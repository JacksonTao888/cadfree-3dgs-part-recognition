"""Background-kept square crops of a home video WITHOUT SAM2: per frame, locate the part as the largest
bright blob (V > median+50, no saturation constraint), crop a square of 2.5x its extent (min 500 px at
4K), background kept.  Writes data_home_test_rawbg/<cls>/.   usage: python crop_home_nomask.py <cls>"""
import os, sys, glob, cv2, numpy as np
cls = sys.argv[1]; ROOT = os.environ.get("CADFREE_ROOT", os.path.expanduser("~/cadfree_work"))
W = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/data_build/home/{cls}/images"
out = sys.argv[3] if len(sys.argv) > 3 else f"{ROOT}/data_home_test_rawbg/{cls}"; os.makedirs(out, exist_ok=True)
for f in glob.glob(f"{out}/*.jpg"): os.remove(f)
n = 0
for ip in sorted(glob.glob(f"{W}/frame_*.jpg")):
    im = cv2.imread(ip); H, Wd = im.shape[:2]; V = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 2]
    cand = (V > np.median(V) + 90).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)); cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    k, lbl, st, cen = cv2.connectedComponentsWithStats(cand, 8)
    def score(j):
        a = st[j, cv2.CC_STAT_AREA]
        if not (0.0003 * H * Wd < a < 0.05 * H * Wd): return -1
        d = (((cen[j][0] - Wd / 2) / Wd) ** 2 + ((cen[j][1] - H / 2) / H) ** 2) ** 0.5
        return -1 if d > 0.4 else a * (1 - d / 0.4)
    ok = [j for j in range(1, k) if score(j) > 0]
    if not ok: print("no blob", os.path.basename(ip)); continue
    i = max(ok, key=score)
    x, y, w, h = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP], st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
    side = int(max(2.5 * max(w, h), 500 * max(H, Wd) / 3840)); cx, cy = x + w // 2, y + h // 2
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2); x1, y1 = min(Wd, x0 + side), min(H, y0 + side)
    cv2.imwrite(f"{out}/{os.path.basename(ip)}", im[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 92]); n += 1
print(f"{cls}: {n} background-kept crops -> {out}")
