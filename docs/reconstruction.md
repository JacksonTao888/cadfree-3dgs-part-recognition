# Turntable → 3D Gaussian Splatting: Single-Ring Object Reconstruction

Reconstruct a single object captured on a **fixed-camera turntable** (object + a random
colored-grid disk rotating in front of a green backdrop) into an **object-only 3D Gaussian
Splatting (3DGS) model**.

This document covers the **single-ring** pipeline end to end: from the raw video to the final
isolated-object mask and the trained 3DGS model. Multi-ring / flip-and-merge for a full 360°
model is **not** covered here (planned separately).

Example used throughout: `screw_a.mp4` — a white hex-socket screw, 4096×2160 @ 50 fps, ~40 s.

---

## 0. TL;DR pipeline

```
video.mp4
  │  ffmpeg                         (1) extract frames
  ▼
images/ (1920px, N frames)
  │  make_masks.py                  (2) TURNTABLE MASK  → remove static green wall, keep disk+object
  ▼
masks/  (COLMAP keep/ignore masks)
  │  est_focal.py                   (3) estimate focal (vanishing points)
  ▼
focal ≈ 5142 px
  │  COLMAP (fixed intrinsics)      (4) camera poses  → clean planar ring, 181/181 registered
  ▼
colmap/sparse/0/  (poses + sparse points)
  │  sam2_screw.py                  (5) OBJECT MASK  → SAM2 video segmentation of the object
  ▼
masks_screw/  (per-frame object masks)
  │  black-out + train.py           (6) object-only 3DGS  → render
  ▼
gs_obj/output/  (point_cloud.ply = the splat model)
  │  spatial prune                  (7) remove black background gaussians
  ▼
clean object-only .ply
```

The two mask steps are the heart of the pipeline and are documented in detail in
[§2 (turntable mask)](#2-turntable-mask--make_maskspy) and [§5 (object mask)](#5-object-mask-sam2--sam2_screwpy).

---

## Setup — requirements (conda envs already on this machine)

| Tool | Path |
|------|------|
| ffmpeg / ffprobe | `~/miniconda3/envs/sibr_viewer_build/bin/ffmpeg` |
| Python + OpenCV (frames, masks, geometry) | `~/miniconda3/envs/nerfstudio/bin/python` (cv2 4.12) |
| COLMAP | `~/miniconda3/envs/nerfstudio/bin/colmap` |
| SAM 2.1 (object masks) | `~/miniconda3/envs/sam3d-objects/bin/python` (`pip install sam2`) |
| SAM 2.1 checkpoint | `~/screw_recon/sam2_ckpt/sam2.1_hiera_large.pt` |
| 3DGS (original INRIA code) | `~/miniconda3/envs/sugar/bin/python` + repo `~/tools/SuGaR/gaussian_splatting` |
| GPU | NVIDIA RTX 5000 Ada (32 GB) |

All work lives under `~/screw_recon/` (the scripts expect this path).

**One-time setup** (SAM2 package + checkpoint are NOT in this archive — fetch them):
```bash
# SAM 2.1 into an env with torch>=2.5.1 (e.g. a fresh env or your sam env)
pip install --no-build-isolation "git+https://github.com/facebookresearch/sam2.git"
# checkpoint (~857 MB) -> the path the scripts read
mkdir -p ~/screw_recon/sam2_ckpt && cd ~/screw_recon/sam2_ckpt
curl -L -o sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
# COLMAP + 3DGS: COLMAP binary on PATH; clone the INRIA gaussian-splatting (or SuGaR's copy)
# and build its rasterizer; point GS_REPO in run_pipeline.sh at it.
```
Put your input video somewhere and pass it to `run_pipeline.sh`. Adjust the env paths at the
top of `run_pipeline.sh` to your machine.

---

## 1. Frame extraction

Sample ~180 frames evenly across the video (one full turntable revolution is plenty;
exhaustive matching later needs overlapping views, not a huge frame count).

```bash
FF=~/miniconda3/envs/sibr_viewer_build/bin/ffmpeg
mkdir -p ~/screw_recon/images
$FF -i ~/screw_a.mp4 -vf "fps=4.5,scale=1920:-1" -q:v 2 ~/screw_recon/images/frame_%04d.jpg
# -> 181 frames at 1920×1013
```

`fps=4.5` over a 40 s clip → 181 frames. **Use the same `fps` value if you re-extract**
(e.g. at 4K) so the frames correspond to the same timestamps and therefore the same poses.

---

## 2. Turntable mask — `make_masks.py`

> **This is the single most important pre-processing step.** Run it before COLMAP.

### Why
- The camera is fixed and the disk only **spins in place**, so the disk silhouette is a
  **static ellipse** in the image. Only the green backdrop is genuinely static *background*.
- The green cloth has folds = static texture. If we don't remove it, **COLMAP anchors on the
  static wall** and concludes the camera barely moves → reconstruction fails.
- In the splatting stage, the wall (which rotates in the disk-anchored world frame) would also
  produce floaters.

So we **keep the entire rotating disk + object** (its colored pattern is what COLMAP uses for
pose) and **remove only the green wall above the disk rim**.

### The trap (why a naive color key fails)
The disk itself contains green/teal squares. A plain "remove green pixels" key punches holes in
the disk. Connectivity-to-top also leaks through the rim. The robust method instead finds the
**rim line** and removes everything above it.

### Algorithm (per-column median horizon)
1. **Per frame, per image column:** walk down from the top row. Green = HSV
   `H∈[30,95] & S≥25 & V≥20`. Find the first row where **T=20 consecutive non-green** rows begin
   — that row is the disk rim for that column (`disk_start`). Everything above is wall.
2. **Across all N frames:** take the **median** `disk_start` per column. The disk silhouette is
   static, so the median is a stable rim; per-frame artifacts (e.g. a column that momentarily
   has stacked green squares near the rim) are washed out.
3. **Smooth across columns** (moving average, k=31) and push the line **8 px below** the rim to
   cover anti-aliasing fringe.
4. This yields **one static horizon** → `wall = rows above horizon`. The same mask is written
   for every frame.

Output: `masks/frame_XXXX.jpg.png`, **255 = keep (disk + object), 0 = ignore (wall)**.
The filename convention `<image_name>.jpg.png` is exactly what COLMAP's `--ImageReader.mask_path`
expects (mask = image filename + `.png`).

### Run
```bash
cd ~/screw_recon
~/miniconda3/envs/nerfstudio/bin/python make_masks.py            # writes all masks
~/miniconda3/envs/nerfstudio/bin/python make_masks.py --preview  # writes red-overlay previews only
# previews -> mask_preview/prev_*.jpg  (sanity-check the horizon before COLMAP)
```

**Always eyeball `mask_preview/` first.** Red = masked-out. You want the green wall fully red and
the entire disk (including its green squares, the black center, and the object) untouched.

### Tuning
- If the wall isn't fully removed → widen the green range or lower `T`.
- If a colored disk row gets cut → the median+smoothing usually fixes it; increase the smoothing
  kernel `k` if a thin stripe survives.

---

## 3. Focal-length estimation — `est_focal.py`

A flat disk is a **planar** scene, and the textureless object adds no off-plane points, so
**all triangulable points are coplanar**. Self-calibrating the focal length from coplanar points
is **degenerate** — COLMAP fails with *"No good initial image pair found."* The fix is to
**provide a known focal and fix it**.

We estimate it from the disk's regular grid using **vanishing points** (no checkerboard needed):

```bash
cd ~/screw_recon
~/miniconda3/envs/nerfstudio/bin/python est_focal.py
# -> focal(px) median ≈ 5142  (for 1920-wide frames)
```

Notes:
- The estimate is noisy (line segments are short); use the **median**.
- **Validation comes later**: if the reconstructed camera centers form a clean **circle** (not an
  ellipse), the focal was good enough. If you have the camera/lens, a proper checkerboard
  calibration is more reliable.

---

## 4. Camera poses — COLMAP (fixed intrinsics)

```bash
COLMAP=~/miniconda3/envs/nerfstudio/bin/colmap
cd ~/screw_recon
DB=colmap/database.db; mkdir -p colmap/sparse

# Features — PINHOLE with FIXED intrinsics (f, f, cx, cy) from §3, plus the turntable masks
$COLMAP feature_extractor --database_path $DB --image_path images \
  --ImageReader.mask_path masks \
  --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE \
  --ImageReader.camera_params "5142,5142,960,506" \
  --SiftExtraction.use_gpu 1 --SiftExtraction.max_num_features 16384 \
  --SiftExtraction.peak_threshold 0.004

# Exhaustive matching — gives the long, cross-rotation tracks the planar scene needs
$COLMAP exhaustive_matcher --database_path $DB --SiftMatching.use_gpu 1

# Mapper — keep intrinsics FIXED (do not refine focal/principal point on a planar scene)
$COLMAP mapper --database_path $DB --image_path images --output_path colmap/sparse \
  --Mapper.multiple_models 0 \
  --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0

# Convert to TXT (used by analysis / the hi-res step)
$COLMAP model_converter --input_path colmap/sparse/0 --output_path colmap/sparse/0 --output_type TXT
```

Expected result: **181/181 images registered, ~1 px reprojection error**, camera centers forming
a clean planar circle (verify by plotting `-R^T t` for each image).

**Gotchas that cost the most time:**
- SIFT works on **grayscale** — it cannot see the random *colors*; the grid is a repetitive
  lattice to it. **Exhaustive** matching is what makes it work (sequential gave tracks too short
  to grow the reconstruction).
- Do **not** add `--SiftExtraction.estimate_affine_shape/--domain_size_pooling` here — they
  crashed with the GPU extractor on this build.

---

## 5. Object mask (SAM2) — `sam2_screw.py`

> Runs **after** poses are solved. The object mask is used for the **3DGS** stage, **not** for
> COLMAP — COLMAP needs the whole disk for pose; a screw-only mask would break it.

### Why SAM2
The video makes SAM 2.1's **video predictor** ideal: prompt the object **once** on frame 0 and it
**propagates** a temporally coherent mask through all frames.

### Auto-prompt (no manual clicking)
The object is white/gray = **low saturation, high value**. The script:
1. Loads frame 0, thresholds `S<50 & V>140`, takes the **largest connected component** = object.
2. Uses its **centroid + two extreme points along the long axis** as positive prompts.

### Frame naming for SAM2
SAM2's `init_state(video_path=dir)` expects frames named as **integers** (`0.jpg`, `1.jpg`, …).
We symlink `images/frame_0001.jpg → sam2_frames/0.jpg`, etc., keeping a mapping back to the
COLMAP names so output masks are saved as `frame_XXXX.jpg.png`.

### Run
```bash
cd ~/screw_recon
# 1) integer-named symlinks for SAM2
mkdir -p sam2_frames
~/miniconda3/envs/nerfstudio/bin/python -c "
import glob,os
for i,f in enumerate(sorted(glob.glob(os.path.expanduser('~/screw_recon/images/frame_*.jpg')))):
    os.symlink(f, os.path.expanduser(f'~/screw_recon/sam2_frames/{i}.jpg'))"

# 2) segment + propagate
~/miniconda3/envs/sam3d-objects/bin/python sam2_screw.py
# -> masks_screw/frame_XXXX.jpg.png  (255 = object, 0 = background)
# -> screw_preview/screw_{0,mid,last}.jpg  (red overlay; check no drift)
```

The `cannot import name '_C'` warning is harmless (optional CUDA post-processing kernel; results
unaffected). **Check `screw_preview/`** to confirm the mask tracks the object across the whole
clip (first / middle / last frames).

---

## 6. Object-only 3DGS

Black out everything except the object (the repo has no mask-loss; baking black background is the
pragmatic equivalent), reuse the COLMAP poses, train, render.

```bash
cd ~/screw_recon
mkdir -p gs_obj/sparse/0

# black out non-object using the SAM2 masks
~/miniconda3/envs/nerfstudio/bin/python - <<'PY'
import cv2,glob,os
for f in sorted(glob.glob(os.path.expanduser('~/screw_recon/images/frame_*.jpg'))):
    n=os.path.splitext(os.path.basename(f))[0]
    m=cv2.imread(os.path.expanduser(f'~/screw_recon/masks_screw/{n}.jpg.png'),0)
    img=cv2.imread(f); img[m==0]=0
    os.makedirs(os.path.expanduser('~/screw_recon/gs_obj/images'),exist_ok=True)
    cv2.imwrite(os.path.expanduser(f'~/screw_recon/gs_obj/images/{n}.jpg'),img,[cv2.IMWRITE_JPEG_QUALITY,95])
PY
cp colmap/sparse/0/*.bin gs_obj/sparse/0/        # reuse poses + sparse init

# train (original INRIA 3DGS in the SuGaR repo)
cd ~/tools/SuGaR/gaussian_splatting
~/miniconda3/envs/sugar/bin/python train.py -s ~/screw_recon/gs_obj -m ~/screw_recon/gs_obj/output \
  --port 6020 --iterations 30000 --save_iterations 7000 30000
~/miniconda3/envs/sugar/bin/python render.py -m ~/screw_recon/gs_obj/output --iteration 30000 --skip_test
```

Output: `gs_obj/output/point_cloud/iteration_30000/point_cloud.ply` — **this IS the splat model**
(each "vertex" is a Gaussian: xyz + SH color + opacity + scale + rotation). Opening it in a plain
point-cloud viewer shows only the centers and looks fuzzy; view the **renders** (`output/train/
ours_30000/renders/`) or a splat viewer (SuperSplat / antimatter15) for the real result.

### Higher resolution (sharper) — recommended for small objects
The object filled only ~10 % of frame width, so at the default 1.6K training downscale it was
~165 px and looked soft. **Re-extract at native 4K and crop a fixed window around the object's
swept region** so it fills ~1367 px:

```bash
# union bbox of all object masks defines the crop; here crop=1800:1200:1095:659 in 4K coords
$FF -i ~/screw_a.mp4 -vf "fps=4.5,crop=1800:1200:1095:659" -q:v 2 images_hi/frame_%04d.jpg
# re-run SAM2 on these (sam2_screw_hi.py), black out -> gs_obj_hi/images
```
Reuse the **same extrinsics** (poses don't change with resolution/crop); only the **intrinsics**
change — scale focal/principal point by `4096/1920` then subtract the crop offset:
`f≈10969.6, cx=2048−1095=953, cy=1080−659=421`, size `1800×1200`. Write these into a new
`sparse/0/cameras.txt` (`PINHOLE 1800 1200 10969.6 10969.6 953 421`), keep `images.txt` /
`points3D.txt`, then train with **`-r 1`** (so 1800 px isn't auto-downscaled). See
`sam2_screw_hi.py` and the `gs_obj_hi/` scene.

---

## 7. Clean up the black-blob (post-process)

Black-background training leaves a blob of **black gaussians** (recolored disk-init points that
weren't pruned). Remove them by a **spatial** prune — define the object's bounding box from
bright+opaque gaussians and keep everything inside it (any color), drop the rest.
**Do not prune by color** — the dark hex-socket interior would be hollowed out.

See the prune snippet that produced `gs_obj_hi/output/point_cloud/iteration_98/point_cloud.ply`
(the clean model). Root-cause alternative (needs alpha-mask loss, not in this repo):
train with `--random_background`.

---

## 8. Novel-view rendering — `novel_views.py`

Renders the splat model from **camera poses not in the training set** (different elevation, zoom,
azimuth). It fits the camera ring (center, axis, radius) from the COLMAP poses, computes the
object center from the bright gaussians, and builds look-at cameras.

```bash
cd ~/screw_recon
~/miniconda3/envs/sugar/bin/python novel_views.py    # -> novel_renders/*.png
```

**Usable envelope (single ring):** training elevation ≈ 19°.
- Azimuth: full 360° (clean).
- Elevation: within **±15°** of the capture angle is clean; beyond ~+30° the under-observed top
  shows streak artifacts.
- The object **bottom** (the face on the disk) is never seen → always empty.
These limits are intrinsic to a single ring and motivate the flip / multi-elevation capture.

---

## File / directory layout

```
~/screw_recon/
├── README.md                 ← this file
├── make_masks.py             (2) turntable disk mask (green-wall removal)
├── est_focal.py              (3) vanishing-point focal estimate
├── sam2_screw.py             (5) SAM2 object mask  (1920px)
├── sam2_screw_hi.py          (5) SAM2 object mask  (4K crop)
├── novel_views.py            (8) novel-view rendering
├── sam2_ckpt/                SAM2.1 checkpoint
│
├── images/        | images_hi/        frames (1920 | 4K-crop)
├── masks/                            (2) COLMAP disk masks (255 keep / 0 wall)
├── mask_preview/                     (2) red-overlay sanity previews
├── masks_screw/   | masks_screw_hi/  (5) SAM2 object masks
├── screw_preview/ | screw_preview_hi/(5) object-mask overlays
├── colmap/sparse/0/                  (4) camera poses + sparse points
├── gs/                               (debug) full-scene 3DGS (disk + object)
├── gs_obj/        | gs_obj_hi/        (6) object-only 3DGS scene + output
│     └── output[/_fine]/point_cloud/iteration_*/point_cloud.ply   ← splat models
└── novel_renders/                    (8) novel-view stills
```

## Key parameters reference

| What | Value |
|------|-------|
| Frame sampling | `fps=4.5` → 181 frames |
| Turntable-mask green | HSV `H∈[30,95] S≥25 V≥20`; `T=20` rows; smooth `k=31`; `+8 px` |
| Focal (1920px) | `f≈5142`, `cx=960`, `cy=506`, model `PINHOLE` |
| Focal (4K crop 1800×1200) | `f≈10969.6`, `cx=953`, `cy=421` |
| COLMAP | exhaustive matcher; intrinsics fixed in mapper |
| SAM2 | `sam2.1_hiera_large`; auto-prompt `S<50 & V>140` largest blob |
| 3DGS | original INRIA code; `-r 1` for the 4K crop; 30k iters |
| Black-blob cleanup | spatial bbox prune (never color-only) |
```
