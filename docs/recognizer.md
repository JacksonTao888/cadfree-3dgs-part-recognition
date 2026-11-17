# CAD-Free 3DGS → Recognition Pipeline

End-to-end pipeline for **CAD-free few-shot / zero-shot recognition of fine-grained parts**:
a turntable video of each part is turned into a 3D Gaussian Splatting (3DGS) digital twin,
rendered into a synthetic *source* domain, and used to meta-train a recognizer (a small head
over a frozen DINOv2 backbone, classified by nearest view / 1-NN). The recognizer is then
tested on **real photographs from a separate video** of the same parts.

You will bring your **own** parts: one *reconstruction* video per part (call it **video A**)
and one *test* video per part (**video B**). Video A and B must be **different captures of the
same physical part** (this is the leakage control: never test on frames used to build the twin).

This README walks through every step. Run the steps in order. Each step says which **conda env**
to use, which **script** to run, what **inputs** it expects, and what **outputs** it produces.

---

## 0. What this produces (the big picture)

```
 video A (per part) ─run_pipeline.sh─► 3DGS twin ─► renders_final/  (synthetic source views)
                                                          │
 test video B (per part) ─assemble_realtest.py─► masked real frames │
                                                          ▼
                          recompose_cropped.py ─► data_3dgs_source/   (training source, cropped)
                                              └─► data_real_test/     (real test set, cropped)
                                                          │
                                       run_meta.py ─► accuracy (10-shot & zero-shot, P1/P2/P3)
                                                   └─► confusion.py / tsne.py (analysis figures)
```

---

## 1. Software you need (conda envs + tools)

The scripts call specific conda environments by their absolute python path. On this machine they are
under `~/miniconda3/envs/`. If your machine differs, **edit the path constants at the top of each
script** (listed in Section 2). The envs and what they provide:

| env | provides | used by |
|-----|----------|---------|
| `nerfstudio` | COLMAP binary + OpenCV (`cv2`) | camera poses, crop math |
| `sam3d-objects` | SAM2 (Segment Anything 2) | object/disk masks |
| `sugar` | 3D Gaussian Splatting trainer (`SuGaR/gaussian_splatting`) + `plyfile` | twin training + render |
| `sibr_viewer_build` | `ffmpeg` | frame extraction, orbit videos |
| `skill-meta` | PyTorch + `torch.hub` DINOv2 + `numpy`/`PIL` (+ `sklearn`, `matplotlib` after `pip install`) | training/testing + analysis figures |

External assets you must have:
- **COLMAP** (comes with the `nerfstudio` env here: `~/miniconda3/envs/nerfstudio/bin/colmap`).
- **SAM2 checkpoint + config** — referenced inside `~/screw_recon/sam2_*.py`. Keep the SAM2 weights where those scripts expect them.
- **SuGaR / gaussian_splatting repo** at `~/tools/SuGaR/gaussian_splatting` (the 3DGS `train.py`).
- **DTD textures** (Describable Textures Dataset) for background randomization, at
  `~/sim2real_industrial_review/datasets/dtd/dtd/images`. Only needed if you use the `*_bg` (random-background) variants; otherwise harmless if missing-ish (the glob just returns fewer/zero images — set a valid path or skip the `_bg` outputs).

If your DINOv2 download is blocked offline, pre-download the `facebookresearch/dinov2` hub weights once on a machine with internet (`torch.hub.load("facebookresearch/dinov2","dinov2_vitb14")`).

---

## 2. Paths you must edit for your machine / data

Every script hard-codes a few constants near the top. **Open each script and change these** to your
own locations before running:

- `~/screw_recon/run_pipeline.sh`: `SCRIPTS` (folder of helper .py), and the env binaries
  `PY_CV`, `PY_SAM`, `PY_GS`, `COLMAP`, `FF`, `GS_REPO`.
- `~/screw_recon/render_final.py`: reads `RECON_WORK` env var (no edit needed) but the virtual-camera
  intrinsics (`FX`, `W`, `H`) and `ELEV_OFFSETS = [0, +8, +15]` are near the top if you want to change views.
- `assemble_realtest.py`: `ROOT`, `TESTVID` (folder holding your test videos), `DTD`, `FF`, `PY_SAM`, `SAM_SCRIPT`, `FPS`.
- `recompose_cropped.py`: `ROOT`, `DTD`, `MARGIN` (crop padding, default 0.18).
- `build_rawframes_cropped.py`: `ROOT`.
- `run_meta.py`: `ROOT`.
- `confusion.py`, `tsne.py`: `ROOT` (+ they save figures to `$CADFREE_FIGOUT` — change that).

`ROOT` is the working directory that holds everything (here `$CADFREE_ROOT`). Keep all the
per-part reconstruction folders, the CSVs, and the `data_*` folders under `ROOT`.

---

## 3. Repository layout (what lives where)

```
~/screw_recon/                         # the 3DGS reconstruction toolkit
   run_pipeline.sh                     # video -> twin -> renders   (MAIN reconstruction script)
   est_focal.py  make_masks.py         # focal estimate; HSV "horizon" disk mask
   sam2_disk.py  sam2_screw.py  sam2_screw_hi.py   # SAM2 masks (disk for COLMAP, object for 3DGS)
   render_final.py                     # render the trained twin at 3 elevations x 120 az
   run_twin_fromframes.sh              # (advanced) build a twin from a chosen K-frame subset

~/recon_runs/   (= ROOT)
   clip_to_class.csv                   # maps each reconstruction folder -> class name   (video A)
   test_clip_to_class.csv              # maps each class -> its test video file           (video B)
   <CLIP_ID>/                          # one folder per reconstruction video (output of run_pipeline.sh)
       images/ images_hi/ masks_screw_hi/ colmap/ gs_obj_hi/ renders_final/
   assemble_realtest.py                # test videos -> extracted+masked real frames
   recompose_cropped.py               # -> data_3dgs_source/  and  data_real_test/  (cropped)
   build_rawframes_cropped.py          # (optional) raw masked frames as an alternative source
   run_meta.py                         # TRAIN + TEST: meta-head over frozen DINOv2, 1-NN
   run_matrix.py                       # (optional) no-meta ProtoNet baseline
   confusion.py  tsne.py               # analysis figures
   data_3dgs_source/<class>/*.jpg      # synthetic training source (object on white)
   data_real_test/<class>/*.jpg        # real test set (object on white)
   data_build/realtest/<class>/        # intermediate: extracted+masked test frames
```

---

## 4. STEP-BY-STEP

### Step A — Prepare your videos and the two CSV files

1. For each part, record **two** turntable videos: a reconstruction video (**A**) and a test video (**B**).
   - Turntable setup: the part sits on a **textured colored-grid disk** in front of a plain backdrop;
     the camera is **fixed**; the disk rotates one full turn. ~35 s is plenty. Higher resolution is better
     (we used 4096×2160). The disk texture is what COLMAP tracks — do **not** use a blank disk.
2. Put the reconstruction videos anywhere; you will point `run_pipeline.sh` at each one.
3. Put the test videos in one folder and set `TESTVID` in `assemble_realtest.py` to it
   (here: `~/turntable_videos_test/extracted/`).
4. Create the two CSVs in `ROOT`:

`clip_to_class.csv` — one row per reconstruction folder (you choose the folder IDs and class names):
```
clip,class
C113,gear_a
C114,gear_b
C115,nut_a
...
```

`test_clip_to_class.csv` — one row per class, giving its test video filename (inside `TESTVID`):
```
class,test_video
gear_a,Gear1.mp4
gear_b,Gear2.mp4
nut_a,Nut1.mp4
...
```
**Class names must match exactly** between the two CSVs. The class name prefix before the first
underscore (`gear`, `nut`, `screw`, …) defines the *category* used for the within-category protocols
(P2/P3) and the figure grouping, so name consistently (e.g. `screw_a`, `screw_b`, …).

### Step B — Reconstruct + render each part (3DGS)  [envs: nerfstudio, sam3d-objects, sugar]

Run **once per reconstruction video**. `RECON_WORK` is the output folder; name it with the CLIP_ID you
put in `clip_to_class.csv`, under `ROOT`:

```bash
cd ~/screw_recon
RECON_WORK=~/recon_runs/C113 GS_ITERS=7000 bash run_pipeline.sh /path/to/your/gear_a_recon.mp4 auto
```
- `GS_ITERS=7000` is enough for source-quality renders (faster than the 30000 default).
- Last arg `auto` estimates focal from the disk's vanishing points; pass a number (px at 1920 width)
  if you know it.
- If COLMAP registers too few frames or the twin looks wrong, the disk mask likely grabbed only the
  black center. Re-run with the HSV "horizon" mask that keeps the full grid:
  ```bash
  RECON_WORK=~/recon_runs/C113 GS_ITERS=7000 DISK_MASK=horizon bash run_pipeline.sh /path/to/gear_a_recon.mp4 auto
  ```

Outputs in `~/recon_runs/C113/`:
- `images/`, `images_hi/` (4K crop), `masks_screw_hi/` (object masks)
- `colmap/sparse/0/` (camera poses)
- `gs_obj_hi/output/point_cloud/iteration_98/point_cloud.ply` (the pruned twin)
- `renders_final/elev_{base,+8,+15}/*.png` — **360 synthetic source views** (3 elevations × 120 az). This is what the next step consumes.

Repeat for every part (C114, C115, …). ~18–20 min/part on one RTX 5000 Ada.

### Step C — Extract + mask the real test videos  [envs: sibr_viewer_build, sam3d-objects]

```bash
cd ~/recon_runs
~/miniconda3/envs/sam3d-objects/bin/python assemble_realtest.py
```
For each class in `test_clip_to_class.csv` it: extracts frames from the test video (fps=3 → ~100 frames),
runs SAM2 to mask the object, and writes them to `data_build/realtest/<class>/{images, masks_screw}`.
(It also writes an uncropped `data_real_test/`; the next step overwrites it with the cropped version.)

### Step D — Build the training source and the test set (cropped)  [env: skill-meta or any with numpy+PIL]

```bash
cd ~/recon_runs
~/miniconda3/envs/skill-meta/bin/python recompose_cropped.py
```
This produces, for every class:
- `data_3dgs_source/<class>/*.jpg` — the 3DGS renders, **object-bbox square-cropped** on white.
- `data_real_test/<class>/*.jpg` — the real test frames, cropped the same way (so source and test framing match — this matters).
- `*_bg/` variants = same objects composited on random DTD textures (for the background-robustness ablation; ignore if not needed).

(Optional) raw-frames source baseline — use the masked **real video-A frames** as the source instead of 3DGS renders:
```bash
~/miniconda3/envs/skill-meta/bin/python build_rawframes_cropped.py   # -> data_rawframes_source/
```

### Step E — Train + test the recognizer  [env: skill-meta]

```bash
cd ~/recon_runs
~/miniconda3/envs/skill-meta/bin/python run_meta.py --cls nn
```
This meta-trains a small head over a **frozen DINOv2 ViT-B/14** on `data_3dgs_source`, then evaluates on
`data_real_test`. `--cls nn` uses the **1-NN (nearest-view)** rule (recommended; `--cls proto` uses a
mean prototype). It prints, for each protocol, three numbers:

```
META over DINOv2 | source=data_3dgs_source test=data_real_test shots=10
protocol           support-only    source-only    source+fewshot
P1_all(18way)      91.6±..         75.5±..        91.9±..
P2_screw(8way)     ...
P3_nut(8way)       ...
```
- **support-only** = 10 real labeled shots per class (the 10-shot number).
- **source-only** = zero real labels (prototypes from the source only) — the **zero-shot** number.
- **source+fewshot** = source ∪ 10 real shots.
- Protocols: **P1** = all classes; **P2** = screws only; **P3** = nuts only (within-category, the hard tests).
  P2/P3 are auto-built from class-name prefixes; rename them in the `protos` dict in `run_meta.py` if your
  categories differ (e.g. `bolt`, `washer`).

Useful flags: `--shots 5`, `--seed 1` (run a few seeds and average — the zero-shot number varies across
seeds), `--variant resnet18` (weaker backbone, for the ablation), `--source data_rawframes_source`
(raw-frames source), `--classes-csv subset.csv` (restrict to a subset of classes).

`--save-head FILE` writes the trained head and `--load-head FILE` evaluates a saved head without
meta-training. `--seed` fixes the episodes and the head's initial weights, so a run can be repeated exactly.

(Optional) no-meta baseline: `~/miniconda3/envs/skill-meta/bin/python run_matrix.py`.

### Step F — Analysis figures (optional)  [env: skill-meta, needs matplotlib + sklearn]

```bash
~/miniconda3/envs/skill-meta/bin/pip install matplotlib   # once, if missing
~/miniconda3/envs/skill-meta/bin/python confusion.py      # -> Figure_confusion.{png,pdf}
~/miniconda3/envs/skill-meta/bin/python tsne.py           # -> Figure_tsne.{png,pdf}
```
`confusion.py` writes 18-way confusion matrices (10-shot & zero-shot) ordered by category; `tsne.py`
writes a t-SNE of the real-test features before vs after the meta-head. Both save to the `figures/`
path set near the top of those scripts — change it to your own.

### Step G — Use the trained head  [env: skill-meta]

The heads trained on the 18 FGIP parts are in `weights/`, with the parts' reference views stored as DINOv2
features, so this step needs neither the data nor the steps above:

```bash
python recognizer/predict.py examples/*.jpg              # label images
python recognizer/predict.py --eval data_real_test       # accuracy on a folder laid out as <class>/<images>
python recognizer/predict.py --refs my_refs photo.jpg    # your own reference images, my_refs/<class>/<images>
```

`weights/README.md` has the accuracy of each head and the details.

---

## 5. Things that will bite you (read this)

- **Leakage.** Never let a reconstruction (video-A) frame appear in the test set. Keep video A and video B
  as separate captures. `clip_to_class.csv` is video A; `test_clip_to_class.csv` is video B.
- **Coplanar scene → fix the focal.** All trackable texture is on the flat disk, so COLMAP cannot
  self-calibrate focal length. `run_pipeline.sh` fixes the intrinsics and uses the **exhaustive** matcher.
  Do not switch to sequential matching or enable focal refinement.
- **SIFT is grayscale.** The disk's *colors* do not help matching; its *geometric grid* does. Use a
  patterned (grid) disk; if SAM2's disk mask grabbed only the black center, use `DISK_MASK=horizon`.
- **3DGS pruning.** The pipeline prunes background Gaussians by a spatial bounding box, never by a color
  threshold (a color threshold hollows out dark hex-socket interiors). The pruned model is saved as
  `iteration_98`.
- **Single ring = limited views.** The underside on the disk is never seen, and renders streak beyond
  ~+30° elevation, so usable novel views are within ~±15° of the capture angle. Keep test photos near the
  capture elevation.
- **Use 1-NN, not the mean prototype** (`--cls nn`). For multi-view objects, averaging 360 source views
  into one prototype washes out the discriminative views; nearest-view is much better.
- **A strong frozen backbone matters.** DINOv2 ≫ ResNet-18 here; do not swap the backbone for a weak one.
- **Crop consistency.** Source and test are object-bbox square-cropped with the same `MARGIN`; if you
  change it, rebuild both `data_3dgs_source` and `data_real_test` together.
- **Honest expectation.** With a *full* video, training on the raw masked frames is as good as or better
  than the 3DGS renders for accuracy; the twin's value is controllability (novel views / backgrounds) and
  the label-free (zero-shot) source. Treat the 3DGS source as one option, not a guaranteed accuracy win.

---

## 6. Minimal end-to-end recipe (copy/paste, adapt paths)

```bash
# (A) reconstruct each part  [run once per video]
cd ~/screw_recon
RECON_WORK=~/recon_runs/C113 GS_ITERS=7000 bash run_pipeline.sh /data/recon/partA.mp4 auto
RECON_WORK=~/recon_runs/C114 GS_ITERS=7000 bash run_pipeline.sh /data/recon/partB.mp4 auto
# ... all parts ...

# (B) write clip_to_class.csv and test_clip_to_class.csv in ~/recon_runs, put test videos in TESTVID

cd ~/recon_runs
# (C) extract + mask test videos
~/miniconda3/envs/sam3d-objects/bin/python assemble_realtest.py
# (D) build cropped source + test sets
~/miniconda3/envs/skill-meta/bin/python recompose_cropped.py
# (E) train + test
~/miniconda3/envs/skill-meta/bin/python run_meta.py --cls nn
# (F) optional analysis
~/miniconda3/envs/skill-meta/bin/python confusion.py
~/miniconda3/envs/skill-meta/bin/python tsne.py
```

Questions about a specific script: open it — they are short and the constants/flags are at the top.
