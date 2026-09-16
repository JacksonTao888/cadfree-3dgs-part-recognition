# CAD-Free Fine-Grained Industrial Part Recognition via 3D Gaussian Splatting

Code for the paper *CAD-Free Fine-Grained Industrial Part Recognition via 3D Gaussian Splatting*,
by Chenxi Tao, Hong-In Won and Seung-Kyum Choi.
Georgia Institute of Technology, and the Korea Institute of Industrial Technology.
Under revision at *Machine Vision and Applications* (Springer).

**Reconstruct → render → recognize.** From one short turntable video per part, we fit a 3D Gaussian
Splatting twin, render it from 360 viewpoints to manufacture a labelled training set that never
existed, and meta-train a small head over a frozen DINOv2 backbone that recognizes a real part by its
most similar reference view. No CAD model, no manual labels, no fine-tuning of the large network.

## The problem

High-mix low-volume production lines have to tell apart many near-identical parts: screw, nut and
gear variants that share surface finish and silhouette and differ only in thread pitch, socket depth
or tooth count. The usual remedy, rendering synthetic training data from CAD, assumes a CAD model
exists, which frequently it does not for legacy, third-party, proprietary or worn parts. We study the
CAD-unavailable but scan-available setting: the class set is known and fixed, a short turntable video
of each part can be captured, and few or no labelled real images are available.

## Abstract

> Fine-grained industrial parts such as screw, nut, and gear variants differ only in subtle
> threading, tooth count, or dimensions, and CAD models and large labeled datasets are rarely
> available in production. We present a CAD-free, video-based recognizer that distinguishes
> near-identical parts from a single short turntable video per part, with no CAD or manual labels.
> From the video we reconstruct a 3D Gaussian Splatting (3DGS) twin of each part and render it from
> 360 viewpoints as a label-free training source. A compact two-layer head, meta-trained over a
> frozen DINOv2 backbone with a 1-nearest-neighbor (1-NN)-consistent objective, classifies a real
> part by its most similar reference view. On a self-collected 18-part dataset, evaluated on a
> separately captured test video from the same rig to control leakage, it reaches 91.6% accuracy
> from 10 real shots and 75.5% with zero real labels; meta-learning and a strong frozen backbone are
> both decisive. We compare the renders with the video's own masked frames, the CAD-free alternative
> that needs no reconstruction. On the same rig the frames are the stronger source, by 2.4 points at
> 10 shots and 6.7 points zero-shot with matched view counts, because fine thread relief is not
> reproduced faithfully. In a hand-held phone capture under different lighting, surface, and
> viewpoints the order reverses: the renders lead by 11.7 points zero-shot with matched views and by
> 14 points on nuts, and labeled shots taken on the rig do not transfer.

## Method

**1. From one video to a twin.** A single continuous turntable pass, 4096×2160 at 50 fps, about 38 s,
with the part rotating on a printed coloured-grid disk. The disk matters: the parts are matte and
textureless, so the feature matcher borrows the disk's texture to recover camera motion. We sample
146–181 frames at 4.5 fps. COLMAP runs with exhaustive matching and a pinhole camera; because the
scene is near-planar, self-calibration is degenerate, so the focal length is estimated once from the
grid's vanishing points and held fixed with the principal point. All frames register at about 1 px
reprojection error. SAM2's video predictor propagates an object mask from one automatic prompt, and
masking happens *after* pose because the pose stage needs the disk. The part is re-extracted at
native 4K and cropped so it fills the frame, then fitted as an object-level 3DGS model for 7,000
iterations, followed by a purely geometric bounding-box prune.

**2. From the twin to a training set.** We fit the real camera ring, its centre, axis and radius, from
the COLMAP poses, so the virtual orbit sits where the camera actually was. We render look-at orbits at
three elevations, the capture elevation, +8° and +15°, 120 views each, for 360 labelled views per
part. Renders can optionally be composited onto randomized natural-texture backgrounds.

**3. The recognizer.** A frozen DINOv2 ViT-B/14 maps any 224×224 image to a 768-dimensional feature
and is never updated. The only trained component is a two-layer MLP head of about 525K parameters
whose output is L2-normalized. The objective is 1-NN-consistent: a class's logit is the log-sum-exp
of the similarities to that class's individual views, which approaches the maximum as the learnable
scale grows, so the training loss is the smoothed form of the test-time rule. A mean-prototype
(ProtoNet) loss, which averages a class's views and blurs away the viewpoints that separate
look-alike parts, is kept only as an ablation. Training is episodic on the renders alone, with zero
real labels. At test time everything is frozen and a query is labelled by its single nearest
reference view.

**Leakage control.** Each part is filmed twice on the same rig. Video A is the only video the twin is
fitted to; video B supplies every real test image, so no test image ever contributed to a
reconstruction.

## Results

The FGIP set is 18 self-3D-printed parts, 2 gears, 8 nuts, 8 screws, near-identical within a family.
P1 is all 18 classes (chance 5.6%), P2 the 8 screws and P3 the 8 nuts (chance 12.5%). Every number is
a mean over 300 episodes and 3 seeds.

**Headline, on the capture rig**

| Protocol | 10-shot | zero-shot |
|---|---|---|
| P1 (18-way) | 91.6 | 75.5 |
| P2 (screws) | 91.0 | 74.0 |
| P3 (nuts) | 90.7 | 76.0 |
| no-meta ProtoNet, P1 | 75.2 | 61.7 |
| linear probe on the 10 shots, P1 | 79.3 | n/a |

**Renders against the video's own frames, matched at 120 views, P1**

| Source | Same rig (10-shot / zero-shot) | Hand-held phone (10-shot / zero-shot) |
|---|---|---|
| Raw masked frames | 94.3 / 83.7 | 63.6 / 64.1 |
| 3DGS renders | 91.8 / 76.9 | 73.4 / 75.8 |

On the rig the frames win, because fine thread relief is not reproduced. In a different environment
the order reverses, and ten labelled shots taken on the rig do not transfer (55.3% as a reference set,
48.5% for a linear probe, both below either label-free source).

**Reconstruction fidelity does not predict recognition.** Across the 18 parts the Spearman correlation
between twin SSIM and zero-shot accuracy is about −0.08 and for PSNR about 0.03, neither significant.
The worst-reconstructed part in the set, `screw h` at 19.9 dB and 0.852 SSIM, is recognized 76.2% of
the time; `screw b`, reconstructed at 27.6 dB and 0.947 SSIM, collapses to 7.3%. A local,
relief-sensitive gradient measure does correlate, at about −0.8 over the eight screws. Per-part
numbers are in [`results/render_quality.csv`](results/render_quality.csv) and the confusion matrices
behind them in [`results/confusion/`](results/confusion/).

**Cost per part.** Reconstruction 7.0 min on one RTX 5000 Ada (3DGS training 4.8 min, 6.3 GB peak).
Recognizer inference 5.0 ms per image; meta-training 4 s on cached features.

## Repository layout

```
reconstruction/   video → focal estimate → SAM2 masks → 3DGS fit → 360-view orbit
data_prep/        renders and frames → cropped, class-foldered source and test sets
recognizer/       run_meta.py is the method; the no-meta, linear-probe and backbone baselines
analysis/         reconstruction fidelity, inference cost, confusion matrices, symmetry index
figures/          scripts that regenerate the paper's figures
scripts/          batch drivers for reconstruction and for the experiment matrix
results/          small curated outputs: per-part fidelity, confusion matrices, cross-environment runs
docs/             pipeline notes, environment setup, reproduction map
```

## Setup

```bash
pip install -r requirements.txt
```

That covers the recognizer and the figures. The reconstruction half needs COLMAP, SAM2 and a CUDA
3DGS rasterizer, which are not pip-installable; see [docs/environments.md](docs/environments.md).
No path is hard-coded: every script reads `CADFREE_ROOT`, `CADFREE_VIDEOS`, `CADFREE_TESTVID`,
`DTD_DIR` and `CADFREE_FIGOUT`, documented in the same file.

## What this repository does and does not give you

It gives you the method: the full reconstruction pipeline, the recognizer, the baselines, the analysis
and figure code, and the small result files behind the paper's tables.

It does not give you the data. The FGIP videos, frames, masks and twins are about 19 GB and are not
published here; the dataset remains available from the corresponding author on reasonable request.
You can therefore run the pipeline end to end on your own turntable capture, but you cannot reproduce
the paper's exact numbers from this repository alone. [docs/reproducing.md](docs/reproducing.md) maps
each table and figure to the script that produced it and is explicit about the gaps, including a few
steps that were run ad hoc and whose drivers did not survive.

## Citation

```bibtex
@article{tao2026cadfree,
  title   = {CAD-Free Fine-Grained Industrial Part Recognition via 3D Gaussian Splatting},
  author  = {Tao, Chenxi and Won, Hong-In and Choi, Seung-Kyum},
  journal = {Machine Vision and Applications},
  note    = {Under revision},
  year    = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE). Third-party components (SAM2, the 3DGS rasterizer, DTD) are not
redistributed here and keep their own terms.
