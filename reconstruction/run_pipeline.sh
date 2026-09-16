#!/usr/bin/env bash
# =============================================================================
# Turntable video  ->  object-only 3D Gaussian Splatting  ->  renders (+/-15deg)
# Single-ring pipeline. SAM2 is used for BOTH masks (disk for COLMAP, object for 3DGS).
# Processes ONE video into $WORK (overwrites). See README.md for the full rationale.
# =============================================================================
set -euo pipefail

# ----------------------------- config ----------------------------------------
VIDEO=${1:-~/screw_a.mp4}                 # input video
SCRIPTS=~/screw_recon                     # helper .py scripts + SAM2 ckpt live here
WORK=${RECON_WORK:-~/screw_recon}         # per-object working/output dir (RECON_WORK to override)
export RECON_WORK="$WORK"
mkdir -p "$WORK"
FOCAL=${2:-auto}                          # known focal in px, or "auto" (vanishing points)
ITERS=${GS_ITERS:-30000}                   # 3DGS training iterations (GS_ITERS to override; 7000 is plenty for source renders)
DISK_MASK=${DISK_MASK:-sam2}               # COLMAP disk mask: 'sam2' (default) or 'horizon' (make_masks.py HSV, keeps full colored grid -> use when sam2 grabbed only the black center square)
FPS=4.5                                   # frame sampling -> ~180 frames for one revolution

PY_CV=~/miniconda3/envs/nerfstudio/bin/python      # python + OpenCV
PY_SAM=~/miniconda3/envs/sam3d-objects/bin/python  # python + SAM2
PY_GS=~/miniconda3/envs/sugar/bin/python           # python + 3DGS rasterizer
COLMAP=~/miniconda3/envs/nerfstudio/bin/colmap
FF=~/miniconda3/envs/sibr_viewer_build/bin/ffmpeg
GS_REPO=~/tools/SuGaR/gaussian_splatting
cd "$WORK"

# ---------------------- 1. extract frames (1920) ------------------------------
echo "[1/9] extract frames"
mkdir -p images
$FF -i "$VIDEO" -vf "fps=$FPS,scale=1920:-1" -q:v 2 -y images/frame_%04d.jpg -loglevel error
read W1 H1 < <($PY_CV -c "import cv2,glob;im=cv2.imread(sorted(glob.glob('images/frame_*.jpg'))[0]);print(im.shape[1],im.shape[0])")
echo "    $(ls images | wc -l) frames @ ${W1}x${H1}"

# ---------------------- 2. estimate focal (px) --------------------------------
echo "[2/9] focal"
if [ "$FOCAL" = "auto" ]; then
  FOCAL=$($PY_CV "$SCRIPTS/est_focal.py" 2>/dev/null | grep -oE "median=[0-9.]+" | cut -d= -f2 | cut -d. -f1)
fi
CX=$(( W1 / 2 )); CY=$(( H1 / 2 ))
echo "    focal=$FOCAL px  cx=$CX cy=$CY  (PINHOLE, fixed)"

# ---------------------- 3. SAM2 turntable (disk) mask -------------------------
if [ "$DISK_MASK" = "horizon" ]; then
  echo "[3/9] disk mask (HSV horizon, keeps full colored grid) -> masks/"
  $PY_CV "$SCRIPTS/make_masks.py" >/dev/null 2>&1
  DISKMASK_DIR=masks
else
  echo "[3/9] SAM2 disk mask  -> masks_disk/"
  rm -rf sam2_frames; mkdir -p sam2_frames
  $PY_CV -c "import glob,os
for i,f in enumerate(sorted(glob.glob('$WORK/images/frame_*.jpg'))): os.symlink(f,f'$WORK/sam2_frames/{i}.jpg')"
  $PY_SAM "$SCRIPTS/sam2_disk.py" >/dev/null 2>&1
  DISKMASK_DIR=masks_disk
fi

# ---------------------- 4. COLMAP poses (fixed intrinsics) --------------------
echo "[4/9] COLMAP poses"
DB=colmap/database.db; rm -rf colmap; mkdir -p colmap/sparse
$COLMAP feature_extractor --database_path $DB --image_path images --ImageReader.mask_path "$DISKMASK_DIR" \
  --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE \
  --ImageReader.camera_params "$FOCAL,$FOCAL,$CX,$CY" \
  --SiftExtraction.use_gpu 1 --SiftExtraction.max_num_features 16384 --SiftExtraction.peak_threshold 0.004 >/dev/null 2>&1
$COLMAP exhaustive_matcher --database_path $DB --SiftMatching.use_gpu 1 >/dev/null 2>&1
$COLMAP mapper --database_path $DB --image_path images --output_path colmap/sparse \
  --Mapper.multiple_models 0 --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 >/dev/null 2>&1
$COLMAP model_converter --input_path colmap/sparse/0 --output_path colmap/sparse/0 --output_type TXT >/dev/null 2>&1
echo "    registered: $($COLMAP model_analyzer --path colmap/sparse/0 2>&1 | grep -i 'Registered' )"

# ---------------------- 5. SAM2 object mask (1920, locate object) -------------
echo "[5/9] SAM2 object mask  -> masks_screw/"
$PY_SAM "$SCRIPTS/sam2_screw.py" >/dev/null 2>&1

# ---------------------- 6. compute 4K crop window from object bbox ------------
echo "[6/9] 4K crop + re-extract"
# Compute the 4K crop window (around the object's swept bbox) and the cropped
# intrinsics. Reuses the §4 extrinsics; only intrinsics change:
#   scale  s = 4096/W1 ;  fx' = FOCAL*s ;  cx' = CX*s - x0 ;  cy' = CY*s - y0
eval $($PY_CV - <<PYEOF
import cv2,glob,numpy as np
X0,Y0,X1,Y1=1e9,1e9,0,0
for f in glob.glob('masks_screw/frame_*.jpg.png'):
    m=cv2.imread(f,0); ys,xs=np.where(m>0)
    if len(xs): X0=min(X0,xs.min());Y0=min(Y0,ys.min());X1=max(X1,xs.max());Y1=max(Y1,ys.max())
s=4096/$W1
cw=int(((X1-X0)*s+440)//2*2); ch=int(((Y1-Y0)*s+440)//2*2)        # +220px margin/side, even
cx=(X0+X1)/2*s; cy=(Y0+Y1)/2*s
x0=int(max(0,min(4096-cw, cx-cw/2))); y0=int(max(0,min(2160-ch, cy-ch/2)))
print(f"CROP=crop={cw}:{ch}:{x0}:{y0}; CW={cw}; CH={ch};"
      f" FXc={$FOCAL*s:.1f}; CXc={$CX*s-x0:.1f}; CYc={$CY*s-y0:.1f}")
PYEOF
)
echo "    $CROP  fx=$FXc cx=$CXc cy=$CYc  (size ${CW}x${CH})"
mkdir -p images_hi
$FF -i "$VIDEO" -vf "fps=$FPS,${CROP}" -q:v 2 -y images_hi/frame_%04d.jpg -loglevel error

# ---------------------- 7. SAM2 object mask on the 4K crop --------------------
echo "[7/9] SAM2 object mask (hi)  -> masks_screw_hi/"
rm -rf sam2_frames_hi; mkdir -p sam2_frames_hi
$PY_CV -c "import glob,os
for i,f in enumerate(sorted(glob.glob('$WORK/images_hi/frame_*.jpg'))): os.symlink(f,f'$WORK/sam2_frames_hi/{i}.jpg')"
$PY_SAM "$SCRIPTS/sam2_screw_hi.py" >/dev/null 2>&1

# ---------------------- 8. build hi scene (reuse extrinsics, scaled intrinsics)
echo "[8/9] black-out + 3DGS train + prune"
rm -rf gs_obj_hi; mkdir -p gs_obj_hi/sparse/0 gs_obj_hi/images
cp colmap/sparse/0/images.txt colmap/sparse/0/points3D.txt gs_obj_hi/sparse/0/
printf '# Camera list\n1 PINHOLE %d %d %s %s %s %s\n' "$CW" "$CH" "$FXc" "$FXc" "$CXc" "$CYc" > gs_obj_hi/sparse/0/cameras.txt
$PY_CV -c "import cv2,glob,os
for f in sorted(glob.glob('$WORK/images_hi/frame_*.jpg')):
    n=os.path.splitext(os.path.basename(f))[0]
    m=cv2.imread(f'$WORK/masks_screw_hi/{n}.jpg.png',0); img=cv2.imread(f); img[m==0]=0
    cv2.imwrite(f'$WORK/gs_obj_hi/images/{n}.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,97])"
# train + render
( cd $GS_REPO && $PY_GS train.py -s $WORK/gs_obj_hi -m $WORK/gs_obj_hi/output \
    --port 6040 -r 1 --iterations $ITERS --save_iterations $ITERS >/dev/null 2>&1 )

# spatial prune of the black-background gaussians -> iteration_98
$PY_CV - <<PYEOF
import numpy as np,os
from plyfile import PlyData,PlyElement
p=PlyData.read('gs_obj_hi/output/point_cloud/iteration_${ITERS}/point_cloud.ply'); v=p['vertex']
C0=0.28209479177387814
dc=np.stack([np.array(v['f_dc_0']),np.array(v['f_dc_1']),np.array(v['f_dc_2'])],1)
lum=np.clip(0.5+C0*dc,0,1)@[0.299,0.587,0.114]; op=1/(1+np.exp(-np.array(v['opacity'])))
xyz=np.stack([np.array(v['x']),np.array(v['y']),np.array(v['z'])],1)
seed=(lum>0.45)&(op>0.3); s=xyz[seed]; lo=np.percentile(s,1,0); hi=np.percentile(s,99,0)
ctr=(lo+hi)/2; half=(hi-lo)/2*1.15; keep=np.all(np.abs(xyz-ctr)<=half,1)&(op>0.05)
os.makedirs('gs_obj_hi/output/point_cloud/iteration_98',exist_ok=True)
PlyData([PlyElement.describe(v.data[keep],'vertex')]).write('gs_obj_hi/output/point_cloud/iteration_98/point_cloud.ply')
print(f'    pruned -> iteration_98  ({keep.sum()} gaussians)')
PYEOF

# ---------------------- 9. final renders: base + (+/-15deg) -------------------
echo "[9/9] final renders -> renders_final/  (base + +15deg orbits + videos)"
$PY_GS "$SCRIPTS/render_final.py" >/dev/null 2>&1
for d in renders_final/*/; do dn=$(basename "$d"); [ -d "$d" ] || continue
  $FF -framerate 24 -i "renders_final/$dn/%04d.png" -c:v libx264 -pix_fmt yuv420p -crf 18 -vf scale=900:-2 \
     -y "renders_final/orbit_$dn.mp4" -loglevel error
done

echo "DONE."
echo "  splat model : gs_obj_hi/output/point_cloud/iteration_98/point_cloud.ply"
echo "  3DGS renders: renders_final/elev_base/  (+ orbit_elev_base.mp4)"
echo "  +15 deg     : renders_final/elev_+15/   (+ orbit_elev_+15.mp4)"
