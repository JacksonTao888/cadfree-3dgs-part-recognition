#!/usr/bin/env bash
# Re-render all reconstructed clips with the current render_final.py ELEV_OFFSETS
# (now [0,+8,+15]). Render-only: reuses each clip's trained+pruned 3DGS model,
# no retraining. Run AFTER the 3DGS batch finishes (uses GPU briefly per clip).
set -uo pipefail
RUNS=${CADFREE_ROOT:-$HOME/cadfree_work}
PY_GS=~/miniconda3/envs/sugar/bin/python
RENDER=~/screw_recon/render_final.py
for d in "$RUNS"/C1*/; do
  ID=$(basename "$d")
  PLY="$d/gs_obj_hi/output/point_cloud/iteration_98/point_cloud.ply"
  [ -f "$PLY" ] || { echo "[skip] $ID (no pruned model)"; continue; }
  echo "[rerender] $ID  $(date +%H:%M:%S)"
  RECON_WORK="$d" $PY_GS "$RENDER" > "$RUNS/logs/${ID}_rerender.log" 2>&1
  echo "[done] $ID bands: $(ls -d "$d"/renders_final/elev_* 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
done
echo "RERENDER DONE"
