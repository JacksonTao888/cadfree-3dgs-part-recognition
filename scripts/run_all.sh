#!/usr/bin/env bash
# Batch turntable->3DGS over all extracted clips. Sequential (GPU-bound).
# Idempotent: skips a clip whose final renders already exist.
set -uo pipefail
VIDEOS=${CADFREE_VIDEOS:-$HOME/turntable_videos/extracted}
RUNS=${CADFREE_ROOT:-$HOME/cadfree_work}
PIPE=${CADFREE_PIPE:-$(dirname "$0")/../reconstruction/run_pipeline.sh}
mkdir -p "$RUNS/logs"

for V in "$VIDEOS"/*.mp4; do
  ID=$(basename "$V" | grep -oE "C1[0-9][0-9]")
  W="$RUNS/$ID"
  if [ -d "$W/renders_final/elev_base" ] && [ "$(ls "$W/renders_final/elev_base" 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "[skip] $ID (renders exist)"; continue
  fi
  echo "[run ] $ID  $(date +%H:%M:%S)"
  RECON_WORK="$W" bash "$PIPE" "$V" auto > "$RUNS/logs/$ID.log" 2>&1
  rc=$?
  reg=$(grep -aoE "Registered images: [0-9]+" "$RUNS/logs/$ID.log" | tail -1)
  echo "[done] $ID exit=$rc  $reg  $(date +%H:%M:%S)"
done
echo "ALL DONE"
