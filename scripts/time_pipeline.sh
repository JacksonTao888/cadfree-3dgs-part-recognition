#!/usr/bin/env bash
# Stage-timed re-run of the production reconstruction pipeline on ONE part (R2 comment 6).
# Usage: bash time_pipeline.sh <video.mp4> <workdir>
# Writes <workdir>/stage_times.txt (epoch-stamped stage lines) and <workdir>/gpu_mem.log (nvidia-smi poll, 2 s).
set -u
VIDEO=$1; WORK=$2; mkdir -p "$WORK"
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits)"; sleep 2; done ) > "$WORK/gpu_mem.log" 2>&1 &
POLL=$!
echo "$(date +%s) START" > "$WORK/stage_times.txt"
RECON_WORK="$WORK" GS_ITERS=7000 bash ~/screw_recon/run_pipeline.sh "$VIDEO" 2>&1 | while IFS= read -r line; do
  echo "$(date +%s) $line" >> "$WORK/stage_times.txt"
done
echo "$(date +%s) END" >> "$WORK/stage_times.txt"
kill $POLL 2>/dev/null
echo "done: $WORK/stage_times.txt"
