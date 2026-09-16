#!/usr/bin/env bash
PY=~/miniconda3/envs/skill-meta/bin/python
cd "${CADFREE_ROOT:-$HOME/cadfree_work}"
run(){ echo "[$(date +%H:%M:%S)] start $1"; $PY run_meta.py "${@:2}" > logs/exp_$1.log 2>&1; echo "[$(date +%H:%M:%S)] done $1"; }
run rawframes  --source data_rawframes_source --test data_real_test --meta-iters 2000 --episodes 300
run resnet     --variant resnet18 --source data_3dgs_source --test data_real_test --meta-iters 2000 --episodes 300
run nn         --cls nn --meta-iters 2000 --episodes 300
run shots1     --shots 1 --episodes 300
run shots5     --shots 5 --episodes 300
run seed1      --seed 1 --episodes 300
run seed2      --seed 2 --episodes 300
echo BATCH2DONE
