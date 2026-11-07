#!/bin/bash
# Design doc 15 stage A: train the calibrated heads on the design-14 traces (NAS copy), one per held-out axis.
cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage14; TR=/nas_data/cjkim/simdist/traces_stage13; mkdir -p $OUT
GPU=${GPU:-3}
head() { local name=$1 axis=$2; shift 2
  [ -f $OUT/$name.json ] && { echo "== $name exists"; return; }
  echo "== head $name (holdout $axis) $(date +%H:%M)"
  GPUS="\"device=$GPU\"" $RUN python scripts/sim2sim/calib_head.py --traces "$@" --holdout_axis $axis --out $OUT/$name 2>&1 | grep -v "WARNING\|cuStream\|^$" | tail -12
}
head head_broadwm_noF friction "$TR/T_broadwm_*.npz" "$TR/G_broadwm_*.npz" "$TR/S_broadwm_*.npz"
head head_broadwm_noM motor    "$TR/T_broadwm_*.npz" "$TR/G_broadwm_*.npz" "$TR/S_broadwm_*.npz"
head head_broadwm_noA mass     "$TR/T_broadwm_*.npz" "$TR/G_broadwm_*.npz" "$TR/S_broadwm_*.npz"
head head_paper_noF friction   "$TR/T_paper_*.npz" "$TR/A_paper*.npz"
head head_paper_noM motor      "$TR/T_paper_*.npz" "$TR/A_paper*.npz"
echo "### heads done $(date)"
