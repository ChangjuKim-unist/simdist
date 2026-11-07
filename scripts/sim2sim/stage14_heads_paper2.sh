#!/bin/bash
# Design doc 15 local supplement L2 follow-up: retrain the go1_paper heads with the 13 NAS traces + 8 new local traces
# (results/stage14/traces_paper on gpu1), one per held-out axis.  Run on gpu1: GPU=3 bash scripts/sim2sim/stage14_heads_paper2.sh
cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage14; TR=/nas_data/cjkim/simdist/traces_stage13; TP=$OUT/traces_paper; mkdir -p $OUT
GPU=${GPU:-3}
head() { local name=$1 axis=$2; shift 2
  [ -f $OUT/$name.json ] && { echo "== $name exists"; return; }
  echo "== head $name (holdout $axis) $(date +%H:%M)"
  GPUS="\"device=$GPU\"" XLA_PYTHON_CLIENT_PREALLOCATE=false $RUN python scripts/sim2sim/calib_head.py --traces "$@" --holdout_axis $axis --out $OUT/$name 2>&1 | grep -v "WARNING\|cuStream\|^$" | tail -14
}
head head_paper2_noF friction "$TR/T_paper_*.npz" "$TR/A_paper*.npz" "/workspace/simdist/$TP/T_paper_*.npz"
head head_paper2_noM motor    "$TR/T_paper_*.npz" "$TR/A_paper*.npz" "/workspace/simdist/$TP/T_paper_*.npz"
echo "### paper heads done $(date)"
