#!/bin/bash
# Design 40 v2 (§11): joint box in every MPPI; K2' + E1/E2 as registered in §10. Run from ~/simdist on gpu1.
#   bash scripts/toy/stage40_e1v2.sh all      (whole pipeline, GPUs 0-3, each GPU <= 11 GB)
set -u
cd ~/simdist
D=results/stage40/e12v2; W1=/mnt/newssd/cjkim/toy/e12; W=/mnt/newssd/cjkim/toy/e12v2; mkdir -p $D $W
BOX="--box_hip 0.4 --box_thigh 0.8 --box_knee 0.8"
run() { GPUS="\"device=$1\"" ~/simdist-docker/run.sh bash -c "cd /workspace/simdist && export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=/mnt/newssd/cjkim/toy_pylibs:scripts/toy && $2"; }
T="python scripts/toy/toy_wm.py"
data_of() { case $1 in P) echo "$W1/d_P.npz";; PN) echo "$W1/d_PN.npz";; R) echo "$W1/d_R.npz";; ME|MC) echo $W/d_MB_s{1,2,3,4}.npz;; esac; }
which_of() { case $1 in ME) echo exec;; MC) echo cand;; *) echo exec,cand;; esac; }
mpc_chunk() { run $1 "$T genmpc --n_ep 25 --T 500 --cand 4 --seed $2 $BOX --out $W/d_MB_s$2.npz" > $D/log_data_M$2.txt 2>&1; }
anchor() { run $1 "$T genmpc --cond $2 --n_ep 8 --T 300 --cand 16 --seed 9 $BOX --out $W/anchor_$2.npz" > $D/log_anchor_$2.txt 2>&1; }
train() { run $1 "$T train --data $(data_of $2) --which $(which_of $2) --max_trans 50000 --steps 30000 --out $W/wm_$2.pkl" > $D/log_train_$2.txt 2>&1; }
e1() {  # GPU source
  for c in mot100 kneeRR0 tow40; do run $1 "$T rank --wm $W/wm_$2.pkl --anchors $W/anchor_$c.npz --out $D/rank_${2}_$c.json"; done
  run $1 "$T evalcl --wm $W/wm_$2.pkl --cond nominal $BOX --out $D/cl_${2}_nominal_before"
}
e2() {  # GPU source
  for c in kneeRR0 kneeFR0 tow40; do
    run $1 "$T evalcl --wm $W/wm_$2.pkl --cond $c $BOX --out $D/cl_${2}_${c}_before && \
            $T train --init $W/wm_$2.pkl --data $D/cl_${2}_${c}_before_trans.npz --which exec,cand --max_trans 8000 --steps 5000 \
                     --replay $(data_of $2) --replay_which $(which_of $2) --replay_frac 0.5 --out $W/wm_${2}_${c}.pkl && \
            $T evalcl --wm $W/wm_${2}_${c}.pkl --cond $c $BOX --out $D/cl_${2}_${c}_after && \
            $T evalcl --wm $W/wm_${2}_${c}.pkl --cond nominal $BOX --out $D/cl_${2}_${c}_afternom"
  done
}
chain() { G=$1; shift; for s in "$@"; do e1 $G $s; e2 $G $s; done > $D/log_e12_g$G.txt 2>&1; }
case $1 in
k2)
  run $2 "python scripts/toy/toy_mppi.py --cond nominal --arm free $BOX --out results/stage40/e0_freebox_nominal" > $D/log_k2.txt 2>&1 ;;
all)
  ( bash $0 k2 0 ) &                                              # K2' (decides whether E1/E2 are read)
  ( train 0 P; train 0 PN; train 0 R ) &                          # PPO-side sources reuse the v1 data
  ( mpc_chunk 1 1; mpc_chunk 1 4 ) &
  ( mpc_chunk 2 2; anchor 2 kneeRR0; anchor 2 mot100 ) &
  ( mpc_chunk 3 3; anchor 3 tow40 ) &
  wait
  ( train 1 ME ) & ( train 2 MC ) & wait
  chain 0 P & chain 1 ME & chain 2 MC & chain 3 PN R & wait
  run 0 "python scripts/toy/stage40_e12_report.py --d $D --md $D/e12.md" > $D/log_report.txt 2>&1 ;;
esac
