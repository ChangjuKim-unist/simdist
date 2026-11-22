#!/bin/bash
# Design 40 §10: E1/E2 pipeline on gpu1. Run from ~/simdist. Stages: data | train | e1 | e2 | report
set -u
cd ~/simdist
D=results/stage40/e12; W=/mnt/newssd/cjkim/toy/e12; mkdir -p $D $W
run() { GPUS="\"device=$1\"" ~/simdist-docker/run.sh bash -c "cd /workspace/simdist && XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=/mnt/newssd/cjkim/toy_pylibs:scripts/toy $2"; }
T="python scripts/toy/toy_wm.py"
SRC="P PN R ME MC"
data_of() { case $1 in P) echo "$W/d_P.npz";; PN) echo "$W/d_PN.npz";; R) echo "$W/d_R.npz";; ME|MC) echo $W/d_M_s{1,2,3,4}.npz;; esac; }
which_of() { case $1 in ME) echo exec;; MC) echo cand;; *) echo exec,cand;; esac; }
case $1 in
data)   # $2 = GPU list for the MPC chunks (4 chunks of 25 episodes), cheap sources on the first GPU
  G=($2)
  run ${G[0]} "$T gen --src ppo --n_ep 100 --T 500 --seed 11 --out $W/d_P.npz && $T gen --src ppo_noise --noise 0.6 --n_ep 100 --T 500 --seed 12 --out $W/d_PN.npz && $T gen --src rand --n_ep 500 --T 100 --seed 13 --out $W/d_R.npz" > $D/log_data_cheap.txt 2>&1
  for i in 1 2 3 4; do g=${G[$(( (i-1) % ${#G[@]} ))]}
    ( run $g "$T genmpc --n_ep 25 --T 500 --cand 4 --seed $i --out $W/d_M_s$i.npz" > $D/log_data_M$i.txt 2>&1 ) &
  done
  for c in mot100 kneeRR0 tow40; do g=${G[0]}
    ( run $g "$T genmpc --cond $c --n_ep 8 --T 300 --cand 16 --seed 9 --out $W/anchor_$c.npz" > $D/log_anchor_$c.txt 2>&1 ) &
  done
  wait ;;
train)  # $2 = GPU
  for s in $SRC; do run $2 "$T train --data $(data_of $s) --which $(which_of $s) --max_trans 50000 --steps 30000 --out $W/wm_$s.pkl" > $D/log_train_$s.txt 2>&1; done ;;
e1)     # $2 = GPU
  for s in $SRC; do
    for c in mot100 kneeRR0 tow40; do run $2 "$T rank --wm $W/wm_$s.pkl --anchors $W/anchor_$c.npz --out $D/rank_${s}_$c.json"; done
    run $2 "$T evalcl --wm $W/wm_$s.pkl --cond nominal --out $D/cl_${s}_nominal_before"
  done > $D/log_e1.txt 2>&1 ;;
e2)     # $2 = GPU, $3 = source list
  for s in $3; do for c in kneeRR0 kneeFR0 tow40; do
    run $2 "$T evalcl --wm $W/wm_$s.pkl --cond $c --out $D/cl_${s}_${c}_before && \
            $T train --init $W/wm_$s.pkl --data $D/cl_${s}_${c}_before_trans.npz --which exec,cand --max_trans 8000 --steps 5000 \
                     --replay $(data_of $s) --replay_which $(which_of $s) --replay_frac 0.5 --out $W/wm_${s}_${c}.pkl && \
            $T evalcl --wm $W/wm_${s}_${c}.pkl --cond $c --out $D/cl_${s}_${c}_after && \
            $T evalcl --wm $W/wm_${s}_${c}.pkl --cond nominal --out $D/cl_${s}_${c}_afternom"
  done; done > $D/log_e2_g$2.txt 2>&1 ;;
report)
  run 0 "python scripts/toy/stage40_e12_report.py --d $D --md $D/e12.md" ;;
esac
