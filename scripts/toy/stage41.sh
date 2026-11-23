#!/bin/bash
# Design 41 (docs/design41_coverage_matched_ko.md): coverage-matched data arms E / EN / M. Run from ~/simdist on gpu1:
#   nohup bash scripts/toy/stage41.sh all > /mnt/newssd/cjkim/toy/e41_all.log 2>&1 &
set -u
cd ~/simdist
D=results/stage41; W=/mnt/newssd/cjkim/toy/e41; V2=/mnt/newssd/cjkim/toy/e12v2; mkdir -p $D $W
BOX="--box_hip 0.4 --box_thigh 0.8 --box_knee 0.8"
run() { GPUS="\"device=$1\"" ~/simdist-docker/run.sh bash -c "cd /workspace/simdist && export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=/mnt/newssd/cjkim/toy_pylibs:scripts/toy && $2"; }
T="python scripts/toy/toy_wm.py"
data_of() { case $1 in E) echo "$W/d_E_s1.npz $W/d_E_s2.npz";; EN) echo "$W/d_EN_s1.npz $W/d_EN_s2.npz";; M) echo "$V2/d_MB_s1.npz $V2/d_MB_s2.npz";; esac; }
anchor_of() { case $1 in kneeFR0) echo "$W/anchor_kneeFR0.npz";; *) echo "$V2/anchor_$1.npz";; esac; }
gen() {   # GPU arm seed
  local drv=ppo; [ $2 = EN ] && drv="ppo_noise --noise 0.6"
  run $1 "$T genmpc --drive $drv --n_ep 25 --T 500 --cand 4 --seed $3 $BOX --out $W/d_$2_s$3.npz" > $D/log_gen_$2_s$3.txt 2>&1; }
train() { # GPU arm seed
  run $1 "$T train --data $(data_of $2) --which cand --max_trans 50000 --steps 30000 --seed $3 --out $W/wm_$2$3.pkl" > $D/log_train_$2$3.txt 2>&1; }
oracle() { run $1 "$T evalcl --oracle --cond $2 $BOX --out $D/oracle_$2" > $D/log_oracle_$2.txt 2>&1; }
chain() { # GPU arm seed
  local G=$1 A=$2 s=$3 n=$2$3
  for c in mot100 kneeRR0 tow40; do run $G "$T rank --wm $W/wm_$n.pkl --anchors $(anchor_of $c) --out $D/rank_${n}_$c.json"; done
  run $G "$T evalcl --wm $W/wm_$n.pkl --cond nominal $BOX --out $D/cl_${n}_nominal_before"
  for c in kneeRR0 kneeFR0 tow40; do
    run $G "$T evalcl --wm $W/wm_$n.pkl --cond $c $BOX --out $D/cl_${n}_${c}_before && \
            $T train --init $W/wm_$n.pkl --data $D/cl_${n}_${c}_before_trans.npz --which exec --max_trans 8000 --steps 5000 --trunc_fall --seed $s \
                     --replay $(data_of $A) --replay_which cand --replay_frac 0.5 --out $W/wm_${n}_${c}.pkl && \
            $T evalcl --wm $W/wm_${n}_${c}.pkl --cond $c $BOX --out $D/cl_${n}_${c}_after && \
            $T evalcl --wm $W/wm_${n}_${c}.pkl --cond nominal $BOX --out $D/cl_${n}_${c}_afternom"
  done
  for c in kneeRR0 kneeFR0 tow40; do      # positive control: privileged true-condition candidates
    run $G "$T train --init $W/wm_$n.pkl --data $(anchor_of $c) --which cand --max_trans 8000 --steps 5000 --seed $s \
                     --replay $(data_of $A) --replay_which cand --replay_frac 0.5 --out $W/wm_${n}_${c}_priv.pkl && \
            $T evalcl --wm $W/wm_${n}_${c}_priv.pkl --cond $c $BOX --out $D/cl_${n}_${c}_pafter && \
            $T evalcl --wm $W/wm_${n}_${c}_priv.pkl --cond nominal $BOX --out $D/cl_${n}_${c}_pafternom"
  done
}
case $1 in
all)
  # phase 1: E / EN shadow data (same seeds 1, 2 as M's d_MB_s1, s2); M world models meanwhile
  gen 0 E 1 & gen 1 E 2 & gen 2 EN 1 & gen 3 EN 2 &
  ( for s in 0 1 2; do train 0 M $s; done ) &
  wait
  # phase 2: oracle ceilings, kneeFR0 anchor (same settings as design 40 v2 anchors), E / EN world models
  oracle 0 nominal & oracle 1 kneeRR0 & oracle 2 kneeFR0 & oracle 3 tow40 &
  ( run 3 "$T genmpc --cond kneeFR0 --n_ep 8 --T 300 --cand 16 --seed 9 $BOX --out $W/anchor_kneeFR0.npz" > $D/log_anchor_kneeFR0.txt 2>&1 ) &
  ( for s in 0 1 2; do train 1 E $s; train 2 EN $s; done ) &
  wait
  # phase 3: 9 chains over 4 GPUs
  i=0
  for A in M E EN; do for s in 0 1 2; do
    ( chain $((i % 4)) $A $s > $D/log_chain_$A$s.txt 2>&1 ) & i=$((i + 1))
  done; done
  wait
  run 0 "python scripts/toy/stage41_report.py --d $D --md $D/report.md" > $D/log_report.txt 2>&1 ;;
esac
