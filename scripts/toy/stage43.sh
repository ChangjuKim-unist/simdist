#!/bin/bash
# Design 43 (docs/design43_own_state_coverage_ko.md): does true-model data at the WM planner's OWN states make WM-MPPI walk on knee faults?
# Run from ~/simdist on gpu1:  nohup bash scripts/toy/stage43.sh all > /mnt/newssd/cjkim/toy/e43_all.log 2>&1 &
set -u
cd ~/simdist
D=results/stage43; W=/mnt/newssd/cjkim/toy/e43; W42=/mnt/newssd/cjkim/toy/e42
mkdir -p $D $W
BOX="--box_hip 0.4 --box_thigh 0.8 --box_knee 0.8"
run() { GPUS="\"device=$1\"" ~/simdist-docker/run.sh bash -c "cd /workspace/simdist && export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=/mnt/newssd/cjkim/toy_pylibs:scripts/toy && $2"; }
T="python scripts/toy/toy_wm.py"
priv_of() { echo "$W42/p_$1_s1.npz $W42/p_$1_s2.npz"; }
frac() { python3 -c "print(round(25 * $1 / (50 + 25 * $1), 4))"; }   # on-policy share for 25k x r next to 50k base

dg() { # GPU cond seed: round 0 (base data only), then 4 own-state rounds
  local G=$1 c=$2 s=$3 n=${2}_s$3 files=""
  run $G "$T train --data $(priv_of $c) --which cand --max_trans 50000 --steps 30000 --seed $s --out $W/wm_DG_${n}_r0.pkl && \
          $T evalcl --wm $W/wm_DG_${n}_r0.pkl --cond $c $BOX --out $D/cl_DG_${n}_r0"
  for r in 1 2 3 4; do
    local p=$((r - 1))
    run $G "$T evalcl --wm $W/wm_DG_${n}_r$p.pkl --cond $c --episodes 32 --T 150 --seed0 $((20000 + 32 * p)) --record_cand 4 $BOX --out $W/col_${n}_r$r"
    files="$files $W/col_${n}_r${r}_cand.npz"
    run $G "$T train --data $(priv_of $c) --which cand --max_trans 50000 --steps 30000 --seed $s \
                     --replay $files --replay_which cand --replay_frac $(frac $r) --out $W/wm_DG_${n}_r$r.pkl && \
            $T evalcl --wm $W/wm_DG_${n}_r$r.pkl --cond $c $BOX --out $D/cl_DG_${n}_r$r"
  done
}
ct() { # GPU cond seed: amount-matched control, more base (walking-planner) data only
  local G=$1 c=$2 s=$3 n=${2}_s$3
  for r in 1 2 3 4; do
    run $G "$T train --data $(priv_of $c) --which cand --max_trans $((50000 + 25000 * r)) --steps 30000 --seed $s --out $W/wm_CT_${n}_r$r.pkl && \
            $T evalcl --wm $W/wm_CT_${n}_r$r.pkl --cond $c $BOX --out $D/cl_CT_${n}_r$r"
  done
}

case $1 in
all)
  U=(); for c in kneeRR0 kneeFR0; do for s in 0 1 2; do U+=("dg $c $s"); done; done
  for c in kneeRR0 kneeFR0; do for s in 0 1 2; do U+=("ct $c $s"); done; done
  for l in 0 1 2 3 4 5 6 7; do
    ( G=$((l % 4)); i=$l
      while [ $i -lt ${#U[@]} ]; do set -- ${U[$i]}; $1 $G $2 $3 >> $D/log_lane_$l.txt 2>&1; i=$((i + 8)); done ) &
  done
  wait
  run 0 "python scripts/toy/stage43_report.py --d $D --d41 results/stage41 --md $D/report.md" > $D/log_report.txt 2>&1 ;;
esac
