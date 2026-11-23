#!/bin/bash
# Design 42 (docs/design42_dose_ladder_ko.md): privileged dose ladder on knee faults, then E vs M with reset-after-fall collection.
# Run from ~/simdist on gpu1:  nohup bash scripts/toy/stage42.sh all > /mnt/newssd/cjkim/toy/e42_all.log 2>&1 &
set -u
cd ~/simdist
D=results/stage42; D41=results/stage41; W=/mnt/newssd/cjkim/toy/e42; W41=/mnt/newssd/cjkim/toy/e41; V2=/mnt/newssd/cjkim/toy/e12v2
mkdir -p $D $W
BOX="--box_hip 0.4 --box_thigh 0.8 --box_knee 0.8"
run() { GPUS="\"device=$1\"" ~/simdist-docker/run.sh bash -c "cd /workspace/simdist && export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=/mnt/newssd/cjkim/toy_pylibs:scripts/toy && $2"; }
T="python scripts/toy/toy_wm.py"
data_of() { case $1 in E) echo "$W41/d_E_s1.npz $W41/d_E_s2.npz";; M) echo "$V2/d_MB_s1.npz $V2/d_MB_s2.npz";; esac; }
priv_of() { echo "$W/p_$1_s1.npz $W/p_$1_s2.npz"; }
cell_args() { # cell arm seed -> init / dose flags
  case $1 in A1) echo "--init $W41/wm_$2$3.pkl --max_trans 8000";; A2) echo "--init $W41/wm_$2$3.pkl --max_trans 50000";; A3) echo "--max_trans 50000";; esac; }
dose_of() { case $1 in A1) echo 8000;; *) echo 50000;; esac; }

gen_priv() { # GPU cond seed: privileged true-condition MPPI candidates, same format as the pretraining data
  run $1 "$T genmpc --cond $2 --n_ep 25 --T 500 --cand 4 --seed $3 $BOX --out $W/p_$2_s$3.npz" > $D/log_gen_$2_s$3.txt 2>&1; }

ladder() { # GPU arm seed cond
  local G=$1 A=$2 s=$3 c=$4 n=$2$3
  for cell in A1 A2 A3; do
    run $G "$T train --data $(priv_of $c) --which cand $(cell_args $cell $A $s) --steps 30000 --seed $s \
                     --replay $(data_of $A) --replay_which cand --replay_frac 0.5 --out $W/wm_${n}_${c}_$cell.pkl && \
            $T evalcl --wm $W/wm_${n}_${c}_$cell.pkl --cond $c $BOX --out $D/cl_${n}_${c}_$cell && \
            $T evalcl --wm $W/wm_${n}_${c}_$cell.pkl --cond nominal $BOX --out $D/cl_${n}_${c}_${cell}nom"
  done
}
diag() { # GPU cond: A4, world model trained only on the privileged fault data, from scratch
  run $1 "$T train --data $(priv_of $2) --which cand --max_trans 50000 --steps 30000 --seed 0 --out $W/wm_A4_$2.pkl && \
          $T evalcl --wm $W/wm_A4_$2.pkl --cond $2 $BOX --out $D/cl_A4_$2"
}
stageB() { # GPU arm seed cond cell
  local G=$1 A=$2 s=$3 c=$4 cell=$5 n=$2$3
  local N=$(dose_of $cell) files="" b=0 u=0 r=0 e=0 f=0
  while [ $b -lt 16 ]; do               # reset after fall: 128 episodes x 3 s per batch, own pretrained WM-MPPI
    local out=$W/col_${n}_${c}_b$b
    run $G "$T evalcl --wm $W41/wm_$n.pkl --cond $c --episodes 128 --T 150 --seed0 $((10000 + 128 * b)) $BOX --out $out"
    files="$files ${out}_trans.npz"
    read u r e f < <(run $G "python scripts/toy/count_prefall.py $files" | tail -n 1)
    b=$((b + 1))
    if [ $u -ge $N ] || [ $r -ge 90000 ]; then break; fi
  done
  echo "$u $r $e $f $b" > $D/col_${n}_${c}.txt
  local INIT="--init $W41/wm_$n.pkl"; [ $cell = A3 ] && INIT=""
  run $G "$T train $INIT --data $files --max_trans $N --steps 30000 --trunc_fall --seed $s \
                   --replay $(data_of $A) --replay_which cand --replay_frac 0.5 --out $W/wm_${n}_${c}_B.pkl && \
          $T evalcl --wm $W/wm_${n}_${c}_B.pkl --cond $c $BOX --out $D/cl_${n}_${c}_B && \
          $T evalcl --wm $W/wm_${n}_${c}_B.pkl --cond nominal $BOX --out $D/cl_${n}_${c}_Bnom"
}

case $1 in
all)
  # phase 1: privileged data (2 conditions x seeds 1, 2) on GPUs 0-3
  gen_priv 0 kneeRR0 1 & gen_priv 1 kneeRR0 2 & gen_priv 2 kneeFR0 1 & gen_priv 3 kneeFR0 2 &
  wait
  # phase 2: ladder, 12 units + 2 diagnostics over 8 lanes (2 per GPU)
  U=(); for A in E M; do for s in 0 1 2; do for c in kneeRR0 kneeFR0; do U+=("$A $s $c"); done; done; done
  for l in 0 1 2 3 4 5 6 7; do
    ( G=$((l % 4)); i=$l
      while [ $i -lt ${#U[@]} ]; do ladder $G ${U[$i]} >> $D/log_ladder_$l.txt 2>&1; i=$((i + 8)); done
      [ $l -eq 4 ] && diag $G kneeRR0 >> $D/log_ladder_$l.txt 2>&1
      [ $l -eq 5 ] && diag $G kneeFR0 >> $D/log_ladder_$l.txt 2>&1 ) &
  done
  wait
  run 0 "python scripts/toy/stage42_report.py --d $D --d41 $D41 --decide" > $D/log_decide.txt 2>&1
  # phase 3: stage B only for conditions with a passing procedure
  B=(); for c in kneeRR0 kneeFR0; do
    cell=$(python3 -c "import json; print(json.load(open('$D/ladder.json')).get('$c', {}).get('cell', ''))")
    [ -n "$cell" ] && for A in E M; do for s in 0 1 2; do B+=("$A $s $c $cell"); done; done
  done
  for l in 0 1 2 3; do                  # one lane per GPU (128-episode collection needs more memory)
    ( G=$l; i=$l
      while [ $i -lt ${#B[@]} ]; do stageB $G ${B[$i]} >> $D/log_B_$l.txt 2>&1; i=$((i + 4)); done ) &
  done
  wait
  run 0 "python scripts/toy/stage42_report.py --d $D --d41 $D41 --md $D/report.md" > $D/log_report.txt 2>&1 ;;
esac
