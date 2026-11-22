#!/bin/bash
# Design 40 E0 on gpu1 (run from ~/simdist on the host). Phase 1: PPO x5 + calibration; phase 2: M-free / M-clock x5.
set -u
cd ~/simdist
D=results/stage40; mkdir -p $D
PPO=/mnt/newssd/cjkim/toy/runs/ppo_s0/params.pkl
run() {  # run GPU "args..."
  GPUS="\"device=$1\"" ~/simdist-docker/run.sh bash -c "cd /workspace/simdist && XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=/mnt/newssd/cjkim/toy_pylibs:scripts/toy $2"
}
M="python scripts/toy/toy_mppi.py"
# ---- phase 1 ----
( for c in nominal legR50 mot50 kneeRR0 kneeFR0; do run 0 "python scripts/toy/toy_ppo_eval.py --params $PPO --cond $c --out $D/e0_ppo_$c"; done
  run 0 "$M --cond nominal --arm free --episodes 8 --seed0 100 --out $D/tune_free_sq0.2_lam0.1" ) > $D/log_p1_g0.txt 2>&1 &
i=1; for w in 1 3 10 30; do
  g=$(( i <= 3 ? i : 0 ))
  ( run $g "$M --cond nominal --arm clock --w_clock $w --episodes 8 --seed0 100 --out $D/tune_clock_w$w" ) > $D/log_p1_w$w.txt 2>&1 &
  i=$((i+1))
done
wait
run 0 "python scripts/toy/stage40_e0_report.py tune --d $D" > $D/tune.txt 2>&1
W=$(python3 -c "import json; p=json.load(open('$D/tune.json'))['clock_pick']; print(p[0] if p else 'NONE')")
echo "w_clock = $W" | tee -a $D/tune.txt
[ "$W" = "NONE" ] && { echo "no clock weight without falls; stop"; exit 1; }
# ---- phase 2 ----
J=( "free nominal" "free kneeRR0" "free kneeFR0" "free legR50" "free mot50" "clock nominal" "clock kneeRR0" "clock kneeFR0" "clock legR50" "clock mot50" )
for g in 0 1 2 3; do
  ( for k in $(seq $g 4 9); do set -- ${J[$k]}
      run $g "$M --cond $2 --arm $1 --w_clock $W --out $D/e0_$1_$2"; done ) > $D/log_p2_g$g.txt 2>&1 &
done
wait
run 0 "python scripts/toy/stage40_e0_report.py report --d $D --md $D/e0.md" > $D/report.txt 2>&1
echo done
