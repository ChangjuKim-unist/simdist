#!/bin/bash
# Design doc 15 §7 distribution-shift check on gpu1: calibrated / plain MPPI runs at 8 and 16 iterations with traces,
# so that the noF head can be scored on the candidates the calibrated planner itself selected.
cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage14; TR=$OUT/traces_shift; mkdir -p $OUT $TR
GPUS_LIST=${GPUS_LIST:-"0 1 3"}
phys() { local m=1.0 f=0.8 a=0.0 tok; IFS=_ read -ra toks <<< "$1"
  for tok in "${toks[@]}"; do case $tok in motor*) m=${tok#motor};; friction*) f=${tok#friction};; mass*) a=${tok#mass};; esac; done
  echo "sim.friction=$f sim.add_mass=$a sim.motor_strength=$m"; }
run() { local gpu=$1 name=$2 cond=$3 head=$4 trace=$5; shift 5
  local csv=$OUT/eval_shift_gpu$gpu.csv extra=""
  if cat $OUT/eval_shift*.csv 2>/dev/null | grep -q "^$name,$cond,"; then return; fi
  [ "$head" != none ] && extra="control.calib.enabled=true control.calib.weights=/workspace/simdist/$OUT/$head.npz"
  [ "$trace" = yes ] && extra="$extra output.trace_npz=/workspace/simdist/$TR/${name}_$cond.npz"
  echo "== $name on $cond (head $head trace $trace: $*) gpu$gpu $(date +%H:%M)"
  GPUS="\"device=$gpu\"" $RUN python scripts/simulate_batch.py model.checkpoint=go1_broadwm $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' 'batch.seeds=[11,12,13,14,15,16]' output.csv=/workspace/simdist/$csv output.model_name=$name output.condition=$cond \
    $extra "$@" --headless 2>&1 | grep -E "Traceback|rror:|trace written"
}
QUEUE=$OUT/queue_shift.txt; LOCK=$OUT/queue_shift.lock
cat > $QUEUE <<Q
D_q10_i16 friction0.15 head_broadwm_noF yes control.calib.tau=0.1 control.iterations=16
D_jhat_i16 friction0.15 none yes control.iterations=16
D_q10_i16 nominal head_broadwm_noF yes control.calib.tau=0.1 control.iterations=16
D_q10_i8 friction0.15 head_broadwm_noF yes control.calib.tau=0.1
D_jhat_i16 nominal none no control.iterations=16
D_q10_i16 mass3 head_broadwm_noF yes control.calib.tau=0.1 control.iterations=16
Q
pop() { ( flock 9; local j; j=$(head -1 $QUEUE); [ -n "$j" ] && sed -i '1d' $QUEUE; echo "$j" ) 9>$LOCK; }
worker() { local gpu=$1; while :; do local j; j=$(pop); [ -z "$j" ] && break; run $gpu $j; done; echo "== worker gpu$gpu idle $(date +%H:%M)"; }
for g in $GPUS_LIST; do worker $g & sleep 20; done
wait
python3 - <<'PY'
import csv, glob
rows=[]
for f in sorted(glob.glob('results/stage14/eval_shift_gpu*.csv')): rows+=list(csv.DictReader(open(f)))
if rows:
    with open('results/stage14/eval_shift.csv','w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('merged', len(rows), 'rows')
PY
echo "### shift check done $(date)"
