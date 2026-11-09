#!/bin/bash
# Design doc 16 §4: generative-prior candidate planner evaluations on gpu1 (queue over GPUS_LIST).  Needs
# results/stage15/gen_broad.npz (generator) and results/stage14/head_broadwm_*.npz (calibrated heads) on the server.
cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage15; mkdir -p $OUT
GPUS_LIST=${GPUS_LIST:-"0 1 3"}
phys() { local m=1.0 f=0.8 a=0.0 tok; IFS=_ read -ra toks <<< "$1"
  for tok in "${toks[@]}"; do case $tok in motor*) m=${tok#motor};; friction*) f=${tok#friction};; mass*) a=${tok#mass};; esac; done
  echo "sim.friction=$f sim.add_mass=$a sim.motor_strength=$m"; }
GEN="control.gen.enabled=true control.gen.weights=/workspace/simdist/$OUT/gen_broad.npz"
B0="control.init_mode=prior control.init_std=0.3 control.mixture_coef=0.0"
cal() { echo "control.calib.enabled=true control.calib.weights=/workspace/simdist/results/stage14/$1.npz control.calib.tau=0.5"; }
run() { local gpu=$1 name=$2 cond=$3; shift 3
  local csv=$OUT/eval_gen_gpu$gpu.csv
  if cat $OUT/eval_gen*.csv 2>/dev/null | grep -q "^$name,$cond,"; then return; fi
  echo "== $name on $cond ($*) gpu$gpu $(date +%H:%M)"
  GPUS="\"device=$gpu\"" $RUN python scripts/simulate_batch.py model.checkpoint=go1_broadwm $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' 'batch.seeds=[11,12,13,14,15,16]' output.csv=/workspace/simdist/$csv output.model_name=$name output.condition=$cond \
    "$@" --headless 2>&1 | grep -E "Traceback|rror:"
}
QUEUE=$OUT/queue_gen.txt; LOCK=$OUT/queue_gen.lock
cat > $QUEUE <<Q
G_gen_q50_i16 friction0.15 $GEN $(cal head_broadwm_noF) control.iterations=16
G_B0_q50_i16 friction0.15 $B0 $(cal head_broadwm_noF) control.iterations=16
G_gen_jhat_i16 friction0.15 $GEN control.iterations=16
G_gen_q50 friction0.15 $GEN $(cal head_broadwm_noF)
G_B0_q50 friction0.15 $B0 $(cal head_broadwm_noF)
G_gen_jhat friction0.15 $GEN
G_gen_q50 motor0.4 $GEN $(cal head_broadwm_noM)
G_gen_jhat motor0.4 $GEN
G_gen_q50 nominal $GEN $(cal head_broadwm_noF)
G_gen_q50 mass3 $GEN $(cal head_broadwm_noA)
Q
pop() { ( flock 9; local j; j=$(head -1 $QUEUE); [ -n "$j" ] && sed -i '1d' $QUEUE; echo "$j" ) 9>$LOCK; }
worker() { local gpu=$1; while :; do local j; j=$(pop); [ -z "$j" ] && break; run $gpu $j; done; echo "== worker gpu$gpu idle $(date +%H:%M)"; }
for g in $GPUS_LIST; do worker $g & sleep 20; done
wait
python3 - <<'PY'
import csv, glob
rows=[]
for f in sorted(glob.glob('results/stage15/eval_gen_gpu*.csv')): rows+=list(csv.DictReader(open(f)))
if rows:
    with open('results/stage15/eval_gen.csv','w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('merged', len(rows), 'rows')
PY
echo "### gen eval done $(date)"
