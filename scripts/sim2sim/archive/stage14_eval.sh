#!/bin/bash
# Design doc 15 stage B: calibrated-score evaluations on gpu1 (GPUS_LIST = free GPUs), flock-shared job queue.
cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage14; mkdir -p $OUT
GPUS_LIST=${GPUS_LIST:-"0 1 3"}
phys() { local m=1.0 f=0.8 a=0.0 tok; IFS=_ read -ra toks <<< "$1"
  for tok in "${toks[@]}"; do case $tok in motor*) m=${tok#motor};; friction*) f=${tok#friction};; mass*) a=${tok#mass};; esac; done
  echo "sim.friction=$f sim.add_mass=$a sim.motor_strength=$m"; }
run() { local gpu=$1 name=$2 ckpt=$3 cond=$4 head=$5; shift 5
  local csv=$OUT/eval_calib_gpu$gpu.csv
  if cat $OUT/eval_calib*.csv 2>/dev/null | grep -q "^$name,$cond,"; then return; fi
  echo "== $name on $cond (head $head: $*) gpu$gpu $(date +%H:%M)"
  GPUS="\"device=$gpu\"" $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' 'batch.seeds=[11,12,13,14,15,16]' output.csv=/workspace/simdist/$csv output.model_name=$name output.condition=$cond \
    control.calib.enabled=true control.calib.weights=/workspace/simdist/$OUT/$head.npz "$@" --headless 2>&1 | grep -E "Traceback|rror:"
}
QUEUE=$OUT/queue.txt; LOCK=$OUT/queue.lock
cat > $QUEUE <<Q
C_broadwm_q10_i16 go1_broadwm friction0.15 head_broadwm_noF control.calib.tau=0.1 control.iterations=16
C_broadwm_q10 go1_broadwm friction0.15 head_broadwm_noF control.calib.tau=0.1
C_broadwm_q10 go1_broadwm motor0.4 head_broadwm_noM control.calib.tau=0.1
C_broadwm_q50 go1_broadwm friction0.15 head_broadwm_noF control.calib.tau=0.5
C_broadwm_q50 go1_broadwm motor0.4 head_broadwm_noM control.calib.tau=0.5
C_broadwm_q50_i16 go1_broadwm friction0.15 head_broadwm_noF control.calib.tau=0.5 control.iterations=16
C_broadwm_q10_i16 go1_broadwm motor0.4 head_broadwm_noM control.calib.tau=0.1 control.iterations=16
C_broadwm_q10_fall go1_broadwm friction0.15 head_broadwm_noF control.calib.tau=0.1 control.calib.fall_max=0.5
C_broadwm_q10_fall go1_broadwm motor0.4 head_broadwm_noM control.calib.tau=0.1 control.calib.fall_max=0.5
C_broadwm_q10 go1_broadwm nominal head_broadwm_noF control.calib.tau=0.1
C_broadwm_q10 go1_broadwm mass3 head_broadwm_noA control.calib.tau=0.1
C_paperfr_q10 go1_paper_friction0.15_dyn_b45000_term motor0.4 head_paper_noM control.calib.tau=0.1
C_papermotor_q10 go1_paper_very_weak_motor_dyn_b45000_term friction0.15 head_paper_noF control.calib.tau=0.1
Q
pop() { ( flock 9; local j; j=$(head -1 $QUEUE); [ -n "$j" ] && sed -i '1d' $QUEUE; echo "$j" ) 9>$LOCK; }
worker() { local gpu=$1; while :; do local j; j=$(pop); [ -z "$j" ] && break; run $gpu $j; done; echo "== worker gpu$gpu idle $(date +%H:%M)"; }
for g in $GPUS_LIST; do worker $g & sleep 20; done
wait
python3 - <<'PY'
import csv, glob
rows=[]
for f in sorted(glob.glob('results/stage14/eval_calib_gpu*.csv')): rows+=list(csv.DictReader(open(f)))
if rows:
    with open('results/stage14/eval_calib.csv','w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('merged', len(rows), 'rows')
PY
echo "### calib eval done $(date)"
