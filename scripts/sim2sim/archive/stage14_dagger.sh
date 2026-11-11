#!/bin/bash
# Design doc 15 §7 follow-up A (user approved 2026-11-08): DAgger-style round for the noF head.
# Retrain head_broadwm_noF2 on the design-14 traces + the calibrated 16-iteration planner's own traces on in-coverage
# conditions (D_q10_i16 nominal / mass3), then evaluate with the new head (queue over GPUS_LIST, traces saved).
cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage14; TR=/nas_data/cjkim/simdist/traces_stage13; TS=$OUT/traces_shift; mkdir -p $OUT
GPUS_LIST=${GPUS_LIST:-"0 1 3"}
if [ ! -f $OUT/head_broadwm_noF2.json ]; then
  echo "== head head_broadwm_noF2 (holdout friction, + D_q10_i16 nominal/mass3) $(date +%H:%M)"
  GPUS="\"device=${GPUS_LIST%% *}\"" $RUN env JAX_PLATFORMS=cpu python scripts/sim2sim/calib_head.py --holdout_axis friction --out $OUT/head_broadwm_noF2 \
    --traces "$TR/T_broadwm_*.npz" "$TR/G_broadwm_*.npz" "$TR/S_broadwm_*.npz" "/workspace/simdist/$TS/D_q10_i16_nominal.npz" "/workspace/simdist/$TS/D_q10_i16_mass3.npz" \
    2>&1 | grep -v "WARNING\|cuStream\|^$" | tail -8
fi
[ -f $OUT/head_broadwm_noF2.npz ] || { echo "head missing, abort"; exit 1; }
phys() { local m=1.0 f=0.8 a=0.0 tok; IFS=_ read -ra toks <<< "$1"
  for tok in "${toks[@]}"; do case $tok in motor*) m=${tok#motor};; friction*) f=${tok#friction};; mass*) a=${tok#mass};; esac; done
  echo "sim.friction=$f sim.add_mass=$a sim.motor_strength=$m"; }
run() { local gpu=$1 name=$2 cond=$3; shift 3
  local csv=$OUT/eval_dagger_gpu$gpu.csv
  if cat $OUT/eval_dagger*.csv 2>/dev/null | grep -q "^$name,$cond,"; then return; fi
  echo "== $name on $cond (head noF2: $*) gpu$gpu $(date +%H:%M)"
  GPUS="\"device=$gpu\"" $RUN python scripts/simulate_batch.py model.checkpoint=go1_broadwm $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' 'batch.seeds=[11,12,13,14,15,16]' output.csv=/workspace/simdist/$csv output.model_name=$name output.condition=$cond \
    control.calib.enabled=true control.calib.weights=/workspace/simdist/$OUT/head_broadwm_noF2.npz output.trace_npz=/workspace/simdist/$TS/${name}_$cond.npz \
    "$@" --headless 2>&1 | grep -E "Traceback|rror:|trace written"
}
QUEUE=$OUT/queue_dagger.txt; LOCK=$OUT/queue_dagger.lock
cat > $QUEUE <<Q
E_q10_i16 friction0.15 control.calib.tau=0.1 control.iterations=16
E_q50_i16 friction0.15 control.calib.tau=0.5 control.iterations=16
E_q50_i16 nominal control.calib.tau=0.5 control.iterations=16
E_q50_i8 friction0.15 control.calib.tau=0.5
Q
pop() { ( flock 9; local j; j=$(head -1 $QUEUE); [ -n "$j" ] && sed -i '1d' $QUEUE; echo "$j" ) 9>$LOCK; }
worker() { local gpu=$1; while :; do local j; j=$(pop); [ -z "$j" ] && break; run $gpu $j; done; echo "== worker gpu$gpu idle $(date +%H:%M)"; }
for g in $GPUS_LIST; do worker $g & sleep 20; done
wait
python3 - <<'PY'
import csv, glob
rows=[]
for f in sorted(glob.glob('results/stage14/eval_dagger_gpu*.csv')): rows+=list(csv.DictReader(open(f)))
if rows:
    with open('results/stage14/eval_dagger.csv','w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('merged', len(rows), 'rows')
PY
echo "### dagger done $(date)"
