#!/bin/bash
# Design doc 14 S0-5 supplement: separate samples from iterations in the MPPI budget sweep (go1_broadwm, no trace).
cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage13; mkdir -p $OUT
phys() { local m=1.0 f=0.8 a=0.0 tok; IFS=_ read -ra toks <<< "$1"
  for tok in "${toks[@]}"; do case $tok in motor*) m=${tok#motor};; friction*) f=${tok#friction};; mass*) a=${tok#mass};; esac; done
  echo "sim.friction=$f sim.add_mass=$a sim.motor_strength=$m"; }
run() { local gpu=$1 name=$2 cond=$3; shift 3
  local csv=$OUT/eval_budget_gpu$gpu.csv
  if cat $OUT/eval_budget*.csv 2>/dev/null | grep -q "^$name,$cond,"; then return; fi
  echo "== $name on $cond ($*) gpu$gpu $(date +%H:%M)"
  GPUS="\"device=$gpu\"" $RUN python scripts/simulate_batch.py model.checkpoint=go1_broadwm $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' 'batch.seeds=[11,12,13,14,15,16]' output.csv=/workspace/simdist/$csv output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:"
}
( run 0 B_broadwm_n450i16 friction0.15 control.num_samples=450 control.iterations=16 ) &
( run 1 B_broadwm_n450i16 motor0.4 control.num_samples=450 control.iterations=16 ) &
( run 2 B_broadwm_n100i8 friction0.15 control.num_samples=100 control.iterations=8; run 2 B_broadwm_n450i3 friction0.15 control.num_samples=450 control.iterations=3 ) &
( run 3 B_broadwm_n100i8 motor0.4 control.num_samples=100 control.iterations=8; run 3 B_broadwm_n450i3 motor0.4 control.num_samples=450 control.iterations=3 ) &
wait
python3 - <<'PY'
import csv, glob
rows=[]
for f in sorted(glob.glob('results/stage13/eval_budget_gpu*.csv')): rows+=list(csv.DictReader(open(f)))
if rows:
    with open('results/stage13/eval_budget.csv','w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('merged', len(rows), 'rows')
PY
echo "### budget done $(date)"
