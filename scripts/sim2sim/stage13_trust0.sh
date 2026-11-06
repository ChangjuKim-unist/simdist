#!/usr/bin/env bash
# Design doc 14 stage 0 (docs/design_trust_ko.md): prior-only vs planner on go1_broadwm / go1_paper with per-step
# prediction traces. One queue per GPU in $GPUS_LIST; per-GPU CSVs merged at the end.
# Usage: GPUS_LIST="0 1 2 3" bash scripts/sim2sim/stage13_trust0.sh
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage13; TR=/data1/cjkim/stage13; mkdir -p $OUT $TR
GPUS_LIST=${GPUS_LIST:-"0"}; export SIMDIST_DIR=~/simdist
echo "### trust0 start $(date) GPUs: $GPUS_LIST"
phys() { case $1 in nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";;
  motor0.4) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";;
  motor0.5) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.5";;
  friction0.15) echo "sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0";;
  mass3) echo "sim.friction=0.8 sim.add_mass=3.0 sim.motor_strength=1.0";; esac; }
SEEDS="[11,12,13,14,15,16]"
run() { local gpu=$1 name=$2 ckpt=$3 cond=$4; shift 4
  local csv=$OUT/eval_trust0_gpu$gpu.csv
  if cat $OUT/eval_trust0*.csv 2>/dev/null | grep -q "^$name,$cond,"; then return; fi
  echo "== $name on $cond ($*) gpu$gpu $(date +%H:%M)"
  GPUS="\"device=$gpu\"" $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$csv output.model_name=$name output.condition=$cond \
    output.trace_npz=$TR/${name}_${cond}.npz "$@" --headless 2>&1 | grep -E "Traceback|rror:|trace written" | head -4
}
PO=control.prior_only=true
JOBS=(
 "T_broadwm_prior go1_broadwm motor0.4 $PO"
 "T_broadwm_planner go1_broadwm motor0.4"
 "T_broadwm_prior go1_broadwm friction0.15 $PO"
 "T_broadwm_planner go1_broadwm friction0.15"
 "T_broadwm_prior go1_broadwm mass3 $PO"
 "T_broadwm_planner go1_broadwm mass3"
 "T_broadwm_prior go1_broadwm nominal $PO"
 "T_broadwm_planner go1_broadwm nominal"
 "T_paper_prior go1_paper motor0.4 $PO"
 "T_paper_planner go1_paper motor0.4"
 "T_paper_prior go1_paper friction0.15 $PO"
 "T_paper_planner go1_paper friction0.15"
 "T_broadwm_prior go1_broadwm motor0.5 $PO"
 "T_paper_prior go1_paper nominal $PO"
)
# round-robin the jobs over the GPUs, one sequential queue per GPU
gpus=($GPUS_LIST); n=${#gpus[@]}
for ((g=0; g<n; g++)); do
  ( for ((j=g; j<${#JOBS[@]}; j+=n)); do run ${gpus[$g]} ${JOBS[$j]}; done ) &
  sleep 20
done
wait
python3 - <<'PY'
import csv, glob
rows=[]; 
for f in sorted(glob.glob('results/stage13/eval_trust0_gpu*.csv')): rows+=list(csv.DictReader(open(f)))
if rows:
    with open('results/stage13/eval_trust0.csv','w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('merged', len(rows), 'rows')
PY
echo "### trust0 done $(date)"
