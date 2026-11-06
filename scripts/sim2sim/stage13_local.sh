#!/usr/bin/env bash
# Design doc 14 stage 0, local supplement (go1_ctx): prior-only vs planner with traces + motor switch runs.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage13; TR=$OUT/traces; mkdir -p $OUT $TR
CSV=$OUT/eval_trust0_local.csv
echo "### trust0-local start $(date)"
phys() { case $1 in nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";;
  motor0.4) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";;
  friction0.15) echo "sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0";;
  mass3) echo "sim.friction=0.8 sim.add_mass=3.0 sim.motor_strength=1.0";; esac; }
SEEDS="[11,12,13,14,15,16]"
run() { local name=$1 ckpt=$2 cond=$3; shift 3
  if [ -f $TR/${name}_${cond}.npz ] && [ -f $CSV ] && grep -q "^$name,$cond," $CSV; then return; fi
  echo "== $name on $cond ($*) $(date +%H:%M)"
  GPUS='"device=0"' $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$CSV output.model_name=$name output.condition=$cond \
    output.trace_npz=/workspace/simdist/$TR/${name}_${cond}.npz "$@" --headless 2>&1 | grep -E "Traceback|rror:|trace written|switched"
}
PO=control.prior_only=true; SW="sim.motor_switch_step=2500 sim.motor_switch_strength=0.8"
run L_ctx_prior go1_ctx friction0.15 $PO
run L_ctx_planner go1_ctx friction0.15
run L_ctx_prior go1_ctx mass3 $PO
run L_ctx_planner go1_ctx mass3
run L_ctx_prior_switch go1_ctx motor0.4 $PO $SW
run L_ctx_planner_switch go1_ctx motor0.4 $SW
run L_ctx_prior go1_ctx motor0.4 $PO
run L_ctx_planner go1_ctx motor0.4
run L_ctx_prior go1_ctx nominal $PO
run L_ctx_planner go1_ctx nominal
# design doc 14 §2 proposition 2: the adapted model (friction-adapted, forgot the motor axis) — does planning help where the model is right and hurt where it forgot?
run L_adapt_prior go1_ctx_friction0.15_dyn_b45000_term friction0.15 $PO
run L_adapt_planner go1_ctx_friction0.15_dyn_b45000_term friction0.15
run L_adapt_prior go1_ctx_friction0.15_dyn_b45000_term motor0.4 $PO
run L_adapt_planner go1_ctx_friction0.15_dyn_b45000_term motor0.4
echo "### trust0-local done $(date)"
