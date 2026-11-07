#!/bin/bash
# Design doc 15 §7 follow-up A (NOT launched without the user's decision): DAgger-style round for the noF head.
# Retrain head_broadwm_noF2 on the design-14 traces + the calibrated 16-iteration planner's own traces on in-coverage
# conditions (D_q10_i16 nominal / mass3), then re-evaluate friction 0.15 at 16 iterations with q10 and q50.
cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage14; TR=/nas_data/cjkim/simdist/traces_stage13; TS=$OUT/traces_shift; mkdir -p $OUT
GPUS_LIST=${GPUS_LIST:-"0 1"}
if [ ! -f $OUT/head_broadwm_noF2.json ]; then
  echo "== head head_broadwm_noF2 (holdout friction, + D_q10_i16 nominal/mass3) $(date +%H:%M)"
  GPUS="\"device=${GPUS_LIST%% *}\"" $RUN env JAX_PLATFORMS=cpu python scripts/sim2sim/calib_head.py --holdout_axis friction --out $OUT/head_broadwm_noF2 \
    --traces "$TR/T_broadwm_*.npz" "$TR/G_broadwm_*.npz" "$TR/S_broadwm_*.npz" "/workspace/simdist/$TS/D_q10_i16_nominal.npz" "/workspace/simdist/$TS/D_q10_i16_mass3.npz" \
    2>&1 | grep -v "WARNING\|cuStream\|^$" | tail -8
fi
run() { local gpu=$1 name=$2 tau=$3
  local csv=$OUT/eval_dagger_gpu$gpu.csv
  if cat $OUT/eval_dagger*.csv 2>/dev/null | grep -q "^$name,friction0.15,"; then return; fi
  echo "== $name on friction0.15 (head noF2 tau $tau, iterations 16) gpu$gpu $(date +%H:%M)"
  GPUS="\"device=$gpu\"" $RUN python scripts/simulate_batch.py model.checkpoint=go1_broadwm sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0 sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' 'batch.seeds=[11,12,13,14,15,16]' output.csv=/workspace/simdist/$csv output.model_name=$name output.condition=friction0.15 \
    control.calib.enabled=true control.calib.weights=/workspace/simdist/$OUT/head_broadwm_noF2.npz control.calib.tau=$tau control.iterations=16 \
    output.trace_npz=/workspace/simdist/$TS/${name}_friction0.15.npz --headless 2>&1 | grep -E "Traceback|rror:|trace written"
}
set -- $GPUS_LIST
run $1 E_q10_i16 0.1 & sleep 20; run ${2:-$1} E_q50_i16 0.5 & wait
echo "### dagger done $(date)"
