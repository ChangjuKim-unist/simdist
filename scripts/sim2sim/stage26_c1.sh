#!/bin/bash
# Design doc 26 stage 2, C1 (triggered: PB3 held in stage 1): D6 and D7 again with the forward command capped at 0.9 m/s.
#   usage (gpu1): tmux new -d -s s26c1 'bash scripts/sim2sim/stage26_c1.sh > results/stage26_c1_server.log 2>&1'
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage26/c1; mkdir -p $OUT
PRIOR=control.prior_only=true
run() { local g=$1 name=$2 ckpt=$3 extra=$4
  echo "== $name gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt \
    sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0 sim.terrain=plane sim.total_steps=5000 \
    sim.episode_length_s=100 'batch.speeds=[0.9]' 'batch.seeds=[11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]' \
    sim.cmd_ramp_steps=50 sim.cmd_vx_max=0.9 \
    output.csv=/workspace/simdist/$OUT/eval_gpu$g.csv output.falls_csv=/workspace/simdist/$OUT/falls_gpu$g.csv \
    output.diag_npz=/workspace/simdist/$OUT/diag_$name.npz \
    output.model_name=$name output.condition=friction0.15 $extra --headless 2>&1 | grep -E "Traceback|rror:|falls written|diagnostics written" | head -4
  echo "-- $name done gpu$g $(date +%H:%M)"; }
echo "### stage26 C1 start $(date)"
run 0 C1D6_broad_plan_r50_cap0.9 go1_broadwm "" &
sleep 20
run 1 C1D7_broad_pol_r50_cap0.9 go1_broadwm "$PRIOR" &
wait
echo "### stage26 C1 done $(date)"
