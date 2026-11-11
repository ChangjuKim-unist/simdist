#!/usr/bin/env bash
# Stage 0 (docs/research_plan_ko.md): student-alone and expert-alone performance per physics axis,
# to compare the axis ranking by visibility (probe) with the ranking by performance (P0).
# Adds to results/prior_coverage/eval_long_batch.csv (motor 0.4 / nominal / combo_hard / motor 0.25 already there).
set -u; cd ~/simdist
RUN=~/simdist-docker/run.sh
RL=2026-10-20_07-40-06
OUT=results/prior_coverage; CSV=$OUT/eval_long_batch.csv; PRE=go1_paper
NEWEXP="policy.rl_run=$RL policy.iteration=4999"
phys() { case $1 in
  motor0.6)     echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.6";;
  motor0.8)     echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.8";;
  payload)      echo "sim.friction=0.8 sim.add_mass=3.0 sim.motor_strength=1.0";;
  low_friction) echo "sim.friction=0.3 sim.add_mass=0.0 sim.motor_strength=1.0";;
  esac; }
SEEDS="[11,12,13,14,15,16]"
run() { local name=$1 ckpt=$2 cond=$3; shift 3
  if [ -f $CSV ] && grep -q "^$name,$cond," $CSV; then return; fi
  echo "== $name on $cond ($*) $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$CSV output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:" | head -3
}
echo "### stage0 student conditions start $(date)"
for cond in motor0.6 motor0.8 payload low_friction; do
  run D_student_broad $PRE $cond prior.student=student_broad policy.use_as=student_act
  run F_expert_broad  $PRE $cond $NEWEXP policy.use_as=act
done
echo "### stage0 student conditions done $(date)"
