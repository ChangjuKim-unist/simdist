#!/usr/bin/env bash
# Design doc 7, X4: RAPP-style per-axis feasibility sweep of the students alone (no planner): where does the
# prior still walk? Axes motor / friction / mass, 6 points each, students w10 (nominal expert) and w03 (broad).
# 18 robots x 100 s per point. Results: results/stage2/eval_feasibility.csv.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage2; mkdir -p $OUT; CSV=$OUT/eval_feasibility.csv; PRE=go1_paper
SEEDS="[11,12,13,14,15,16]"
run() { local name=$1 cond=$2 phys=$3; shift 3
  if [ -f $CSV ] && grep -q "^$name,$cond," $CSV; then return; fi
  echo "== $name on $cond ($phys) $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$PRE $phys sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$CSV output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:" | head -3
}
echo "### feasibility sweep start $(date)"
for w in w10 w03; do
  for m in 0.3 0.4 0.5 0.6 0.8 1.0; do run D_$w motor$m "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=$m" prior.student=student_$w policy.use_as=student_act; done
  for f in 0.1 0.15 0.2 0.3 0.5 0.8; do run D_$w friction$f "sim.friction=$f sim.add_mass=0.0 sim.motor_strength=1.0" prior.student=student_$w policy.use_as=student_act; done
  for k in 1 2 3 4 5 6; do run D_$w mass$k "sim.friction=0.8 sim.add_mass=$k.0 sim.motor_strength=1.0" prior.student=student_$w policy.use_as=student_act; done
done
echo "### feasibility sweep done $(date)"
