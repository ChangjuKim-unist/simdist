#!/usr/bin/env bash
# Head-quality check (2026-11-04): is targethead's weakness at motor 0.5/0.6/friction 0.15 the narrow DR or this one seed?
# Evaluate the seed-2 targethead on the same three conditions, then resume the stage10 chain (it skips finished rows).
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; CSV=results/stage10/eval_core.csv
until [ -z "$(docker ps -q)" ]; do sleep 30; done
phys() { case $1 in motor0.5) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.5";;
  motor0.6) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.6";;
  friction0.15) echo "sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
for c in motor0.5 motor0.6 friction0.15; do
  grep -q "^H_targethead_s2_planner,$c," $CSV && continue
  echo "== H_targethead_s2_planner on $c $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=go1_paper_targethead_s2 $(phys $c) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' 'batch.seeds=[11,12,13,14,15,16]' output.csv=/workspace/simdist/$CSV output.model_name=H_targethead_s2_planner output.condition=$c --headless 2>&1 | grep -E "Traceback|rror:" | head -3
done
echo "### s2 check done $(date)"; bash scripts/sim2sim/stage10_reeval.sh
