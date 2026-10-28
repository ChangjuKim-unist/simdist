#!/usr/bin/env bash
# Design doc 8 evaluation: the broad SimDist-structure prior (go1_paper_broadhead) vs SimDist (go1_paper) vs the
# broad student, at nominal / motor 0.4 / friction 0.15; planner with its own policy head (no external student).
# P8d: dynamics adapted on narrow-prior data (dyn_b45000_term) under the broadhead prior -> coupling.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage3; mkdir -p $OUT; CSV=$OUT/eval_broadhead.csv
phys() { case $1 in nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";; motor0.4) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";; friction0.15) echo "sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
SEEDS="[11,12,13,14,15,16]"
run() { local name=$1 ckpt=$2 cond=$3; shift 3
  if [ -f $CSV ] && grep -q "^$name,$cond," $CSV; then return; fi
  echo "== $name on $cond ($*) $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$CSV output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:" | head -3
}
until grep -q "### stage3 done" results/stage3_broad_head.log 2>/dev/null; do sleep 300; done
ls -d checkpoints/models/go1_paper_broadhead/[0-9]* >/dev/null 2>&1 || { echo "no broadhead checkpoint"; exit 1; }
echo "### stage3 eval start $(date)"
for cond in nominal motor0.4 friction0.15; do
  run A_simdist_planner go1_paper $cond
  run H_broadhead_planner go1_paper_broadhead $cond
  run H_broadhead_alone go1_paper_broadhead $cond control.prior_only=true
done
# P8d: coupling with the narrow-prior real-format data under the broad head. The adapted dynamics checkpoint keeps its own
# (narrow) policy head, so the broad head is taken from go1_paper_broadhead through the external-prior hook.
run H_broadhead_dynB go1_paper_very_weak_motor_dyn_b45000_term motor0.4 prior.head_from=go1_paper_broadhead
echo "### stage3 eval done $(date)"
