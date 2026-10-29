#!/usr/bin/env bash
# Design doc 10, P10e: 3-axis identification (motor x friction x mass) on logs with known conditions. Runs after the P10d chain.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage5; mkdir -p $OUT
until grep -q "### identified-adapt done" results/stage5_identified_adapt.log 2>/dev/null; do sleep 300; done
echo "### identify3 start $(date)"
G="grid.lo=0.3 grid.hi=1.1 grid.n=9 friction_grid=[0.15,0.3,0.5,0.8,1.0] mass_grid=[0.0,1.5,3.0,4.5] replicas=4 rounds=60"
for spec in "sim2sim_combo_hard_go1_paper_b45000:combo_0.6_0.3_3" "sim2sim_low_friction_priorfirst_b45000:lowfric_1.0_0.15_0" "sim2sim_very_weak_motor_priorfirst_b45000:motor_0.4_0.8_0" "sim2sim_nominal_priorfirst_b45000:nominal_1.0_0.8_0"; do IFS=: read ds tag <<< "$spec"
  echo "== identify3 $tag ($ds) $(date +%H:%M)"
  $RUN python scripts/sim2sim/identify_physics.py dataset=$ds $G out=$OUT/id3_$tag.json --headless 2>&1 | grep -E "marginal|MAP|Error|Traceback|File \"/workspace" | cut -c1-230
done
echo "### identify3 done $(date)"
