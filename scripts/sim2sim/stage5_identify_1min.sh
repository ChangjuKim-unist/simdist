#!/usr/bin/env bash
# Design doc 10: how few logged steps does the replay identifier need? Subsets of 3000 steps (1 min at 50 Hz) and 9000 steps.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage5; mkdir -p $OUT
until grep -q "### targethead done" results/stage6_targethead.log 2>/dev/null; do sleep 300; done
echo "### identify-1min start $(date)"
for b in 3000 9000; do for spec in "sim2sim_very_weak_motor_priorfirst:pf_motor04" "sim2sim_very_weak_motor_go1_paper:narrow_motor04" "sim2sim_nominal_priorfirst:pf_nominal"; do IFS=: read ds tag <<< "$spec"
  [ -d datasets/real/${ds}_b$b/processed_data_go1_H-25_T-25 ] || COND=nominal DATASET=$ds BUDGETS=$b scripts/sim2sim/run_pipeline.sh process 2>&1 | grep -E "Error|Traceback" | head -2
  echo "== $tag budget $b $(date +%H:%M)"
  $RUN python scripts/sim2sim/identify_physics.py dataset=${ds}_b$b out=$OUT/id1m_${tag}_b$b.json --headless 2>&1 | grep -E "error per|MAP|Error|Traceback" | cut -c1-200
done; done
echo "### identify-1min done $(date)"
