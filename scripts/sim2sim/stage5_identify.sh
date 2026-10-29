#!/usr/bin/env bash
# Design doc 10: replay-based motor-strength identification on three logs (truth: 0.4, 0.4, 1.0). Runs after stage 4.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage5; mkdir -p $OUT
until grep -q "### stage4 done" results/stage4_matched.log 2>/dev/null; do sleep 300; done
echo "### identify start $(date)"
for spec in "sim2sim_very_weak_motor_priorfirst_b45000:pf_motor04" "sim2sim_very_weak_motor_go1_paper_b45000:narrow_motor04" "sim2sim_nominal_priorfirst_b45000:pf_nominal"; do IFS=: read ds tag <<< "$spec"
  echo "== identify $tag ($ds) $(date +%H:%M)"
  $RUN python scripts/sim2sim/identify_physics.py dataset=$ds out=$OUT/id_$tag.json --headless 2>&1 | grep -E "^round|error per|MAP|Error|Traceback|File \"/workspace" | cut -c1-220
done
echo "### identify done $(date)"
