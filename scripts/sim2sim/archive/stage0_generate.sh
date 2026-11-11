#!/usr/bin/env bash
# Stage 0 of docs/research_plan_ko.md: regenerate narrow (nominal expert, go1) and broad
# (motor-DR expert, go1dr) expert-only datasets with the physics recorder, then process them.
set -u; cd ~/simdist
RUN=~/simdist-docker/run.sh; OUT=results/stage0; mkdir -p $OUT
gen() {  # gen <tag> <rl_run> <system>
  local tag=$1 rl=$2 sys=$3
  [ -f $OUT/dataset_$tag ] && return
  local before=$(ls datasets/sim | sort | tail -1)
  echo "== generate $tag ($rl, $sys) $(date +%H:%M)"
  $RUN python scripts/generate_data.py rl_run=$rl system=$sys envs=384 steps=5000 expert_prob=1.0 action_corruption.never_noise_prob=1.0 2>&1 | grep -E "steps will be|Error|Traceback" | head -3
  local ds=$(ls datasets/sim | sort | tail -1); [ "$ds" != "$before" ] || { echo "no new dataset for $tag"; exit 1; }
  echo "$ds" > $OUT/dataset_$tag
  $RUN python scripts/process_data.py dataset_name=$ds system=$sys 2>&1 | grep -E "Error|Traceback|num_trajectories" | head -2
}
echo "### stage0 generate start $(date)"
gen narrow 2026-10-01_10-38-36 go1
gen broad  2026-10-20_07-40-06 go1dr
echo "### stage0 generate done $(date)"
