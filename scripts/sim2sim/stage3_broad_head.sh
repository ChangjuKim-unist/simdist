#!/usr/bin/env bash
# Design doc 8: a broad prior with SimDist's own structure. Generate pretraining-style data (expert + corrupted
# actions, as the SimDist pretraining set) with the broad (motor-DR) expert, then retrain ONLY the world model's
# policy head on it (encoder/dynamics/reward/value frozen) -> checkpoint go1_paper_broadhead.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage3; mkdir -p $OUT; RL=2026-10-20_07-40-06
echo "### stage3 start $(date)"
if [ ! -f $OUT/dataset_broad_pretrain ]; then
  before=$(ls datasets/sim | sort | tail -1); echo "== generate broad pretraining-style data (go1dr, expert_prob 0.5, 384 x 20000) $(date +%H:%M)"
  $RUN python scripts/generate_data.py rl_run=$RL system=go1dr envs=384 steps=20000 2>&1 | grep -E "steps will be|Error|Traceback" | head -3
  ds=$(ls datasets/sim | sort | tail -1); [ "$ds" != "$before" ] || { echo "no dataset"; exit 1; }; echo "$ds" > $OUT/dataset_broad_pretrain
  $RUN python scripts/process_data.py dataset_name=$ds system=go1dr 2>&1 | grep -E "Error|Traceback|num_trajectories" | head -2
fi
DS=$(cat $OUT/dataset_broad_pretrain); echo "== dataset $DS"
if ! ls -d checkpoints/models/go1_paper_broadhead/[0-9]* >/dev/null 2>&1; then
  echo "== retrain policy head on broad data (heads=policy_only, 60k steps) $(date +%H:%M)"
  $RUN python scripts/finetune_model.py data.dataset_name=$DS system=go1dr checkpoint.resume_checkpoint=go1_paper run_name=go1_paper_broadhead \
    loss=world_model heads=policy_only training.max_steps=60000 training.eval_interval=10000 2>&1 | grep -E "Parameters|Steps: |Error|Traceback|override" | cut -c1-200 | tail -8
fi
echo "### stage3 done $(date)"
