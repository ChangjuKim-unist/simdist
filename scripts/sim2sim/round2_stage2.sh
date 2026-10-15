#!/usr/bin/env bash
# Second half of round2_combo_hard.sh (dataset build / finetune / evaluate), run after its collections.
set -u; cd ~/simdist
P=scripts/sim2sim/run_pipeline.sh; RUN=~/simdist-docker/run.sh
export COND=combo_hard RESULTS=results/sim2sim/combo_hard FT_EXTRA="training.warmup_steps=200"
export EVAL_SEEDS="42 7 123 5 99 1 2 3 4 8" NOMINAL_SEEDS="42 7 123 5 99"
until grep -q "^### r2 done" results/sim2sim/combo_hard/round2.log; do sleep 60; done
echo "### stage2 start $(date)"
for M in dyn res_anchor_tf; do
  C1=go1_paper_combo_hard_${M}_b45000; COLL=sim2sim_combo_hard_r2_${M}_collect; DS=sim2sim_combo_hard_r2${M}
  echo "### [$M] build ${DS}_b45000"
  $RUN python scripts/sim2sim/build_round_dataset.py ${DS}_b45000 sim2sim_combo_hard_go1_paper:45000 $COLL 2>&1 | grep -vE "Warning|absl"
  $RUN python scripts/aggregate_realworld_data.py dataset_name=${DS}_b45000 2>&1 | grep -E "Total|Error|Traceback"
  $RUN python scripts/process_data.py dataset_name=${DS}_b45000 system=go1 2>&1 | grep -E "Error|Traceback"
  echo "### [$M] finetune from $C1"
  DATASET=$DS BUDGETS=45000 TAG=r2 VARIANTS=$M RESUME_CKPT=$C1 FT_STEPS=3000 $P finetune
  echo "### [$M] evaluate"
  TAG=r2 BUDGETS=45000 VARIANTS=$M $P evaluate
  TAG=r2 BUDGETS=45000 VARIANTS=$M $P forget
done
echo "### stage2 done $(date)"
