#!/usr/bin/env bash
# Pre-registered experiment of docs/design_value_adaptation_ko.md (section 7):
#   dyn vs res_anchor_tf vs res_anchor_term, conditions very_weak_motor and combo_hard,
#   15-minute budget, long-horizon evaluation (5000-step runs x 3 speeds x 3 seeds).
set -u; cd ~/simdist
P=scripts/sim2sim/run_pipeline.sh; RUN=~/simdist-docker/run.sh
export FT_EXTRA="training.warmup_steps=200" LONG_SEEDS="11 12 13" LONG_SPEEDS="0.3 0.6 0.9"
echo "### terminal experiment start $(date)"
for COND in very_weak_motor combo_hard; do
  export COND RESULTS=results/sim2sim/terminal_${COND}
  mkdir -p $RESULTS
  DS=sim2sim_${COND}_go1_paper_b45000
  echo "### [$COND] re-aggregate and re-process $DS with terminal information"
  $RUN python scripts/aggregate_realworld_data.py dataset_name=$DS max_steps=45000 2>&1 | grep -E "Total|Error|Traceback"
  $RUN python scripts/process_data.py dataset_name=$DS system=go1 2>&1 | grep -E "Error|Traceback"
  # all three recipes retrained on the re-processed windows (TAG=term) so they share data and steps
  echo "### [$COND] finetune dyn / res_anchor_tf / res_anchor_term"
  TAG=term BUDGETS=45000 VARIANTS="dyn res_anchor_tf res_anchor_term" FT_STEPS=3000 $P finetune
  echo "### [$COND] long-horizon evaluation"
  TAG=term BUDGETS=45000 VARIANTS="dyn res_anchor_tf res_anchor_term" LONG_CONDS="$COND nominal" $P evaluate_long
  TAG=term BUDGETS=45000 VARIANTS="dyn res_anchor_tf res_anchor_term" $P forget
done
echo "### terminal experiment done $(date)"
