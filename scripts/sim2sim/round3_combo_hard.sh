#!/usr/bin/env bash
# Third sim-to-sim round: moderate shift (combo_hard), oracle upper bounds, expert reference, 10 seeds.
set -u
cd ~/simdist
export COND=combo_hard SPEEDS="0.3 0.6 0.9" COLLECT_STEPS=5000 RESULTS=results/sim2sim/combo_hard
export FT_EXTRA="training.warmup_steps=200" EVAL_SEEDS="42 7 123 5 99 1 2 3 4 8" NOMINAL_SEEDS="42 7 123 5 99"
export EXPERT_RUN=2026-10-01_10-38-36 EXPERT_ITER=4999
P=scripts/sim2sim/run_pipeline.sh
echo "### start $(date)"
ROUNDS=12 $P collect                                   # 12 x 3 x 100 s = 60 min of target-condition data
COND=nominal RESULTS=results/sim2sim/nominal_ref ROUNDS=1 $P collect   # 5 min nominal reference for value_error
BUDGETS="15000 45000 90000 180000" $P process
BUDGETS="15000 45000 90000" VARIANTS="dyn full full_replay res_anchor_tf res_replay res_anchor_replay" FT_STEPS=3000 $P finetune
TAG=oracle BUDGETS="180000" VARIANTS="res_replay full_replay" FT_STEPS=9000 $P finetune
BUDGETS="15000 45000 90000" VARIANTS="dyn full full_replay res_anchor_tf res_replay res_anchor_replay" $P evaluate
TAG=oracle BUDGETS="180000" VARIANTS="res_replay full_replay" EXPERT_RUN= $P evaluate
BUDGETS="15000 45000 90000" VARIANTS="dyn full full_replay res_anchor_tf res_replay res_anchor_replay" $P forget
TAG=oracle BUDGETS="180000" VARIANTS="res_replay full_replay" $P forget
$P summarize
echo "### done $(date)"
