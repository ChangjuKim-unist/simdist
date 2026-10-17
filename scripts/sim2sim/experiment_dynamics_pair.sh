#!/usr/bin/env bash
# Complete the 18-robot pairing: evaluate base / dyn / full recipe on seeds 14-16 (the new
# recipes already have seeds 11-16), then merge into eval_long_batch.csv.
set -u; cd ~/simdist
P=scripts/sim2sim/run_pipeline.sh
until grep -q "^### dynamics experiment done" results/sim2sim/dynamics_experiment.log; do sleep 120; done
echo "### pairing start $(date)"
export COND=very_weak_motor RESULTS=results/sim2sim/terminal_very_weak_motor/seeds14_16 TAG=term BUDGETS=45000
export LONG_SPEEDS="0.3 0.6 0.9" LONG_SEEDS="14 15 16"
mkdir -p $RESULTS
VARIANTS="dyn res_anchor_term_replay" LONG_CONDS="very_weak_motor nominal" $P evaluate_long_batch
tail -n +2 $RESULTS/eval_long_batch.csv >> results/sim2sim/terminal_very_weak_motor/eval_long_batch.csv
echo "### pairing done $(date)"
