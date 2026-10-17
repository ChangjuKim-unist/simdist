#!/usr/bin/env bash
# Pre-registered experiment of docs/design_dynamics_adaptation_ko.md (section 6):
#   dyn / full (anchor+replay+terminal) / full+ret / full+boundary-weight on very_weak_motor,
#   15-minute budget, 3000 steps, batched long-horizon evaluation with 18 robots.
set -u; cd ~/simdist
P=scripts/sim2sim/run_pipeline.sh; RUN=~/simdist-docker/run.sh
export COND=very_weak_motor RESULTS=results/sim2sim/terminal_very_weak_motor FT_EXTRA="training.warmup_steps=200"
export LONG_SPEEDS="0.3 0.6 0.9" LONG_SEEDS="11 12 13 14 15 16" TAG=term BUDGETS=45000
V="dyn res_anchor_term_replay full_ret full_bw"
echo "### dynamics experiment start $(date)"
VARIANTS="$V" FT_STEPS=3000 $P finetune
VARIANTS="$V" LONG_CONDS="very_weak_motor nominal" $P evaluate_long_batch
echo "### diagnostics $(date)"
M=""; for v in $V; do M="$M go1_paper_${COND}_${v}_b45000_term"; done
$RUN python scripts/sim2sim/return_error.py --models $M --dataset sim2sim_${COND}_go1_paper_b45000 --out $RESULTS/return_error.json 2>&1 | grep -vE "Warning|absl|WARNING|Checkpoint"
$RUN python scripts/sim2sim/value_error.py --models $M --dataset sim2sim_${COND}_go1_paper --stride 10 --out $RESULTS/value_error.json 2>&1 | grep -vE "Warning|absl|WARNING|Checkpoint"
VARIANTS="$V" $P forget 2>&1 | tail -2
echo "### dynamics experiment done $(date)"
