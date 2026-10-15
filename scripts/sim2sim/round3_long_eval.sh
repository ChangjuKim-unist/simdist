#!/usr/bin/env bash
# Long-horizon evaluation (5000-step runs x 3 speeds x 2 seeds) of the round-3 models on combo_hard.
set -u; cd ~/simdist
export COND=combo_hard RESULTS=results/sim2sim/combo_hard EXPERT_RUN=2026-10-01_10-38-36 EXPERT_ITER=4999
echo "### long start $(date)"
BUDGETS="15000 45000 90000" VARIANTS="dyn res_anchor_tf" \
  LONG_EXTRA_MODELS="go1_paper_combo_hard_res_replay_b45000 go1_paper_combo_hard_full_b45000 go1_paper_combo_hard_full_replay_b45000 go1_paper_combo_hard_dyn_b45000_r2 go1_paper_combo_hard_res_anchor_tf_b45000_r2" \
  scripts/sim2sim/run_pipeline.sh evaluate_long
echo "### long done $(date)"
