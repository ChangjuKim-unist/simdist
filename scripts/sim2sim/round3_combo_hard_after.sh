#!/usr/bin/env bash
# Runs after run_r3.sh: value-error analysis on the target and the nominal reference data.
cd ~/simdist
until grep -q "^### done" results/sim2sim/combo_hard/pipeline.log; do sleep 120; done
M="go1_paper go1_paper_combo_hard_dyn_b45000 go1_paper_combo_hard_full_b45000 go1_paper_combo_hard_full_replay_b45000 go1_paper_combo_hard_res_anchor_tf_b45000 go1_paper_combo_hard_res_replay_b45000 go1_paper_combo_hard_res_anchor_replay_b45000 go1_paper_combo_hard_res_replay_b180000_oracle"
echo "### value_error $(date)"
~/simdist-docker/run.sh python scripts/sim2sim/value_error.py --models $M --dataset sim2sim_combo_hard_go1_paper --out results/sim2sim/combo_hard/value_error_target.json 2>&1 | grep -vE "Warning|absl|WARNING"
~/simdist-docker/run.sh python scripts/sim2sim/value_error.py --models $M --dataset sim2sim_nominal_go1_paper --out results/sim2sim/combo_hard/value_error_nominal.json 2>&1 | grep -vE "Warning|absl|WARNING"
echo "### value_error done $(date)"
