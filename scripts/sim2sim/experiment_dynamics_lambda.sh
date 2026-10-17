#!/usr/bin/env bash
# Pre-registered single follow-up of design doc 2 (section 6, third branch): lambda
# sensitivity of the return-matching term (0.1, 0.3), then the 18-robot pairing of the
# baselines (seeds 14-16).
set -u; cd ~/simdist
P=scripts/sim2sim/run_pipeline.sh; RUN=~/simdist-docker/run.sh
until grep -q "^### dynamics experiment done" results/sim2sim/dynamics_experiment.log; do sleep 120; done
echo "### lambda start $(date)"
export COND=very_weak_motor RESULTS=results/sim2sim/terminal_very_weak_motor FT_EXTRA="training.warmup_steps=200"
export LONG_SPEEDS="0.3 0.6 0.9" LONG_SEEDS="11 12 13" TAG=term BUDGETS=45000
DS=sim2sim_${COND}_go1_paper_b45000
BASEOVR="loss=world_model_value_adapt heads=dynamics_value_residual loss.bootstrap=anchor loss.terminal.enabled=true loss.weights.value_reg=0.0 loss.weights.value_replay=1.0 adapt.replay.dataset_name=2026-10-02_01-18-59 adapt.value_residual.enabled=true adapt.value_residual.type=transformer adapt.value_anchor=true adapt.residual_lr=1e-3 training.warmup_steps=200 training.eval_interval=3000"
for lam in 0.1 0.3; do
  name=go1_paper_${COND}_full_ret${lam}_b45000_term
  ls -d checkpoints/models/$name/[0-9]* >/dev/null 2>&1 && continue
  echo "== finetune $name"
  $RUN python scripts/finetune_model.py data.dataset_name=$DS system=go1 checkpoint.resume_checkpoint=go1_paper run_name=$name training.max_steps=3000 $BASEOVR loss.weights.ret=$lam 2>&1 | grep -E "^Steps: |Stopping|Error|Traceback" | cut -c1-240
done
LONG_EXTRA_MODELS="go1_paper_${COND}_full_ret0.1_b45000_term go1_paper_${COND}_full_ret0.3_b45000_term" VARIANTS="" LONG_CONDS="very_weak_motor nominal" $P evaluate_long_batch
$RUN python scripts/sim2sim/return_error.py --models go1_paper_${COND}_full_ret0.1_b45000_term go1_paper_${COND}_full_ret0.3_b45000_term --dataset $DS 2>&1 | grep -vE "Warning|absl|WARNING|Checkpoint"
echo "### lambda done $(date)"
echo "### pairing start $(date)"
RESULTS=results/sim2sim/terminal_very_weak_motor/seeds14_16 LONG_SEEDS="14 15 16" VARIANTS="dyn res_anchor_term_replay" LONG_CONDS="very_weak_motor nominal" $P evaluate_long_batch
tail -n +2 results/sim2sim/terminal_very_weak_motor/seeds14_16/eval_long_batch.csv >> results/sim2sim/terminal_very_weak_motor/eval_long_batch.csv
echo "### pairing done $(date)"
