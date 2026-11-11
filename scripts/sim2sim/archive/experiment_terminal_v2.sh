#!/usr/bin/env bash
# Pre-registered follow-ups of docs/design_value_adaptation_ko.md (sections 5 and 9.3):
# the full recipe (anchor + simulation replay + terminal-aware targets) at 3000 and 9000
# steps on very_weak_motor (15 min), then a batched long-horizon evaluation of all recipes
# (4 from experiment_terminal.sh + 2 new) in the target and the nominal condition.
set -u; cd ~/simdist
RUN=~/simdist-docker/run.sh
until grep -q "^### terminal experiment done" results/sim2sim/terminal_experiment.log; do sleep 120; done
echo "### v2 start $(date)"
COND=very_weak_motor; DS=sim2sim_${COND}_go1_paper_b45000; RES=results/sim2sim/terminal_${COND}
OVR="loss=world_model_value_adapt heads=dynamics_value_residual loss.bootstrap=anchor loss.terminal.enabled=true loss.weights.value_reg=0.0 loss.weights.value_replay=1.0 adapt.replay.dataset_name=2026-10-02_01-18-59 adapt.value_residual.enabled=true adapt.value_residual.type=transformer adapt.value_anchor=true adapt.residual_lr=1e-3 training.warmup_steps=200 training.eval_interval=1000"
for spec in "res_anchor_term_replay_b45000_term:3000" "res_anchor_term_replay_b45000_term9k:9000"; do
  name=go1_paper_${COND}_${spec%%:*}; steps=${spec##*:}
  if ls -d checkpoints/models/$name/[0-9]* >/dev/null 2>&1; then echo "== $name exists"; continue; fi
  echo "== finetune $name ($steps steps)"
  $RUN python scripts/finetune_model.py data.dataset_name=$DS system=go1 checkpoint.resume_checkpoint=go1_paper run_name=$name training.max_steps=$steps $OVR 2>&1 | grep -E "Parameters|^Steps: |Stopping|Error|Traceback" | cut -c1-240
done
echo "### batched long-horizon evaluation $(date)"
CSV=$RES/eval_long_batch.csv
for m in go1_paper go1_paper_${COND}_dyn_b45000_term go1_paper_${COND}_res_anchor_tf_b45000_term go1_paper_${COND}_res_anchor_term_b45000_term go1_paper_${COND}_res_anchor_term_replay_b45000_term go1_paper_${COND}_res_anchor_term_replay_b45000_term9k; do
  for cond in very_weak_motor nominal; do
    if [ -f $CSV ] && grep -q "^$m,$cond," $CSV; then continue; fi
    case $cond in very_weak_motor) phys="sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";; nominal) phys="sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";; esac
    echo "== batch eval $m on $cond"
    $RUN python scripts/simulate_batch.py model.checkpoint=$m $phys sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 'batch.speeds=[0.3,0.6,0.9]' 'batch.seeds=[11,12,13]' output.csv=/workspace/simdist/$CSV output.condition=$cond --headless 2>&1 | grep -E "^RESULT|Traceback|rror:" | cut -c1-200
  done
done
echo "### value error / forgetting $(date)"
$RUN python scripts/sim2sim/value_error.py --models go1_paper_${COND}_res_anchor_term_replay_b45000_term go1_paper_${COND}_res_anchor_term_replay_b45000_term9k --dataset sim2sim_${COND}_go1_paper --stride 10 --out $RES/value_error.json 2>&1 | grep -vE "Warning|absl|WARNING|Checkpoint"
$RUN python scripts/sim2sim/value_forgetting.py --base go1_paper --adapted go1_paper_${COND}_res_anchor_term_replay_b45000_term go1_paper_${COND}_res_anchor_term_replay_b45000_term9k --dataset 2026-10-02_01-18-59 --out $RES/forgetting.json 2>&1 | grep -vE "Warning|absl|WARNING" | tail -3
echo "### v2 done $(date)"
