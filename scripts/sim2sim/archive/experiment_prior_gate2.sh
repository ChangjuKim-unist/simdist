#!/usr/bin/env bash
# Gate 2 of docs/design_prior_refresh_ko.md: refresh the planner prior in imagination with the
# adapted dynamics (loss=policy_refresh, heads=policy_only, anchor to the simulation policy),
# then batched long-horizon evaluation (18 robots) in the target, nominal and combo conditions.
set -u; cd ~/simdist
RUN=~/simdist-docker/run.sh
BASE=go1_paper_very_weak_motor_dyn_b45000_term
DS=sim2sim_very_weak_motor_go1_paper_b45000
OUT=results/sim2sim/prior_gate2; mkdir -p $OUT; CSV=$OUT/eval_long_batch.csv
phys() { case $1 in very_weak_motor) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";; combo_hard) echo "sim.friction=0.3 sim.add_mass=3.0 sim.motor_strength=0.6";; nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
run() {  # run <name> <ckpt> <condition> <extra...>
  local name=$1 ckpt=$2 cond=$3; shift 3
  if [ -f $CSV ] && grep -q "^$name,$cond," $CSV; then return; fi
  echo "== $name on $cond ($*) $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' 'batch.seeds=[11,12,13,14,15,16]' output.csv=/workspace/simdist/$CSV output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:" | head -3
}
echo "### gate2 start $(date)"
for beta in 0.1 1.0; do
  name=${BASE}_prior_b${beta}
  if ! ls -d checkpoints/models/$name/[0-9]* >/dev/null 2>&1; then
    echo "== finetune $name (beta=$beta, 2000 steps) $(date +%H:%M)"
    $RUN python scripts/finetune_model.py data.dataset_name=$DS system=go1 checkpoint.resume_checkpoint=$BASE run_name=$name \
      training.max_steps=2000 training.eval_interval=500 training.warmup_steps=100 loss=policy_refresh heads=policy_only adapt.policy_anchor=true \
      loss.weights.policy_anchor=$beta 2>&1 | grep -E "Parameters|^Steps: |Stopping|Error|Traceback" | sed -E "s/'steps_per_second': [0-9.]+, //" | cut -c1-260
  fi
done
for beta in 0.1 1.0; do
  name=${BASE}_prior_b${beta}
  run prior_b${beta}          $name very_weak_motor
  run prior_b${beta}_mix50    $name very_weak_motor control.mixture_coef=0.5
  run prior_b${beta}          $name nominal
  run prior_b${beta}          $name combo_hard
done
echo "### gate2 done $(date)"
