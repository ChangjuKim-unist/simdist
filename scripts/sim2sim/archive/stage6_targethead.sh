#!/usr/bin/env bash
# Design doc 10 §8: identification-driven ("targeted") prior. Expert with motor DR 0.3-0.7 (covering the identified
# ~0.5 with margin), pretraining-style data, policy-head retraining on go1_paper, evaluation vs SimDist / broadhead.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage6; mkdir -p $OUT; CSV=$OUT/eval_targethead.csv
echo "### targethead start $(date)"
EXP=$OUT/expert.txt
if [ ! -f $EXP ]; then
  before=$(ls -t checkpoints/rl | head -1); echo "== expert motor DR 0.3,0.7 $(date +%H:%M)"
  $RUN env GO1_MOTOR_DR=0.3,0.7 python scripts/train_rl.py --task Go1DR --headless --num_envs 1024 2>&1 | grep -E "Traceback|rror:|Learning iteration 4999" | head -3
  after=$(ls -t checkpoints/rl | head -1); [ "$after" != "$before" ] && [ -f checkpoints/rl/$after/model_4999.pt ] && echo "$after" > $EXP || { echo "expert failed"; exit 1; }
fi
RL=$(cat $EXP); echo "== expert $RL"
[ -f checkpoints/rl/$RL/policies/policy_4999.pt ] || $RUN python scripts/export_policies.py --task Go1DRPlay -r $RL --headless 2>&1 | grep -E "Error|Traceback" | tail -2
if [ ! -f $OUT/dataset ]; then
  before=$(ls datasets/sim | sort | tail -1); echo "== generate pretraining-style data (DR 0.3-0.7) $(date +%H:%M)"
  $RUN env GO1_MOTOR_DR=0.3,0.7 python scripts/generate_data.py rl_run=$RL system=go1dr envs=384 steps=20000 2>&1 | grep -E "Error|Traceback" | head -2
  ds=$(ls datasets/sim | sort | tail -1); [ "$ds" != "$before" ] || { echo "no dataset"; exit 1; }; echo "$ds" > $OUT/dataset
  $RUN python scripts/process_data.py dataset_name=$ds system=go1dr 2>&1 | grep -E "Error|Traceback" | head -2
fi
DS=$(cat $OUT/dataset)
ls -d checkpoints/models/go1_paper_targethead/[0-9]* >/dev/null 2>&1 || $RUN python scripts/finetune_model.py data.dataset_name=$DS system=go1dr checkpoint.resume_checkpoint=go1_paper run_name=go1_paper_targethead loss=world_model heads=policy_only training.max_steps=60000 training.eval_interval=10000 2>&1 | grep -E "Steps: |Error|Traceback" | cut -c1-160 | tail -2
phys() { case $1 in nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";; motor0.4) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";; friction0.15) echo "sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
SEEDS="[11,12,13,14,15,16]"
run() { local name=$1 ckpt=$2 cond=$3; shift 3
  if [ -f $CSV ] && grep -q "^$name,$cond," $CSV; then return; fi
  echo "== $name on $cond $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$CSV output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:" | head -3
}
for cond in motor0.4 nominal friction0.15; do run H_targethead_planner go1_paper_targethead $cond; done
echo "### targethead done $(date)"
