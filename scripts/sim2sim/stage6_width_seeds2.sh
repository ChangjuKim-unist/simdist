#!/usr/bin/env bash
# Reviewer request: expert seeds for widths 0.9-1.1 and 0.7-1.1 (two more each), datasets, students, D/B0 at nominal and motor 0.4.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/prehw; mkdir -p $OUT; EXP=$OUT/experts.txt; PRE=go1_paper
train() { local tag=$1 motor=$2 seed=$3
  grep -q "^$tag " $EXP && return; echo "== expert $tag (motor DR $motor, seed $seed) $(date +%H:%M)"
  local before=$(ls -t checkpoints/rl | head -1)
  $RUN env GO1_MOTOR_DR=$motor python scripts/train_rl.py --task Go1DR --headless --num_envs 1024 --seed $seed 2>&1 | grep -E "Traceback|rror:|Learning iteration 4999" | head -3
  local after=$(ls -t checkpoints/rl | head -1)
  if [ "$after" != "$before" ] && [ -f checkpoints/rl/$after/model_4999.pt ]; then echo "$tag $after $motor" >> $EXP; echo "   -> $after"; else echo "   !! $tag failed"; fi
}
echo "### width seeds 2 start $(date)"
train w09s2 0.9,1.1 2; train w09s3 0.9,1.1 3; train w07s2 0.7,1.1 2; train w07s3 0.7,1.1 3
for tag in w09s2 w09s3 w07s2 w07s3; do
  rl=$(grep "^$tag " $EXP | awk '{print $2}'); motor=$(grep "^$tag " $EXP | awk '{print $3}'); [ -n "$rl" ] || continue
  if [ ! -f $OUT/dataset_$tag ]; then
    [ -f checkpoints/rl/$rl/policies/policy_4999.pt ] || $RUN python scripts/export_policies.py --task Go1DRPlay -r $rl --headless 2>&1 | grep -E "Error|Traceback" | tail -2
    before=$(ls datasets/sim | sort | tail -1); echo "== generate $tag $(date +%H:%M)"
    $RUN env GO1_MOTOR_DR=$motor python scripts/generate_data.py rl_run=$rl system=go1dr envs=384 steps=5000 expert_prob=1.0 action_corruption.never_noise_prob=1.0 2>&1 | grep -E "Error|Traceback" | head -2
    ds=$(ls datasets/sim | sort | tail -1); [ "$ds" != "$before" ] || { echo "no dataset for $tag"; continue; }; echo "$ds" > $OUT/dataset_$tag
    $RUN python scripts/process_data.py dataset_name=$ds system=go1dr 2>&1 | grep -E "Error|Traceback" | head -2
  fi
  ds=$(cat $OUT/dataset_$tag)
  [ -f checkpoints/students/student_$tag/params.pkl ] || { echo "== student $tag $(date +%H:%M)"; $RUN python scripts/train_student_prior.py --dataset $ds --system go1dr --name student_$tag --num_windows 1500000 --steps 20000 2>&1 | grep -E "^step 20000|Error|Traceback" | tail -1; }
done
phys() { case $1 in nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";; motor0.4) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";; esac; }
SEEDS="[11,12,13,14,15,16]"; CSV=$OUT/eval_width_seeds.csv
run() { local name=$1 ckpt=$2 cond=$3; shift 3
  if [ -f $CSV ] && grep -q "^$name,$cond," $CSV; then return; fi
  echo "== $name on $cond $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$CSV output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:" | head -3
}
for cond in nominal motor0.4; do for tag in w09s2 w09s3 w07s2 w07s3; do
  [ -f checkpoints/students/student_$tag/params.pkl ] || continue
  run D_$tag $PRE $cond prior.student=student_$tag policy.use_as=student_act
  run B0_$tag $PRE $cond prior.student=student_$tag
done; done
echo "### width seeds 2 done $(date)"
