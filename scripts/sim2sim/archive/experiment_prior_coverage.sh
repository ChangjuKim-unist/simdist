#!/usr/bin/env bash
# Design doc 4 pipeline: wait for the broad expert (Go1DR PPO), export it, generate expert data
# under motor-strength DR, train the broad student prior, then run the pre-registered arms.
set -u; cd ~/simdist
RUN=~/simdist-docker/run.sh
RL=2026-10-20_07-40-06
OUT=results/prior_coverage; mkdir -p $OUT; CSV=$OUT/eval_long_batch.csv
DYN=go1_paper_very_weak_motor_dyn_b45000_term; PRE=go1_paper
OLDEXP="policy.rl_run=2026-10-01_10-38-36 policy.iteration=4999"
NEWEXP="policy.rl_run=$RL policy.iteration=4999"
until [ -f checkpoints/rl/$RL/model_4999.pt ]; do sleep 300; done
sleep 120
echo "### expert ready $(date)"
if [ ! -f checkpoints/rl/$RL/policies/policy_4999.pt ]; then
  $RUN python scripts/export_policies.py --task Go1DRPlay -r $RL --headless 2>&1 | grep -E "Export|Error|Traceback" | tail -3
fi
if [ ! -f $OUT/dataset_name ]; then
  echo "### generate broad-expert data $(date)"
  before=$(ls datasets/sim | sort | tail -1)
  $RUN python scripts/generate_data.py rl_run=$RL system=go1dr envs=384 steps=20000 expert_prob=1.0 action_corruption.never_noise_prob=1.0 2>&1 | grep -E "steps will be|Error|Traceback|Saved|dataset" | tail -4
  DS=$(ls datasets/sim | sort | tail -1); [ "$DS" != "$before" ] || { echo "no new dataset"; exit 1; }
  echo "$DS" > $OUT/dataset_name
fi
DS=$(cat $OUT/dataset_name); echo "### dataset $DS"
[ -d datasets/sim/$DS/processed_data_go1dr_H-25_T-25 ] || $RUN python scripts/process_data.py dataset_name=$DS system=go1dr 2>&1 | grep -E "Error|Traceback|num_trajectories" | tail -2
[ -f checkpoints/students/student_broad/params.pkl ] || $RUN python scripts/train_student_prior.py --dataset $DS --system go1dr --name student_broad --num_windows 2000000 --steps 20000 2>&1 | grep -E "^step|saved|Error|Traceback" | tail -4
phys() { case $1 in motor0.4) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";; motor0.2) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.2";; combo_hard) echo "sim.friction=0.3 sim.add_mass=3.0 sim.motor_strength=0.6";; nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
SEEDS="[11,12,13,14,15,16]"   # main arms: 18 robots; ablation arms set SEEDS="[11,12,13]" (9 robots)
run() {  # run <name> <ckpt> <condition> <extra...>
  local name=$1 ckpt=$2 cond=$3; shift 3
  if [ -f $CSV ] && grep -q "^$name,$cond," $CSV; then return; fi
  echo "== $name on $cond ($*) $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$CSV output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:" | head -3
}
echo "### evaluation $(date)"
for cond in motor0.4 motor0.2 nominal combo_hard; do
  run F_expert_broad   $PRE $cond $NEWEXP policy.use_as=act
  run D_student_broad  $DYN $cond prior.student=student_broad policy.use_as=student_act
  run B_dyn_studentB   $DYN $cond prior.student=student_broad
done
run A_base           $DYN motor0.2
run E_expert_init    $DYN motor0.2 $OLDEXP policy.use_as=init control.init_mode=external
SEEDS="[11,12,13]"
for cond in motor0.4 nominal; do
  run B0_pre_studentB  $PRE $cond prior.student=student_broad
  run C_dyn_studentN   $DYN $cond prior.student=student_narrow
  run D_student_narrow $DYN $cond prior.student=student_narrow policy.use_as=student_act
done
run F_expert_old     $PRE motor0.2 $OLDEXP policy.use_as=act
echo "### prior coverage done $(date)"
