#!/usr/bin/env bash
# Everything to do on the GPU before the hardware sessions (design doc 6 §9, 2026-10-25):
#  1. M5: adapt go1_paper dynamics on the round-3 real data (collected by the narrow SimDist prior) -> go1_paper_r3data_dyn
#  2. sanity sims of the student prior / prior-only / M5 code paths (single simulator, same code as the robot node)
#  3. expert seeds for widths 0.5-1.1 and 0.3-1.1 (2 more each) and a friction-extended broad expert (feet 0.05-1.2)
#  4. datasets, students, S probes for each new expert
#  5. evaluations: width seeds (D/B0 at nominal, motor 0.4) and the friction axis (friction 0.15 and nominal)
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/prehw; mkdir -p $OUT; PRE=go1_paper
echo "### prehw chain start $(date)"
# ---- 1. M5 -------------------------------------------------------------------------------
if ! ls -d checkpoints/models/go1_paper_r3data_dyn/[0-9]* >/dev/null 2>&1; then
  echo "== M5 finetune go1_paper on go1_simdist_controller (round-3 real data) $(date +%H:%M)"
  $RUN python scripts/finetune_model.py data.dataset_name=go1_simdist_controller system=go1 checkpoint.resume_checkpoint=go1_paper run_name=go1_paper_r3data_dyn training.num_epochs=40 2>&1 | grep -E "Steps: |Error|Traceback|test" | tail -3 | cut -c1-200
fi
# ---- 2. sanity sims ----------------------------------------------------------------------
sanity() { local name=$1; shift
  echo "== sanity $name $(date +%H:%M)"
  $RUN python scripts/simulate_go2.py sim.terrain=plane sim.total_steps=500 sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0 task.forward_vel=0.5 "$@" --headless 2>&1 | grep -E "^\{'total_reward|Planner prior|Error|Traceback" | cut -c1-160
}
sanity student_prior      model.checkpoint=$PRE prior.student=student_broad
sanity student_prior_only model.checkpoint=$PRE prior.student=student_broad control.prior_only=true
sanity M5_student         model.checkpoint=go1_paper_r3data_dyn prior.student=student_broad
# ---- 3. experts --------------------------------------------------------------------------
EXP=$OUT/experts.txt; touch $EXP
train() { local tag=$1 motor=$2 seed=$3 fr=$4   # fr="" = default friction DR
  grep -q "^$tag " $EXP && return
  echo "== expert $tag (motor DR $motor, seed $seed, friction DR ${fr:-default}) $(date +%H:%M)"
  local before=$(ls -t checkpoints/rl | head -1)
  $RUN env GO1_MOTOR_DR=$motor ${fr:+GO1_FRICTION_DR=$fr} python scripts/train_rl.py --task Go1DR --headless --num_envs 1024 --seed $seed 2>&1 | grep -E "Traceback|rror:|Learning iteration 4999" | head -3
  local after=$(ls -t checkpoints/rl | head -1)
  if [ "$after" != "$before" ] && [ -f checkpoints/rl/$after/model_4999.pt ]; then echo "$tag $after $motor ${fr:-}" >> $EXP; echo "   -> $after"; else echo "   !! $tag failed"; fi
}
train w05s2 0.5,1.1 2 ""; train w05s3 0.5,1.1 3 ""; train w03s2 0.3,1.1 2 ""; train w03s3 0.3,1.1 3 ""
train fr03 0.3,1.1 1 0.05,1.2
# ---- 4. datasets, students, probes -------------------------------------------------------
while read tag rl motor fr; do
  [ -n "$tag" ] || continue
  if [ ! -f $OUT/dataset_$tag ]; then
    [ -f checkpoints/rl/$rl/policies/policy_4999.pt ] || $RUN python scripts/export_policies.py --task Go1DRPlay -r $rl --headless 2>&1 | grep -E "Error|Traceback" | tail -2
    before=$(ls datasets/sim | sort | tail -1); echo "== generate $tag $(date +%H:%M)"
    $RUN env GO1_MOTOR_DR=$motor ${fr:+GO1_FRICTION_DR=$fr} python scripts/generate_data.py rl_run=$rl system=go1dr envs=384 steps=5000 expert_prob=1.0 action_corruption.never_noise_prob=1.0 2>&1 | grep -E "Error|Traceback" | head -2
    ds=$(ls datasets/sim | sort | tail -1); [ "$ds" != "$before" ] || { echo "no dataset for $tag"; continue; }
    echo "$ds" > $OUT/dataset_$tag
    $RUN python scripts/process_data.py dataset_name=$ds system=go1dr 2>&1 | grep -E "Error|Traceback" | head -2
  fi
  ds=$(cat $OUT/dataset_$tag)
  [ -f checkpoints/students/student_$tag/params.pkl ] || { echo "== student $tag $(date +%H:%M)"; $RUN python scripts/train_student_prior.py --dataset $ds --system go1dr --name student_$tag --num_windows 1500000 --steps 20000 2>&1 | grep -E "^step 20000|Error|Traceback" | tail -1; }
  [ -f $OUT/probe_$tag.json ] || ( $RUN env JAX_PLATFORMS=cpu python scripts/sim2sim/visibility_probe.py --dataset $ds --system go1dr --num_windows 300000 --steps 6000 --out $OUT/probe_$tag.json > $OUT/probe_$tag.log 2>&1 ) &
done < $EXP
# ---- 5. evaluations ----------------------------------------------------------------------
phys() { case $1 in nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";; motor0.4) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";; friction0.15) echo "sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
SEEDS="[11,12,13,14,15,16]"
run() { local csv=$1 name=$2 ckpt=$3 cond=$4; shift 4
  if [ -f $csv ] && grep -q "^$name,$cond," $csv; then return; fi
  echo "== $name on $cond ($*) $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$csv output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:" | head -3
}
# friction axis first (shorter): existing broad student vs friction-extended student vs SimDist planner (policy head)
for cond in friction0.15 nominal; do
  run $OUT/eval_friction.csv A_simdist_planner $PRE $cond
  run $OUT/eval_friction.csv D_w03 $PRE $cond prior.student=student_w03 policy.use_as=student_act
  [ -f checkpoints/students/student_fr03/params.pkl ] && run $OUT/eval_friction.csv D_fr03 $PRE $cond prior.student=student_fr03 policy.use_as=student_act
  run $OUT/eval_friction.csv B0_w03 $PRE $cond prior.student=student_w03
  [ -f checkpoints/students/student_fr03/params.pkl ] && run $OUT/eval_friction.csv B0_fr03 $PRE $cond prior.student=student_fr03
done
for cond in nominal motor0.4; do
  for tag in w05s2 w05s3 w03s2 w03s3; do
    [ -f checkpoints/students/student_$tag/params.pkl ] || continue
    run $OUT/eval_width_seeds.csv D_$tag $PRE $cond prior.student=student_$tag policy.use_as=student_act
    run $OUT/eval_width_seeds.csv B0_$tag $PRE $cond prior.student=student_$tag
  done
done
wait
echo "### prehw chain done $(date)"
