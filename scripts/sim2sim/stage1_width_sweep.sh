#!/usr/bin/env bash
# Design doc 6, E1: motor-DR width sweep. For each width: export the expert, generate expert data under
# that DR (physics labels), process, train a student (identical recipe for all widths), probe S_motor;
# then evaluate student-alone (D) and student + planner on the pretrained dynamics (B0) at nominal,
# motor 0.6 and motor 0.4 with 18 robots each. Results: results/stage1/eval_width.csv.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage1; mkdir -p $OUT; CSV=$OUT/eval_width.csv; PRE=go1_paper
# tag  range    rl_run                 system  (datasets for w03/w10 come from stage 0)
WIDTHS="w10:1.0,1.0:2026-10-01_10-38-36:go1 w09:0.9,1.1:2026-10-21_15-49-45:go1dr w07:0.7,1.1:2026-10-21_13-21-39:go1dr w05:0.5,1.1:2026-10-21_10-55-09:go1dr w03:0.3,1.1:2026-10-20_07-40-06:go1dr"
echo "### stage1 width sweep start $(date)"
for spec in $WIDTHS; do IFS=: read tag range rl sys <<< "$spec"
  case $tag in w10) echo $(cat results/stage0/dataset_narrow) > $OUT/dataset_$tag;; w03) echo $(cat results/stage0/dataset_broad) > $OUT/dataset_$tag;; esac
  if [ ! -f $OUT/dataset_$tag ]; then
    [ -f checkpoints/rl/$rl/policies/policy_4999.pt ] || $RUN python scripts/export_policies.py --task Go1DRPlay -r $rl --headless 2>&1 | grep -E "Export|Error|Traceback" | tail -2
    before=$(ls datasets/sim | sort | tail -1); echo "== generate $tag (DR $range, $rl) $(date +%H:%M)"
    $RUN env GO1_MOTOR_DR=$range python scripts/generate_data.py rl_run=$rl system=$sys envs=384 steps=5000 expert_prob=1.0 action_corruption.never_noise_prob=1.0 2>&1 | grep -E "steps will be|GO1_MOTOR_DR|Error|Traceback" | head -3
    ds=$(ls datasets/sim | sort | tail -1); [ "$ds" != "$before" ] || { echo "no new dataset for $tag"; exit 1; }
    echo "$ds" > $OUT/dataset_$tag
    $RUN python scripts/process_data.py dataset_name=$ds system=$sys 2>&1 | grep -E "Error|Traceback|num_trajectories" | head -2
  fi
  ds=$(cat $OUT/dataset_$tag)
  if [ ! -f checkpoints/students/student_$tag/params.pkl ]; then
    echo "== student $tag from $ds $(date +%H:%M)"
    $RUN python scripts/train_student_prior.py --dataset $ds --system $sys --name student_$tag --num_windows 1500000 --steps 20000 2>&1 | grep -E "^step 20000|saved|loss|Error|Traceback" | tail -2
  fi
  if [ "$sys" = go1dr ] && [ ! -f $OUT/probe_$tag.json ]; then   # S_motor (CPU, in the background)
    ( $RUN env JAX_PLATFORMS=cpu python scripts/sim2sim/visibility_probe.py --dataset $ds --system $sys --num_windows 300000 --steps 6000 --out $OUT/probe_$tag.json > $OUT/probe_$tag.log 2>&1 ) &
  fi
done
phys() { case $1 in nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";; motor0.6) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.6";; motor0.4) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";; esac; }
SEEDS="[11,12,13,14,15,16]"
run() { local name=$1 ckpt=$2 cond=$3; shift 3
  if [ -f $CSV ] && grep -q "^$name,$cond," $CSV; then return; fi
  echo "== $name on $cond ($*) $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$CSV output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:" | head -3
}
echo "### evaluation $(date)"
for cond in nominal motor0.6 motor0.4; do
  for spec in $WIDTHS; do IFS=: read tag range rl sys <<< "$spec"
    run D_$tag  $PRE $cond prior.student=student_$tag policy.use_as=student_act
    run B0_$tag $PRE $cond prior.student=student_$tag
  done
done
wait
echo "### stage1 width sweep done $(date)"
