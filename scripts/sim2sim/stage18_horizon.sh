#!/bin/bash
# Design 18 section 9.7: F_cl - A0 versus the replanning horizon L = 25 / 125 / 500 steps, motor-adapted model on friction0.15,
# same branch points (seeds 11, 12 of the check-2 anchor), candidates [A0, Fcl] only, one GPU per run. gpu1.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; export SIMDIST_DIR=~/simdist
BIG=/mnt/newssd/cjkim/simdist/results/stage18; OUT=$BIG  # mounted at the same path inside the container
GPUS_LIST=${GPUS_LIST:-"0 1 2"}; read -ra GPUS <<< "$GPUS_LIST"
MODEL=go1_paper_very_weak_motor_dyn_b45000_term
COMMON="control.prior_only=true sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0"
echo "### horizon start $(date) GPUs: $GPUS_LIST"
i=0
for L in 25 125 500; do
  gpu=${GPUS[$i]}; i=$((i+1))
  ( echo "== horizon L=$L gpu$gpu $(date +%H:%M)"
    GPUS="\"device=$gpu\"" $RUN python scripts/sim2sim/branch_eval.py mode=branch model.checkpoint=$MODEL $COMMON \
      branch.anchor=$OUT/anchor_papermotor_friction0.15.npz branch.csv=$OUT/branch_h${L}_papermotor_friction0.15.csv \
      'branch.groups=[A0,Fcl]' branch.chunk=$L branch.horizon=$((L+100)) 'branch.seeds=[11,12]' branch.W=8 \
      output.model_name=papermotor output.condition=friction0.15 --headless 2>&1 | grep -E "RESULT branch|Traceback|rror:|dropping|branch points:|round [0-9]*0/" | tail -6
    echo "== horizon L=$L done $(date +%H:%M)" ) &
done
wait
echo "### horizon end $(date)"
