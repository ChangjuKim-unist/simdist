#!/bin/bash
# Design doc 27 stage 2 K (conditional, §7): split the broad planner's deviation by joint group on the same slip events.
#   candidates: A0 policy head, Fcl planner, Pcalfpol planner with the calves from the policy head, Ponlycalf policy head
#   with the calves from the planner. Run only if stage 1 gives the verdict "direction" and calf / other deviation >= 1.5.
#   usage (gpu1): tmux new -d -s s27k 'bash scripts/sim2sim/stage27_k.sh > results/stage27_k_server.log 2>&1'
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh
BIG=/mnt/newssd/cjkim/simdist/results/stage27; OUT=$BIG/k; mkdir -p $OUT
run() { local g=$1 name=$2
  echo "== K $name gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" $RUN python scripts/sim2sim/branch_eval.py mode=branch model.checkpoint=go1_broadwm control.prior_only=true \
    sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0 sim.terrain=plane sim.episode_length_s=100 \
    branch.anchor=$BIG/anchor_$name.npz branch.csv=$OUT/branch_$name.csv branch.steps_dir=$OUT/steps_$name \
    'branch.groups=[A0,Fcl,Pcalfpol,Ponlycalf]' branch.chunk=150 branch.horizon=150 branch.W=8 branch.J=3 \
    branch.max_events=200 branch.max_ctrl=80 branch.only_events=true \
    output.model_name=K_$name output.condition=friction0.15 --headless 2>&1 | grep -aE "Traceback|rror:|slip events|RESULT branch|round (1|[0-9]*0)/" | head -20
  echo "-- K $name done gpu$g $(date +%H:%M)"; }
echo "### stage27 K start $(date)"
run ${G0:-0} AB_P &
sleep 20; run ${G1:-1} AB_pi &
wait
echo "### stage27 K done $(date)"
