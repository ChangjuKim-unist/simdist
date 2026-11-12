#!/bin/bash
# Design 18 section 9.2: branch-point features + world-model imagination for the check-3 (generator base) branch points, local GPU.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; export SIMDIST_DIR=~/simdist
OUT=/workspace/simdist/results/stage18; GEN=/workspace/simdist/results/stage15/gen_broad.npz
phys() { local m=1.0 f=0.8 a=0.0 tok; IFS=_ read -ra toks <<< "$1"
  for tok in "${toks[@]}"; do case $tok in motor*) m=${tok#motor};; friction*) f=${tok#friction};; mass*) a=${tok#mass};; esac; done
  echo "sim.friction=$f sim.add_mass=$a sim.motor_strength=$m"; }
echo "### features start $(date)"
for cond in nominal motor0.4 friction0.15 mass3 motor0.35 friction0.1; do
  [ -f results/stage18/feat_gen_$cond.npz ] && continue
  echo "== features $cond $(date +%H:%M)"
  $RUN python scripts/sim2sim/branch_eval.py mode=features model.checkpoint=go1_broadwm $(phys $cond) \
    control.prior_only=true sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    branch.anchor=$OUT/anchor_gen_$cond.npz branch.out=$OUT/feat_gen_$cond.npz branch.W=64 branch.base=gen branch.gen_weights=$GEN \
    output.condition=$cond --headless 2>&1 | grep -E "RESULT|Traceback|rror:" | tail -3
done
echo "### features end $(date)"
