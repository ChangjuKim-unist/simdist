#!/bin/bash
# Design doc 16 L1: go1_ctx friction 0.15 with generator candidates + q50 (head_ctx_noF), local RTX 3060.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage15; mkdir -p $OUT
run() { local name=$1 ckpt=$2 cond=$3 f=$4; shift 4
  if grep -q "^$name,$cond," $OUT/eval_local.csv 2>/dev/null; then return; fi
  echo "== $name on $cond ($*) $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt sim.friction=$f sim.add_mass=0.0 sim.motor_strength=1.0 sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' 'batch.seeds=[11,12,13,14,15,16]' output.csv=/workspace/simdist/$OUT/eval_local.csv output.model_name=$name output.condition=$cond \
    "$@" --headless 2>&1 | grep -E "Traceback|rror:"
}
GEN="control.gen.enabled=true control.gen.weights=/workspace/simdist/$OUT/gen_broad.npz"
CAL="control.calib.enabled=true control.calib.weights=/workspace/simdist/results/stage14/head_ctx_noF.npz control.calib.tau=0.5"
run L_ctx_gen_q50 go1_ctx friction0.15 0.15 $GEN $CAL
echo "### stage15-local done $(date)"
