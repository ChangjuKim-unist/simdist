#!/bin/bash
# Design doc 16 §7 S1b (local): adapted go1_ctx (friction 0.15 dynamics) with generative candidates + q50 (head_ctx_noF).
cd /home/rml/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage16; mkdir -p $OUT
GEN="control.gen.enabled=true control.gen.weights=/workspace/simdist/results/stage15/gen_broad.npz"
CAL="control.calib.enabled=true control.calib.weights=/workspace/simdist/results/stage14/head_ctx_noF.npz control.calib.tau=0.5"
for cond in friction0.15 motor0.4; do
  case $cond in friction0.15) P="sim.friction=0.15 sim.motor_strength=1.0";; motor0.4) P="sim.friction=0.8 sim.motor_strength=0.4";; esac
  grep -q "^S1b_ctxadapt_gen_q50,$cond," $OUT/eval_local.csv 2>/dev/null && continue
  echo "== S1b_ctxadapt_gen_q50 on $cond $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=go1_ctx_friction0.15_dyn_b45000_term $P sim.add_mass=0.0 sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' 'batch.seeds=[11,12,13,14,15,16]' output.csv=/workspace/simdist/$OUT/eval_local.csv output.terms_csv=/workspace/simdist/$OUT/terms_local.csv \
    output.model_name=S1b_ctxadapt_gen_q50 output.condition=$cond $GEN $CAL --headless 2>&1 | grep -E "Traceback|rror:"
done
echo "### local supp done $(date)"
