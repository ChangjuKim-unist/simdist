#!/bin/bash
# Design doc 15 local supplement (RTX 3060): (1) go1_ctx replication of C4 with the locally trained head_ctx_noF;
# (2) extra go1_paper traces (prior / planner x 4 conditions) so that the paper heads can be retrained for C5.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage14; TR=$OUT/traces; mkdir -p $OUT $TR
phys() { local m=1.0 f=0.8 a=0.0 tok; IFS=_ read -ra toks <<< "$1"
  for tok in "${toks[@]}"; do case $tok in motor*) m=${tok#motor};; friction*) f=${tok#friction};; mass*) a=${tok#mass};; esac; done
  echo "sim.friction=$f sim.add_mass=$a sim.motor_strength=$m"; }
run() { local name=$1 ckpt=$2 cond=$3; shift 3
  if grep -q "^$name,$cond," $OUT/eval_local.csv 2>/dev/null; then return; fi
  echo "== $name on $cond ($*) $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' 'batch.seeds=[11,12,13,14,15,16]' output.csv=/workspace/simdist/$OUT/eval_local.csv output.model_name=$name output.condition=$cond \
    "$@" --headless 2>&1 | grep -E "Traceback|rror:|trace written"
}
CAL="control.calib.enabled=true control.calib.weights=/workspace/simdist/$OUT/head_ctx_noF.npz"
PO=control.prior_only=true
run L_ctx_q10 go1_ctx friction0.15 $CAL control.calib.tau=0.1
for c in motor0.6 friction0.3 mass3 motor0.5_friction0.3; do
  run T_paper_prior go1_paper $c $PO output.trace_npz=/workspace/simdist/$TR/T_paper_prior_$c.npz
  run T_paper_planner go1_paper $c output.trace_npz=/workspace/simdist/$TR/T_paper_planner_$c.npz
done
run L_ctx_q50 go1_ctx friction0.15 $CAL control.calib.tau=0.5
echo "### stage14-local done $(date)"
