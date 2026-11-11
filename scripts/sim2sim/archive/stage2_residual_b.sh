#!/usr/bin/env bash
# Design doc 7, X1b: residual dynamics WITHOUT the direct action channel (dynres_na), after the dynres residual
# adapted on prior-first data collapsed under the planner (X1). Same data/evaluation as stage2_residual.sh.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; P=scripts/sim2sim/run_pipeline.sh
OUT=results/stage2; mkdir -p $OUT; CSV=$OUT/eval_residual.csv; STUDENT=student_broad
phys() { case $1 in very_weak_motor) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";; nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
SEEDS="[11,12,13,14,15,16]"
run() { local name=$1 ckpt=$2 cond=$3; shift 3
  if [ -f $CSV ] && grep -q "^$name,$cond," $CSV; then return; fi
  echo "== $name on $cond ($*) $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$CSV output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:" | head -3
}
echo "### residual-b experiment start $(date)"
for COND in very_weak_motor nominal; do
  for spec in "sim2sim_${COND}_go1_paper:narrow" "sim2sim_${COND}_priorfirst:pf"; do IFS=: read DS TAG <<< "$spec"
    CK=go1_paper_${COND}_dynres_na_b45000_$TAG
    ls -d checkpoints/models/$CK/[0-9]* >/dev/null 2>&1 || COND=$COND DATASET=$DS TAG=$TAG BUDGETS=45000 VARIANTS=dynres_na FT_STEPS=3000 $P finetune 2>&1 | grep -E "^==|Parameters|Steps: 3|Error|Traceback" | cut -c1-160
    run BresNA_${TAG}_studentB $CK $COND prior.student=$STUDENT
  done
done
echo "### residual-b experiment done $(date)"
