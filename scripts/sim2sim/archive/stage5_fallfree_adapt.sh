#!/usr/bin/env bash
# Design doc 10 §7 follow-up: why did on-policy data at the true condition (0.4, fall-heavy) hurt the broad head?
# Re-adapt on the same data with fall-boundary windows removed (loss.boundary_weight=0), and evaluate the harmful
# checkpoint at nominal to see whether the harm is condition-specific.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; P=scripts/sim2sim/run_pipeline.sh; OUT=results/stage5; CSV=$OUT/eval_identified.csv
phys() { case $1 in motor0.4) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";; nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
SEEDS="[11,12,13,14,15,16]"
run() { local name=$1 ckpt=$2 cond=$3; shift 3
  if [ -f $CSV ] && grep -q "^$name,$cond," $CSV; then return; fi
  echo "== $name on $cond ($*) $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$CSV output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:" | head -3
}
echo "### fallfree start $(date)"
DS2=sim2sim_very_weak_motor_broadhead_onpolicy; CK=go1_paper_broadhead_very_weak_motor_dyn_b45000_nofall
ls -d checkpoints/models/$CK/[0-9]* >/dev/null 2>&1 || BASE=go1_paper_broadhead COND=very_weak_motor DATASET=$DS2 TAG=nofall BUDGETS=45000 VARIANTS=dyn FT_STEPS=3000 RESUME_CKPT=go1_paper_broadhead FT_EXTRA="loss.boundary_weight=0.0" $P finetune 2>&1 | grep -E "^==|Steps: 3|Error|Traceback" | cut -c1-160
run H_broadhead_realadapt_nofall $CK motor0.4
run H_broadhead_realadapt go1_paper_broadhead_very_weak_motor_dyn_b45000_real nominal
echo "### fallfree done $(date)"
