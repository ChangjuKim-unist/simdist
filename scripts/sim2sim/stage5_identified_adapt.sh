#!/usr/bin/env bash
# Design doc 10, P10d: adapt the broad-head world model on on-policy data generated in the IDENTIFIED simulator
# (motor 0.5 = MAP of the 0.4 logs) by the deployed prior (broadhead + planner), then evaluate at the true condition (0.4).
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; P=scripts/sim2sim/run_pipeline.sh; OUT=results/stage5; mkdir -p $OUT; CSV=$OUT/eval_identified.csv
phys() { case $1 in motor0.4) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";; nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
SEEDS="[11,12,13,14,15,16]"
run() { local name=$1 ckpt=$2 cond=$3; shift 3
  if [ -f $CSV ] && grep -q "^$name,$cond," $CSV; then return; fi
  echo "== $name on $cond ($*) $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$CSV output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:" | head -3
}
echo "### identified-adapt start $(date)"
DS=sim2sim_motor0.5_broadhead_onpolicy
[ -d datasets/real/$DS/raw ] || COND=motor0.5 DATASET=$DS COLLECT_CKPT=go1_paper_broadhead ROUNDS=3 COLLECT_STEPS=5000 $P collect 2>&1 | grep -E "^==|total_reward|Error|Traceback" | cut -c1-120
[ -d datasets/real/${DS}_b45000/processed_data_go1_H-25_T-25 ] || COND=motor0.5 DATASET=$DS BUDGETS=45000 $P process 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-160
CK=go1_paper_broadhead_motor0.5_dyn_b45000_id
ls -d checkpoints/models/$CK/[0-9]* >/dev/null 2>&1 || BASE=go1_paper_broadhead COND=motor0.5 DATASET=$DS TAG=id BUDGETS=45000 VARIANTS=dyn FT_STEPS=3000 RESUME_CKPT=go1_paper_broadhead $P finetune 2>&1 | grep -E "^==|Steps: 3|Error|Traceback" | cut -c1-160
run H_broadhead_idadapt $CK motor0.4
run H_broadhead_idadapt $CK nominal
# reference: real-data (true 0.4) on-policy adaptation of the broad head -- the ideal that identification approximates
DS2=sim2sim_very_weak_motor_broadhead_onpolicy
[ -d datasets/real/$DS2/raw ] || COND=very_weak_motor DATASET=$DS2 COLLECT_CKPT=go1_paper_broadhead ROUNDS=3 COLLECT_STEPS=5000 $P collect 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-120
[ -d datasets/real/${DS2}_b45000/processed_data_go1_H-25_T-25 ] || COND=very_weak_motor DATASET=$DS2 BUDGETS=45000 $P process 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-160
CK2=go1_paper_broadhead_very_weak_motor_dyn_b45000_real
ls -d checkpoints/models/$CK2/[0-9]* >/dev/null 2>&1 || BASE=go1_paper_broadhead COND=very_weak_motor DATASET=$DS2 TAG=real BUDGETS=45000 VARIANTS=dyn FT_STEPS=3000 RESUME_CKPT=go1_paper_broadhead $P finetune 2>&1 | grep -E "^==|Steps: 3|Error|Traceback" | cut -c1-160
run H_broadhead_realadapt $CK2 motor0.4
echo "### identified-adapt done $(date)"
