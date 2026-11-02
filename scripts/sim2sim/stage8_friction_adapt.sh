#!/usr/bin/env bash
# Design doc 11 S7: does SimDist's own real-data adaptation buy coverage on the CONTACT axis (friction 0.15),
# where replay identification fails and SimDist's hardware result (slippery slope) lives?
# SimDist head + planner: pretrained vs adapted on 45k steps collected by itself at friction 0.15 (and combo_hard).
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; P=scripts/sim2sim/run_pipeline.sh; OUT=results/stage8; mkdir -p $OUT; CSV=$OUT/eval_friction_adapt.csv
until grep -q "### idcurve done" results/stage7b_idcurve.log 2>/dev/null; do sleep 300; done
phys() { case $1 in friction0.15) echo "sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0";; combo_hard) echo "sim.friction=0.3 sim.add_mass=3.0 sim.motor_strength=0.6";; nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
SEEDS="[11,12,13,14,15,16]"
run() { local name=$1 ckpt=$2 cond=$3; shift 3
  if [ -f $CSV ] && grep -q "^$name,$cond," $CSV; then return; fi
  echo "== $name on $cond ($*) $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$CSV output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:" | head -3
}
echo "### friction-adapt start $(date)"
DS=sim2sim_friction0.15_go1_paper
[ -d datasets/real/$DS/raw ] || COND=friction0.15 DATASET=$DS COLLECT_CKPT=go1_paper ROUNDS=3 COLLECT_STEPS=5000 $P collect 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-120
[ -d datasets/real/${DS}_b45000/processed_data_go1_H-25_T-25 ] || COND=friction0.15 DATASET=$DS BUDGETS=45000 $P process 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-120
CK=go1_paper_friction0.15_dyn_b45000_term
ls -d checkpoints/models/$CK/[0-9]* >/dev/null 2>&1 || COND=friction0.15 DATASET=$DS TAG=term BUDGETS=45000 VARIANTS=dyn FT_STEPS=3000 $P finetune 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-120
run A_simdist_pretrained go1_paper friction0.15
run A_simdist_adapted $CK friction0.15
run A_simdist_adapted $CK nominal
# combo_hard: adapted checkpoint from design doc 1 era exists
run A_simdist_pretrained go1_paper combo_hard
run A_simdist_adapted_combo go1_paper_combo_hard_dyn_b45000_term combo_hard
echo "### friction-adapt done $(date)"
