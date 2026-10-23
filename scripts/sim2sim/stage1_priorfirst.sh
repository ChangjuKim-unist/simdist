#!/usr/bin/env bash
# Design doc 6, E2: prior-first adaptation. Collect 45k steps with the broad student + planner on the
# pretrained dynamics (B0) in the target condition, adapt the dynamics (same recipe as dyn_b45000_term),
# and compare B' (adapted on prior-first data) with B0 (pretrained) and B (adapted on narrow-prior data),
# all with the broad student prior, 18 robots x 100 s. Results: results/stage1/eval_priorfirst.csv.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; P=scripts/sim2sim/run_pipeline.sh
OUT=results/stage1; CSV=$OUT/eval_priorfirst.csv; STUDENT=student_broad
phys() { case $1 in very_weak_motor) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";; nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
SEEDS="[11,12,13,14,15,16]"
run() { local name=$1 ckpt=$2 cond=$3; shift 3
  if [ -f $CSV ] && grep -q "^$name,$cond," $CSV; then return; fi
  echo "== $name on $cond ($*) $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$CSV output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:" | head -3
}
echo "### prior-first start $(date)"
for COND in very_weak_motor nominal; do
  DS=sim2sim_${COND}_priorfirst
  if [ ! -d datasets/real/$DS/raw ]; then
    echo "== collect $DS with B0 (student $STUDENT + planner, go1_paper) $(date +%H:%M)"
    COND=$COND DATASET=$DS COLLECT_CKPT=go1_paper PRIOR_STUDENT=$STUDENT ROUNDS=3 COLLECT_STEPS=5000 $P collect 2>&1 | grep -E "^==|total_reward|Error|Traceback|prior" | cut -c1-160
  fi
  [ -d datasets/real/${DS}_b45000/processed_data_go1_H-25_T-25 ] || COND=$COND DATASET=$DS BUDGETS=45000 $P process 2>&1 | grep -E "^==|Error|Traceback|num_traj" | cut -c1-160
  CK=go1_paper_${COND}_dyn_b45000_pf
  ls -d checkpoints/models/$CK/[0-9]* >/dev/null 2>&1 || COND=$COND DATASET=$DS TAG=pf BUDGETS=45000 VARIANTS=dyn FT_STEPS=3000 $P finetune 2>&1 | grep -E "^==|Steps|Error|Traceback" | cut -c1-160
  run Bp_pf_studentB $CK $COND prior.student=$STUDENT
  run B0_pre_studentB go1_paper $COND prior.student=$STUDENT
  B=go1_paper_${COND}_dyn_b45000_term; ls -d checkpoints/models/$B/[0-9]* >/dev/null 2>&1 && run B_dyn_studentB $B $COND prior.student=$STUDENT
done
echo "### prior-first done $(date)"
