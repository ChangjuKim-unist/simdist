#!/usr/bin/env bash
# Design doc 7, experiment 1: residual-on-frozen dynamics vs full latent-dynamics adaptation, under the
# coupling test. Adapt on (a) the narrow-prior data (sim2sim_<cond>_go1_paper, the data behind B) and
# (b) the prior-first data (sim2sim_<cond>_priorfirst, behind B'), evaluate with the broad student prior
# + planner, 18 robots x 100 s. Compare with B / B0 / B' in results/stage1/eval_priorfirst.csv.
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
echo "### residual experiment start $(date)"
for COND in very_weak_motor nominal; do
  # (a) narrow-prior data (B's data). The nominal narrow-prior dataset is the reference log sim2sim_nominal_go1_paper.
  DSN=sim2sim_${COND}_go1_paper
  [ -d datasets/real/${DSN}_b45000/processed_data_go1_H-25_T-25 ] || COND=$COND DATASET=$DSN BUDGETS=45000 $P process 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-160
  CKN=go1_paper_${COND}_dynres_b45000_narrow
  ls -d checkpoints/models/$CKN/[0-9]* >/dev/null 2>&1 || COND=$COND DATASET=$DSN TAG=narrow BUDGETS=45000 VARIANTS=dynres FT_STEPS=3000 $P finetune 2>&1 | grep -E "^==|Parameters|Steps: 3|Error|Traceback" | cut -c1-160
  # (b) prior-first data (B''s data)
  DSP=sim2sim_${COND}_priorfirst; CKP=go1_paper_${COND}_dynres_b45000_pf
  ls -d checkpoints/models/$CKP/[0-9]* >/dev/null 2>&1 || COND=$COND DATASET=$DSP TAG=pf BUDGETS=45000 VARIANTS=dynres FT_STEPS=3000 $P finetune 2>&1 | grep -E "^==|Parameters|Steps: 3|Error|Traceback" | cut -c1-160
  run Bres_narrow_studentB $CKN $COND prior.student=$STUDENT
  run Bres_pf_studentB     $CKP $COND prior.student=$STUDENT
done
# full adaptation on narrow-prior nominal data (B at nominal did not exist): same recipe as dyn_b45000_term
CKF=go1_paper_nominal_dyn_b45000_narrow
ls -d checkpoints/models/$CKF/[0-9]* >/dev/null 2>&1 || COND=nominal DATASET=sim2sim_nominal_go1_paper TAG=narrow BUDGETS=45000 VARIANTS=dyn FT_STEPS=3000 $P finetune 2>&1 | grep -E "^==|Steps: 3|Error|Traceback" | cut -c1-160
run B_dyn_narrow_studentB $CKF nominal prior.student=$STUDENT
echo "### residual experiment done $(date)"
