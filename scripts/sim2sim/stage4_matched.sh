#!/usr/bin/env bash
# Design doc 9: X9a prior-consistent adaptation, X9b density-ratio weighted prior. Runs after the stage 3 evaluation.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; P=scripts/sim2sim/run_pipeline.sh; OUT=results/stage4; mkdir -p $OUT; STUDENT=student_broad
BROAD=$(cat results/stage3/dataset_broad_pretrain)
phys() { case $1 in very_weak_motor|motor0.4) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";; nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";; low_friction|friction0.15) echo "sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
SEEDS="[11,12,13,14,15,16]"
run() { local csv=$1 name=$2 ckpt=$3 cond=$4; shift 4
  if [ -f $csv ] && grep -q "^$name,$cond," $csv; then return; fi
  echo "== $name on $cond ($*) $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$csv output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:" | head -3
}
until grep -q "### stage3 eval done" results/stage3_eval.log 2>/dev/null; do sleep 300; done
echo "### stage4 start $(date)"
# ---- X9a: prior-consistent adaptation on the narrow-prior data (B's data), sigma 0.5 (main) and 0.3 (secondary)
CSV=$OUT/eval_pw.csv
for COND in very_weak_motor nominal; do
  DS=sim2sim_${COND}_go1_paper_b45000
  for SIG in 0.5 0.3; do
    $RUN python scripts/prior_consistency_weights.py --dataset $DS --student $STUDENT --sigma $SIG 2>&1 | grep -E "mean_weight|effective|Error|Traceback" | tr -d '\n'; echo " <- $DS sigma $SIG"
    CK=go1_paper_${COND}_dyn_pw_b45000_s$SIG
    ls -d checkpoints/models/$CK/[0-9]* >/dev/null 2>&1 || COND=$COND DATASET=sim2sim_${COND}_go1_paper TAG=s$SIG BUDGETS=45000 VARIANTS=dyn_pw FT_STEPS=3000 $P finetune 2>&1 | grep -E "^==|Steps: 3|Error|Traceback|prior_weight" | cut -c1-160
    run $CSV Bpw_s${SIG}_studentB $CK $COND prior.student=$STUDENT
  done
done
# P9a'': mixed data (narrow + prior-first, 90k) with weights, motor 0.4
MIX=sim2sim_very_weak_motor_mix
if [ ! -d datasets/real/$MIX/raw ]; then mkdir -p datasets/real/$MIX/raw; for src in sim2sim_very_weak_motor_go1_paper sim2sim_very_weak_motor_priorfirst; do for f in datasets/real/$src/raw/*; do ln -s ../../$src/raw/$(basename $f) datasets/real/$MIX/raw/${src##sim2sim_very_weak_motor_}_$(basename $f); done; done; fi
[ -d datasets/real/${MIX}_b90000/processed_data_go1_H-25_T-25 ] || COND=very_weak_motor DATASET=$MIX BUDGETS=90000 $P process 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-160
$RUN python scripts/prior_consistency_weights.py --dataset ${MIX}_b90000 --student $STUDENT --sigma 0.5 2>&1 | grep -E "mean_weight|Error" | tr -d '\n'; echo
CK=go1_paper_very_weak_motor_dyn_pw_b90000_mix; ls -d checkpoints/models/$CK/[0-9]* >/dev/null 2>&1 || COND=very_weak_motor DATASET=$MIX TAG=mix BUDGETS=90000 VARIANTS=dyn_pw FT_STEPS=3000 $P finetune 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-160
run $CSV Bpw_mix90k_studentB $CK very_weak_motor prior.student=$STUDENT
CK=go1_paper_very_weak_motor_dyn_b90000_mix; ls -d checkpoints/models/$CK/[0-9]* >/dev/null 2>&1 || COND=very_weak_motor DATASET=$MIX TAG=mix BUDGETS=90000 VARIANTS=dyn FT_STEPS=3000 $P finetune 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-160
run $CSV B_mix90k_studentB $CK very_weak_motor prior.student=$STUDENT
# ---- X9b: density-ratio weighted student on the broad pretraining-style data
CSV=$OUT/eval_dr.csv
DSF=sim2sim_low_friction_priorfirst
[ -d datasets/real/$DSF/raw ] || COND=low_friction DATASET=$DSF COLLECT_CKPT=go1_paper PRIOR_STUDENT=$STUDENT ROUNDS=3 COLLECT_STEPS=5000 $P collect 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-160
[ -d datasets/real/${DSF}_b45000/processed_data_go1_H-25_T-25 ] || COND=low_friction DATASET=$DSF BUDGETS=45000 $P process 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-160
[ -f $OUT/w_deploy.npz ] || $RUN python scripts/density_ratio_weights.py --sim $BROAD --real sim2sim_very_weak_motor_priorfirst_b45000 ${DSF}_b45000 sim2sim_nominal_priorfirst_b45000 --out $OUT/w_deploy.npz 2>&1 | grep -E "windows|AUC|weights:|Error|Traceback" | cut -c1-200
for cond_ds in "motor:sim2sim_very_weak_motor_priorfirst_b45000" "friction:${DSF}_b45000" "nominal:sim2sim_nominal_priorfirst_b45000"; do IFS=: read c d <<< "$cond_ds"
  [ -f $OUT/w_$c.npz ] || $RUN python scripts/density_ratio_weights.py --sim $BROAD --real $d --steps 2000 --out $OUT/w_$c.npz 2>&1 | grep -E "AUC" | cut -c1-200; done
[ -f checkpoints/students/student_u77/params.pkl ] || $RUN python scripts/train_student_prior.py --dataset $BROAD --system go1dr --name student_u77 --num_windows 1500000 --steps 20000 2>&1 | grep -E "^step 20000|Error|Traceback" | tail -1
[ -f checkpoints/students/student_dr77/params.pkl ] || $RUN python scripts/train_student_prior.py --dataset $BROAD --system go1dr --name student_dr77 --num_windows 1500000 --steps 20000 --weights $OUT/w_deploy.npz 2>&1 | grep -E "weighted|^step 20000|Error|Traceback" | tail -2
for cond in friction0.15 motor0.4 nominal; do for s in u77 dr77; do
  run $CSV D_$s go1_paper $cond prior.student=student_$s policy.use_as=student_act
  run $CSV B0_$s go1_paper $cond prior.student=student_$s
done; done
echo "### stage4 done $(date)"
