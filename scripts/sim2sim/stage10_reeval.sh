#!/usr/bin/env bash
# Code review 2026-11-03 (design doc 12 §5): re-derive every paper number with the fixed evaluator / identifier.
#  1. identification logs re-collected under the fixed physics (torque cap = effort_limit only, nominal gains, 100 m slab)
#     and re-identified (joint-order fix, cross-segment demand weighting, bootstrap / z-score posterior)
#  2. SimDist dynamics-only adaptation on the motor axis re-done (its 45k data were collected under the old eval physics)
#  3. core heads re-evaluated (num_falls = contact/orientation terminations only)
#  4. stage 9 (rule width) continues on the new numbers
#  5. width sweep (3 seeds) re-evaluated last (optional; kill if the hardware week needs the GPU)
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; P=scripts/sim2sim/run_pipeline.sh; OUT=results/stage10; mkdir -p $OUT; CSV=$OUT/eval_core.csv
until [ -z "$(docker ps -q)" ]; do sleep 60; done   # let a running container finish first
echo "### reeval start $(date)"

collect() { local cond=$1 ds=$2 ckpt=$3 rounds=$4
  [ -d datasets/real/$ds/raw ] || COND=$cond DATASET=$ds COLLECT_CKPT=$ckpt ROUNDS=$rounds COLLECT_STEPS=5000 $P collect 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-120; }
process() { local cond=$1 ds=$2 budget=$3
  [ -d datasets/real/${ds}_b$budget/processed_data_go1_H-25_T-25 ] || COND=$cond DATASET=$ds BUDGETS=$budget $P process 2>&1 | grep -E "Error|Traceback" | head -2; }
ident() { local ds=$1 out=$2; shift 2
  [ -f $out ] || { echo "== identify $ds -> $out $(date +%H:%M)"; $RUN python scripts/sim2sim/identify_physics.py dataset=$ds out=$out "$@" --headless 2>&1 | grep -E "MAP|articulation|Unitree|Error|Traceback" | cut -c1-200; }; }

# ---- 1. identification curve (5-minute logs, walking prior = broadhead, SimDist prior for contrast) ----
for s in 0.3 0.5 0.6 0.8; do for spec in "go1_paper_broadhead:bh" "go1_paper:sd"; do IFS=: read ck tag <<< "$spec"
  ds=sim2sim_motor${s}_${tag}_5min_v2; collect motor$s $ds $ck 1; process motor$s $ds 15000
  ident ${ds}_b15000 $OUT/id_${tag}_s$s.json replicas=24 rounds=100
done; done
# friction 0.15 (expected flat) and payload +5 kg mass axis, broadhead logs
ds=sim2sim_friction0.15_bh_5min_v2; collect friction0.15 $ds go1_paper_broadhead 1; process friction0.15 $ds 15000
ident ${ds}_b15000 $OUT/id_bh_friction0.15.json "grid.lo=0.8" "grid.hi=1.1" "grid.n=4" "friction_grid=[0.15,0.3,0.5,0.8,1.0]" "base_channel=true" replicas=8 rounds=100
ds=sim2sim_payload5_bh_5min_v2; collect payload5 $ds go1_paper_broadhead 1; process payload5 $ds 15000
ident ${ds}_b15000 $OUT/id_bh_payload5.json "grid.lo=0.8" "grid.hi=1.1" "grid.n=4" "mass_grid=[0.0,1.5,3.0,4.5,6.0,7.5]" "base_channel=true" replicas=8 rounds=100

# ---- 2. SimDist dynamics-only adaptation on motor 0.4 (design doc 1 result), re-collected under the fixed physics ----
DS=sim2sim_very_weak_motor_go1_paper_v2
[ -d datasets/real/$DS/raw ] || COND=very_weak_motor DATASET=$DS COLLECT_CKPT=go1_paper ROUNDS=3 COLLECT_STEPS=5000 $P collect 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-120
process very_weak_motor $DS 45000
CKM=go1_paper_very_weak_motor_dyn_b45000_term_v2   # run_pipeline names it ${BASE}_${COND}_${VARIANT}_b${BUDGET}_${TAG}
ls -d checkpoints/models/$CKM/[0-9]* >/dev/null 2>&1 || COND=very_weak_motor DATASET=$DS TAG=term_v2 BUDGETS=45000 VARIANTS=dyn FT_STEPS=3000 $P finetune 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-120

# ---- 3. core evaluations ----
phys() { case $1 in nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";;
  motor0.4) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";;
  motor0.5) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.5";;
  motor0.6) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.6";;
  friction0.15) echo "sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0";;
  payload5) echo "sim.friction=0.8 sim.add_mass=5.0 sim.motor_strength=1.0";; esac; }
SEEDS="[11,12,13,14,15,16]"
run() { local csv=$1 name=$2 ckpt=$3 cond=$4; shift 4
  if [ -f $csv ] && grep -q "^$name,$cond," $csv; then return; fi
  echo "== $name on $cond ($*) $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$csv output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:" | head -3
}
# headline first
for c in motor0.4 nominal; do run $CSV A_simdist_planner go1_paper $c; run $CSV H_targethead_planner go1_paper_targethead $c; run $CSV H_broadhead_planner go1_paper_broadhead $c; done
run $CSV A_simdist_adapted_motor $CKM motor0.4
for c in motor0.4 nominal; do run $CSV H_targethead_s2_planner go1_paper_targethead_s2 $c; run $CSV H_targethead_alone go1_paper_targethead $c control.prior_only=true; done
for c in motor0.5 motor0.6 friction0.15; do run $CSV A_simdist_planner go1_paper $c; run $CSV H_targethead_planner go1_paper_targethead $c; run $CSV H_broadhead_planner go1_paper_broadhead $c; done
for c in friction0.15 nominal; do run $CSV A_simdist_adapted_friction go1_paper_friction0.15_dyn_b45000_term $c; done
run $CSV H_targethead_alone go1_paper_targethead motor0.5 control.prior_only=true

# ---- 4. stage 9 (E1 evals, k, rule-width heads, friction blanket) on the new evaluator ----
bash scripts/sim2sim/stage9_rule_width.sh

# ---- 5. width sweep re-evaluation (students, 3 seeds): D = student alone at motor 0.4 and nominal ----
CSVW=$OUT/eval_width.csv
for tag in w10 w09 w07 w05 w03 w03s2 w05s2 w07s2 w03s3 w05s3 w07s3; do
  [ -f checkpoints/students/student_$tag/params.pkl ] || continue
  for c in motor0.4 nominal; do run $CSVW D_$tag go1_paper $c prior.student=student_$tag policy.use_as=student_act; done
done
echo "### reeval done $(date)"
