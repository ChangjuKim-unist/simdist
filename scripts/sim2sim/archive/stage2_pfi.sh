#!/usr/bin/env bash
# Design doc 7, X3: Prior-First Iteration, two rounds in sim-to-sim, per condition.
#   round k: collect 45k steps with (pi_k + planner + f_k) -> f_{k+1} = residual adaptation on that data
#            -> pi_{k+1} = distill the executed actions into pi_k (beta 0.3, sim windows mixed)
#            -> evaluate D_{k+1} (student alone) and B_{k+1} (student + planner + f_{k+1}).
# Round 0 = student_broad, go1_paper; the round-1 data already exists (sim2sim_<cond>_priorfirst).
# NOTE 2026-10-27: the residual (dynres) adapted on prior-first data collapsed under the planner (X1: 9.7 falls/1000),
# so the rounds use the full dynamics adaptation (dyn, E2's B' = 0.93) until the residual variant is fixed (X1b).
# Control P3d: distill from the narrow-prior data instead (student_pfi1n_<cond>).
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; P=scripts/sim2sim/run_pipeline.sh; OUT=results/stage2; mkdir -p $OUT; CSV=$OUT/eval_pfi.csv
SIMDS=$(cat results/stage0/dataset_broad)   # the broad student's simulation dataset (go1dr)
phys() { case $1 in very_weak_motor) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";; nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
SEEDS="[11,12,13,14,15,16]"
run() { local name=$1 ckpt=$2 cond=$3; shift 3
  if [ -f $CSV ] && grep -q "^$name,$cond," $CSV; then return; fi
  echo "== $name on $cond ($*) $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$CSV output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:" | head -3
}
distill() { local name=$1 init=$2 real=$3   # name init real_dataset
  [ -f checkpoints/students/$name/params.pkl ] && return
  echo "== distill $name <- $init + $real $(date +%H:%M)"
  $RUN python scripts/distill_planner_to_student.py --init $init --name $name --real $real --sim $SIMDS --sim_system go1dr --beta 0.3 --steps 5000 2>&1 | grep -E "windows|step   4999|saved|Error|Traceback" | cut -c1-160
}
echo "### PFI start $(date)"
for COND in very_weak_motor nominal; do
  # ---- round 1 (data: prior-first collection by student_broad + planner + go1_paper)
  D1=sim2sim_${COND}_priorfirst; F1=go1_paper_${COND}_dyn_b45000_pf   # f_1 = E2's B' checkpoint (full adaptation)
  ls -d checkpoints/models/$F1/[0-9]* >/dev/null 2>&1 || COND=$COND DATASET=$D1 TAG=pf BUDGETS=45000 VARIANTS=dyn FT_STEPS=3000 $P finetune 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-160
  distill student_pfi1_$COND student_broad ${D1}_b45000
  distill student_pfi1n_$COND student_broad sim2sim_${COND}_go1_paper_b45000      # control: narrow-prior data
  run D1_pfi   go1_paper $COND prior.student=student_pfi1_$COND policy.use_as=student_act
  run D1n_ctrl go1_paper $COND prior.student=student_pfi1n_$COND policy.use_as=student_act
  run B1_pfi   $F1 $COND prior.student=student_pfi1_$COND
  # ---- round 2 (collect with pi_1 + planner + f_1)
  D2=sim2sim_${COND}_pfi2
  [ -d datasets/real/$D2/raw ] || COND=$COND DATASET=$D2 COLLECT_CKPT=$F1 PRIOR_STUDENT=student_pfi1_$COND ROUNDS=3 COLLECT_STEPS=5000 $P collect 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-160
  [ -d datasets/real/${D2}_b45000/processed_data_go1_H-25_T-25 ] || COND=$COND DATASET=$D2 BUDGETS=45000 $P process 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-160
  F2=go1_paper_${COND}_dyn_b45000_pfi2
  ls -d checkpoints/models/$F2/[0-9]* >/dev/null 2>&1 || COND=$COND DATASET=$D2 TAG=pfi2 BUDGETS=45000 VARIANTS=dyn FT_STEPS=3000 $P finetune 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-160
  distill student_pfi2_$COND student_pfi1_$COND ${D2}_b45000
  run D2_pfi go1_paper $COND prior.student=student_pfi2_$COND policy.use_as=student_act
  run B2_pfi $F2 $COND prior.student=student_pfi2_$COND
done
echo "### PFI done $(date)"
