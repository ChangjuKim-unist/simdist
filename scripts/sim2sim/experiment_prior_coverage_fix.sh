#!/usr/bin/env bash
# Follow-up to experiment_prior_coverage.sh: (1) condition-matched dynamics for combo_hard
# (the first pass used the weak-motor-adapted model everywhere), (2) motor 0.25 instead of
# the infeasible 0.2 as the outside-DR condition (design doc 4, section 7).
set -u; cd ~/simdist
RUN=~/simdist-docker/run.sh
RL=2026-10-20_07-40-06
OUT=results/prior_coverage; CSV=$OUT/eval_long_batch.csv
DYNW=go1_paper_very_weak_motor_dyn_b45000_term; DYNC=go1_paper_combo_hard_dyn_b45000_term; PRE=go1_paper
OLDEXP="policy.rl_run=2026-10-01_10-38-36 policy.iteration=4999"; NEWEXP="policy.rl_run=$RL policy.iteration=4999"
phys() { case $1 in motor0.25) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.25";; combo_hard) echo "sim.friction=0.3 sim.add_mass=3.0 sim.motor_strength=0.6";; esac; }
SEEDS="[11,12,13,14,15,16]"
run() { local name=$1 ckpt=$2 cond=$3; shift 3
  if [ -f $CSV ] && grep -q "^$name,$cond," $CSV; then return; fi
  echo "== $name on $cond ($*) $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$CSV output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:" | head -3
}
echo "### fix start $(date)"
run F_expert_broad   $PRE  motor0.25 $NEWEXP policy.use_as=act
run A_base           $DYNW motor0.25
run D_student_broad  $DYNW motor0.25 prior.student=student_broad policy.use_as=student_act
run B_dyn_studentB   $DYNW motor0.25 prior.student=student_broad
run Ac_base_combodyn $DYNC combo_hard
run Bc_combodyn_studentB $DYNC combo_hard prior.student=student_broad
echo "### fix done $(date)"
