#!/usr/bin/env bash
# Gate 1 of docs/design_prior_refresh_ko.md: does the planner's search region explain the falls?
# All runs use the dynamics-only adapted model (dyn_b45000_term), batched evaluation with 18 robots.
set -u; cd ~/simdist
RUN=~/simdist-docker/run.sh
M=go1_paper_very_weak_motor_dyn_b45000_term
EXP="policy.rl_run=2026-10-01_10-38-36 policy.iteration=4999"
OUT=results/sim2sim/prior_gate1; mkdir -p $OUT; CSV=$OUT/eval_long_batch.csv
phys() { case $1 in very_weak_motor) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";; combo_hard) echo "sim.friction=0.3 sim.add_mass=3.0 sim.motor_strength=0.6";; nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
run() {  # run <name> <condition> <extra overrides...>
  local name=$1 cond=$2; shift 2
  if [ -f $CSV ] && grep -q "^$name,$cond," $CSV; then return; fi
  echo "== $name on $cond ($*) $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$M $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' 'batch.seeds=[11,12,13,14,15,16]' output.csv=/workspace/simdist/$CSV output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:" | head -3
}
echo "### gate1 start $(date)"
# target condition: baseline + manipulations (design doc 3, section 3)
run base            very_weak_motor
run M4_expert_init  very_weak_motor $EXP policy.use_as=init control.init_mode=external
run M1_zero         very_weak_motor control.init_mode=zero
run M1_prior        very_weak_motor control.init_mode=prior
run M3_prior0       very_weak_motor control.mixture_coef=0.0
run M3_prior50      very_weak_motor control.mixture_coef=0.5
run M2_std4         very_weak_motor control.init_std=4.0
run M2_std8         very_weak_motor control.init_std=8.0
run M2_elite16      very_weak_motor control.num_elites=16
# controls: nominal (forgetting / side effects) and combo_hard (negative control, P5)
run base            nominal
run M4_expert_init  nominal $EXP policy.use_as=init control.init_mode=external
run base            combo_hard
run M4_expert_init  combo_hard $EXP policy.use_as=init control.init_mode=external
echo "### gate1 done $(date)"
