#!/usr/bin/env bash
# Design doc 7, X2: planner ablation (noise schedule, consistency gate, both) over students {w10 narrow, w03 broad}
# x conditions {nominal, motor0.4}, pretrained dynamics go1_paper, 18 robots x 100 s. Baseline rows = B0_* in
# results/stage1/eval_width.csv (same seeds/speeds). Results: results/stage2/eval_planner.csv.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage2; mkdir -p $OUT; CSV=$OUT/eval_planner.csv; PRE=go1_paper
phys() { case $1 in nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";; motor0.4) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";; esac; }
SEEDS="[11,12,13,14,15,16]"
run() { local name=$1 ckpt=$2 cond=$3; shift 3
  if [ -f $CSV ] && grep -q "^$name,$cond," $CSV; then return; fi
  echo "== $name on $cond ($*) $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$CSV output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:|gate" | head -3
}
SCHED="control.noise_schedule.enabled=true"; GATE="control.consistency_gate.enabled=true control.consistency_gate.factor=3.0"
echo "### planner ablation start $(date)"
for cond in nominal motor0.4; do for w in w10 w03; do
  run B0sched_$w $PRE $cond prior.student=student_$w $SCHED
  run B0gate_$w  $PRE $cond prior.student=student_$w $GATE
  run B0both_$w  $PRE $cond prior.student=student_$w $SCHED $GATE
done; done
echo "### planner ablation done $(date)"
