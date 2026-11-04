#!/usr/bin/env bash
# Design doc 13 (docs/design_context_wm_ko.md): context slot psi.
# Waits for stage 9 (rule width) to finish, pauses the stage10 width-sweep re-evaluation, then:
#   1. physics-labelled broad-DR data (broadhead expert, go1dr record env: motor 0.3-1.1, mass -1..+3, friction 0.2-1.2)
#   2. go1_ctx (context slot) and go1_broadwm (same data, no slot) warm-started from go1_paper, identical settings
#   3. evaluations P1-P5 (fixed evaluator, 6 seeds x 3 speeds, 100 s) -> results/stage11/eval_context.csv
#   4. resumes stage10 (width sweep)
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage11; mkdir -p $OUT
CSV=$OUT/eval_context.csv; CTX=$OUT/context_estimates.csv; LOG=results/stage10_reeval.log
until grep -q "### rule-width done" $LOG 2>/dev/null; do sleep 120; done
for p in $(pgrep -f "stage10_reeval.s[h]"); do kill $p; done; sleep 3
for c in $(docker ps -q); do docker stop -t 5 $c; done
echo "### context start $(date)"

# ---- 1. data ----
EXPERT=2026-10-20_07-40-06   # broadhead expert (motor DR 0.3-1.1), design doc 6
DSF=$OUT/dataset.txt
if [ ! -f $DSF ]; then
  before=$(ls datasets/sim | sort | tail -1); echo "== data $(date +%H:%M)"
  $RUN python scripts/generate_data.py rl_run=$EXPERT system=go1dr envs=384 steps=50000 2>&1 | grep -E "Error|Traceback|steps will be" | head -3
  ds=$(ls datasets/sim | sort | tail -1); [ "$ds" != "$before" ] || { echo "no dataset"; exit 1; }
  $RUN python scripts/process_data.py dataset_name=$ds system=go1dr 2>&1 | grep -E "Error|Traceback" | head -2
  [ -d datasets/sim/$ds/processed_data_go1dr_H-25_T-25 ] && echo $ds > $DSF || { echo "process failed"; exit 1; }
  free=$(df --output=avail -BG datasets | tail -1 | tr -dc 0-9)
  [ "$free" -lt 40 ] && { echo "free ${free}G < 40G: deleting datasets/sim/$ds/raw_data.hdf5"; rm -f datasets/sim/$ds/raw_data.hdf5; }
fi
DS=$(cat $DSF); echo "== dataset $DS"

# ---- 2. models (same recipe; the only difference is adapt.context.enabled) ----
train() { local name=$1; shift
  ls -d checkpoints/models/$name/[0-9]* >/dev/null 2>&1 && return 0
  echo "== train $name $(date +%H:%M)"
  $RUN python scripts/finetune_model.py data.dataset_name=$DS system=go1dr checkpoint.resume_checkpoint=go1_paper run_name=$name \
    loss=world_model heads=all training.max_steps=60000 training.batch_size=128 training.warmup_steps=2000 training.decay_steps=60000 \
    training.end_learning_rate=0.00005 training.eval_interval=10000 "$@" 2>&1 | grep -E "Steps: |Error|Traceback|Added" | cut -c1-220 | tail -3
  ls -d checkpoints/models/$name/[0-9]* >/dev/null 2>&1 || { echo "train $name failed"; exit 1; }
}
train go1_ctx adapt.context.enabled=true
train go1_broadwm

# ---- 3. evaluations ----
phys() { case $1 in nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";;
  motor0.4) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";;
  motor0.5) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.5";;
  motor0.6) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.6";;
  friction0.15) echo "sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
SEEDS="[11,12,13,14,15,16]"
run() { local name=$1 ckpt=$2 cond=$3; shift 3
  if [ -f $CSV ] && grep -q "^$name,$cond," $CSV; then return; fi
  echo "== $name on $cond ($*) $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$CSV output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:|switched|Oracle" | head -4
}
CTXOPT=output.context_csv=/workspace/simdist/$CTX
# P1 / P3 / P2 (headline first)
for c in motor0.4 nominal; do
  run C_ctx_planner go1_ctx $c $CTXOPT
  run C_broadwm_planner go1_broadwm $c
  run C_ctx_oracle_planner go1_ctx $c sim.context_oracle=true
done
# P4: motor 0.4 -> 0.8 at step 2500 (estimate trace in $CTX)
run C_ctx_switch go1_ctx motor0.4 sim.motor_switch_step=2500 sim.motor_switch_strength=0.8 $CTXOPT
run C_broadwm_switch go1_broadwm motor0.4 sim.motor_switch_step=2500 sim.motor_switch_strength=0.8
# E4-style reference points and P5
for c in motor0.5 motor0.6 friction0.15; do run C_ctx_planner go1_ctx $c $CTXOPT; run C_broadwm_planner go1_broadwm $c; done
run C_ctx_oracle_planner go1_ctx friction0.15 sim.context_oracle=true
run C_ctx_alone go1_ctx motor0.4 control.prior_only=true
echo "### context done $(date)"

# ---- 4. resume stage10 (width sweep) ----
nohup bash scripts/sim2sim/stage10_reeval.sh >> $LOG 2>&1 &
