#!/bin/bash
# Design doc 34 (docs/design34_condition_screen_ko.md): mid-difficulty conditions for the paper model.
#   A  screen, no training: policy head alone (POL) and pretrained planner (PRE) on every candidate level + nominal
#   S  pick one level per family with the registered rule (scripts/sim2sim/stage34_select.py -> chosen.txt)
#   B  SimDist adaptation on each chosen level: collect 45000 steps with the pretrained planner, dynamics-only
#      fine-tune (3000 steps, seed 42), then the adapted planner on its own condition (ADP) and on nominal (ADPnom)
# Protocol: 18 robots (3 speeds x 6 seeds) x 5000 steps, flat ground, as designs 29-33.
# usage (gpu1): tmux new -d -s s34 'bash scripts/sim2sim/stage34.sh > results/stage34_server.log 2>&1'
# SMOKE=1: two levels, 150-step runs, 30-step fine-tune (local check, not a result)
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; P=scripts/sim2sim/run_pipeline.sh
BIG=${BIG:-/mnt/newssd/cjkim/simdist/results/stage34}; mkdir -p $BIG
GPU_LIST=${GPU_LIST:-"0 1 2 3"}; SMOKE=${SMOKE:-0}; M=go1_paper
CONDS="legF70 legF50 legR50 delay1 delay2 slope3 slope5 tow10 tow20 mot50"
TOTAL=5000; FT_STEPS=3000; ROUNDS=3; CSTEPS=5000; BUDGET=45000; SEEDS="[11,12,13,14,15,16]"; TAG=s42
[ $SMOKE = 1 ] && { CONDS="delay2 tow20"; TOTAL=150; FT_STEPS=30; ROUNDS=1; CSTEPS=400; BUDGET=1200; SEEDS="[11]"; TAG=smk; }
phys() { BASE=$M COND=$1 bash -c "source <(sed -n '/^condition_overrides()/,/^}/p' $P); condition_overrides $1"; }
FILT="Traceback|rror:|^condition:|falls written|diagnostics written|Killed|OOM|RESOURCE_EXHAUSTED"
LOCK=$BIG/queue.lock
evalrun() { local g=$1 name=$2 ck=$3 c=$4 extra=$5
  [ -f $BIG/eval_$name.csv ] && { echo "== $name exists"; return; }
  echo "== $name gpu$g $ck $(date +%H:%M)"
  GPUS="\"device=$g\"" $RUN python scripts/simulate_batch.py model.checkpoint=$ck $extra $(phys $c) sim.terrain=plane sim.episode_length_s=100 \
    "batch.speeds=[0.3,0.6,0.9]" "batch.seeds=$SEEDS" sim.total_steps=$TOTAL output.csv=$BIG/eval_$name.csv output.falls_csv=$BIG/falls_$name.csv \
    output.diag_npz=$BIG/diag_$name.npz output.model_name=$name output.condition=$c --headless 2>&1 | grep -aE "$FILT" | head -8
  echo "-- $name done gpu$g $(date +%H:%M)"; }
echo "### stage34 start $(date)"

# ---- phase A: screen (queue line: run name|arm|condition) -----------------------------------------------------------
Q=$BIG/queue.txt; : > $Q
for c in nominal $CONDS; do for a in POL PRE; do echo "${c}_${a}|$a|$c" >> $Q; done; done
a_worker() { local g=$1
  while true; do
    local job; job=$(flock $LOCK bash -c "head -1 $Q; sed -i 1d $Q"); [ -z "$job" ] && break
    IFS='|' read -r name a c <<< "$job"
    [ $a = POL ] && evalrun $g $name $M $c "control.prior_only=true" || evalrun $g $name $M $c "control.prior_only=false"
  done; }
for g in $GPU_LIST; do a_worker $g & sleep 20; done; wait
echo "== phase A done $(date +%H:%M)"

# ---- phase S: registered selection rule -----------------------------------------------------------------------------
GPUS="\"device=$(echo $GPU_LIST | cut -d' ' -f1)\"" $RUN python scripts/sim2sim/stage34_select.py --d $BIG --conds "$CONDS" 2>&1 | grep -aE "^chosen|^family|rror|Traceback"
[ -s $BIG/chosen.txt ] || { echo "!! no chosen.txt"; exit 1; }

# ---- phase B: SimDist adaptation on the chosen levels (queue line: condition) ---------------------------------------
BQ=$BIG/b_queue.txt; awk '{print $2}' $BIG/chosen.txt > $BQ
b_worker() { local g=$1
  while true; do
    local c; c=$(flock $LOCK bash -c "head -1 $BQ; sed -i 1d $BQ"); [ -z "$c" ] && break
    local ds=sim2sim_${c}_${M}; local ck=${M}_${c}_dyn_b${BUDGET}_${TAG}
    echo "== adapt $c gpu$g $(date +%H:%M)"
    export GPUS="\"device=$g\""
    [ -d datasets/real/$ds/raw ] || BASE=$M COND=$c DATASET=$ds COLLECT_CKPT=$M ROUNDS=$ROUNDS COLLECT_STEPS=$CSTEPS $P collect 2>&1 | grep -aE "^==|Error|Traceback" | cut -c1-160
    BASE=$M COND=$c DATASET=$ds BUDGETS=$BUDGET $P process 2>&1 | grep -aE "^==|Total|Error|Traceback" | cut -c1-200
    BASE=$M RESUME_CKPT=$M COND=$c DATASET=$ds TAG=$TAG BUDGETS=$BUDGET VARIANTS=dyn FT_STEPS=$FT_STEPS FT_EXTRA="training.seed=42" $P finetune 2>&1 | grep -aE "^==|optimized|Steps: |Error|Traceback" | cut -c1-260
    ls -d checkpoints/models/$ck/[0-9]* >/dev/null 2>&1 || { echo "!! $ck missing after the fine-tune"; continue; }
    echo "-- adapt $c done $(date +%H:%M)"
    evalrun $g ${c}_ADP $ck $c "control.prior_only=false"
    evalrun $g ${c}_ADPnom $ck nominal "control.prior_only=false"
  done; }
for g in $GPU_LIST; do b_worker $g & sleep 20; done; wait
echo "### stage34 done $(date)"
