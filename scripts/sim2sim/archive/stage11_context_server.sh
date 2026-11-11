#!/usr/bin/env bash
# Design doc 13 (docs/design_context_wm_ko.md) on the lab server (gpu1, 4x RTX 2080 Ti 11 GB).
# Same recipe as stage11_context.sh, but: no waiting for stage 9, no stage10 pause/resume,
# and the evaluations are spread over the GPUs listed in $GPUS_LIST (one queue per GPU).
# Usage (in tmux):  GPUS_LIST="0 1 2 3" bash scripts/sim2sim/stage11_context_server.sh
# Results: results/stage11/eval_context.csv, context_estimates.csv (merged from per-GPU files).
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage11; mkdir -p $OUT
GPUS_LIST=${GPUS_LIST:-"0"}; G0=$(echo $GPUS_LIST | awk '{print $1}')
export SIMDIST_DIR=~/simdist
echo "### context(server) start $(date) GPUs: $GPUS_LIST"

# ---- 1. data (one GPU) ----
EXPERT=2026-10-20_07-40-06
DSF=$OUT/dataset.txt
if [ ! -f $DSF ]; then
  before=$(ls datasets/sim 2>/dev/null | sort | tail -1); echo "== data $(date +%H:%M)"
  GPUS="\"device=$G0\"" $RUN python scripts/generate_data.py rl_run=$EXPERT system=go1dr envs=384 steps=50000 2>&1 | grep -E "Error|Traceback|steps will be" | head -3
  ds=$(ls datasets/sim | sort | tail -1); [ -n "$ds" ] && [ "$ds" != "$before" ] || { echo "no dataset"; exit 1; }
  GPUS="\"device=$G0\"" $RUN python scripts/process_data.py dataset_name=$ds system=go1dr 2>&1 | grep -E "Error|Traceback" | head -2
  [ -d datasets/sim/$ds/processed_data_go1dr_H-25_T-25 ] && echo $ds > $DSF || { echo "process failed"; exit 1; }
  rm -f datasets/sim/$ds/raw_data.hdf5
fi
DS=$(cat $DSF); echo "== dataset $DS"

# ---- 2. models: the two trainings run in parallel on two GPUs if available ----
train() { local gpu=$1 name=$2; shift 2
  ls -d checkpoints/models/$name/[0-9]* >/dev/null 2>&1 && return 0
  echo "== train $name on gpu$gpu $(date +%H:%M)"
  GPUS="\"device=$gpu\"" $RUN python scripts/finetune_model.py data.dataset_name=$DS system=go1dr checkpoint.resume_checkpoint=go1_paper run_name=$name \
    loss=world_model heads=all training.max_steps=60000 training.batch_size=128 training.warmup_steps=2000 training.decay_steps=60000 \
    training.end_learning_rate=0.00005 training.eval_interval=10000 "$@" 2>&1 | grep -E "Steps: |Error|Traceback|Added" | cut -c1-220 | tail -3
  ls -d checkpoints/models/$name/[0-9]* >/dev/null 2>&1 || { echo "train $name failed"; exit 1; }
}
G1=$(echo $GPUS_LIST | awk '{print ($2!="")?$2:$1}')
train $G0 go1_ctx adapt.context.enabled=true &
[ "$G1" != "$G0" ] && sleep 60   # stagger the Isaac Sim start-up
train $G1 go1_broadwm &
wait
ls -d checkpoints/models/go1_ctx/[0-9]* checkpoints/models/go1_broadwm/[0-9]* >/dev/null 2>&1 || { echo "training failed"; exit 1; }

# ---- 3. evaluations: one queue per GPU, per-GPU CSVs merged at the end ----
phys() { case $1 in nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";;
  motor0.4) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";;
  motor0.5) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.5";;
  motor0.6) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.6";;
  friction0.15) echo "sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
SEEDS="[11,12,13,14,15,16]"
run() { local gpu=$1 name=$2 ckpt=$3 cond=$4; shift 4
  local csv=$OUT/eval_context_gpu$gpu.csv ctx=$OUT/context_estimates_gpu$gpu.csv
  if cat $OUT/eval_context*.csv 2>/dev/null | grep -q "^$name,$cond,"; then return; fi
  echo "== $name on $cond ($*) gpu$gpu $(date +%H:%M)"
  GPUS="\"device=$gpu\"" $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$csv output.model_name=$name output.condition=$cond \
    $(echo "$*" | sed "s#CTXCSV#output.context_csv=/workspace/simdist/$ctx#") --headless 2>&1 | grep -E "Traceback|rror:|switched|Oracle" | head -4
}
# job list in priority order (P1/P3/P2 first, then P4, then reference points / P5)
JOBS=(
 "C_ctx_planner go1_ctx motor0.4 CTXCSV"
 "C_broadwm_planner go1_broadwm motor0.4"
 "C_ctx_oracle_planner go1_ctx motor0.4 sim.context_oracle=true"
 "C_ctx_planner go1_ctx nominal CTXCSV"
 "C_broadwm_planner go1_broadwm nominal"
 "C_ctx_oracle_planner go1_ctx nominal sim.context_oracle=true"
 "C_ctx_switch go1_ctx motor0.4 sim.motor_switch_step=2500 sim.motor_switch_strength=0.8 CTXCSV"
 "C_broadwm_switch go1_broadwm motor0.4 sim.motor_switch_step=2500 sim.motor_switch_strength=0.8"
 "C_ctx_planner go1_ctx motor0.5 CTXCSV"
 "C_broadwm_planner go1_broadwm motor0.5"
 "C_ctx_planner go1_ctx motor0.6 CTXCSV"
 "C_broadwm_planner go1_broadwm motor0.6"
 "C_ctx_planner go1_ctx friction0.15 CTXCSV"
 "C_broadwm_planner go1_broadwm friction0.15"
 "C_ctx_oracle_planner go1_ctx friction0.15 sim.context_oracle=true"
 "C_ctx_alone go1_ctx motor0.4 control.prior_only=true"
)
gpus=($GPUS_LIST); n=${#gpus[@]}
for ((q=0; q<n; q++)); do
  ( for ((j=q; j<${#JOBS[@]}; j+=n)); do run ${gpus[$q]} ${JOBS[$j]}; done ) &
  sleep 60
done
wait
# merge (header once)
for kind in eval_context context_estimates; do
  files=$(ls $OUT/${kind}_gpu*.csv 2>/dev/null); [ -n "$files" ] || continue
  head -1 $(echo $files | awk '{print $1}') > $OUT/$kind.csv
  for f in $files; do tail -n +2 $f >> $OUT/$kind.csv; done
done
echo "### context(server) done $(date)"
