#!/usr/bin/env bash
# Sim-to-sim adaptation experiment for the "selective value adaptation" study.
#
# A perturbed Isaac Sim world plays the role of the real world. The pipeline:
#   collect   roll out the pretrained world model + MPPI in the target condition and
#             log the episodes in the real-world data format (with simulator rewards)
#   process   aggregate the logs under a step budget and build the training windows
#   finetune  adapt the pretrained model with several recipes
#   evaluate  deploy every checkpoint in the target and the nominal condition
#   forget    value-forgetting metrics on the nominal simulation dataset
#
# Usage (from the host, uses simdist-docker/run.sh):
#   scripts/sim2sim/run_pipeline.sh <stage|all> [options]
# Options (environment variables):
#   BASE        pretrained checkpoint                         (default: go1_paper)
#   SYSTEM      system config                                 (default: go1)
#   COND        target condition name, see CONDITIONS below   (default: low_friction)
#   COLLECT_STEPS steps logged per collection run             (default: 5000 = 100 s at 50 Hz)
#   SPEEDS      forward velocities for collection runs        (default: "0.3 0.6 0.9")
#   ROUNDS      collection rounds over SPEEDS; runs are interleaved so that any
#               step-budget prefix contains every speed            (default: 1)
#   COLLECT_EPISODE_S episode time limit during collection [s]  (default: 100; evaluation keeps 20)
#   BUDGETS     real-data budgets in steps for finetuning     (default: "15000")
#   VARIANTS    finetune recipes, see VARIANTS below          (default: "dyn res res_anchor full full_anchor")
#   FT_STEPS    gradient steps per finetune                   (default: 3000)
#   EVAL_SEEDS  simulation seeds for evaluation               (default: "42 7 123")
#   EVAL_STEPS  steps per evaluation run                      (default: 1000)
#   NOMINAL_DATASET processed sim dataset for the forgetting metric (default: 2026-10-02_01-18-59)
#   RESULTS     results directory                             (default: results/sim2sim/<COND>)
set -euo pipefail

STAGE="${1:-all}"
BASE="${BASE:-go1_paper}"
SYSTEM="${SYSTEM:-go1}"
COND="${COND:-low_friction}"
COLLECT_STEPS="${COLLECT_STEPS:-5000}"
ROUNDS="${ROUNDS:-1}"
COLLECT_EPISODE_S="${COLLECT_EPISODE_S:-100}"
SPEEDS="${SPEEDS:-0.3 0.6 0.9}"
BUDGETS="${BUDGETS:-15000}"
VARIANTS="${VARIANTS:-dyn res res_anchor full full_anchor}"
FT_STEPS="${FT_STEPS:-3000}"
EVAL_SEEDS="${EVAL_SEEDS:-42 7 123}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
NOMINAL_DATASET="${NOMINAL_DATASET:-2026-10-02_01-18-59}"
TERRAIN="${TERRAIN:-plane}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="${RUN:-$HOME/simdist-docker/run.sh}"
RESULTS="${RESULTS:-$REPO/results/sim2sim/$COND}"
DATASET="sim2sim_${COND}_${BASE}"
mkdir -p "$RESULTS"
RESULTS="$(cd "$RESULTS" && pwd)"
# the same directory as seen inside the container (the repo is mounted at /workspace/simdist)
RESULTS_REL="${RESULTS#$REPO/}"

# ---- target conditions: hydra overrides for scripts/simulate_go2.py ------------
condition_overrides() {
  case "$1" in
    nominal)      echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0" ;;
    low_friction) echo "sim.friction=0.3 sim.add_mass=0.0 sim.motor_strength=1.0" ;;
    payload)      echo "sim.friction=0.8 sim.add_mass=3.0 sim.motor_strength=1.0" ;;
    weak_motor)   echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.6" ;;
    very_weak_motor) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4" ;;
    combo)        echo "sim.friction=0.4 sim.add_mass=2.0 sim.motor_strength=0.8" ;;
    combo_hard)   echo "sim.friction=0.3 sim.add_mass=3.0 sim.motor_strength=0.6" ;;
    *) echo "unknown condition: $1" >&2; exit 1 ;;
  esac
}

# ---- finetune recipes: hydra overrides for scripts/finetune_model.py ----------
variant_overrides() {
  case "$1" in
    dyn)         echo "loss=world_model_dynamics_only heads=world_model_dynamics_only" ;;
    res)         echo "loss=world_model_value_adapt heads=dynamics_value_residual adapt.value_residual.enabled=true adapt.value_anchor=false" ;;
    res_anchor)  echo "loss=world_model_value_adapt heads=dynamics_value_residual adapt.value_residual.enabled=true adapt.value_anchor=true" ;;
    full)        echo "loss=world_model_value_adapt heads=dynamics_value_full adapt.value_anchor=false" ;;
    full_anchor) echo "loss=world_model_value_adapt heads=dynamics_value_full adapt.value_anchor=true" ;;
    *) echo "unknown variant: $1" >&2; exit 1 ;;
  esac
}

sim() {  # sim <checkpoint> <condition> <seed> <steps> [extra overrides...]
  local ckpt="$1" cond="$2" seed="$3" steps="$4"; shift 4
  "$RUN" python scripts/simulate_go2.py model.checkpoint="$ckpt" $(condition_overrides "$cond") \
    sim.terrain="$TERRAIN" sim.seed="$seed" control.seed="$seed" sim.total_steps="$steps" --headless "$@" 2>&1
}

stage_collect() {
  echo "== collect: $DATASET ($COND, $ROUNDS round(s) over speeds: $SPEEDS, $COLLECT_STEPS steps each)"
  local r i
  for r in $(seq 1 "$ROUNDS"); do
    i=0
    for v in $SPEEDS; do
      i=$((i + 1))
      sim "$BASE" "$COND" "$((1000 + 10 * r + i))" "$COLLECT_STEPS" task.forward_vel="$v" \
        sim.episode_length_s="$COLLECT_EPISODE_S" \
        logging.enabled=true logging.dataset_name="$DATASET" | grep -E "^\{'total_reward|Error|Traceback" || true
    done
  done
}

stage_process() {
  for b in $BUDGETS; do
    local name="${DATASET}_b${b}"
    echo "== process: $name (budget $b steps)"
    local real_dir="$REPO/datasets/real/$name"
    mkdir -p "$real_dir"
    [ -e "$real_dir/raw" ] || ln -s "../$DATASET/raw" "$real_dir/raw"
    "$RUN" python scripts/aggregate_realworld_data.py dataset_name="$name" max_steps="$b" 2>&1 | grep -E "Total|Error|Traceback" || true
    "$RUN" python scripts/process_data.py dataset_name="$name" system="$SYSTEM" 2>&1 | grep -E "Error|Traceback" || true
    cat "$real_dir/processed_data_${SYSTEM}_H-25_T-25/dataset_metrics.json" 2>/dev/null | tr -d '\n' | cut -c1-200; echo
  done
}

ckpt_name() { echo "${BASE}_${COND}_${1}_b${2}"; }

stage_finetune() {
  for b in $BUDGETS; do
    for var in $VARIANTS; do
      local name; name="$(ckpt_name "$var" "$b")"
      echo "== finetune: $name"
      "$RUN" python scripts/finetune_model.py data.dataset_name="${DATASET}_b${b}" system="$SYSTEM" \
        checkpoint.resume_checkpoint="$BASE" run_name="$name" training.max_steps="$FT_STEPS" \
        training.eval_interval="$FT_STEPS" $(variant_overrides "$var") 2>&1 \
        | grep -E "Parameters to be optimized|Steps: |Stopping|Error|Traceback" | cut -c1-300 || true
    done
  done
}

stage_evaluate() {
  local csv="$RESULTS/eval.csv"
  [ -f "$csv" ] || echo "model,condition,seed,total_reward,reward_per_step,episode_length,forward_progress" > "$csv"
  local models="$BASE"
  for b in $BUDGETS; do for var in $VARIANTS; do models="$models $(ckpt_name "$var" "$b")"; done; done
  for m in $models; do
    [ -d "$REPO/checkpoints/models/$m" ] || { echo "skip $m (no checkpoint)"; continue; }
    for cond in "$COND" nominal; do
      for seed in $EVAL_SEEDS; do
        if grep -q "^$m,$cond,$seed," "$csv"; then continue; fi
        echo "== evaluate: $m on $cond seed $seed"
        local out; out="$(sim "$m" "$cond" "$seed" "$EVAL_STEPS" | grep -E "^\{'total_reward" || true)"
        local r rs el fp
        r="$(echo "$out" | grep -oE "'total_reward': [-0-9.e]+" | grep -oE "[-0-9.e]+$" || true)"
        rs="$(echo "$out" | grep -oE "'reward_per_step': [-0-9.e]+" | grep -oE "[-0-9.e]+$" || true)"
        el="$(echo "$out" | grep -oE "'episode_length': [0-9]+" | grep -oE "[0-9]+$" || true)"
        fp="$(echo "$out" | grep -oE "'forward_progress': [-0-9.e]+" | grep -oE "[-0-9.e]+$" || true)"
        [ -n "$r" ] || echo "WARNING: no result for $m on $cond seed $seed (simulation failed?)" >&2
        echo "$m,$cond,$seed,${r:-nan},${rs:-nan},${el:-nan},${fp:-nan}" | tee -a "$csv"
      done
    done
  done
}

stage_forget() {
  local adapted=""
  for b in $BUDGETS; do for var in $VARIANTS; do
    local n; n="$(ckpt_name "$var" "$b")"; [ -d "$REPO/checkpoints/models/$n" ] && adapted="$adapted $n"
  done; done
  [ -n "$adapted" ] || { echo "no adapted checkpoints"; return; }
  echo "== value forgetting on $NOMINAL_DATASET"
  "$RUN" python scripts/sim2sim/value_forgetting.py --base "$BASE" --adapted $adapted \
    --dataset "$NOMINAL_DATASET" --out "$RESULTS_REL/forgetting.json" 2>&1 | grep -vE "Warning|absl" | tail -40
}

stage_summarize() {
  "$RUN" python scripts/sim2sim/summarize.py "$RESULTS_REL" 2>&1 | grep -vE "Warning|absl"
}

case "$STAGE" in
  collect)   stage_collect ;;
  process)   stage_process ;;
  finetune)  stage_finetune ;;
  evaluate)  stage_evaluate ;;
  forget)    stage_forget ;;
  summarize) stage_summarize ;;
  all)       stage_collect; stage_process; stage_finetune; stage_evaluate; stage_forget; stage_summarize ;;
  *) echo "usage: $0 {collect|process|finetune|evaluate|forget|summarize|all}" >&2; exit 1 ;;
esac
