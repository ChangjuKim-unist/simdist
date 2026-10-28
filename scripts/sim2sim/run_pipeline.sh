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
FT_EXTRA="${FT_EXTRA:-}"  # extra hydra overrides for every finetune, e.g. training.warmup_steps=200
EVAL_SEEDS="${EVAL_SEEDS:-42 7 123}"
NOMINAL_SEEDS="${NOMINAL_SEEDS:-$EVAL_SEEDS}"  # seeds for the nominal-condition (forgetting) evaluation
EVAL_STEPS="${EVAL_STEPS:-1000}"
TAG="${TAG:-}"            # optional suffix for checkpoint names (e.g. oracle)
EXPERT_RUN="${EXPERT_RUN:-}"   # RL run whose exported expert policy is evaluated as a reference
EXPERT_ITER="${EXPERT_ITER:-}"
LONG_STEPS="${LONG_STEPS:-5000}"; LONG_SPEEDS="${LONG_SPEEDS:-0.3 0.6 0.9}"; LONG_SEEDS="${LONG_SEEDS:-11 12}"
LONG_CONDS="${LONG_CONDS:-$COND}"; LONG_EXTRA_MODELS="${LONG_EXTRA_MODELS:-}"  # extra checkpoint names for evaluate_long
NOMINAL_DATASET="${NOMINAL_DATASET:-2026-10-02_01-18-59}"
TERRAIN="${TERRAIN:-plane}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="${RUN:-$HOME/simdist-docker/run.sh}"
RESULTS="${RESULTS:-$REPO/results/sim2sim/$COND}"
DATASET="${DATASET:-sim2sim_${COND}_${BASE}}"   # raw/processed dataset name (override for later rounds)
COLLECT_CKPT="${COLLECT_CKPT:-$BASE}"        # model that collects the data (round k: the model adapted in round k-1)
PRIOR_STUDENT="${PRIOR_STUDENT:-}"           # student prior used by the collecting planner (design doc 6, prior-first adaptation)
RESUME_CKPT="${RESUME_CKPT:-$BASE}"          # checkpoint the finetune starts from
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
    dyn_pw)      echo "loss=world_model_dynamics_only heads=world_model_dynamics_only loss.prior_weight=true" ;;
    dynres)      echo "loss=world_model_dynamics_only heads=dynamics_residual_only adapt.dynamics_residual.enabled=true" ;;
    dynres_na)   echo "loss=world_model_dynamics_only heads=dynamics_residual_only adapt.dynamics_residual.enabled=true adapt.dynamics_residual.action_input=false" ;;
    res)         echo "loss=world_model_value_adapt heads=dynamics_value_residual adapt.value_residual.enabled=true adapt.value_anchor=false" ;;
    res_anchor)  echo "loss=world_model_value_adapt heads=dynamics_value_residual adapt.value_residual.enabled=true adapt.value_anchor=true" ;;
    full)        echo "loss=world_model_value_adapt heads=dynamics_value_full adapt.value_anchor=false" ;;
    full_anchor) echo "loss=world_model_value_adapt heads=dynamics_value_full adapt.value_anchor=true" ;;
    # --- second round of variants ---
    full_self)   echo "loss=world_model_value_adapt heads=dynamics_value_full loss.bootstrap=self" ;;
    full_replay) echo "loss=world_model_value_adapt heads=dynamics_value_full loss.bootstrap=self loss.weights.value_replay=1.0 adapt.replay.dataset_name=$NOMINAL_DATASET" ;;
    res_self)    echo "loss=world_model_value_adapt heads=dynamics_value_residual loss.bootstrap=self loss.weights.value_reg=0.0 adapt.value_residual.enabled=true adapt.value_residual.type=transformer adapt.residual_lr=1e-3" ;;
    res_self_mlp) echo "loss=world_model_value_adapt heads=dynamics_value_residual loss.bootstrap=self loss.weights.value_reg=0.0 adapt.value_residual.enabled=true adapt.value_residual.type=mlp adapt.residual_lr=1e-3" ;;
    res_anchor_tf) echo "loss=world_model_value_adapt heads=dynamics_value_residual loss.bootstrap=anchor loss.weights.value_reg=0.0 adapt.value_residual.enabled=true adapt.value_residual.type=transformer adapt.value_anchor=true adapt.residual_lr=1e-3" ;;
    res_anchor_term) echo "loss=world_model_value_adapt heads=dynamics_value_residual loss.bootstrap=anchor loss.terminal.enabled=true loss.weights.value_reg=0.0 adapt.value_residual.enabled=true adapt.value_residual.type=transformer adapt.value_anchor=true adapt.residual_lr=1e-3" ;;
    res_anchor_term_replay) echo "loss=world_model_value_adapt heads=dynamics_value_residual loss.bootstrap=anchor loss.terminal.enabled=true loss.weights.value_reg=0.0 loss.weights.value_replay=1.0 adapt.replay.dataset_name=$NOMINAL_DATASET adapt.value_residual.enabled=true adapt.value_residual.type=transformer adapt.value_anchor=true adapt.residual_lr=1e-3" ;;
    full_ret)    echo "loss=world_model_value_adapt heads=dynamics_value_residual loss.bootstrap=anchor loss.terminal.enabled=true loss.weights.value_reg=0.0 loss.weights.value_replay=1.0 loss.weights.ret=1.0 adapt.replay.dataset_name=$NOMINAL_DATASET adapt.value_residual.enabled=true adapt.value_residual.type=transformer adapt.value_anchor=true adapt.residual_lr=1e-3" ;;
    full_bw)     echo "loss=world_model_value_adapt heads=dynamics_value_residual loss.bootstrap=anchor loss.terminal.enabled=true loss.weights.value_reg=0.0 loss.weights.value_replay=1.0 loss.boundary_weight=5.0 adapt.replay.dataset_name=$NOMINAL_DATASET adapt.value_residual.enabled=true adapt.value_residual.type=transformer adapt.value_anchor=true adapt.residual_lr=1e-3" ;;
    res_anchor_replay) echo "loss=world_model_value_adapt heads=dynamics_value_residual loss.bootstrap=anchor loss.weights.value_reg=0.0 loss.weights.value_replay=1.0 adapt.replay.dataset_name=$NOMINAL_DATASET adapt.value_residual.enabled=true adapt.value_residual.type=transformer adapt.value_anchor=true adapt.residual_lr=1e-3" ;;
    res_replay)  echo "loss=world_model_value_adapt heads=dynamics_value_residual loss.bootstrap=self loss.weights.value_reg=0.0 loss.weights.value_replay=1.0 adapt.replay.dataset_name=$NOMINAL_DATASET adapt.value_residual.enabled=true adapt.value_residual.type=transformer adapt.residual_lr=1e-3" ;;
    *) echo "unknown variant: $1" >&2; exit 1 ;;
  esac
}

sim() {  # sim <checkpoint> <condition> <seed> <steps> [extra overrides...]
  local ckpt="$1" cond="$2" seed="$3" steps="$4"; shift 4
  "$RUN" python scripts/simulate_go2.py model.checkpoint="$ckpt" $(condition_overrides "$cond") \
    sim.terrain="$TERRAIN" sim.seed="$seed" control.seed="$seed" sim.total_steps="$steps" ${PRIOR_STUDENT:+prior.student=$PRIOR_STUDENT} --headless "$@" 2>&1
}

stage_collect() {
  echo "== collect: $DATASET ($COND, $ROUNDS round(s) over speeds: $SPEEDS, $COLLECT_STEPS steps each)"
  local r i
  for r in $(seq 1 "$ROUNDS"); do
    i=0
    for v in $SPEEDS; do
      i=$((i + 1))
      sim "$COLLECT_CKPT" "$COND" "$((1000 + 10 * r + i))" "$COLLECT_STEPS" task.forward_vel="$v" \
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

ckpt_name() { echo "${BASE}_${COND}_${1}_b${2}${TAG:+_$TAG}"; }

stage_finetune() {
  for b in $BUDGETS; do
    for var in $VARIANTS; do
      local name; name="$(ckpt_name "$var" "$b")"
      if ls -d "$REPO/checkpoints/models/$name"/[0-9]* >/dev/null 2>&1; then echo "== finetune: $name exists, skipping"; continue; fi
      echo "== finetune: $name"
      "$RUN" python scripts/finetune_model.py data.dataset_name="${DATASET}_b${b}" system="$SYSTEM" \
        checkpoint.resume_checkpoint="$RESUME_CKPT" run_name="$name" training.max_steps="$FT_STEPS" \
        training.eval_interval="$FT_STEPS" $(variant_overrides "$var") $FT_EXTRA 2>&1 \
        | grep -E "Parameters to be optimized|Steps: |Stopping|Error|Traceback" | cut -c1-300 || true
    done
  done
}

stage_evaluate() {
  local csv="$RESULTS/eval.csv"
  [ -f "$csv" ] || echo "model,condition,seed,total_reward,reward_per_step,episode_length,forward_progress" > "$csv"
  local models="$BASE"
  for b in $BUDGETS; do for var in $VARIANTS; do models="$models $(ckpt_name "$var" "$b")"; done; done
  # the privileged RL expert acting directly (EXPERT_RUN / EXPERT_ITER): an upper
  # bound for shifts it observes (friction, mass), blind to motor strength
  [ -n "$EXPERT_RUN" ] && models="$models expert_${EXPERT_ITER}"
  for m in $models; do
    local ckpt="$m" extra=""
    if [[ "$m" == expert_* ]]; then
      ckpt="$BASE"; extra="policy.rl_run=$EXPERT_RUN policy.iteration=$EXPERT_ITER"
    else
      [ -d "$REPO/checkpoints/models/$m" ] || { echo "skip $m (no checkpoint)"; continue; }
    fi
    for cond in "$COND" nominal; do
      local seeds="$EVAL_SEEDS"; [ "$cond" = nominal ] && seeds="$NOMINAL_SEEDS"
      for seed in $seeds; do
        if grep -q "^$m,$cond,$seed," "$csv"; then continue; fi
        echo "== evaluate: $m on $cond seed $seed"
        local out; out="$(sim "$ckpt" "$cond" "$seed" "$EVAL_STEPS" $extra | grep -E "^\{'total_reward" || true)"
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

# long-horizon evaluation: LONG_STEPS-step runs (episodes end on a fall or after
# COLLECT_EPISODE_S) at each speed of LONG_SPEEDS; reports reward per step over the
# whole run and the number of resets (falls) -- much less seed-sensitive than a
# single 1000-step episode
stage_evaluate_long() {
  local csv="$RESULTS/eval_long.csv"
  [ -f "$csv" ] || echo "model,condition,speed,seed,run_reward_per_step,num_resets,first_episode_length,first_episode_reward" > "$csv"
  local models="$BASE"
  for b in $BUDGETS; do for var in $VARIANTS; do models="$models $(ckpt_name "$var" "$b")"; done; done
  [ -n "$EXPERT_RUN" ] && models="$models expert_${EXPERT_ITER}"
  for m in $models $LONG_EXTRA_MODELS; do
    local ckpt="$m" extra=""
    if [[ "$m" == expert_* ]]; then
      ckpt="$BASE"; extra="policy.rl_run=$EXPERT_RUN policy.iteration=$EXPERT_ITER"
    else
      [ -d "$REPO/checkpoints/models/$m" ] || { echo "skip $m (no checkpoint)"; continue; }
    fi
    for cond in $LONG_CONDS; do
      for v in $LONG_SPEEDS; do
        for seed in $LONG_SEEDS; do
          if grep -q "^$m,$cond,$v,$seed," "$csv"; then continue; fi
          echo "== evaluate_long: $m on $cond speed $v seed $seed"
          local out; out="$(sim "$ckpt" "$cond" "$seed" "$LONG_STEPS" task.forward_vel="$v" sim.episode_length_s="$COLLECT_EPISODE_S" $extra | grep -E "^\{'total_reward" || true)"
          local rr nr el r
          rr="$(echo "$out" | grep -oE "'run_reward_per_step': [-0-9.e]+" | grep -oE "[-0-9.e]+$" || true)"
          nr="$(echo "$out" | grep -oE "'num_resets': [0-9]+" | grep -oE "[0-9]+$" || true)"
          el="$(echo "$out" | grep -oE "'episode_length': [0-9]+" | grep -oE "[0-9]+$" || true)"
          r="$(echo "$out" | grep -oE "'total_reward': [-0-9.e]+" | grep -oE "[-0-9.e]+$" || true)"
          [ -n "$rr" ] || echo "WARNING: no result for $m ($cond, $v, $seed)" >&2
          echo "$m,$cond,$v,$seed,${rr:-nan},${nr:-nan},${el:-nan},${r:-nan}" | tee -a "$csv"
        done
      done
    done
  done
}

# batched long-horizon evaluation: one Isaac Sim instance per (model, condition) with a
# robot per (speed, seed) -> $RESULTS/eval_long_batch.csv (same columns as eval_long)
stage_evaluate_long_batch() {
  local csv="$RESULTS/eval_long_batch.csv"
  local models="$BASE"
  for b in $BUDGETS; do for var in $VARIANTS; do models="$models $(ckpt_name "$var" "$b")"; done; done
  [ -n "$EXPERT_RUN" ] && models="$models expert_${EXPERT_ITER}"
  local speeds seeds
  speeds="[$(echo $LONG_SPEEDS | tr ' ' ',')]"; seeds="[$(echo $LONG_SEEDS | tr ' ' ',')]"
  for m in $models $LONG_EXTRA_MODELS; do
    local ckpt="$m" extra=""
    if [[ "$m" == expert_* ]]; then
      ckpt="$BASE"; extra="policy.rl_run=$EXPERT_RUN policy.iteration=$EXPERT_ITER"
    else
      [ -d "$REPO/checkpoints/models/$m" ] || { echo "skip $m (no checkpoint)"; continue; }
    fi
    for cond in $LONG_CONDS; do
      if [ -f "$csv" ] && grep -q "^$m,$cond," "$csv"; then continue; fi
      echo "== evaluate_long_batch: $m on $cond"
      "$RUN" python scripts/simulate_batch.py model.checkpoint="$ckpt" $(condition_overrides "$cond") \
        sim.terrain="$TERRAIN" sim.seed=42 sim.total_steps="$LONG_STEPS" sim.episode_length_s="$COLLECT_EPISODE_S" \
        "batch.speeds=$speeds" "batch.seeds=$seeds" output.csv="/workspace/simdist/${csv#$REPO/}" \
        output.model_name="$m" output.condition="$cond" $extra --headless 2>&1 | grep -E "^RESULT|Traceback|rror:" | cut -c1-200
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
  evaluate_long) stage_evaluate_long ;;
  evaluate_long_batch) stage_evaluate_long_batch ;;
  forget)    stage_forget ;;
  summarize) stage_summarize ;;
  all)       stage_collect; stage_process; stage_finetune; stage_evaluate; stage_forget; stage_summarize ;;
  *) echo "usage: $0 {collect|process|finetune|evaluate|forget|summarize|all}" >&2; exit 1 ;;
esac
