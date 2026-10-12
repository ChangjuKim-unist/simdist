#!/usr/bin/env bash
# One real-robot adaptation round for the Go1 (SimDist Algorithm 1), run from the host after a
# collection session:
#   aggregate raw episodes -> process windows -> finetune from the previous checkpoint
#   -> (optional) simulation sanity check -> point the ROS controller at the new checkpoint
#   -> back up the raw data and the new checkpoint
#
#   PREV=go1_finetuned_r2 NEW=go1_finetuned_r3 scripts/go1_round.sh
#
# Environment:
#   PREV     checkpoint to continue from                     (required)
#   NEW      name of the new checkpoint                      (required)
#   DATASET  real dataset name                               (default: go1_simdist_controller)
#   EPOCHS   finetuning epochs over the aggregated data      (default: 40)
#   RECIPE   dyn  = latent dynamics only (SimDist)           (default)
#            res  = dynamics + transformer residual value head, rewards from the frozen reward head
#   EXTRA    extra hydra overrides for the finetune
#   SANITY   1 = also deploy NEW in simulation (plane, 1 seed) and print its reward (default: 1)
#   NO_ROS   1 = do not touch the ROS controller config / container
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$HOME/simdist-docker/run.sh"
DATASET="${DATASET:-go1_simdist_controller}"
EPOCHS="${EPOCHS:-40}"
RECIPE="${RECIPE:-dyn}"
EXTRA="${EXTRA:-}"
SANITY="${SANITY:-1}"
NO_ROS="${NO_ROS:-0}"
: "${PREV:?set PREV=<checkpoint to continue from>}"
: "${NEW:?set NEW=<name of the new checkpoint>}"

[ -d "$REPO/checkpoints/models/$PREV" ] || { echo "no checkpoint $PREV" >&2; exit 1; }
[ -d "$REPO/checkpoints/models/$NEW" ] && { echo "checkpoint $NEW already exists" >&2; exit 1; }
if docker ps --format '{{.Image}} {{.Names}}' | grep -q "^simdist"; then
  echo "another simdist container is running (sim experiment?); stop it first:" >&2
  docker ps --format '  {{.Names}}  {{.Image}}  {{.Status}}' | grep simdist >&2
  exit 1
fi

case "$RECIPE" in
  dyn) RECIPE_OVR="" ;;
  res) RECIPE_OVR="loss=world_model_value_adapt heads=dynamics_value_residual loss.bootstrap=self loss.reward_source=model loss.weights.value_reg=0.0 adapt.value_residual.enabled=true adapt.value_residual.type=transformer adapt.residual_lr=1e-3" ;;
  *) echo "unknown RECIPE $RECIPE" >&2; exit 1 ;;
esac

n_raw=$(find "$REPO/datasets/real/$DATASET/raw" -name '*.hdf5' | wc -l)
echo "== $n_raw raw episodes in datasets/real/$DATASET/raw"

echo "== aggregate"
"$RUN" python scripts/aggregate_realworld_data.py dataset_name="$DATASET" 2>&1 | grep -vE "Warning|absl" | tail -5
echo "== process"
"$RUN" python scripts/process_data.py dataset_name="$DATASET" system=go1 2>&1 | grep -vE "Warning|absl" | tail -3
cat "$REPO/datasets/real/$DATASET/processed_data_go1_H-25_T-25/dataset_metrics.json"; echo

echo "== finetune $PREV -> $NEW ($RECIPE, $EPOCHS epochs)"
"$RUN" python scripts/finetune_model.py data.dataset_name="$DATASET" system=go1 \
  checkpoint.resume_checkpoint="$PREV" run_name="$NEW" training.num_epochs="$EPOCHS" $RECIPE_OVR $EXTRA 2>&1 \
  | grep -E "Parameters to be optimized|Residual|^Steps: |Stopping|Error|Traceback" | cut -c1-300
ls -d "$REPO/checkpoints/models/$NEW"/[0-9]* >/dev/null 2>&1 || { echo "finetune produced no checkpoint" >&2; exit 1; }

if [ "$SANITY" = "1" ]; then
  echo "== simulation sanity check of $NEW (plane, seed 42; go1_paper ≈ 41-42 reward, 1000/1000 steps)"
  "$RUN" python scripts/simulate_go2.py model.checkpoint="$NEW" sim.terrain=plane sim.seed=42 control.seed=42 \
    control.num_samples=200 control.iterations=3 control.num_elites=32 2>&1 | grep -E "^\{'total_reward" || echo "WARNING: simulation gave no result"
fi

if [ "$NO_ROS" != "1" ]; then
  cfg="$REPO/go2_ros2_ws/src/config/config/simdist_controller_go1.yaml"
  sed -i -E "s|^  checkpoint: .*|  checkpoint: $NEW  # set by scripts/go1_round.sh on $(date +%F) (previous: $PREV)|" "$cfg"
  echo "== controller checkpoint set to $NEW in $(basename "$cfg")"
  if docker ps --format '{{.Names}}' | grep -qx go2_ros2; then
    docker exec go2_ros2 bash -lc "cd /home/go2/simdist/go2_ros2_ws && source scripts/setup.sh >/dev/null 2>&1; colcon build --packages-select config 2>&1 | tail -1"
  else
    echo "   (ROS container not running: run 'colcon build --packages-select config' inside it before the next session)"
  fi
fi

stamp="$(date +%Y-%m-%d)"
backup="$HOME/simdist-docker/backup_go1_real_${NEW}_${stamp}.tgz"
echo "== backup -> $backup"
tar czf "$backup" -C "$REPO" "datasets/real/$DATASET/raw" "checkpoints/models/$NEW"
du -h "$backup" | cut -f1
echo "== done: $NEW"
